"""Exercise seed isolation and resumable gauntlet stages without costly attacks."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from security_tests.gauntlet import gauntlet


class GauntletSeedsTest(unittest.TestCase):
    def test_seed_cells_resume_independently_and_keep_unmixed_controls_once(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            binaries = root / "bin"
            binaries.mkdir()
            for name in ("gauntlet_gen", "gauntlet_audit"):
                (binaries / name).touch()
            outdir = root / "runs"
            arms = (
                "embedded_masking_balanced",
                "embedded_masking_shuffled",
                "embedded_masking_shuffled_carried",
            )
            calls = []

            def generate(**kwargs):
                calls.append((kwargs["arm"], kwargs["mix_seed"]))
                for artifact in gauntlet.generation_artifacts(kwargs["cdir"] / "bundle", "native"):
                    artifact.write_text("fixture\n")

            def audit(**kwargs):
                # A skipped attack stays visibly skipped through report regeneration.
                result = "RESULT a1_nt=0 xtrace_nt=0 xtrace_status=skipped-cap w1flag=0\n"
                (kwargs["cdir"] / "audit.log").write_text(result)
                (kwargs["cdir"] / "bundle.hits.jsonl").write_text("")
                return result

            def maps(cdir):
                destination = cdir / "heatmaps"
                destination.mkdir(exist_ok=True)
                for attack in ("a1", "xrows", "xtrace", "w1", "w2", "w3"):
                    (destination / f"{attack}.png").write_bytes(b"fixture")

            arguments = [
                "all", "--ks", "1", "--arms", ",".join(arms),
                "--mix", "both", "--mix-seeds", "777,778", "--jobs", "2",
                "--outdir", str(outdir), "--bin-dir", str(binaries),
            ]
            with patch.object(gauntlet, "generate_cell", side_effect=generate), \
                 patch.object(gauntlet, "audit_cell", side_effect=audit), \
                 patch.object(gauntlet, "maps_cell", side_effect=maps), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(gauntlet.main(arguments), 0)
                self.assertEqual(len(calls), 9)
                index = json.loads((outdir / "index.json").read_text())
                self.assertEqual(len(index), 9)
                for arm in arms:
                    self.assertEqual(
                        {row["mix_seed"] for row in index if row["arm"] == arm},
                        {None, 777, 778},
                    )
                self.assertIn("skipped-cap", (outdir / "REPORT.md").read_text())
                self.assertEqual(gauntlet.main(arguments), 0)
                self.assertEqual(len(calls), 9, "a current run must resume without regenerating")

                # Corruption invalidates only this seed/arm and its dependent stages.
                damaged = outdir / "k1" / "embedded_masking_shuffled_random_mix_seed778"
                (damaged / "bundle.trace.bin").write_bytes(b"corrupt")
                self.assertEqual(gauntlet.main(arguments), 0)
                self.assertEqual(calls[9:], [("embedded_masking_shuffled", 778)])
                saved = json.loads((damaged / "cell-config.json").read_text())["stages"]
                self.assertEqual(saved["generation"]["config"]["mix_seed"], 778)
                self.assertEqual(saved["generation"]["config"]["shuffling_segments"], 8)
                self.assertEqual(
                    saved["audit"]["config"]["generation_digest"], saved["generation"]["digest"]
                )
                self.assertEqual(saved["maps"]["config"]["audit_digest"], saved["audit"]["digest"])
                # Report generation must not relabel an audit from another seed.
                saved["generation"]["config"]["mix_seed"] = 779
                (damaged / "cell-config.json").write_text(json.dumps({"stages": saved}))
                with self.assertRaisesRegex(RuntimeError, "provenance does not match"):
                    gauntlet.main(["report", *arguments[1:]])

    def test_invalid_axes_fail_before_creating_output(self):
        for option, value in (
            ("--mix-seeds", "777,777"), ("--mix-seeds", "-1"),
            ("--mix-seeds", str(2**64)), ("--mix-seeds", ""),
            ("--mix-seeds", "777,"), ("--ks", "1,1"),
            ("--arms", "none,none"), ("--shuffling-segments", "7"),
        ):
            with self.subTest(option=option, value=value), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "not-created"
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    gauntlet.main(["gen", option, value, "--outdir", str(output)])
                self.assertEqual(error.exception.code, 2)
                self.assertFalse(output.exists())

    def test_native_generation_forwards_seed_and_shuffling_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            cdir = Path(temporary)
            command = []

            def run(argv, *args, **kwargs):
                command.extend(argv)
                (cdir / "bundle.meta").write_text("behavioral_ok\ttrue\nmix_seed\t123\n")

            with patch.object(gauntlet, "run", side_effect=run):
                gauntlet.generate_cell(
                    cdir=cdir, arm="embedded_masking_shuffled_carried", aux="random", k=1,
                    mix_on=True, mix_seed=123, shuffling_segments=16, chain=cdir / "chain",
                    gen_binary=cdir / "gen", correlation_samples=64, mix_moves=17,
                    pool_keys=120, rayon_threads=1,
                )
            self.assertEqual(command[command.index("--mix-seed") + 1], "123")
            self.assertEqual(command[command.index("--shuffling-segments") + 1], "16")
            self.assertIn("--shuffling-carry-layout", command)

    def test_file_probe_and_final_generation_use_the_same_mixer_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            cdir = Path(temporary)
            commands = []

            def run(argv, log_path, **kwargs):
                commands.append(argv)
                if "--size-only" in argv:
                    log_path.write_text("[size] gates=291\n")
                elif str(gauntlet.BUILD_SCRIPT) in argv:
                    (cdir / "bundle.buildmeta").write_text("builder_checked\ttrue\nn_wires\t121\n")
                else:
                    (cdir / "bundle.meta").write_text("behavioral_ok\ttrue\nmix_seed\t456\n")

            with patch.object(gauntlet, "run", side_effect=run):
                gauntlet.generate_cell(
                    cdir=cdir, arm="nonlinear291", aux="builder", k=1,
                    mix_on=True, mix_seed=456, shuffling_segments=8, chain=cdir / "chain",
                    gen_binary=cdir / "gen", correlation_samples=64, mix_moves=17,
                    pool_keys=120, rayon_threads=1,
                )
            generator_calls = [argv for argv in commands if argv[0] == str(cdir / "gen")]
            self.assertEqual(len(generator_calls), 2)
            for argv in generator_calls:
                self.assertEqual(argv[argv.index("--mix-seed") + 1], "456")


if __name__ == "__main__":
    unittest.main()
