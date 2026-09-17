import unittest

import numpy as np

from security_tests.gadgetization import nonlinear291


def decode_encoding(wires):
    """Decode E(x) = x0 + x1 + majority(x2, x3, x4)."""

    x0, x1, x2, x3, x4 = wires
    majority = (x2 + x3 + x4) >= 2
    return x0 ^ x1 ^ majority.astype(np.uint8)


def run_direct_dirty_gadget(scratch2):
    """Exercise the public emitter with caller-provided secondary scratch."""

    samples = 256
    rng = np.random.default_rng(71)
    target_share1 = tuple(range(0, 5))
    target_share2 = tuple(range(5, 10))
    fresh_share = tuple(range(10, 15))
    a_blocks = (tuple(range(15, 20)), tuple(range(20, 25)))
    b_blocks = (tuple(range(25, 30)), tuple(range(30, 35)))
    output_majority = tuple(range(35, 38))
    primary_scratch = 38
    scratch2_wires = (39, 40)
    chaff = tuple(range(41, 45))
    next_wire = 45

    decomposition = (next_wire, next_wire + 1)
    next_wire += 2
    persistent = tuple(range(next_wire, next_wire + 24))
    next_wire += 24
    temporary = (next_wire,)
    next_wire += 1

    wires = [
        rng.integers(0, 2, samples).astype(np.uint8)
        for _ in range(next_wire)
    ]
    for wire in output_majority + (primary_scratch,) + scratch2_wires:
        wires[wire] = np.zeros(samples, np.uint8)

    for wire in decomposition + persistent + temporary:
        wires[wire] = np.zeros(samples, np.uint8)
    circuit = nonlinear291.Weight2Circuit(
        wires,
        decomposition,
        set(),
        set(),
        persist=persistent,
        temp=temporary,
    )

    def decode(blocks):
        value = np.zeros(samples, np.uint8)
        for block in blocks:
            value ^= decode_encoding([circuit.s[wire] for wire in block])
        return value.astype(np.uint8)

    a = decode(a_blocks)
    b = decode(b_blocks)
    c_in = decode((target_share1, target_share2))
    expected = (c_in ^ 1 ^ b ^ (a & b)).astype(np.uint8)
    nonlinear291.gadget_gate(
        circuit,
        a_blocks,
        b_blocks,
        target_share1,
        target_share2,
        fresh_share,
        output_majority,
        primary_scratch,
        scratch2,
        chaff,
        dirty=True,
    )
    output_share2 = (
        target_share2[0],
        target_share2[1],
        *output_majority,
    )
    actual = (
        decode_encoding([circuit.s[wire] for wire in fresh_share])
        ^ decode_encoding([circuit.s[wire] for wire in output_share2])
    ).astype(np.uint8)
    return circuit, expected, actual, scratch2_wires


class NonlinearGadgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.circuit291, cls.info291 = nonlinear291.run_gate(samples=512, seed=17)

    def test_nonlinear291_r57_correctness_count_and_physical_fanin(self):
        info = self.info291
        self.assertTrue(info["correct"])
        np.testing.assert_array_equal(
            info["gate_ab"], 1 ^ info["b"] ^ (info["a"] & info["b"])
        )
        np.testing.assert_array_equal(info["c_out_actual"], info["c_out"])
        self.assertEqual(info["n_gates"], nonlinear291.CANONICAL_R57_GATE_COUNT)
        self.assertEqual(
            len(self.circuit291.gate_log), nonlinear291.CANONICAL_R57_GATE_COUNT
        )
        physical_fanin = max(
            len(controls) for _, _, controls in self.circuit291.gate_log
        )
        self.assertEqual(physical_fanin, 2)
        self.assertEqual(
            info["max_physical_fanin"],
            nonlinear291.CANONICAL_MAX_PHYSICAL_FANIN,
        )
        self.assertEqual(
            info["max_requested_fanin"],
            nonlinear291.CANONICAL_MAX_REQUESTED_FANIN,
        )

    def test_clean_scratch_is_restored(self):
        self.assertTrue(self.info291["scratch_restored"])
        self.assertTrue(self.info291["required_ancillas_restored"])
        for wire in self.info291["layout"]["scratch"]:
            self.assertFalse(np.any(self.circuit291.s[wire]))

        self.assertTrue(self.info291["decomposition_ancillas_restored"])
        for group in ("decomposition", "persistent", "temporary"):
            for wire in self.info291["layout"][group]:
                self.assertFalse(np.any(self.circuit291.s[wire]))

    def test_random_batch_covers_the_r57_truth_table(self):
        rows = set(
            zip(
                self.info291["a"].tolist(),
                self.info291["b"].tolist(),
                self.info291["c_in"].tolist(),
            )
        )
        self.assertEqual(rows, {(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)})

    def test_two_gadget_chains_compose(self):
        chain = nonlinear291.build_chain(samples=256, seed=29)
        self.assertTrue(chain["correct"])
        self.assertTrue(chain["scratch_restored"])
        self.assertTrue(chain["required_ancillas_restored"])
        self.assertEqual(len(chain["circ"].gate_log), 582)
        self.assertEqual(
            max(len(controls) for _, _, controls in chain["circ"].gate_log), 2
        )
        np.testing.assert_array_equal(
            chain["targets"]["g2:a"], chain["targets"]["g1:c_out"]
        )

    def test_dirty_restoration_metadata_is_literal_and_mode_aware(self):
        circuit, info = nonlinear291.run_gate(samples=512, seed=43, dirty=True)
        self.assertTrue(info["correct"])
        self.assertFalse(info["scratch_restored"])
        self.assertTrue(info["required_ancillas_restored"])
        self.assertTrue(
            any(np.any(circuit.s[wire]) for wire in info["layout"]["scratch"])
        )
        self.assertTrue(info["decomposition_ancillas_restored"])

    def test_dirty_scratch_accepts_general_sequences(self):
        circuit, expected, actual, scratch2_wires = run_direct_dirty_gadget(
            range(39, 41)
        )
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(len(circuit.gate_log), 273)
        self.assertTrue(any(np.any(circuit.s[wire]) for wire in scratch2_wires))

    def test_dirty_scratch_requires_one_distinct_wire_per_group(self):
        for invalid in (39, range(39, 40), (39, 39)):
            with self.subTest(scratch2=invalid):
                with self.assertRaises(ValueError):
                    run_direct_dirty_gadget(invalid)


if __name__ == "__main__":
    unittest.main()
