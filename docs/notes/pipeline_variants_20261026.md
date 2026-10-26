# Pipeline-variant resilience campaign (launched 2026-10-25 23:13 UTC, server tree `~/variants`)

## Question (RC)
Generate a 64-bit circuit with each of four variants of the preferred pipeline and test its resilience with
C-unaware compression, exposure measurement and inversion attempts.

1. **base**: stage 1 as is; stage 2 three-mask modified (v2: `--random-helpers --unique-cover`); stages 3, 4, 5 as is;
   stage 6 adaptive, C-aware, v2, rerun until two consecutive rounds compress jointly by less than 2%.
2. **skip3**: as base without stage 3.
3. **skip5**: as base without stage 5.
4. **bal**: as base with the balanced C (every internal value read exactly twice) in stage 1.
5. **k24** (added 2026-10-25 23:30 UTC): as base, with stage 5 run at `--k-max 24` instead of 12 (identical to base through
   stage 4: reuses base's stage-3/4 outputs and resumes from the same stage-4 state).

## Design
- One fresh 40-bit seed from the OS CSPRNG shared by all arms (`SEED`, 0600, never in chat). Arms 1-3 share one random C
  and one stage-2 circuit; arm 4 uses a fresh balanced C built by `py/balanced_c.py`, a port of the Codex
  `balanced_c_build.generate` sampler whose self-test reproduces RC's published balanced C gate for gate; the fresh C passed
  the topology audit (936 internal births, every one read exactly twice, max control fanout 2, write spread 1).
- Stage 2 generator = the cdesign build (`--source-file` for the balanced C). Stage-2 sizes: 504,163 (random C),
  503,865 (balanced C); sandwich 2,832 gates.
- Stages 3-5 = the threemask_cmp protocol (reduced profile 2,2,2,2,2; stage-3 moves 5.2 × gates with the DB; stage 4
  split-stop; stage 5 resumed from the stage-4 state to 2 × the stage-4 size, temp max(64, target/25), split-base 3,
  split-damp 1, k-max 12). skip5 reuses base's stage-3 and stage-4 outputs verbatim (same input, same seeds).
- Stage 6 = `stage6_run.py` variant v2 (exposure-aware acceptance, two seeds, partitions shifted away from the previous
  round's boundaries, scaled budgets), S6_PROCS=3 × 8 threads per arm; stop when the last two rounds' joint compression is
  below 2% (`S6_STOP_FRAC`). The mixed output F = the smallest round output.
- Resilience on F (`resilience.py`): raw ledger and the nine-arm narrow portfolio; C-blind adaptive resynthesis of F to the
  1% rule (the previous campaign's attacker rule, kept for comparability); stage-3 remix of F for 5.2 and 2 effs followed
  by C-blind resynthesis; validated exposure census of every produced circuit; campaign tally = internal C births exposed
  in any derived circuit.
- Extra C-blind shortening of the direct plateau P (`shorten.py`): fcompress over function-preserving reorderings, a
  128-gate-cap blind pass, fcompress, a 48-gate-cap pass, fcompress; strict equivalence vs P and the payload contract;
  validated census of the result. Motivation: the 2026-10-25 shortening campaign showed the blind plateau of the baseline
  was a block-size artefact (351,744 → 274,097).
- Inversion attempts (`inversion.py`, derived circuits only): greedy structural walk (`blind_recon.py`, backward from
  the payload outputs and forward from the inputs, meet ⇒ preimage) and a CryptoMiniSat preimage of 0^64 on the honest
  slice with a 24 h cap and 8 threads, on F and on P.

## Early observations
- skip3: the split stage (stage 4) acts on g57-form gates that stage 3 produces (819k in the reference run); on the raw
  stage-2 circuit it found 1,024 and expanded the circuit by 0.4% in 6 s (504,163 → 506,006). Stage 5 then doubles it
  (target 1,012,012). So "skip stage 3" effectively removes stages 3 and 4.
- The server was already loaded (~120 of 160 cores: six SAT preimage jobs of the baseline campaign, the cdesign v2 arms
  and their early attackers); 44 GB of disk were free at launch.

- k24: stage 5 at k-max 24 gives 3,497,310 gates with at most 14 controls (19 gates above 12), so the wider limit hardly
  changes the width distribution (split probability 3^-(c-1) dominates). The ANF/adversary/payload tools cap input gates at
  12 controls (fixed-size loader arrays); k24 runs on private MAXK=24 builds (`~/variants/k24tools`, env `VAR_TOOLS`),
  the other arms on the original binaries. Both checkers agree on 12-control circuits.

- skip3 stage 6 (07:15 UTC): degenerate. The synthesizer converges on 3,499 of 3,900 round-1 windows but the
  resynthesized windows are ~3.6× larger than the originals (4,043 candidates rejected as growing; 1,549 gates of
  952,774 synthesized); round 2 the same; the 2% rule fires and F = 952,498 gates (census 0/0), essentially the stage-5
  circuit. Without stage 3 the three-mask circuit is ANF-incompressible in windows.
- skip5 stage 6: r1 1,889,509 → 1,722,812 (55% of input gates synthesized), r2 → 1,691,599; 0/0 deeper.

## Results
(to be filled from `run/REPORT.md`; `ssh machine1 '~/variants/status.sh'`)
