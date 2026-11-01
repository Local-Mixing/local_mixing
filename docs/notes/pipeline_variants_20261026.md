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

6. **rust** (added 2026-10-28 11:35 UTC): as base, with the new stage 2 from the Rust port (session "Rust version with
   circuit/stage updates", local_mixing_dev commit 5b18bff3): the three-mask modified design plus per-wire band turnover
   (promotion cohorts of 16 wires; 112 rotations, 109 cohorts drained, per-wire limbo mean 62.5 fires) and fire
   interleaving (window 4096, all 2,832 fires laned). The other session's caveat: the interleaving's measured effect is
   small (mean displacement 31 gates; a 250-gate window spans ~755 original positions) because carrier migration still
   chains consecutive fires, so the wire turnover is the substantive change. Generated on base's C and sandwich (hashes
   identical), 502,755 gates vs 504,163 for the Codex generator; payload check 8,192 states per direction passed.

7. **rust2** (added 2026-10-28 22:42 UTC, commit 4a5fe63d): as rust, with the interleaving made effective: the mask refresh
   leaves band coordinates read-only, dirty helpers come from 8 per-fire lanes of a 32-wire reserve that growth never
   writes, and the persistent carrier migration runs only in every 8th fire (2,079 transfers vs 16,756; stage 2 ~9%
   smaller: 455,884 gates). Measured: mean gate displacement 344 (rust 31), a 250-gate window spans 3,101 original
   positions (rust 755), dependency depth 17% of the gates; side effect 91 rotations instead of 112, so serving
   coordinates live ~20% longer. Same C and sandwich as base; all gates have at most 2 controls.

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

## Block-cap experiment (2026-10-26/27): does giving stage 6 larger blocks (and more search) early pay off?

Setup: base's stage-5 circuit, the first round-1 planner segment (gates 0-218,605; 232 production blocks), C-aware planning
from the round-1 census, two synthesizer seeds, keep the smaller converged result per block, initial pass only (no refinement).
Arms differ in the planner's ANF budgets (final term cap; pair cap = products of monomial pairs while composing a gate, the
budget that stops 93% of blocks; intermediate caps scale with the term cap); x1 and the cap arms use monomial-scaled time
budgets (60 s per 16k terms, 60-960 s) instead of the production gate-scaled 60/120/240 s. Experiment planner:
`~/variants/exptools/stage6_driver` (env AB_PAIRS, AB_LOOKAHEAD); driver `py6/blockcap_exp.py`; results
`run/exp_blockcap/<arm>/result.json`, table `run/exp_blockcap/REPORT.md`.

| arm | term cap | pair cap | blocks | block gates median / p90 / max | ANF terms median / p90 | plan s | ratio on the span | ok s1/s2 | timeouts | wall h | peak RSS GB |
|---|--:|--:|--:|---|---|--:|--:|---|---|--:|--:|
| control (production round 1) | 131072 | 4M | 232 | 887 / 1489 / 2862 | 17.8k / 76.7k | 740 | 0.9062 | 207/208 | 45 | - | - |
| x05 | 65536 | 2M | 262 | 780 / 1339 / 2865 | 10.4k / 42.2k | 204 | 0.9148 | 238/239 | 43 | 0.83 | 1.8 |
| x1 (= control caps, monomial budgets) | 131072 | 4M | 232 | 887 / 1489 / 2862 | 17.8k / 76.7k | 740 | 0.9080 | 209/208 | 44 | 1.28 | 3.4 |
| x2 | 262144 | 8M | 207 | 1000 / 1674 / 4791 | 28.3k / 155k | 1110 | 0.9184 | 194/187 | 32 | 2.83 | 4.1 |
| x4 | 524288 | 16M | 195 | 1036 / 1837 / 4791 | 45.1k / 277k | 2971 | 0.9184 | 179/179 | 29 | 3.44 | 11.6 |

Findings:
- Compression per gate is flat (0.906-0.918) across an 8x range of ANF budget; the production cap is at the optimum of this
  family. Larger budgets make blocks only 12-17% longer (ANF grows superlinearly along the circuit) and cost 2-3x the time
  and up to 6x the memory; halving the cap loses 0.9 points at 65% of the time.
- Time budgets are not the limit: monomial-scaled budgets up to 480 s on the same blocks changed nothing (0.908), and the
  same ~22 windows per seed time out regardless.
- What compresses is ANF sparsity, not block length. Control by ANF size: ratio 0.80 below 4k terms, 0.87 at 4-16k,
  0.95 at 16-65k, 0.98 at 65-131k; x2/x4's new 131k-524k-term windows come back at 0.95-0.98 after 600-1000 s each. The
  control's large blocks compress well (2-3k gates: 0.73) only because they are the sparse ones (8.7k terms).
- Consequence for the "large early, small late" schedule: the first half is a no; the per-round compression is set by the
  sparse windows, which exist at every cap. The useful levers are elsewhere: keep windows in the productive ANF range (the
  65k-131k windows give 2% for the most time), and use a different mechanism for dense windows (transport compression:
  fcompress took skip3's blind plateau from 952k to 544k gates and skip5's from 1.64M to 1.29M in one step).

## Stage-6 round anatomy and speed-ups (2026-10-30)

Per-phase timing of regular rounds (base round 6: 4.8 h; rust2 round 4: 5.4 h): synthesizer 92-95% of thread-time, of
which the 120 s budget group (blocks 1,000-4,000 gates) 54% for the two seeds run one after the other, the 60 s and 240 s
groups 5-7% each per seed, refinement 10-12%, the two candidate censuses 7%, censuses/audits/planning/assembly ~5%.
Base round 7 took 25.7 h because one window aborted on the memory cap in both seeds' 120 s chunks and the fallback
synthesized those chunks one window at a time (11.8 h and 11.4 h). Value per thread-hour (base round 2 / round 6):
sparse windows < 4k terms 9,341 / 2,426 gates, 4-16k 3,752 / 961, 16-65k 1,648 / 186, >= 65k 792 / 142; timeouts 13-17%
of the time for nothing; both seeds win symmetrically; refinement 8% of the time for 27% of a late round's saving.

Changes to the driver (new launches; rust2 restarted at a round boundary with 12 processes): one job pool over all
seeds and budgets; crashed chunks retried once then bisected; dense windows (>= 65,536 planner terms) get 60 s from round
3; one seed after a round with < 3% gain; candidate censuses in parallel; the input census is reused from the previous
round's output census. The C-blind attacker's rules are unchanged (comparability with the earlier campaign).

## Results
(to be filled from `run/REPORT.md`; `ssh machine1 '~/variants/status.sh'`)
