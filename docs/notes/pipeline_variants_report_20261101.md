# Pipeline-variant resilience campaign: summary of results so far

Data as of 2026-11-01 13:25 UTC. Server tree `~/variants` on machine1; status tables regenerate every 10 minutes; this document freezes them.

## 1. Question and arms

Starting point: the preferred pipeline = stage 1 (seeded sandwich around a random 64-bit, 1000-gate g57 source C), stage 2 three-mask modified (random helpers, unique cover), stages 3-5 as is (DB mixing 5.2 effs with the frozen identity store; split; crossing to 2x), stage 6 adaptive C-aware resynthesis v2 iterated until two consecutive rounds compress jointly by less than 2%. One fresh 40-bit seed from the OS CSPRNG is shared by all arms (stage seeds SEED+2/+3), so arms 1-3 and 5-8 share one random C and one sandwich (2,832 gates), and the balanced arm shares the seed but uses a fresh balanced C (every internal value read exactly twice; built by a port of the Codex sampler that reproduces RC's published balanced C gate for gate).

| arm | description | stage 2 | stage 3 | stage 4 | stage 5 | stage-6 input |
|---|---|--:|--:|--:|--:|---|
| base | stage 1 as is, stage 2 three-mask modified, stages 3-5, stage 6 v2 (2% rule). | 504,163 | 1,064,981 | 1,889,509 | 3,497,695 | stage5 3,497,695 |
| skip3 | as base without stage 3. Stage 4 (split) acts on g57-form gates that stage 3 produces; on the raw stage-2 circuit it found 1,024 and grew the circuit by 0.4% in 6 s, so this arm is effectively "stage 2 -> stage 5 -> stage 6". | 504,163 | - | 506,006 | 952,774 | stage5 952,774 |
| skip5 | as base without stage 5: stage 6 runs on base's stage-4 output (base's stage-3/4 outputs reused verbatim). | 504,163 | 1,064,981 | 1,889,509 | - | stage4 1,889,509 |
| bal | as base with a fresh balanced C in stage 1 (884 deeper internal births instead of 886). | 503,865 | 1,062,178 | 1,883,952 | 3,487,126 | stage5 3,487,126 |
| k24 | as base with stage 5 at --k-max 24 instead of 12 (base's stage-3/4 outputs and stage-4 state reused). The wider cap produced only 19 gates above 12 controls (max 14); the ANF/adversary/payload tools cap input gates at 12 controls, so this arm runs on private MAXK=24 builds of those crates. | 504,163 | 1,064,981 | 1,889,509 | 3,497,310 | stage5 3,497,310 |
| rust | base's stage 1 + the Rust port's stage 2 (session "Rust version with circuit/stage updates", commit 5b18bff3): three-mask modified plus per-wire band turnover (promotion cohorts of 16) and fire interleaving whose measured effect is small (mean fire displacement 31 gates); 502,755 gates. | 502,755 | 1,042,261 | 1,872,736 | 3,466,788 | stage5 3,466,788 |
| rust2 | as rust with the interleaving made effective (read-only mask refresh, dirty helpers from 8 per-fire lanes of a 32-wire reserve, persistent carrier migration only every 8th fire): mean displacement 344 gates, a 250-gate window spans 3,101 original positions, 91 rotations instead of 112; 455,884 gates (commit 4a5fe63d). Both Rust arms use base's C and sandwich (identical hashes). | 455,884 | 1,014,079 | 1,874,988 | 3,462,940 | stage5 3,462,940 |
| rust2_skip3 | rust2's stage 2 straight into stages 4-5 (no stage 3), then stage 6; like skip3, stage 4 is a near no-op and stage 6 finds nothing to resynthesize. | 455,884 | - | 457,631 | 853,767 | stage5 853,767 |

## 2. What was measured (the columns)

- **Stage 6 v2 (C-aware)**: each round plans blocks that avoid cutting through exposed C values, resynthesizes every block from its ANF with two seeds and keeps a replacement only if it is smaller and not leakier; rounds repeat until two consecutive rounds compress jointly by less than 2%. The smallest round output is F, the delivered circuit. Its *census* (validated: every witness re-checked on 8,192 fresh plus structured inputs) counts the deeper internal C births (886 of 936 for the random C, 884 for the balanced C; the 50-52 excluded are degree-2 functions of public inputs or outputs) present directly (verbatim up to NOT at some cut) / as an affine combination of simultaneously live wires at some cut, written d/a.
- **Raw ledger on F**: C-blind observers on F itself (direct ledger plus affine-window observers at ranks 10 and 12), labeled against C afterwards and audited; R/O = births found with >= 2 distinct representations (repeated) / exactly one (once).
- **Narrow portfolio on F**: ten C-blind window rewrites of F followed by the same observers: factor rewrites with 125-, 250-, 500- and 1,000-gate windows, factor 125 shifted by 63, factor 125 on a wire-permuted circuit, pairs 125, and three payload-view cases (factor 125, factor 250, delta 250); the 500/1,000-gate windows were added in this campaign because a stage-5 fire spans ~1,200 gates, and 2,000/4,000-gate windows were added later ("+2" in the count). The same portfolio was run on the shortened circuits, and on the 5.2-eff remixed circuit before and after its resynthesis for some arms.
- **Blind direct**: C-unaware adaptive resynthesis of F (same machinery as stage 6 with an empty census, one seed, keep-smaller, iterated to a 1% two-round rule, the previous campaign's attacker rule), with the validated census of its plateau P.
- **Remix 5.2 / remix 2**: stage-3 remixing of F with the frozen store for 5.2 or 2 effective units of work (the circuit grows about 2.1x / 1.7x), then the blind resynthesis of the remix with its census.
- **Shorten**: extra C-blind compression of P (fcompress transport compression over function-preserving reorderings, a 128-gate-cap blind pass, fcompress, a 48-gate-cap pass, fcompress; verified against P and the payload contract) with the validated census of the result; motivated by the finding that blind plateaus are not size floors.
- **Candidates C/all**: a C-blind scan on 8,192 honest samples counts the internal functions born at least twice (up to NOT) that are not affine in the inputs, then labels them against C: "deeper C births among them / all such candidates".
- **Degree-2 replicas mid-circuit**: at interior cuts (depths 0.40, 0.45, 0.50, 0.55, 0.60), how many of the 64 payload outputs are degree <= 2 polynomials of the wires live at the cut (C-blind; the degree-2 span is kept below the 36,864 samples so the test is non-vacuous). The direct payload replicas near the output (54-56 of 64 in F for most arms) are the sandwich's own copies and are not reported.
- **Sandwich classes** (rust family): the census run against the whole 128-wire sandwich as source (every gate with target < 64), classifying each sandwich gate as C g57 (936 internal, C's outputs excluded), D g57 (951), C/D transition CNOT (the 64 copy CNOTs carry C's outputs; counted only when a replica interval starts before 90% of the circuit), slicing CNOT (262), slicing Toffoli (491).
- **Tally**: deeper C births exposed in any circuit produced or measured since the campaign started on F (raw and narrow ledgers of F and of its derived circuits, every blind round output, both remixes and their rounds); strong = >= 2 supports in at least one circuit. No tallied birth is exposed only in the last 10% of a circuit (checked).
- **Inversion**: C-blind attempts to compute a preimage of the all-zero payload: a greedy structural walk (backward from the payload outputs, forward from the inputs; a meet would reconstruct C) and a CryptoMiniSat preimage search with a 24 h wall cap (8 threads), each on F and on P.

## 3. Results

### 3.1 Sizes and the C-aware stage 6

| arm | stage-6 input | rounds | cumulative ratio per round | F gates | F census d/a |
|---|---|--:|---|--:|---|
| base | stage5 3,497,695 | 7 | 0.923/0.856/0.824/0.804 0.790/0.783/0.778 | 2,720,511 | 0/0 |
| skip3 | stage5 952,774 | 2 | 1.000/1.000 | 952,498 | 0/0 |
| skip5 | stage4 1,889,509 | 4 | 0.912/0.895/0.889/0.884 | 1,670,132 | 0/0 |
| bal | stage5 3,487,126 | 7 | 0.926/0.862/0.833/0.813 0.800/0.793/0.786 | 2,742,554 | 0/0 |
| k24 | stage5 3,497,310 | 7 | 0.926/0.857/0.830/0.809 0.797/0.788/0.782 | 2,735,148 | 0/0 |
| rust | stage5 3,466,788 | 11 | 0.905/0.805/0.752/0.717 0.696/0.679/0.668/0.659 0.652/0.644/0.641 | 2,221,174 | 0/0 |
| rust2 | stage5 3,462,940 | 8 | 0.972/0.942/0.922/0.905 0.895/0.882/0.873/0.866 | 2,997,188 | 0/0 |
| rust2_skip3 | stage5 853,767 | 2 | 1.000/1.000 | 853,767 | 0/0 |

### 3.2 Exposure of F to the C-blind observers

| arm | raw R/O | narrow union R/O (cases) | factor 1000 | factor 2000 | factor 4000 | candidates C/all on F |
|---|---|---|---|---|---|---|
| base | 0/0 | 254/61 (10/10+2) | 231/180 | 232/181 | 232/181 | 0/205,622 |
| skip3 | 0/0 | 361/29 (10/10+1) | 290/258 | 290/257 | - | 0/10,063 |
| skip5 | 0/0 | 343/69 (10/10+2) | 345/282 | 350/286 | 348/284 | 0/119,501 |
| bal | 0/0 | 295/109 (10/10+2) | 282/192 | 283/195 | 279/191 | 0/208,884 |
| k24 | 0/0 | 229/79 (10/10) | 224/164 | - | - | 0/205,859 |
| rust | 0/0 | 322/65 (10/10+2) | 281/215 | 284/216 | 284/215 | 0/190,181 |
| rust2 | 0/0 | 101/27 (10/10+2) | 81/52 | 85/60 | 85/60 | 0/165,649 |
| rust2_skip3 | 0/0 | 42/9 (10/10+2) | 44/38 | 44/38 | 44/38 | 0/8,873 |

### 3.3 C-unaware compression lines

| arm | blind direct: rounds, ratios -> P gates, d/a | remix 5.2: gates (x F); blind of the remix | remix 2: gates (x F); blind of the remix | shorten: final (x P), d/a |
|---|---|---|---|---|
| base | 4r 0.992/0.983/0.978/0.974 -> 2,650,806 9/9 | 5,549,367 (2.040) blind 12r 0.691/0.640/0.619/0.606 0.596/0.590/0.583/0.577 0.573/0.568/0.565/0.563 -> 3,125,171 37/37 | 4,573,901 (1.681) blind 13r 0.800/0.744/0.719/0.703 0.692/0.686/0.680/0.675 0.670/0.667/0.662/0.660 0.657 -> 3,003,030 23/23 | 1,786,769 (0.674) 9/9 |
| skip3 | 2r 1.000/1.000 -> 952,168 0/0 | 1,975,704 (2.074) blind 7r 0.970/0.925/0.913/0.905 0.898/0.894/0.891 -> 1,761,278 7/7 | 1,652,357 (1.735) blind 4r 0.986/0.965/0.960/0.955 -> 1,578,452 1/1 | 539,812 (0.567) 0/0 |
| skip5 | 3r 0.991/0.986/0.982 -> 1,640,742 2/2 | 3,567,260 (2.136) blind 9r 0.633/0.610/0.601/0.593 0.588/0.585/0.581/0.578 0.576 -> 2,056,153 16/16 | 2,939,382 (1.760) blind 8r 0.740/0.714/0.699/0.690 0.685/0.680/0.677/0.675 -> 1,983,686 21/21 | 944,984 (0.576) 2/2 |
| bal | 6r 0.990/0.981/0.973/0.964 0.961/0.956 -> 2,622,475 14/14 | 6,039,148 (2.202) blind 11r 0.643/0.595/0.573/0.555 0.544/0.538/0.532/0.528 0.525/0.522/0.520 -> 3,139,876 33/33 | 4,399,457 (1.604) blind 12r 0.815/0.760/0.731/0.715 0.701/0.687/0.679/0.675 0.670/0.666/0.663/0.661 -> 2,905,958 29/29 | 1,797,939 (0.686) 15/15 |
| k24 | 8r 0.990/0.980/0.974/0.969 0.962/0.955/0.950/0.946 -> 2,587,709 9/9 | 6,042,867 (2.209) blind 10r 0.646/0.596/0.571/0.556 0.547/0.540/0.533/0.529 0.527/0.525 -> 3,173,128 36/36 | 4,597,623 (1.681) blind 10r 0.791/0.737/0.710/0.690 0.676/0.668/0.662/0.657 0.653/0.651 -> 2,993,584 22/22 | 1,806,170 (0.698) 8/8 |
| rust | 7r 0.989/0.976/0.966/0.958 0.947/0.943/0.938 -> 2,083,769 20/20 | 4,587,484 (2.065) blind 13r 0.665/0.608/0.584/0.572 0.563/0.557/0.550/0.543 0.539/0.536/0.532/0.529 0.526 -> 2,414,137 37/37 (running) | 3,703,893 (1.668) blind 13r 0.780/0.727/0.696/0.677 0.665/0.655/0.648/0.643 0.638/0.631/0.627/0.624 0.621 -> 2,300,403 48/48 | 1,570,496 (0.754) 21/21 |
| rust2 | 4r 0.992/0.986/0.980/0.976 -> 2,924,806 8/8 | 6,333,844 (2.113) blind 11r 0.697/0.640/0.614/0.594 0.579/0.567/0.556/0.548 0.541/0.536/0.531 -> 3,361,128 16/16 (running) | 4,884,323 (1.630) blind 13r 0.815/0.760/0.733/0.712 0.698/0.687/0.677/0.670 0.663/0.655/0.650/0.645 0.642 -> 3,135,322 24/24 (running) | 1,559,530 (0.533) 8/8 |
| rust2_skip3 | 2r 1.000/1.000 -> 853,767 0/0 | 1,783,803 (2.089) blind 2r 0.996/0.995 -> 1,774,087 0/0 | 1,476,377 (1.729) blind - | 476,870 (0.559) 0/0 |

### 3.4 Narrow portfolio on the derived circuits, tally, inversion

| arm | narrow on shortened | narrow on remix 5.2 | narrow on remix-5.2 final | tally deeper (strong) | inversion |
|---|---|---|---|---|---|
| base | 251/79 (9/10) | 106/52 (10/10) | 284/91 (10/10) | 492 (443) | recon_F: no meet sat_F: timeout recon_P: no meet sat_P: timeout |
| skip3 | 498/17 (10/10) | - | - | 532 (515) | recon_F: no meet sat_F: timeout recon_P: no meet sat_P: timeout |
| skip5 | 339/72 (10/10) | - | - | 491 (440) | recon_F: no meet sat_F: timeout recon_P: no meet sat_P: timeout |
| bal | 331/139 (10/10) | - | - | 603 (483) | recon_F: no meet sat_F: timeout recon_P: no meet sat_P: timeout |
| k24 | 246/96 (10/10) | - | - | 432 (351) | recon_F: no meet sat_F: timeout recon_P: no meet sat_P: timeout |
| rust | 294/66 (9/10) | - | - | 455 (414) (running) | recon_F: no meet sat_F: timeout recon_P: no meet sat_P: timeout |
| rust2 | 129/47 (10/10+1) | 29/19 (10/10) | - | 234 (194) (running) | recon_F: no meet sat_F: timeout recon_P: no meet (running) |
| rust2_skip3 | 285/38 (10/10) | 5/4 (10/10) | 9/5 (10/10) | 336 (300) (running) | recon_F: no meet (running) |

Inversion detail: the walks recover 6-41 forward gates and 0-2 backward gates and never meet; every SAT run hit its 24 h cap without a result (rust2 and rust2_skip3 still running).

### 3.5 Exposure by test and circuit (deeper C births found / strong)

| test / circuit | base F | base short | base remix5.2 | base remix5.2 final | skip3 F | skip3 short | skip5 F | skip5 short | bal F | bal short | k24 F | k24 short | rust F | rust short | rust2 F | rust2 short | rust2 remix5.2 | rust2_skip3 F | rust2_skip3 short | rust2_skip3 remix5.2 | rust2_skip3 remix5.2 final |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| factor 125 | 67/41 | 56/35 | 2/0 | 117/79 | 60/48 | 277/231 | 39/21 | 67/39 | 60/31 | 54/29 | 68/34 | 62/37 | 110/70 | 83/50 | 26/18 | 21/16 | 1/1 | 2/0 | 108/85 | 0/0 | 0/0 |
| factor 125 shifted 63 | 60/29 | 49/25 | 1/1 | 126/85 | 62/49 | 257/215 | 40/22 | 70/46 | 55/27 | 60/34 | 61/31 | 58/27 | 120/82 | 78/53 | 20/12 | 19/11 | 0/0 | 2/2 | 104/77 | 0/0 | 1/0 |
| factor 125 wire-permuted | 88/48 | 64/36 | 3/1 | 150/93 | 66/53 | 281/241 | 51/22 | 84/44 | 81/46 | 62/38 | 84/47 | 72/38 | 132/90 | 90/56 | 34/23 | 25/19 | 0/0 | 3/1 | 113/86 | 0/0 | 0/0 |
| factor 250 | 114/71 | - | 12/4 | 187/121 | 252/221 | 374/350 | 111/73 | 215/149 | 113/61 | 149/79 | 121/51 | 148/92 | 193/126 | - | 45/26 | 53/36 | 1/1 | 12/9 | 194/166 | 1/0 | 1/0 |
| factor 500 | 170/118 | 231/157 | 51/28 | 246/164 | 289/257 | 403/382 | 262/206 | 320/253 | 210/124 | 314/203 | 176/122 | 232/153 | 237/167 | 249/170 | 67/44 | 111/78 | 16/9 | 44/38 | 229/200 | 7/5 | 11/7 |
| factor 1000 | 231/180 | 277/201 | 131/86 | 295/208 | 290/258 | 403/382 | 345/282 | 340/277 | 282/192 | 371/253 | 224/164 | 257/175 | 281/215 | 308/237 | 81/52 | 131/90 | 39/22 | 44/38 | 229/200 | 8/5 | 12/9 |
| factor 2000 | 232/181 | - | - | - | 290/257 | - | 350/286 | - | 283/195 | - | - | - | 284/216 | - | 85/60 | 132/90 | - | 44/38 | - | - | - |
| factor 4000 | 232/181 | - | - | - | - | - | 348/284 | - | 279/191 | - | - | - | 284/215 | - | 85/60 | - | - | 44/38 | - | - | - |
| pairs 125 | 1/1 | 10/10 | 0/0 | 39/39 | 0/0 | 1/1 | 0/0 | 3/3 | 1/1 | 15/15 | 0/0 | 8/8 | 2/2 | 24/24 | 0/0 | 8/8 | 1/1 | 0/0 | 0/0 | 0/0 | 0/0 |
| payload factor 125 | 74/27 | 47/18 | 1/1 | 150/76 | 50/34 | 214/182 | 61/24 | 67/35 | 90/23 | 58/28 | 70/31 | 64/29 | 136/69 | 91/58 | 22/12 | 25/18 | 0/0 | 2/0 | 94/77 | 0/0 | 0/0 |
| payload factor 250 | 130/64 | 126/73 | 21/6 | 209/106 | 198/169 | 306/285 | 145/71 | 168/112 | 134/57 | 133/76 | 113/58 | 135/66 | 201/113 | 164/92 | 39/16 | 43/29 | 9/4 | 7/6 | 147/129 | 0/0 | 0/0 |
| payload delta 250 | 149/79 | 127/71 | 36/13 | 222/114 | 200/170 | 308/290 | 159/75 | 170/110 | 156/67 | 137/77 | 130/65 | 138/65 | 207/121 | 166/93 | 43/23 | 46/32 | 12/5 | 9/8 | 149/128 | 0/0 | 0/0 |
| narrow portfolio (union) | 315/254 | 330/251 | 158/106 | 375/284 | 390/361 | 515/498 | 412/343 | 411/339 | 404/295 | 470/331 | 308/229 | 342/246 | 387/322 | 360/294 | 128/101 | 176/129 | 48/29 | 51/42 | 323/285 | 9/5 | 14/9 |
| raw ledger (no rewrite) | 0/0 | 9/5 | 0/0 | 37/11 | 0/0 | 0/0 | 0/0 | 2/0 | 0/0 | 15/4 | 0/0 | 8/4 | 0/0 | 21/12 | 0/0 | 8/5 | 0/0 | 0/0 | 0/0 | 0/0 | 0/0 |
| blind direct resynthesis (all rounds) | 9/9 | n/a | n/a | n/a | 0/0 | n/a | 2/2 | n/a | 17/17 | n/a | 13/13 | n/a | 22/22 | n/a | 8/8 | n/a | n/a | 0/0 | n/a | n/a | n/a |
| blind resynthesis of the 5.2-eff remix | 41/41 | n/a | n/a | n/a | 7/7 | n/a | 27/27 | n/a | 42/42 | n/a | 44/43 | n/a | 47/47 | n/a | 23/23 | n/a | n/a | 0/0 | n/a | n/a | n/a |
| blind resynthesis of the 2-eff remix | 27/27 | n/a | n/a | n/a | 1/1 | n/a | 26/26 | n/a | 42/42 | n/a | 35/35 | n/a | 61/61 | n/a | 29/29 | n/a | n/a | n/a | n/a | n/a | n/a |
| census of the shortened circuit | n/a | 9/9 | n/a | n/a | n/a | 0/0 | n/a | 2/2 | n/a | 15/15 | n/a | 8/8 | n/a | 21/21 | n/a | 8/8 | n/a | n/a | 0/0 | n/a | n/a |
| TOTAL union of all tests on this circuit | 334/283 | 330/251 | 158/106 | 375/284 | 393/364 | 515/498 | 419/358 | 411/339 | 435/336 | 470/331 | 329/264 | 342/246 | 407/350 | 360/294 | 159/135 | 176/129 | 48/29 | 51/42 | 323/285 | 9/5 | 14/9 |

### 3.6 Sandwich gate classes (rust family; direct/affine exposed of internal values in the class)

| gate class / circuit | rust F | rust short | rust2 F | rust2 short | rust2_skip3 F | rust2_skip3 short |
|---|---|---|---|---|---|---|
| C g57 | 2/3 of 936 | 24/24 of 936 | 0/1 of 936 | 8/8 of 936 | 0/1 of 936 | 0/0 of 936 |
| D g57 | 21/25 of 951 | 30/33 of 951 | 9/12 of 951 | 11/14 of 951 | 0/3 of 951 | 0/3 of 951 |
| C/D transition CNOT | 7/8 of 64 | 9/10 of 64 | 4/4 of 64 | 5/5 of 64 | 0/0 of 64 | 0/0 of 64 |
| slicing CNOT | 23/26 of 262 | 31/34 of 262 | 22/26 of 262 | 23/27 of 262 | 21/25 of 262 | 21/25 of 262 |
| slicing Toffoli | 43/43 of 491 | 53/53 of 491 | 38/38 of 491 | 40/40 of 491 | 36/36 of 491 | 36/36 of 491 |

### 3.7 Degree-2 replicas at interior cuts

Every arm, every circuit (F, P, shortened), every tested depth (0.40-0.60): 0 of 64 payload outputs lie in the degree-2 span of the live wires (wire rank 256, span dimension 32,897 of 36,864 samples, non-vacuous). The old baseline pipeline (managed stage 2, as-is stage 6) had 36 of 64 on F and 50 of 64 on its blind plateau at depth 0.45-0.48. With the three-mask stage 2 the band carries the payload behind cubic masks, so no interior cut exposes the outputs at degree 2, and neither the blind resynthesis nor the shortening recreates such a cut.

### 3.8 Gates with more than two controls

Stage 5 creates products up to the k-max and the ANF synthesizer only replaces the windows it resynthesizes; every circuit keeps wide gates except skip5 (no stage 5): base F 322,149 of 2.72M (max 12), bal 344,725, k24 338,654 (max 13), rust 123,985, rust2 936,164 (31% of its F), skip3 603,917 of 0.95M, rust2_skip3 571,393 of 0.85M, skip5 1,335 (max 5). Shortening removes most of them (about 1% of the circuit for the stage-6-compressed arms).

### 3.9 Block-cap experiment (does stage 6 benefit from larger blocks or more search?)

Span = gates 0-218,605 of base's stage-5 circuit (its first round-1 planner segment), two seeds, keep smaller per block, initial pass.

| arm | term cap | pair cap | blocks | block gates median / p90 / max | ANF terms median / p90 | plan s | orig -> best gates | ratio | ok s1/s2 | timeouts | crashes | wall h | peak RSS GB |
|---|--:|--:|--:|---|---|--:|---|--:|---|---|---|--:|--:|
| control | 131072 | 4000000 | 232 | 887 / 1489 / 2862 | 17761.0 / 76685 | - | 218605 -> 198103 | 0.9062 | 207/208 | 45 | 0 | - | - |
| x05 | 65536 | 2000000 | 262 | 780 / 1339 / 2865 | 10431.0 / 42214 | 204 | 218605 -> 199987 | 0.9148 | 238/239 | 43 | 0 | 0.83 | 1.8 |
| x1 | 131072 | 4000000 | 232 | 887 / 1489 / 2862 | 17761.0 / 76685 | 740 | 218605 -> 198487 | 0.9080 | 209/208 | 44 | 0 | 1.28 | 3.4 |
| x2 | 262144 | 8000000 | 207 | 1000 / 1674 / 4791 | 28304 / 155321 | 1110 | 218605 -> 200757 | 0.9184 | 194/187 | 32 | 0 | 2.83 | 4.1 |
| x4 | 524288 | 16000000 | 195 | 1036 / 1837 / 4791 | 45131 / 277108 | 2971 | 218605 -> 200777 | 0.9184 | 179/179 | 29 | 0 | 3.44 | 11.6 |

Compression per gate is flat (0.906-0.918) across an 8x range of planner ANF budgets; the production cap is at the optimum; monomial-scaled time budgets up to 480 s change nothing (the same ~22 windows per seed time out). What compresses is ANF sparsity: ratio 0.80 below 4k terms, 0.87 at 4-16k, 0.95 at 16-65k, 0.98 at 65-131k; the windows larger budgets add (131k-524k terms) return at 0.95-0.98 after 600-1,000 s each. Per-round anatomy: the synthesizer takes 92-95% of thread-time, both seeds win symmetrically, refinement is 8% of the time for 27% of a late round's saving, and a crashed chunk's per-window fallback cost base's round 7 twenty extra hours. The driver was changed accordingly (one job pool over seeds and budgets, bisection on crashes, 60 s for dense windows from round 3, one seed after a round with < 3% gain, parallel candidate censuses, census reuse) for arms launched afterwards; rust2 ran its last rounds with 12 synthesizer processes.

## 4. Findings

- **The 1,000-gate window is the strongest test everywhere, and larger windows add nothing.** On the stage-6 outputs the 1,000-gate factor rewrite finds 224-345 deeper births for the Codex-generator arms where the 125-gate cases find 40-130; 2,000 and 4,000 gates give the same counts (base 232/232, skip5 350/348, bal 283/279, rust 284/284, rust2 85/85). The earlier campaign's portfolio, which stopped at 250 gates, under-reported exposure two to three times; with the long windows, base, skip3, skip5, bal, k24 and rust all end between 329 and 435 deeper births exposed, i.e. a third to a half of C.
- **rust2 is the only design that resists the window rewrites**: 128 narrow / 159 total on F (85 in the long windows), three times fewer than every other arm, and its blind lines expose 8-29 instead of 17-61. rust (turnover alone, weak interleave) is as exposed as base (387 narrow, 407 total), so the effective fire interleaving is the element that works, not the turnover.
- **Skipping stage 3 does not reduce exposure and removes stage 6's leverage**: without stage 3's identities the stage-6 synthesizer cannot compress (skip3 and rust2_skip3 plateau at their stage-5 size with 0/0 census), the narrow portfolio exposes 390 (skip3) and 51 on F (rust2_skip3), and the shortened circuits are the most exposed of all (515 and 323). Skipping stage 5 (skip5) halves the circuit but leaks like base (412 narrow, 419 total).
- **The balanced C does not help against the C-blind observers** (bal: 404 narrow, 435 total, the highest of the Codex-generator arms), consistent with the per-read exposure model of the previous campaign.
- **k-max 24 changes nothing**: stage 5 produced 19 gates above 12 controls and k24 tracks base within 1% at every stage (F 2,735,148 vs 2,720,511; narrow 308 vs 315).
- **Compression attacks expose little by themselves but shortening and remixing move the picture**: the blind direct plateaus expose 0-22 births (census) and are not size floors; fcompress-plus-small-window shortening compresses P by 25-47% and the shortened circuits are more exposed to the window rewrites than F for every arm; remixing then recompressing exposes 23-61 (census) and, for base, 375 births through the narrow portfolio of the remix final versus 158 on the remix itself.
- **No preimage from any inversion attempt**; the C-blind candidate haystack is 9k-209k repeated non-affine functions per circuit with at most 8 deeper C births among them.
- **The payload is not replicated at degree 2 anywhere inside any circuit** (0 of 64 at all tested depths for all arms), unlike the old baseline's 36-50, which closes the midpoint-replication avenue of the preimage campaign for this stage-2 family.
- **The sandwich leaks through its slicing gates, not through C**: on rust2's F, 0 of 936 C births but 9 of 951 D births, 22 of 262 slicing CNOT outputs and 38 of 491 slicing Toffoli outputs are directly present; the same pattern holds for rust and rust2_skip3.

## 5. Status and caveats

Still running at the time of writing: the blind remix lines of rust and rust2, rust2_skip3's resilience (its remixes were produced on the Mac Pro against the local store after the server's store was deleted to free 299 GB; the store is re-fetched automatically when the campaign ends), the remix gauntlets (rust2 and base first), the 2,000/4,000-gate windows of the arms not yet covered, the shortened-circuit portfolios of rust and rust2_skip3, and the remaining SAT inversion runs. Caveats: one seed and one C per arm (the Rust arms share base's); the attacker rules (1% stop) and the ten standard cases are the previous campaign's for comparability; the stage-6 drivers of base, skip3, skip5, bal, k24, rust and the first rounds of rust2 ran the original driver, later rounds of rust2 and all of rust2_skip3 the revised one (identical results for bit-identical inputs except for the dense-window budget and the late single seed); the degree-2 scan tests only depths 0.40-0.60.

## Appendix: where things are

Server `~/variants` (machine1): `run/STATUS.md` and `run/STATUS.html` (tables above, refreshed every 10 min), `run/report.json`, `run/exp_blockcap/`, `run/d2/`, `run/replicas/`, `run/sandwich_exposure/`, `run/inversion/`, `run/ledgers/`, `run/resilience/`, `run/shorten/`; drivers in `py/` and `py6/`; campaign loop `py6/variants_loops10.sh`. Local: `docs/notes/pipeline_variants_20261026.md` (design and chronology), `docs/notes/pipeline_variants_status.html` (live mirror).
