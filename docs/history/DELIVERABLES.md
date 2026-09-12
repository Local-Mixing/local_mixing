# Deliverables

## SAT-scaling study

- [Refined final estimate](experiments/gss_sat_scaling_20260810/FINAL_SAT_SCALING_ESTIMATE.md)
- [Public summary visualization](experiments/gss_sat_scaling_20260810/analysis/public-scaling-summary.png)
- [Accessible SVG visualization](experiments/gss_sat_scaling_20260810/analysis/public-scaling-summary.svg)
- [Concise model comparison](experiments/gss_sat_scaling_20260810/analysis/PUBLIC_MODEL_COMPARISON.md)
- [Knee-extension decision](experiments/gss_sat_scaling_20260810/KNEE_EXTENSION_DECISION.md)
- [Public terminal aggregate](experiments/gss_sat_scaling_20260810/aggregate/current/results.tsv)
- [Public censored-model analysis](experiments/gss_sat_scaling_20260810/aggregate/current/conflicts_analysis.json)

## Red-team challenges

- [Safe-to-share challenge index](red_team_tests/red_team_challenges.txt)
- [Public challenge archive](red_team_tests/circuits/challenges.zip)
- [Circuit-generation provenance](red_team_tests/CIRCUIT_GENERATION_INFO.txt)
- [Public C1](red_team_tests/circuits/challenges/public_c1)
- [Public C2](red_team_tests/circuits/challenges/public_c2)
- [Public C3](red_team_tests/circuits/challenges/public_c3)

The challenge archive contains only the three public circuit/challenge pairs.
Private source circuits, initial circuits, seeds, and answer records remain in
mode-600 files under mode-700 per-circuit directories.  The private answer
index is `red_team_tests/red_team_answers.txt` (mode 600); do not share it with
challenge participants.
