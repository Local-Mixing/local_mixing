#!/usr/bin/env python3
"""Generate the xpanel arm list: r x b x c x temp, one line per arm.

Fields: tag target temp moves base damp
Budgets: reach (~1.8 moves/net gate measured in the local probe) plus >=10
moves/gate of linger, rounded to 6M/8M/12M part-2 moves; --moves is ABSOLUTE
on a resume, so the split state's move counter (5358) is added.
"""
BASE = 362995      # gates in nR20_k2p_split.state
RESUMED_AT = 5358  # the state's move counter
MOVES = {1.5: 6_000_000, 2.0: 8_000_000, 3.0: 12_000_000}

for r in (1.5, 2.0, 3.0):
    for b in ("1.2", "1.5", "1.8"):
        for c in (2, 3, 4):
            for td in (100, 25, 8):
                tgt = round(BASE * r)
                temp = max(64, round(tgt / td))
                moves = MOVES[r] + RESUMED_AT
                tag = f"xp_r{r}_b{b}_c{c}_t{td}"
                print(tag, tgt, temp, moves, b, c)
