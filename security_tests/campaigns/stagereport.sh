#!/bin/bash
# stagereport.sh <run-name> — per-stage compression summary for a mixing run.
# For each stage: post-shoot peak, 0.4 target, compressed result, ratio, floor_gen,
# and HOW compression stopped: "stall" (aggressive COMPRESS_STALL_FRAC rule),
# "legacy" (<50 gates over the long window), or "target" (compressed all the way).
SERVER=ai-claude@129.114.108.89
RUN=${1:?usage: stagereport.sh <run-name>   e.g. stagereport.sh cdcnot_m3000_es}

ssh -o ConnectTimeout=15 "$SERVER" "cat ~/tds/$RUN.log" | awk '
  /\[stage-D\] stage [0-9]+ \| last_compressed/ {
    for (i = 1; i <= NF; i++) if ($i == "stage") cur = $(i + 1)
    next
  }
  /compress target/ {
    for (i = 1; i <= NF; i++) {
      if ($i == "of") peak[cur] = $(i + 1)
      if ($i == "=")  tgt[cur]  = $(i + 1)
    }
    next
  }
  /Early stop/ {
    reason[cur] = "legacy"
    for (i = 1; i <= NF; i++)
      if ($i == "threshold") { th = $(i + 1) + 0; if (th > 50) reason[cur] = "stall" }
    next
  }
  /Early-stop target reached/ { reason[cur] = "target"; next }
  /Light compression target/  { reason[cur] = "light";  next }
  /stage [0-9]+ progress:/ {
    s = ""
    for (i = 1; i <= NF; i++) {
      if ($i == "stage")     s = $(i + 1)
      if (s != "" && $i == "of") comp[s] = $(i + 1)
      if ($i == "floor_gen") fg[s] = $(i + 1)
      if ($i == "progress:") pc[s] = $(i + 1)
    }
    if (s != "" && s + 0 > last) last = s + 0
    next
  }
  END {
    if (last == 0) { print "no completed stages yet"; exit }
    printf "%-6s %10s %10s %10s %7s %9s %8s  %s\n",
           "stage", "peak", "target.4", "compressed", "ratio", "floor_gen", "at-tgt%", "stop"
    for (k = 1; k <= last; k++) {
      if (comp[k] == "") continue
      p = peak[k] + 0
      c = comp[k] + 0
      rstr = (p > 0) ? sprintf("%.2f", c / p) : "-"
      printf "%-6d %10d %10d %10d %7s %9s %8s  %s\n",
             k, p, tgt[k] + 0, c, rstr, fg[k], pc[k],
             (reason[k] != "" ? reason[k] : "?")
    }
  }'
