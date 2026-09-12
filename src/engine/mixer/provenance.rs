//! Per-gate metadata, ancestry universes, sidecars and event/litter identity.
use super::*;

impl Mixer {
    /// Write the per-gate ancestry sidecar: a header naming the universe, the
    /// tracer list in sampled mode, then one line per gate (current arena
    /// order — call after the final float so the order matches the written
    /// circuit) holding the gate's ancestor set as `anc_words` decimal u64s.
    /// Exact-mode implicit singletons are materialised, so the file is
    /// self-contained.
    pub fn write_anc_sidecar(&self, path: &str) -> std::io::Result<()> {
        use std::fmt::Write as _;
        assert!(
            self.anc_words > 0,
            "--anc-out needs ancestry armed (--ancestors, --anc-samples or --anc-in)"
        );
        let mut o = String::with_capacity(self.arena.len() * self.anc_words * 8);
        let _ = writeln!(
            o,
            "fmix-anc 1 {} m={} words={} gates={}",
            if self.anc_sampled { "sampled" } else { "exact" },
            self.anc_m,
            self.anc_words,
            self.arena.len()
        );
        if self.anc_sampled {
            let _ = write!(o, "tracers {}", self.anc_tracers.len());
            for t in &self.anc_tracers {
                let _ = write!(o, " {t}");
            }
            o.push('\n');
        }
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            let mut first = true;
            for w in &bits {
                if !first {
                    o.push(' ');
                }
                let _ = write!(o, "{w}");
                first = false;
            }
            o.push('\n');
            cur = self.arena.neighbor(cur, Dir::R);
        }
        std::fs::write(path, o)
    }

    /// Parse a sidecar written by `write_anc_sidecar`.
    pub fn read_anc_sidecar(path: &str) -> std::io::Result<AncSidecar> {
        let bad = |m: &str| std::io::Error::other(m.to_string());
        let text = std::fs::read_to_string(path)?;
        let mut lines = text.lines();
        let hdr = lines.next().ok_or_else(|| bad("empty ancestry sidecar"))?;
        let f: Vec<&str> = hdr.split_whitespace().collect();
        if f.len() != 6 || f[0] != "fmix-anc" || f[1] != "1" {
            return Err(bad(
                "ancestry sidecar header: want `fmix-anc 1 <mode> m= words= gates=`",
            ));
        }
        let sampled = match f[2] {
            "sampled" => true,
            "exact" => false,
            _ => return Err(bad("ancestry sidecar mode: want exact|sampled")),
        };
        let num = |s: &str, pre: &str| -> std::io::Result<usize> {
            s.strip_prefix(pre)
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad(&format!("ancestry sidecar field {pre}")))
        };
        let m = num(f[3], "m=")?;
        let words = num(f[4], "words=")?;
        let gates = num(f[5], "gates=")?;
        let tracers: Vec<u32> = if sampled {
            let tl = lines.next().ok_or_else(|| bad("missing tracers line"))?;
            let mut it = tl.split_whitespace();
            if it.next() != Some("tracers") {
                return Err(bad("want `tracers K t0 t1 ...`"));
            }
            let k: usize = it
                .next()
                .and_then(|x| x.parse().ok())
                .ok_or_else(|| bad("tracer count"))?;
            let v: Vec<u32> = it.filter_map(|x| x.parse().ok()).collect();
            if v.len() != k {
                return Err(bad("tracer list length mismatch"));
            }
            v
        } else {
            Vec::new()
        };
        let want_words = if sampled {
            tracers.len().div_ceil(64)
        } else {
            m.div_ceil(64)
        };
        if words != want_words {
            return Err(bad(
                "ancestry sidecar words= inconsistent with its universe",
            ));
        }
        let mut sets: Vec<Vec<u64>> = Vec::with_capacity(gates);
        for l in lines {
            let row: Vec<u64> = l
                .split_whitespace()
                .filter_map(|x| x.parse().ok())
                .collect();
            if row.len() != words {
                return Err(bad("ancestry sidecar row width mismatch"));
            }
            sets.push(row);
        }
        if sets.len() != gates {
            return Err(bad("ancestry sidecar gate count mismatch"));
        }
        Ok(AncSidecar {
            sampled,
            m,
            words,
            tracers,
            sets,
        })
    }

    /// Install imported ancestor lists as this run's INITIAL ancestry. Fresh
    /// runs only: the mixer must have been constructed with ancestry OFF
    /// (ancestors false, anc_samples 0), so input gates already sit on their
    /// singleton litters 0..n; this replaces the universe and attaches one
    /// imported set per input litter.
    pub fn import_ancestry(&mut self, sc: AncSidecar) {
        assert_eq!(
            sc.sets.len(),
            self.arena.len(),
            "--anc-in: sidecar has {} sets but the input circuit has {} gates",
            sc.sets.len(),
            self.arena.len()
        );
        assert!(
            self.anc_words == 0,
            "--anc-in replaces the ancestry universe; construct with --ancestors/--anc-samples off"
        );
        assert!(
            sc.sampled || sc.m <= 20_000,
            "--anc-in exact mode stores m={} bits per litter, past the small-input envelope \
             (re-run the producing chain with --anc-samples)",
            sc.m
        );
        self.anc_sampled = sc.sampled;
        self.anc_m = sc.m;
        self.anc_words = sc.words;
        self.anc_tracers = sc.tracers;
        self.anc.clear();
        for (i, bits) in sc.sets.into_iter().enumerate() {
            // Exact mode stores EVERY row, an all-zero one included: the
            // implicit-singleton rule reads a missing id < anc_m as {id}, and
            // after an import gate index i is unrelated to original-input
            // index i. Sampled mode keeps the missing-means-empty convention.
            if !self.anc_sampled || bits.iter().any(|&w| w != 0) {
                self.anc.insert(i as u64, bits);
            }
        }
        // Fresh litter ids must clear the exact-mode implicit-singleton range
        // [0, anc_m): a circuit smaller than the ORIGINAL input would
        // otherwise mint union ids that alias it.
        self.next_litter = self.next_litter.max(self.anc_m as u64);
        // Reporting reads these fields, not the universe params.
        self.params.ancestors = !self.anc_sampled;
        self.params.anc_samples = self.anc_tracers.len();
    }
    pub(crate) fn set_meta(&mut self, id: u32, m: Meta) {
        let i = id as usize;
        if i >= self.meta.len() {
            self.meta.resize(
                i + 1,
                Meta {
                    origin: ORIGIN_SYNTH,
                    event: 0,
                    dir: Dir::R,
                    dgen: GEN_FRESH,
                    litter: 0,
                    litter_size: 1,
                },
            );
        }
        self.meta[i] = m;
    }

    pub(crate) fn meta_of(&self, id: u32) -> Meta {
        self.meta.get(id as usize).copied().unwrap_or(Meta {
            origin: ORIGIN_SYNTH,
            event: 0,
            dir: Dir::R,
            dgen: GEN_FRESH,
            litter: 0,
            litter_size: 1,
        })
    }

    // Litter census of a window: (distinct litters, is-exactly-one-complete-
    // litter). "Complete" requires every gate to share one id AND the count to
    // still equal the size recorded when that litter was emitted — so a litter
    // that has since been split or partly merged reads as incomplete, making
    // the test conservative under churn.
    //
    // Singleton litters are excluded by construction: input gates and
    // born-random material carry no earlier spelling to be returned to, and a
    // ban on them would also refuse the descent's length-1 rung, which is the
    // one rung that always makes progress.
    pub(super) fn litter_census(&self, ids: &[u32]) -> (usize, bool) {
        if ids.is_empty() {
            return (0, false);
        }
        let mut distinct: Vec<u64> = Vec::with_capacity(ids.len());
        for &id in ids {
            let l = self.meta_of(id).litter;
            if !distinct.contains(&l) {
                distinct.push(l);
            }
        }
        let size = self.meta_of(ids[0]).litter_size;
        let full = distinct.len() == 1 && size >= 2 && ids.len() == size as usize;
        (distinct.len(), full)
    }

    // A new litter id. Unlike events these carry no tabu bookkeeping — a litter
    // is pure provenance.
    /// OR litter `l`'s ancestor set into `out`. Singleton sets are NOT stored:
    /// input gate `i` is litter `i` by construction, so any id below `anc_m`
    /// with no map entry denotes `{id}`. Ids at or above it with no entry are
    /// born-random material (twist brackets, insert pairs) and contribute
    /// nothing. That keeps init O(1) instead of O(input^2).
    pub(super) fn anc_or_into(&self, l: u64, out: &mut [u64]) {
        if let Some(v) = self.anc.get(&l) {
            for (o, x) in out.iter_mut().zip(v.iter()) {
                *o |= *x;
            }
        } else if !self.anc_sampled && (l as usize) < self.anc_m {
            // Exact mode only: bit `l` IS input gate `l`. In sampled mode the
            // bit space is the tracer set, tracer singletons are stored
            // explicitly, and a missing entry means "descends from no tracer".
            out[l as usize / 64] |= 1u64 << (l as usize % 64);
        }
    }

    /// Choose `k` distinct input-gate indices to trace, uniformly without
    /// replacement, from a DEDICATED rng: tracer choice must not perturb the
    /// mixing trajectory, so an exact-mode and a sampled-mode run with the same
    /// `--seed` follow the identical chain and can be compared gate for gate.
    /// Rejection sampling is O(k) expected for k << n and needs no O(n) buffer,
    /// which matters at production input sizes.
    pub(super) fn pick_tracers(n: usize, k: usize, sample_seed: u64) -> Vec<u32> {
        let mut rng = StdRng::seed_from_u64(
            sample_seed ^ 0x7ACE_5EED_0000_0000 ^ ((n as u64) << 17) ^ ((k as u64) << 3),
        );
        let mut set: std::collections::HashSet<u32> = std::collections::HashSet::with_capacity(k);
        while set.len() < k {
            set.insert(rng.random_range(0..n) as u32);
        }
        let mut v: Vec<u32> = set.into_iter().collect();
        v.sort_unstable();
        v
    }

    /// Union the ancestor sets of `srcs`' litters and record it under a fresh
    /// litter id, which is returned. The union is what makes this survive
    /// mixed-lineage replacement, where the scalar `origin` label is discarded.
    pub(crate) fn anc_union_litter(&mut self, srcs: &[u64]) -> u64 {
        let l = self.fresh_litter();
        if self.anc_words == 0 {
            return l;
        }
        let mut bits = vec![0u64; self.anc_words];
        for &src in srcs {
            self.anc_or_into(src, &mut bits);
        }
        // An all-zero union needs no entry: fresh litter ids are always >= anc_m
        // (next_litter starts at the input count), so a missing entry can never
        // alias an implicit singleton and reads back as empty either way. In
        // sampled mode this is the main memory win -- only litters that actually
        // carry a tracer are stored, and most carry none.
        if bits.iter().any(|&w| w != 0) {
            self.anc.insert(l, bits);
        }
        l
    }

    /// Mean ancestor-set cardinality and mean normalised ancestor SPAN over
    /// live gates. Cardinality answers "what is a mixed gate made of"; span --
    /// (max index - min index) / (input - 1) -- answers "how far has input
    /// material travelled to meet". Both are immune to the ORIGIN_SYNTH erosion
    /// that makes odiff/oadj unreadable (see osyn=).
    pub(super) fn anc_stats(&self) -> (f64, f64) {
        // Sampled mode reports through `tracer_report` instead: a sampled
        // popcount is not `anc` and a sampled index range is a biased `span`,
        // so leaving anc=/ancspan= at zero keeps those fields from silently
        // changing meaning (the mistake the g57=/shaped= split had to undo).
        if self.anc_words == 0 || self.anc_sampled {
            return (0.0, 0.0);
        }
        let (mut card_sum, mut span_sum, mut n) = (0f64, 0f64, 0u64);
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            let card: u32 = bits.iter().map(|w| w.count_ones()).sum();
            if card > 0 {
                let lo = bits
                    .iter()
                    .enumerate()
                    .find(|(_, w)| **w != 0)
                    .map(|(i, w)| i * 64 + w.trailing_zeros() as usize)
                    .unwrap_or(0);
                let hi = bits
                    .iter()
                    .enumerate()
                    .rev()
                    .find(|(_, w)| **w != 0)
                    .map(|(i, w)| i * 64 + 63 - w.leading_zeros() as usize)
                    .unwrap_or(0);
                card_sum += card as f64;
                span_sum += (hi.saturating_sub(lo)) as f64 / (self.anc_m.max(2) - 1) as f64;
                n += 1;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        if n == 0 {
            (0.0, 0.0)
        } else {
            (card_sum / n as f64, span_sum / n as f64)
        }
    }

    /// Log-bucketed histogram, rendered as `lo-hi:count` for non-empty buckets.
    pub(super) fn log_hist(vals: &[u64]) -> String {
        let mut b = [0u64; 32];
        for &v in vals {
            let k = if v == 0 {
                0
            } else {
                64 - (v.leading_zeros() as usize)
            };
            b[k.min(31)] += 1;
        }
        let mut out: Vec<String> = Vec::new();
        for (k, &c) in b.iter().enumerate() {
            if c == 0 {
                continue;
            }
            if k == 0 {
                out.push(format!("0:{c}"));
            } else {
                let lo = 1u64 << (k - 1);
                let hi = (1u64 << k) - 1;
                if lo == hi {
                    out.push(format!("{lo}:{c}"))
                } else {
                    out.push(format!("{lo}-{hi}:{c}"))
                }
            }
        }
        out.join(" ")
    }

    /// Ancestry in absolute units, with shapes rather than means.
    ///
    /// Three views of the same sets. `anc` is per-gate cardinality -- how many
    /// ORIGINAL gates a current gate descends from. `span` is per-gate, the
    /// index distance between the first and last of those ancestors, in
    /// original-circuit gate positions: how far apart in the input the material
    /// meeting in one gate came from. `fanout` is the dual, per INPUT gate: how
    /// many current gates carry any information about it. The two are linked by
    /// double counting, mean_fanout = mean_anc * gates / inputs, so quoting only
    /// the mean of one hides nothing -- but the distributions differ, and it is
    /// the tails that say whether spreading is uniform or a few gates are doing
    /// all the mixing.
    pub fn anc_report(&self) -> String {
        if self.anc_words == 0 {
            return String::new();
        }
        if self.anc_sampled {
            return self.tracer_report();
        }
        let mut cards: Vec<u64> = Vec::new();
        let mut spans: Vec<u64> = Vec::new();
        let mut fanout = vec![0u64; self.anc_m];
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            let mut lo = usize::MAX;
            let mut hi = 0usize;
            let mut card = 0u64;
            for (wi, &w) in bits.iter().enumerate() {
                let mut x = w;
                while x != 0 {
                    let b = x.trailing_zeros() as usize;
                    let idx = wi * 64 + b;
                    if idx < self.anc_m {
                        fanout[idx] += 1;
                        card += 1;
                        lo = lo.min(idx);
                        hi = hi.max(idx);
                    }
                    x &= x - 1;
                }
            }
            if card > 0 {
                cards.push(card);
                spans.push((hi - lo) as u64);
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        let mean = |v: &[u64]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<u64>() as f64 / v.len() as f64
            }
        };
        format!(
            "[fmix] ancestry: anc mean={:.1} [{}] | span(input gates) mean={:.0} [{}] | fanout/input mean={:.0} [{}]",
            mean(&cards),
            Self::log_hist(&cards),
            mean(&spans),
            Self::log_hist(&spans),
            mean(&fanout),
            Self::log_hist(&fanout)
        )
    }

    /// Total gate x input-gate incidence: the sum over live gates of how many
    /// original gates each descends from. This is the one transport quantity
    /// both modes can report on the same footing -- exactly in exact mode, and
    /// in sampled mode as the Horvitz-Thompson estimate (every input has
    /// inclusion probability K/m, so the sampled sum scaled by m/K is unbiased).
    /// It is also the schedule-invariant measure: `anc` per gate is this divided
    /// by the gate count, which compression inflates.
    pub fn anc_incidence(&self) -> f64 {
        if self.anc_words == 0 {
            return 0.0;
        }
        let mut sum = 0u64;
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            sum += bits.iter().map(|w| w.count_ones() as u64).sum::<u64>();
            cur = self.arena.neighbor(cur, Dir::R);
        }
        if self.anc_sampled {
            sum as f64 * self.anc_m as f64 / self.anc_tracers.len().max(1) as f64
        } else {
            sum as f64
        }
    }

    /// Sampled-ancestry readout: for each traced input gate, the set of current
    /// gates descended from it, summarised three ways.
    ///
    /// - **`desc`** -- how many current gates descend from one input gate. This
    ///   is the per-input FANOUT, measured exactly for the traced gates, so its
    ///   mean over tracers is an unbiased estimate of the mean fanout over all
    ///   input gates (each input has inclusion probability K/m). Everything else
    ///   global follows from it: `incid = desc x m` is the total gate x input
    ///   incidence, and `anc = incid / size` is the mean ancestors per gate --
    ///   the exact-mode `anc`, estimated without storing |input| bits anywhere.
    /// - **`cov`** -- of `POS_BUCKETS` equal slices of the CURRENT circuit, the
    ///   fraction that hold at least one descendant. 1.0 means one input gate's
    ///   influence is present everywhere in the mixed circuit.
    /// - **`ent`** -- normalised entropy of the descendant positions over those
    ///   buckets. `cov` says how far the influence reaches, `ent` says how
    ///   evenly: cov can be 1.0 while the mass sits in one slice.
    ///
    /// `cov`/`ent` are the security-facing quantities and have no exact-mode
    /// analogue -- they ask directly whether an adversary can localise which
    /// part of the mixed circuit a given original gate went to. They are also
    /// natively samplable, unlike `span`, which a column sample can only
    /// underestimate.
    pub fn tracer_report(&self) -> String {
        const POS_BUCKETS: usize = 64;
        let k = self.anc_tracers.len();
        if k == 0 {
            return String::new();
        }
        let size = self.arena.len().max(1);
        let mut cnt = vec![0u64; k];
        let (mut lo, mut hi) = (vec![usize::MAX; k], vec![0usize; k]);
        let mut buckets = vec![0u32; k * POS_BUCKETS];
        let mut sampled_card_sum = 0u64;
        let mut carriers = 0u64;
        // BACKWARD direction: for each gate in the CURRENT circuit, how spread
        // out are its ancestors in the INPUT circuit? The forward measures
        // (reach/cov/ent) answer the mirror question -- where a given input
        // gate's descendants ended up -- and say nothing about whether an
        // output gate draws on a narrow band of the original or on all of it.
        //
        // A min/max RANGE cannot be sampled: with K of m tracers the sample's
        // extremes sit strictly inside the true ones, and the bias grows as the
        // ancestor count shrinks, so the statistic would mean different things
        // at different points in a run (which is why ancspan= is switched off
        // in sampled mode). Bucket occupancy and entropy degrade gracefully
        // instead, and the sample standard deviation is outright unbiased for
        // the population one. All three are capped by the ancestor count, so
        // they are only readable next to sampled_card.
        let mut aspan_cov = 0f64;
        let mut aspan_ent = 0f64;
        let mut aspan_sd = 0f64;
        let mut aspan_sd_n = 0u64;
        let m_in = self.anc_m.max(1);
        let mut in_buckets = [0u32; POS_BUCKETS];
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        let mut pos = 0usize;
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            let b = (pos * POS_BUCKETS / size).min(POS_BUCKETS - 1);
            in_buckets.iter_mut().for_each(|c| *c = 0);
            let (mut ppos_sum, mut ppos_sq) = (0f64, 0f64);
            let mut card = 0u64;
            for (wi, &w) in bits.iter().enumerate() {
                let mut x = w;
                while x != 0 {
                    let t = wi * 64 + x.trailing_zeros() as usize;
                    if t < k {
                        cnt[t] += 1;
                        lo[t] = lo[t].min(pos);
                        hi[t] = hi[t].max(pos);
                        buckets[t * POS_BUCKETS + b] += 1;
                        // This ancestor's home in the INPUT circuit.
                        let gp = self.anc_tracers[t] as usize;
                        in_buckets[(gp * POS_BUCKETS / m_in).min(POS_BUCKETS - 1)] += 1;
                        ppos_sum += gp as f64;
                        ppos_sq += (gp as f64) * (gp as f64);
                        card += 1;
                    }
                    x &= x - 1;
                }
            }
            if card > 0 {
                carriers += 1;
                sampled_card_sum += card;
                let c = card as f64;
                aspan_cov +=
                    in_buckets.iter().filter(|&&x| x > 0).count() as f64 / POS_BUCKETS as f64;
                let h: f64 = in_buckets
                    .iter()
                    .filter(|&&x| x > 0)
                    .map(|&x| {
                        let q = x as f64 / c;
                        -q * q.log2()
                    })
                    .sum();
                aspan_ent += h / (POS_BUCKETS as f64).log2();
                if card >= 2 {
                    let mean = ppos_sum / c;
                    // Unbiased sample variance: with tracers drawn uniformly
                    // from the input, this estimates the spread of the gate's
                    // TRUE ancestor set, not just of the sampled part.
                    let var = (ppos_sq / c - mean * mean) * c / (c - 1.0);
                    aspan_sd += var.max(0.0).sqrt() / m_in as f64;
                    aspan_sd_n += 1;
                }
            }
            pos += 1;
            cur = self.arena.neighbor(cur, Dir::R);
        }
        let (mut reach, mut cov, mut ent) = (Vec::new(), Vec::new(), Vec::new());
        for t in 0..k {
            if cnt[t] == 0 {
                reach.push(0.0);
                cov.push(0.0);
                ent.push(0.0);
                continue;
            }
            reach.push((hi[t] - lo[t] + 1) as f64 / size as f64);
            let row = &buckets[t * POS_BUCKETS..(t + 1) * POS_BUCKETS];
            cov.push(row.iter().filter(|&&c| c > 0).count() as f64 / POS_BUCKETS as f64);
            let n_t = cnt[t] as f64;
            let h: f64 = row
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let q = c as f64 / n_t;
                    -q * q.log2()
                })
                .sum();
            ent.push(h / (POS_BUCKETS as f64).log2());
        }
        let meanf = |v: &[f64]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f64>() / v.len() as f64
            }
        };
        // desc: exact per-tracer fanout, so its mean estimates mean fanout/input.
        let desc = cnt.iter().sum::<u64>() as f64 / k as f64;
        let incid = desc * self.anc_m as f64;
        let anc_all = incid / size as f64;
        // Reported for honesty about the sample's resolution: gates whose
        // ancestry misses every tracer look empty, and at small K most do.
        let hit = carriers as f64 / size as f64;
        format!(
            "[fmix] tracers: K={} of m={} | desc mean={:.0} [{}] | cov mean={:.3} ent mean={:.3} reach mean={:.3} | est anc={:.1} incid={:.3e} | carriers={:.3} sampled_card={:.2} | ancspan cov={:.3} ent={:.3} sd={:.3}",
            k,
            self.anc_m,
            desc,
            Self::log_hist(&cnt),
            meanf(&cov),
            meanf(&ent),
            meanf(&reach),
            anc_all,
            incid,
            hit,
            if carriers > 0 {
                sampled_card_sum as f64 / carriers as f64
            } else {
                0.0
            },
            if carriers > 0 {
                aspan_cov / carriers as f64
            } else {
                0.0
            },
            if carriers > 0 {
                aspan_ent / carriers as f64
            } else {
                0.0
            },
            if aspan_sd_n > 0 {
                aspan_sd / aspan_sd_n as f64
            } else {
                0.0
            },
        )
    }

    /// JOINT census of re-encoding depth against ancestry: for each generation
    /// band, how many gates are in it and what their mean ancestor count and
    /// mean ancestor span are.
    ///
    /// `anc` and `dgen` have only ever been reported as separate marginals, which
    /// cannot answer the question that matters: does depth BUY ancestry? A
    /// protocol where anc rises steeply with generation is compounding -- each
    /// re-encoding folds in genuinely new lineage. One where anc is flat across
    /// generations is re-spelling the same material over and over, and its depth
    /// counter is measuring effort rather than mixing. Different mode schedules
    /// can produce the same mean anc with very different shapes here.
    ///
    /// `r` is the Pearson correlation of (dgen, anc) over gates with a real
    /// generation. GEN_FRESH is a sentinel (born-random material: twist
    /// brackets, insert pairs), not a large number, so it is excluded from `r`
    /// and reported as its own band. In sampled mode the per-gate ancestor count
    /// is scaled by m/K, so the bands are comparable to exact mode; `span` is
    /// omitted there (a column sample can only underestimate it).
    pub fn gen_anc_report(&self) -> String {
        if self.anc_words == 0 {
            return String::new();
        }
        // Upper bound of each band; the last band is GEN_FRESH alone.
        const EDGES: [u32; 9] = [0, 1, 2, 4, 8, 16, 32, 64, u32::MAX - 1];
        const NAMES: [&str; 9] = [
            "g0", "g1", "g2", "g3-4", "g5-8", "g9-16", "g17-32", "g33-64", "g65+",
        ];
        let nb = EDGES.len();
        let mut n = vec![0u64; nb + 1];
        let mut anc_sum = vec![0f64; nb + 1];
        let mut span_sum = vec![0f64; nb + 1];
        // Pearson accumulators over real-generation gates.
        let (mut cn, mut sx, mut sy, mut sxx, mut syy, mut sxy) =
            (0f64, 0f64, 0f64, 0f64, 0f64, 0f64);
        let scale = if self.anc_sampled {
            self.anc_m as f64 / self.anc_tracers.len().max(1) as f64
        } else {
            1.0
        };
        let mut bits = vec![0u64; self.anc_words];
        let mut cur = self.arena.head();
        while cur != NIL {
            bits.iter_mut().for_each(|w| *w = 0);
            self.anc_or_into(self.meta_of(cur).litter, &mut bits);
            let mut card = 0u64;
            let (mut lo, mut hi) = (usize::MAX, 0usize);
            for (wi, &w) in bits.iter().enumerate() {
                let mut x = w;
                while x != 0 {
                    let idx = wi * 64 + x.trailing_zeros() as usize;
                    card += 1;
                    lo = lo.min(idx);
                    hi = hi.max(idx);
                    x &= x - 1;
                }
            }
            let g = self.meta_of(cur).dgen;
            let bi = if g == GEN_FRESH {
                nb // the born-random band
            } else {
                EDGES.iter().position(|&e| g <= e).unwrap_or(nb - 1)
            };
            let a = card as f64 * scale;
            n[bi] += 1;
            anc_sum[bi] += a;
            if !self.anc_sampled && card > 0 {
                span_sum[bi] += (hi - lo) as f64;
            }
            if g != GEN_FRESH {
                let (x, y) = (g as f64, a);
                cn += 1.0;
                sx += x;
                sy += y;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }
            cur = self.arena.neighbor(cur, Dir::R);
        }
        let r = {
            let num = cn * sxy - sx * sy;
            let den = ((cn * sxx - sx * sx) * (cn * syy - sy * sy)).sqrt();
            if den > 0.0 { num / den } else { 0.0 }
        };
        let mut parts: Vec<String> = Vec::new();
        for bi in 0..=nb {
            if n[bi] == 0 {
                continue;
            }
            let label = if bi == nb { "FRESH" } else { NAMES[bi] };
            let cnt = n[bi] as f64;
            if self.anc_sampled {
                parts.push(format!("{label}:n={} anc={:.1}", n[bi], anc_sum[bi] / cnt));
            } else {
                parts.push(format!(
                    "{label}:n={} anc={:.1} span={:.0}",
                    n[bi],
                    anc_sum[bi] / cnt,
                    span_sum[bi] / cnt
                ));
            }
        }
        format!(
            "[fmix] gen-anc: r={r:.3} (n={:.0} real-gen gates) | {}",
            cn,
            parts.join(" | ")
        )
    }

    /// Drop ancestor sets for litters with no live gates. Without this the map
    /// grows with every splice for the whole run; with it, it is bounded by the
    /// live litter count (plus the litters restorable journal entries hold).
    pub(super) fn anc_prune(&mut self) {
        if self.anc_words == 0 {
            return;
        }
        let mut live: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let mut cur = self.arena.head();
        while cur != NIL {
            live.insert(self.meta_of(cur).litter);
            cur = self.arena.neighbor(cur, Dir::R);
        }
        // A restorable journal entry will put its parents' litters back, and
        // since a cross relabels ALL of its outputs to the union litter, those
        // pre-cross litters may have no live gate left. Dropping their sets
        // would make an undo restore ancestry-less litters silently. Dead
        // entries (any piece touched) can never be restored, so they hold
        // nothing.
        for e in self.journal.iter() {
            if e.after
                .iter()
                .all(|&(id, st)| self.arena.is_linked(id) && self.arena.stamp(id) == st)
            {
                live.insert(e.litters[0]);
                live.insert(e.litters[1]);
            }
        }
        self.anc.retain(|k, _| live.contains(k));
    }

    pub(super) fn fresh_litter(&mut self) -> u64 {
        let l = self.next_litter;
        self.next_litter += 1;
        l
    }

    pub(crate) fn fresh_event(&mut self) -> u64 {
        let e = self.next_event;
        self.next_event += 1;
        self.tabu.push_back((e, self.moves_done));
        while let Some(&(_, mv)) = self.tabu.front() {
            if mv + self.params.tabu_moves <= self.moves_done {
                self.tabu.pop_front();
            } else {
                break;
            }
        }
        e
    }

    pub(crate) fn is_tabu(&self, event: u64) -> bool {
        // `tabu` is only push_back'd with strictly increasing events
        // (fresh_event) and popped from the front, so it stays sorted by event:
        // a binary search finds the only possible match.
        event != 0
            && self
                .tabu
                .binary_search_by_key(&event, |&(ev, _)| ev)
                .is_ok_and(|i| self.tabu[i].1 + self.params.tabu_moves > self.moves_done)
    }

    /// Install per-gate litter ids (circuit order) from an external stage —
    /// e.g. the SGDB substitution, where each replaced gate's block becomes
    /// one litter. litter_size is recomputed per id; next_litter continues
    /// above the max.
    pub fn load_litters(&mut self, ids: &[u64]) {
        let order = self.arena.ids_in_order();
        assert_eq!(
            order.len(),
            ids.len(),
            "litter sidecar length != gate count"
        );
        let mut sizes: std::collections::HashMap<u64, u16> = std::collections::HashMap::new();
        for &l in ids {
            *sizes.entry(l).or_insert(0) += 1;
        }
        for (&id, &l) in order.iter().zip(ids.iter()) {
            let mut m = self.meta_of(id);
            m.litter = l;
            m.litter_size = sizes[&l];
            self.set_meta(id, m);
        }
        self.next_litter = ids.iter().copied().max().unwrap_or(0) + 1;
        println!(
            "[fmix] litters loaded: {} gates, {} litters, largest {}",
            ids.len(),
            sizes.len(),
            sizes.values().max().copied().unwrap_or(0)
        );
    }

    pub fn origins_in_order(&self) -> Vec<u32> {
        self.arena
            .ids_in_order()
            .iter()
            .map(|&id| self.meta_of(id).origin)
            .collect()
    }

    /// Per-gate DB-generation stamps in circuit order (GEN_FRESH = born-random
    /// material that never held input structure).
    pub fn gens_in_order(&self) -> Vec<u32> {
        self.arena
            .ids_in_order()
            .iter()
            .map(|&id| self.meta_of(id).dgen)
            .collect()
    }
}
