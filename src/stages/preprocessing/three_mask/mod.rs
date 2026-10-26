//! Stage-2 three-mask compute (RC, 2026-10): every data role carries
//! `x_i = d_i XOR` three cubic band monomials. Band wires are grouped in banks
//! that cycle through serving, retiring and growing phases; fires are
//! dirty-helper chains over the masked operand products; mask replacement is
//! a differential cubic refresh; the finished stream goes through persistent
//! carrier migration (`preprocessing_shuffling`).
//!
//! Ported from the standalone `stage2_compute_pilot` generator that produced
//! the server campaigns: same RNG streams drawn in the same order (compute
//! `seed ^ 0xC02F_17ED`, witness `seed ^ 0xA0D1_7012`, extra-band ingress
//! `seed ^ 0xF1AE_BA4D`), so one seed reproduces the generator's gates. The
//! assembly around the compute (guards, band seed and reseed) is
//! `construct::preprocess_sandwich`, which also resolves the band width.

pub mod band;
pub mod fire;
pub mod mask_rank;

use super::preprocessing_shuffling::{FireBox, apply_shuffling};
use crate::circuit::{XGate, eval_lanes};
use fire::Mask;
use rand::{Rng, SeedableRng, rngs::StdRng, seq::SliceRandom};

/// Every knob of the generator's `Config`, plus the stage-2 band policy
/// (`band_wires`). `Default` is the generator's default; [`Self::v2`] is the
/// campaign configuration.
#[derive(Clone, Debug)]
pub struct ThreeMaskParams {
    pub seed: u64,
    /// Band width for the stage-2 assembly; `0` = auto
    /// ([`auto_band_wires`]). The compute itself takes the resolved width.
    pub band_wires: usize,
    pub growth_per_fire: usize,
    pub warmup_sweeps: usize,
    pub rotation_interval: usize,
    pub maturity_sweeps: usize,
    pub keep_helper_garbage: bool,
    pub factored: bool,
    pub migration_segments: usize,
    pub schedule: bool,
    pub banks: usize,
    pub retirement_budget: usize,
    /// Zero selects half the banks minus one, retaining a growing bank.
    pub max_retiring: usize,
    pub local_masks: bool,
    pub prefer_fresh: bool,
    pub growth_sweeps_per_visit: usize,
    /// If nonzero and less than r, initialize extra band roles inside compute
    /// using this existing seeded prefix. No change to the external seed block.
    /// `preprocess_sandwich` sets this from the resolved band width.
    pub seeded_band_wires: usize,
    /// Fuse the mask's NOT into its mixed-polarity controlled gate.
    /// This preserves the exact mask function and uses the existing comp bit.
    pub compact_masks: bool,
    /// Put the quadratic mask controls in two distinct banks. The linear
    /// mask wire stays in either of those banks, so at most two are pinned.
    pub cross_bank_pairs: bool,
    /// For local masks, exclude entire banks mentioned by live operand/target
    /// masks. In particular, a data role's two persistent masks use different banks.
    pub bank_disjoint_masks: bool,
    /// Keep the two-mask nonlinear rows independent (a forest of quadratic pairs).
    pub forest_masks: bool,
    /// Experimental one-mask encoding. With forest_masks, its quadratic
    /// label must be unique among roles; the two-mask mode uses a forest.
    pub single_mask: bool,
    /// Pure cubic masks in current band coordinates. With forest_masks,
    /// cubic labels are unique across roles (also in two-mask mode).
    pub cubic_mask: bool,
    /// Use a linear-size dirty chain for cubic firing products.
    pub cubic_chain: bool,
    /// Replace this many growth gates per fire by data/band products. Their
    /// targets remain reference-free, so current decoding masks stay fixed.
    pub feedback_per_fire: usize,
    /// Split the growth budget across mid-fire and post-fire bursts, both
    /// reading this fire's current carrier. Migration boxes end before each
    /// read, so their temporary linear masks are drained correctly.
    pub midfire_feedback: bool,
    /// Differential cubic remasking and factored negative carrier literals.
    pub temporal_masks: bool,
    pub fold_cubic_polarity: bool,
    /// Grow each bank using one control from another bank, at unchanged cost.
    pub cross_bank_growth: bool,
    /// Keep a role's persistent cubic masks in distinct banks.
    pub separate_mask_banks: bool,
    /// Replace both target masks each fire; pair with one idle retirement.
    pub refresh_both_masks: bool,
    pub discard_fire_helpers: bool,
    pub mask_count: usize,
    /// Draw dirty helpers from a shuffled pool of every reference-free band wire
    /// (growing banks plus drained retiring wires), striding per product unit.
    pub random_helpers: bool,
    /// Reject a temporary cover whose cubic label is a live persistent label.
    pub unique_cover: bool,
}

impl Default for ThreeMaskParams {
    fn default() -> Self {
        Self {
            seed: 1,
            band_wires: 0,
            growth_per_fire: 16,
            warmup_sweeps: 4,
            rotation_interval: 16,
            maturity_sweeps: 4,
            keep_helper_garbage: true,
            factored: true,
            migration_segments: 8,
            schedule: true,
            banks: 4,
            retirement_budget: 1,
            max_retiring: 1,
            local_masks: false,
            prefer_fresh: false,
            growth_sweeps_per_visit: 1,
            seeded_band_wires: 0,
            compact_masks: false,
            cross_bank_pairs: false,
            bank_disjoint_masks: false,
            forest_masks: false,
            single_mask: false,
            cubic_mask: false,
            cubic_chain: false,
            feedback_per_fire: 0,
            midfire_feedback: false,
            temporal_masks: false,
            fold_cubic_polarity: false,
            cross_bank_growth: false,
            separate_mask_banks: false,
            refresh_both_masks: false,
            discard_fire_helpers: false,
            mask_count: 2,
            random_helpers: false,
            unique_cover: false,
        }
    }
}

impl ThreeMaskParams {
    /// The campaign configuration ("three-mask modified / v2"): banks of 16
    /// wires (`banks` 0 = auto, [`resolve_banks`]: 8 at the campaign's
    /// 128-wire band), retirement budget 2, rotation every fire, maturity 2, two growth
    /// sweeps per visit, 8 growth gates per fire, 4 warm-up sweeps, 6
    /// migration segments, local fresh compact forest cubic chain masks with
    /// temporal refresh, cross-bank growth, discarded fire helpers, three
    /// masks per role, random helpers and a unique cover.
    pub fn v2(seed: u64, band_wires: usize) -> Self {
        Self {
            seed,
            band_wires,
            banks: 0,
            retirement_budget: 2,
            max_retiring: 0,
            rotation_interval: 1,
            maturity_sweeps: 2,
            growth_sweeps_per_visit: 2,
            growth_per_fire: 8,
            warmup_sweeps: 4,
            migration_segments: 6,
            local_masks: true,
            prefer_fresh: true,
            compact_masks: true,
            forest_masks: true,
            cubic_mask: true,
            cubic_chain: true,
            temporal_masks: true,
            cross_bank_growth: true,
            discard_fire_helpers: true,
            mask_count: 3,
            random_helpers: true,
            unique_cover: true,
            ..Self::default()
        }
    }
}

/// The automatic band width for a sandwich of `np` (= 2n) wires: the larger
/// of `np` and 128, rounded up to a multiple of 16. The compute needs
/// `r >= 64`, `r % 8 == 0` and an even bank size of at least 8; with the
/// default eight banks, 128 is the smallest geometry that admits 64 data
/// roles times three masks (a 64-wire band fails at n = 32).
pub fn auto_band_wires(np: usize) -> usize {
    np.max(128).div_ceil(16) * 16
}

/// `band_wires` resolved against the auto policy (`0` = auto).
pub fn resolve_band_wires(np: usize, band_wires: usize) -> usize {
    if band_wires == 0 {
        auto_band_wires(np)
    } else {
        band_wires
    }
}

/// `banks` resolved against the auto policy (`0` = banks of 16 wires, the
/// design's geometry: 8 at a 128-wire band, 16 at 256).
pub fn resolve_banks(band_wires: usize, banks: usize) -> usize {
    if banks == 0 { band_wires / 16 } else { banks }
}

#[derive(Default, Debug, Clone)]
pub struct ThreeMaskStats {
    pub input_gates: usize,
    pub mask_pair_checkpoints: usize,
    pub mask_pair_bank_overlaps: usize,
    pub compute_gates: usize,
    pub before_migration: usize,
    pub fire_core_gates: usize,
    pub mask_gates: usize,
    pub growth_gates: usize,
    pub migration_gates: usize,
    pub transfers: usize,
    pub skipped_cuts: usize,
    pub retiring_replacements: usize,
    pub retired_banks: usize,
    pub rotations: Vec<band::Rotation>,
    pub fire: fire::FireStats,
    pub internal_checkpoint_samples: usize,
    pub retirement_waits: Vec<usize>,
    pub local_mask_fallbacks: usize,
    pub promotion_stalls: usize,
    pub band_ingress_gates: usize,
    pub mask_rank_rejections: usize,
    pub max_mask_rank_deficit: usize,
    pub fires_with_mask_rank_deficit: usize,
    pub feedback_gates: usize,
}

/// A mask's identity for uniqueness checks: the sorted cubic triple, or the
/// sorted quadratic pair with a zero third entry.
fn mask_key(m: Mask, cubic: bool) -> Mask {
    let mut k = m;
    if cubic {
        k.sort();
    } else {
        k = [m[0].min(m[1]), m[0].max(m[1]), 0];
    }
    k
}

#[allow(clippy::too_many_arguments)]
fn sample_persistent(
    band: &mut band::Band,
    masks: &[Vec<Mask>],
    role: usize,
    remaining: Option<Mask>,
    excluded: &[u16],
    fresh: bool,
    cfg: &ThreeMaskParams,
    rng: &mut impl Rng,
    stats: &mut ThreeMaskStats,
) -> Result<Mask, String> {
    let mut separated = excluded.to_vec();
    if cfg.separate_mask_banks
        && let Some(other) = remaining
    {
        let other_banks: Vec<_> = other.iter().map(|&w| band.bank(w)).collect();
        separated.extend(
            (band.np..band.np + band.refs.len())
                .map(|w| w as u16)
                .filter(|&w| other_banks.contains(&band.bank(w))),
        );
    }
    let excluded = separated.as_slice();
    let used_pairs: std::collections::BTreeSet<_> = masks
        .iter()
        .enumerate()
        .filter(|(w, _)| *w != role)
        .flat_map(|(_, ms)| ms.iter())
        .map(|m| mask_key(*m, cfg.cubic_mask))
        .collect();
    let mut forest =
        if cfg.forest_masks && !cfg.single_mask && !cfg.cubic_mask && remaining.is_some() {
            Some(mask_rank::Forest::without_role(masks, role))
        } else {
            None
        };
    for trial in 0..256 {
        // If the freshest bank's available pairs are all connected, broaden
        // allocation rather than repeatedly sampling that same bank forever.
        let m = if fresh && trial < 8 {
            band.sample_fresh(excluded, rng)
        } else {
            band.sample(excluded, rng)
        };
        if !(cfg.forest_masks
            && (cfg.single_mask || cfg.cubic_mask)
            && used_pairs.contains(&mask_key(m, cfg.cubic_mask)))
            && forest
                .as_mut()
                .is_none_or(|f| !f.connected(m, remaining.unwrap()))
        {
            return Ok(m);
        }
        band.release(m);
        stats.mask_rank_rejections += 1;
    }
    Err(
        "could not find an independent mask after 256 draws; increase band width or loosen bank geometry"
            .into(),
    )
}

fn dh(band: &band::Band, rng: &mut impl Rng, cfg: &ThreeMaskParams) -> Vec<u16> {
    if cfg.random_helpers {
        band.borrow_helpers_pool(rng)
    } else {
        band.borrow_helpers()
    }
}

pub struct Output {
    pub gates: Vec<XGate>,
    /// Same compute before physical carrier migration, for role-wise diagnostics.
    pub logical_gates: Vec<XGate>,
    pub fire_boundaries: Vec<usize>,
    pub stats: ThreeMaskStats,
}

// Conservatively order all accesses to each role. Randomize only among ready
// gates; every reordering consists of disjoint, hence commuting, gates.
fn schedule(gates: &[XGate], np: usize, rng: &mut impl Rng) -> Vec<XGate> {
    let mut last = vec![None; np];
    let mut deps = vec![0; gates.len()];
    let mut next = vec![vec![]; gates.len()];
    for (i, g) in gates.iter().enumerate() {
        let mut parents = vec![];
        for w in std::iter::once(g.target).chain(g.ctrls.iter().map(|l| l.0)) {
            if let Some(p) = last[w as usize]
                && !parents.contains(&p)
            {
                parents.push(p);
            }
            last[w as usize] = Some(i);
        }
        deps[i] = parents.len();
        for p in parents {
            next[p].push(i);
        }
    }
    let mut ready: Vec<_> = (0..gates.len()).filter(|&i| deps[i] == 0).collect();
    let mut out = vec![];
    while !ready.is_empty() {
        let j = rng.random_range(0..ready.len());
        let i = ready.swap_remove(j);
        out.push(gates[i].clone());
        for &k in &next[i] {
            deps[k] -= 1;
            if deps[k] == 0 {
                ready.push(k);
            }
        }
    }
    assert_eq!(out.len(), gates.len());
    out
}

/// The output realizes the input permutation on all data roles for EVERY
/// initial band state. Band outputs are unrestricted; every gate is reversible.
///
/// `r` is the resolved band width (see [`resolve_band_wires`]); the circuit
/// uses wires `0..np` for data and `np..np + r` for the band.
pub fn compute(
    input: &[XGate],
    np: usize,
    r: usize,
    cfg: &ThreeMaskParams,
) -> Result<Output, String> {
    if np < 3 || r < 64 || !r.is_multiple_of(8) || np + r > 65536 {
        return Err("need np>=3, band>=64 divisible by 8, total<=65536".into());
    }
    if cfg.banks < 4
        || !cfg.banks.is_multiple_of(2)
        || !r.is_multiple_of(cfg.banks)
        || r / cfg.banks < 8
        || !(r / cfg.banks).is_multiple_of(2)
        || cfg.max_retiring >= cfg.banks / 2
        || cfg.growth_sweeps_per_visit == 0
    {
        return Err("need even bank count>=4, even bank size>=8, positive visit sweeps, and max_retiring<bank_count/2 (0=auto)".into());
    }
    if cfg.prefer_fresh && !(cfg.local_masks || cfg.cross_bank_pairs) {
        return Err("prefer_fresh requires local_masks or cross_bank_pairs".into());
    }
    if cfg.temporal_masks && !cfg.cubic_mask {
        return Err("temporal_masks requires cubic_mask".into());
    }
    if cfg.refresh_both_masks && (!cfg.temporal_masks || cfg.single_mask) {
        return Err("refresh_both_masks requires two temporal cubic masks".into());
    }
    if cfg.local_masks && cfg.cross_bank_pairs {
        return Err("local_masks and cross_bank_pairs are alternative allocation modes".into());
    }
    if cfg.bank_disjoint_masks && !cfg.local_masks {
        return Err("bank_disjoint_masks requires local_masks".into());
    }
    if cfg.seeded_band_wires > r || (cfg.seeded_band_wires != 0 && cfg.seeded_band_wires < 2) {
        return Err("seeded band prefix must be zero (all) or between 2 and band width".into());
    }
    if input.iter().any(|g| {
        g.ctrls.len() > 2 || g.max_wire() as usize >= np || g.ctrls.iter().any(|l| l.0 == g.target)
    }) {
        return Err(
            "input must use data roles only and at most two controls, excluding its target".into(),
        );
    }
    if !(1..=3).contains(&cfg.mask_count)
        || (cfg.mask_count != 2 && !cfg.cubic_mask)
        || (cfg.discard_fire_helpers && !cfg.cubic_chain)
    {
        return Err("mask count/dirty-fire mode requires cubic chain masks".into());
    }
    if cfg.mask_count != 2 && (cfg.separate_mask_banks || cfg.refresh_both_masks) {
        return Err("bank separation and double refresh currently require two masks".into());
    }
    let mut rng = StdRng::seed_from_u64(cfg.seed ^ 0xC02F_17ED);
    let mut band = band::Band::new(np, r, cfg);
    let ordered = if cfg.schedule {
        schedule(input, np, &mut rng)
    } else {
        input.to_vec()
    };
    let mut masks: Vec<Vec<Mask>> = vec![vec![]; np];
    let mask_cost = if cfg.cubic_mask {
        if cfg.discard_fire_helpers { 3 } else { 4 }
    } else if cfg.compact_masks {
        2
    } else {
        3
    };
    let emit_mask = |t, m: Mask, band: &band::Band| {
        if cfg.cubic_mask {
            if cfg.discard_fire_helpers {
                fire::borrowed_mcx_chain_dirty(t, &m.map(|w| (w, true)), &band.borrow_helpers())
            } else {
                fire::emit_cubic_mask(t, m, &band.borrow_helpers())
            }
        } else if cfg.compact_masks {
            fire::emit_mask_compact(t, m)
        } else {
            fire::emit_mask(t, m)
        }
    };
    let mut gates = vec![];
    let mut boxes = vec![];
    let mut stats = ThreeMaskStats {
        input_gates: input.len(),
        ..ThreeMaskStats::default()
    };
    if cfg.seeded_band_wires > 0 && cfg.seeded_band_wires < r {
        // Reversible XOR fill for extra wires, whose honest input is zero.
        // Without this, a freshly promoted extra bank could carry constants.
        // Every read is from the unchanged existing seeded prefix.
        let mut srng = StdRng::seed_from_u64(cfg.seed ^ 0xF1AE_BA4D);
        let prefix = cfg.seeded_band_wires;
        for w in np + prefix..np + r {
            gates.push(XGate::cnot(
                w as u16,
                (np + srng.random_range(0..prefix)) as u16,
            ));
            for _ in 0..4 {
                let a = srng.random_range(0..prefix);
                let mut b = srng.random_range(0..prefix - 1);
                if b >= a {
                    b += 1;
                }
                gates.push(
                    XGate::conj(
                        w as u16,
                        [
                            ((np + a) as u16, srng.random_bool(0.5)),
                            ((np + b) as u16, srng.random_bool(0.5)),
                        ],
                    )
                    .unwrap(),
                );
            }
        }
        stats.band_ingress_gates = gates.len();
    }
    for w in 0..np {
        for _ in 0..if cfg.single_mask { 1 } else { cfg.mask_count } {
            let excluded: Vec<_> = masks[w].iter().flatten().copied().collect();
            let m = sample_persistent(
                &mut band,
                &masks,
                w,
                masks[w].first().copied(),
                &excluded,
                false,
                cfg,
                &mut rng,
                &mut stats,
            )?;
            masks[w].push(m);
            gates.extend(emit_mask(w as u16, m, &band));
            stats.mask_gates += mask_cost;
        }
    }
    let warm = band.growth(cfg.warmup_sweeps * (r / 2), &mut rng);
    stats.growth_gates += warm.len();
    gates.extend(warm);
    // Independent bit-sliced witness: validates every fire boundary, including
    // unrelated idle data and the changing band, before carrier migration.
    let mut vrng = StdRng::seed_from_u64(cfg.seed ^ 0xA0D1_7012);
    let mut witness: Vec<u64> = (0..np + r).map(|_| vrng.random()).collect();
    let mut reference = witness.clone();
    eval_lanes(&gates, &mut witness);
    let mut checked = gates.len();
    let mut fire_boundaries = vec![checked];
    let mut drain_cursor = 0;
    for (i, g) in ordered.iter().enumerate() {
        band.rotate(i, cfg.rotation_interval, cfg.maturity_sweeps);
        let t = g.target as usize;
        let mut excluded: Vec<_> = masks[t].iter().flatten().copied().collect();
        for &(w, _) in &g.ctrls {
            excluded.extend(masks[w as usize].iter().flatten().copied());
        }
        let cover = if cfg.unique_cover {
            sample_persistent(
                &mut band, &masks, t, None, &excluded, false, cfg, &mut rng, &mut stats,
            )?
        } else {
            band.sample(&excluded, &mut rng)
        };
        excluded.extend(cover);
        let fresh = sample_persistent(
            &mut band,
            &masks,
            t,
            masks[t].get(1).copied(),
            &excluded,
            true,
            cfg,
            &mut rng,
            &mut stats,
        )?;
        let second_fresh = if cfg.refresh_both_masks {
            let mut ex = excluded.clone();
            ex.extend(fresh);
            Some(sample_persistent(
                &mut band,
                &masks,
                t,
                Some(fresh),
                &ex,
                true,
                cfg,
                &mut rng,
                &mut stats,
            )?)
        } else {
            None
        };
        let helpers = band.helpers(&mut rng);
        let pool = dh(&band, &mut rng, cfg);
        let mut units = if cfg.cubic_mask {
            fire::compile_cubic_options_dirty(
                g,
                &masks,
                &pool,
                &mut stats.fire,
                cfg.cubic_chain,
                cfg.temporal_masks || cfg.fold_cubic_polarity,
                cfg.discard_fire_helpers,
            )
        } else {
            fire::compile(
                g,
                &masks,
                helpers,
                cfg.factored,
                cfg.keep_helper_garbage,
                &mut stats.fire,
            )
        };
        assert_eq!(
            stats.fire.pair_guard_fallbacks, 0,
            "partitioned mask allocation must admit a safe helper pairing"
        );
        units.shuffle(&mut rng);
        let mut start = gates.len();
        let mut cuts = vec![start];
        let support: Vec<_> = masks[t]
            .iter()
            .flatten()
            .copied()
            .chain(cover)
            .chain(fresh)
            .chain(second_fresh.iter().flatten().copied())
            .collect();
        gates.extend(emit_mask(g.target, cover, &band));
        stats.mask_gates += mask_cost;
        cuts.push(gates.len());
        let mid = units.len() / 2;
        for (j, unit) in units.into_iter().enumerate() {
            if j == mid {
                if cfg.midfire_feedback {
                    boxes.push(FireBox {
                        target: g.target,
                        start,
                        end: gates.len(),
                        unit_cuts: cuts,
                        mask_support: vec![(start, support.clone())],
                    });
                    let mut growth = band.growth(cfg.growth_per_fire / 2, &mut rng);
                    for gate in &mut growth {
                        *gate = XGate::conj(
                            gate.target,
                            [(g.target, rng.random_bool(0.5)), gate.ctrls[1]],
                        )
                        .unwrap();
                    }
                    stats.feedback_gates += growth.len();
                    stats.growth_gates += growth.len();
                    gates.extend(growth);
                    start = gates.len();
                    cuts = vec![start];
                }
                if !cfg.temporal_masks {
                    gates.extend(emit_mask(g.target, fresh, &band));
                    stats.mask_gates += mask_cost;
                    cuts.push(gates.len());
                }
            }
            stats.fire_core_gates += unit.len();
            gates.extend(unit);
            cuts.push(gates.len());
        }
        // A zero firing expression can still occur through contradictory input
        // literals; validated source gates normally have at least one unit.
        // XGate canonical inputs always leave a carrier product, so units>0.
        let old = masks[t].remove(0);
        masks[t].push(fresh);
        if cfg.temporal_masks {
            let delta = fire::emit_cubic_delta(
                g.target,
                old,
                fresh,
                &dh(&band, &mut rng, cfg),
                cfg.keep_helper_garbage,
            );
            stats.mask_gates += delta.len();
            gates.extend(delta);
        } else {
            gates.extend(emit_mask(g.target, old, &band));
            stats.mask_gates += mask_cost;
        }
        cuts.push(gates.len());
        band.release(old);
        if let Some(new) = second_fresh {
            let old = masks[t].remove(0);
            let delta = fire::emit_cubic_delta(
                g.target,
                old,
                new,
                &dh(&band, &mut rng, cfg),
                cfg.keep_helper_garbage,
            );
            stats.mask_gates += delta.len();
            gates.extend(delta);
            cuts.push(gates.len());
            masks[t].push(new);
            band.release(old);
        }
        gates.extend(emit_mask(g.target, cover, &band));
        stats.mask_gates += mask_cost;
        cuts.push(gates.len());
        band.release(cover);
        boxes.push(FireBox {
            target: g.target,
            start,
            end: gates.len(),
            unit_cuts: cuts,
            mask_support: vec![(start, support)],
        });
        // Bounded replacements drain even idle roles. Open first, close
        // second: the configured number of persistent masks never drops.
        for _ in 0..cfg.retirement_budget {
            // Clear zero-reference retirements between replacements, without
            // starting another promotion in the middle of a fire boundary.
            if cfg.retirement_budget > 1 {
                band.rotate(i, 0, cfg.maturity_sweeps);
            }
            let oldest = band.oldest_retiring();
            if let Some((w, j)) = (0..np).map(|d| (drain_cursor + d) % np).find_map(|w| {
                masks[w]
                    .iter()
                    .position(|m| m.iter().any(|&wire| Some(band.bank(wire)) == oldest))
                    .map(|j| (w, j))
            }) {
                let ex: Vec<_> = masks[w].iter().flatten().copied().collect();
                let new = sample_persistent(
                    &mut band,
                    &masks,
                    w,
                    masks[w]
                        .iter()
                        .enumerate()
                        .find(|(k, _)| *k != j)
                        .map(|(_, m)| *m),
                    &ex,
                    true,
                    cfg,
                    &mut rng,
                    &mut stats,
                )?;
                let old = masks[w][j];
                if cfg.temporal_masks {
                    let delta = fire::emit_cubic_delta(
                        w as u16,
                        old,
                        new,
                        &dh(&band, &mut rng, cfg),
                        cfg.keep_helper_garbage,
                    );
                    stats.mask_gates += delta.len();
                    gates.extend(delta);
                } else {
                    gates.extend(emit_mask(w as u16, new, &band));
                    gates.extend(emit_mask(w as u16, old, &band));
                    stats.mask_gates += 2 * mask_cost;
                }
                masks[w][j] = new;
                band.release(old);
                stats.retiring_replacements += 1;
                drain_cursor = (w + 1) % np;
            } else {
                break;
            }
        }
        let count = cfg.growth_per_fire
            - if cfg.midfire_feedback {
                cfg.growth_per_fire / 2
            } else {
                0
            };
        let mut growth = band.growth(count, &mut rng);
        for gate in growth.iter_mut().take(if cfg.midfire_feedback {
            count
        } else {
            cfg.feedback_per_fire
        }) {
            let data = if cfg.midfire_feedback {
                g.target
            } else {
                rng.random_range(0..np) as u16
            };
            let other = gate.ctrls[1];
            *gate = XGate::conj(gate.target, [(data, rng.random_bool(0.5)), other]).unwrap();
            stats.feedback_gates += 1;
        }
        stats.growth_gates += growth.len();
        gates.extend(growth);
        band.audit(&masks);
        for ms in &masks {
            if ms.len() == 2 {
                stats.mask_pair_checkpoints += 1;
                stats.mask_pair_bank_overlaps += usize::from(
                    ms[0]
                        .iter()
                        .any(|&a| ms[1].iter().any(|&b| band.bank(a) == band.bank(b))),
                );
            }
        }
        if cfg.separate_mask_banks {
            for ms in &masks {
                if ms.len() == 2 {
                    assert!(
                        ms[0]
                            .iter()
                            .all(|&a| ms[1].iter().all(|&b| band.bank(a) != band.bank(b)))
                    );
                }
            }
        }
        let rank_deficit = if cfg.single_mask || cfg.cubic_mask {
            let mut seen = std::collections::BTreeSet::new();
            masks
                .iter()
                .flatten()
                .filter(|m| !seen.insert(mask_key(**m, cfg.cubic_mask)))
                .count()
        } else {
            mask_rank::deficit(&masks)
        };
        stats.max_mask_rank_deficit = stats.max_mask_rank_deficit.max(rank_deficit);
        stats.fires_with_mask_rank_deficit += usize::from(rank_deficit > 0);
        if cfg.forest_masks {
            assert_eq!(rank_deficit, 0);
        }
        eval_lanes(&gates[checked..], &mut witness);
        checked = gates.len();
        eval_lanes([g], &mut reference);
        for w in 0..np {
            let decoded = masks[w].iter().fold(witness[w], |v, &m| {
                v ^ if cfg.cubic_mask {
                    fire::cubic_mask_value(m, &witness)
                } else {
                    fire::mask_value(m, &witness)
                }
            });
            assert_eq!(
                decoded, reference[w],
                "internal decode at fire {i}, role {w}"
            );
        }
        stats.internal_checkpoint_samples += 64;
        fire_boundaries.push(gates.len());
    }
    for (w, ms) in masks.iter().enumerate() {
        for &m in ms {
            gates.extend(emit_mask(w as u16, m, &band));
            stats.mask_gates += mask_cost;
            band.release(m);
        }
    }
    assert!(band.refs.iter().all(|&n| n == 0));
    stats.retired_banks = band.retired;
    stats.retirement_waits = band.retirement_waits;
    stats.local_mask_fallbacks = band.local_mask_fallbacks;
    stats.promotion_stalls = band.promotion_stalls;
    stats.rotations = band.rotations;
    stats.before_migration = gates.len();
    let shifted = apply_shuffling(
        &gates,
        &boxes,
        np,
        np + r,
        if cfg.midfire_feedback {
            cfg.migration_segments.div_ceil(2)
        } else {
            cfg.migration_segments
        },
        false,
        0,
        cfg.seed,
    );
    assert!(
        shifted.final_layout[..np]
            .iter()
            .enumerate()
            .all(|(i, &w)| i == w as usize)
    );
    stats.migration_gates = shifted.stats.added_gates;
    stats.transfers = shifted.stats.transfer_count;
    stats.skipped_cuts = shifted.stats.skipped_cut_count;
    stats.compute_gates = shifted.gates.len();
    assert_eq!(
        stats.before_migration,
        stats.fire_core_gates + stats.mask_gates + stats.growth_gates + stats.band_ingress_gates
    );
    Ok(Output {
        gates: shifted.gates,
        logical_gates: gates,
        fire_boundaries,
        stats,
    })
}

#[cfg(test)]
#[path = "../../../../tests/unit/stages/preprocessing/three_mask.rs"]
mod tests;
