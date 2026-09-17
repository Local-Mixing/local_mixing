//! Physical role transfers inside embedded-masking fire boxes.
//!
//! The emitter records exact unit boundaries and target-mask support in logical
//! coordinates. This pass moves a write-only target onto eligible band roles,
//! retaining each partner value as a temporary XOR mask until the box ends.
//! Gates outside boxes, including the final mask drain, use the same role map.
//! With data returned home, the last transfer removes the first temporary mask
//! after every original gate in the box, fencing the final carrier segment.
//! Band roles stay permuted between boxes.

use crate::circuit::xgate::{XGate, sort_lits};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Exact boundaries supplied by the embedded-masking emitter. Band writes inside
/// a unit must restore their borrowed values before the next unit boundary.
#[derive(Clone, Debug)]
pub(crate) struct FireBox {
    pub target: u16,
    pub start: usize,
    pub end: usize,
    pub unit_cuts: Vec<usize>,
    /// Open target-mask support at absolute cuts, starting with `start`.
    pub mask_support: Vec<(usize, Vec<u16>)>,
}

/// A two-CNOT role transfer, recorded before the source gate at `cut`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShufflingTransfer {
    pub cut: usize,
    pub partner: u16,
    pub from: u16,
    pub to: u16,
    pub returning_home: bool,
}

/// Physical write counts for one fire box. Sparse histograms are wire-sorted.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShufflingBoxStats {
    pub target: u16,
    pub source_start: usize,
    pub source_end: usize,
    pub output_start: usize,
    pub output_end: usize,
    pub original_gates: usize,
    pub added_gates: usize,
    pub skipped_cut_count: usize,
    /// Every emitted write, including transfers and temporary-mask removal.
    pub all_target_counts: Vec<(u16, usize)>,
    /// Only the original gates writing the logical source target.
    pub original_target_counts: Vec<(u16, usize)>,
    pub transfers: Vec<ShufflingTransfer>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ShufflingStats {
    pub original_gates: usize,
    pub shuffled_gates: usize,
    pub added_gates: usize,
    pub transfer_count: usize,
    pub skipped_cut_count: usize,
    pub boxes: Vec<ShufflingBoxStats>,
}

pub(crate) struct ShufflingOutput {
    pub gates: Vec<XGate>,
    /// Physical location for each logical role after the complete gate stream.
    pub final_layout: Vec<u16>,
    pub stats: ShufflingStats,
}

/// Preserve literal polarities and canonical control order under a role map.
pub(crate) fn remap_gate(gate: &XGate, layout: &[u16]) -> XGate {
    let mut mapped = XGate {
        target: layout[gate.target as usize],
        comp: gate.comp,
        ctrls: gate
            .ctrls
            .iter()
            .map(|&(wire, positive)| (layout[wire as usize], positive))
            .collect(),
    };
    sort_lits(&mut mapped.ctrls);
    mapped
}

fn validate_ledger(gates: &[XGate], boxes: &[FireBox], np: usize, total: usize) {
    assert!(np <= total && total <= u16::MAX as usize + 1);
    assert!(gates.iter().all(|gate| (gate.max_wire() as usize) < total));
    let mut previous_end = 0;
    for fire_box in boxes {
        assert!(
            (fire_box.target as usize) < np,
            "fire target must be a data role"
        );
        assert!(previous_end <= fire_box.start && fire_box.start < fire_box.end);
        assert!(fire_box.end <= gates.len());
        assert_eq!(fire_box.unit_cuts.first(), Some(&fire_box.start));
        assert_eq!(fire_box.unit_cuts.last(), Some(&fire_box.end));
        assert!(fire_box.unit_cuts.windows(2).all(|pair| pair[0] < pair[1]));
        assert_eq!(
            fire_box.mask_support.first().map(|item| item.0),
            Some(fire_box.start)
        );
        assert!(
            fire_box
                .mask_support
                .windows(2)
                .all(|pair| pair[0].0 < pair[1].0)
        );
        for (cut, support) in &fire_box.mask_support {
            assert!(fire_box.unit_cuts.binary_search(cut).is_ok());
            assert!(
                support
                    .iter()
                    .all(|&wire| np <= wire as usize && (wire as usize) < total)
            );
        }
        assert!(
            gates[fire_box.start..fire_box.end]
                .iter()
                .all(|gate| !gate.reads(fire_box.target)),
            "preprocessing shuffling requires a target that is never read inside its fire box"
        );
        previous_end = fire_box.end;
    }
}

fn selected_cuts(gates: &[XGate], fire_box: &FireBox, segments: usize) -> Vec<usize> {
    let mut prefix = vec![0usize; fire_box.end - fire_box.start + 1];
    for (offset, gate) in gates[fire_box.start..fire_box.end].iter().enumerate() {
        prefix[offset + 1] = prefix[offset] + usize::from(gate.target == fire_box.target);
    }
    let writes = *prefix.last().unwrap();
    if writes == 0 {
        return Vec::new();
    }
    let mut cuts = vec![fire_box.start];
    let interior: Vec<usize> = fire_box
        .unit_cuts
        .iter()
        .copied()
        .filter(|&cut| fire_box.start < cut && cut < fire_box.end)
        .collect();
    if interior.is_empty() {
        return cuts;
    }
    // For a huge requested segment count, only grid points neighboring each
    // boundary's write count can introduce a new nearest boundary. A boundary's
    // nearest-cut interval contains its own count; if that interval contains
    // any grid point, it contains a neighboring floor/ceil point too. This
    // preserves the exact quantile selection while bounding work by the ledger.
    let quantiles: Vec<usize> = if segments <= interior.len().saturating_mul(2) {
        (1..segments).collect()
    } else {
        let mut neighboring = Vec::with_capacity(interior.len() * 2);
        for &cut in &interior {
            let numerator = prefix[cut - fire_box.start] as u128 * segments as u128;
            let floor = numerator / writes as u128;
            let ceil = floor + u128::from(numerator % writes as u128 != 0);
            for quantile in [floor, ceil] {
                neighboring.push(quantile.clamp(1, (segments - 1) as u128) as usize);
            }
        }
        neighboring.sort_unstable();
        neighboring.dedup();
        neighboring
    };
    for segment in quantiles {
        // Compare rational distances without float rounding; ties prefer the
        // lower absolute cut. u128 also avoids multiplying gate counts in usize.
        let ideal = writes as u128 * segment as u128;
        let cut = interior
            .iter()
            .copied()
            .min_by_key(|&cut| {
                (
                    (prefix[cut - fire_box.start] as u128 * segments as u128).abs_diff(ideal),
                    cut,
                )
            })
            .unwrap();
        cuts.push(cut);
    }
    cuts.sort_unstable();
    cuts.dedup();
    cuts
}

fn sparse_counts(counts: &[usize]) -> Vec<(u16, usize)> {
    counts
        .iter()
        .enumerate()
        .filter_map(|(wire, &count)| (count != 0).then_some((wire as u16, count)))
        .collect()
}

/// Last target-writing unit that reads each role, in the original coordinates.
/// A dirty-helper bracket can read a role only in its helper writes while that
/// role still participates in the bracket's net target update. Excluding only
/// controls on individual target writes would miss that dependency.
fn target_unit_read_ends(gates: &[XGate], fire_box: &FireBox, total: usize) -> Vec<usize> {
    let mut last_read_end = vec![0; total];
    for cuts in fire_box.unit_cuts.windows(2) {
        let unit = &gates[cuts[0]..cuts[1]];
        if unit.iter().any(|gate| gate.target == fire_box.target) {
            for gate in unit {
                for &(role, _) in &gate.ctrls {
                    last_read_end[role as usize] = cuts[1];
                }
            }
        }
    }
    last_read_end
}

/// Apply transfers using a separate deterministic RNG; disabled builds do not
/// consume any of the emitter's random draws. The seeded stream is specific to
/// this implementation, rather than byte-compatible with the research harness.
pub(crate) fn apply_shuffling(
    gates: &[XGate],
    boxes: &[FireBox],
    np: usize,
    total: usize,
    segments: usize,
    return_home: bool,
    seed: u64,
) -> ShufflingOutput {
    assert!(total <= u16::MAX as usize + 1);
    let mut layout: Vec<u16> = (0..total).map(|wire| wire as u16).collect();
    if segments == 0 {
        return ShufflingOutput {
            gates: gates.to_vec(),
            final_layout: layout,
            stats: ShufflingStats {
                original_gates: gates.len(),
                shuffled_gates: gates.len(),
                ..ShufflingStats::default()
            },
        };
    }
    assert!(
        segments >= 8,
        "preprocessing shuffling requires at least eight segments"
    );
    validate_ledger(gates, boxes, np, total);
    let route_seed =
        xxhash_rust::xxh3::xxh3_64_with_seed(b"embedded-masking/preprocessing-shuffling/v1", seed);
    let mut rng = StdRng::seed_from_u64(route_seed);
    // One start cut plus interior cuts, each costing at most three gates.
    // Reserve once so the first insertion does not double a large gate buffer.
    let additional_capacity: usize = boxes
        .iter()
        .map(|fire_box| segments.min(fire_box.unit_cuts.len() - 1) * 3)
        .sum();
    let mut out = Vec::with_capacity(gates.len() + additional_capacity);
    let mut stats = ShufflingStats {
        original_gates: gates.len(),
        ..ShufflingStats::default()
    };
    let mut cursor = 0;
    for fire_box in boxes {
        out.extend(
            gates[cursor..fire_box.start]
                .iter()
                .map(|gate| remap_gate(gate, &layout)),
        );
        let target = fire_box.target as usize;
        if return_home {
            assert_eq!(layout[target], fire_box.target);
        }
        let mut cuts = selected_cuts(gates, fire_box, segments);
        // Reserve the last selected transfer for returning home. It must be
        // after the box's final target write: an earlier return leaves the last
        // segment unfenced by the carrier-transfer dependencies.
        if return_home && let Some(last) = cuts.last_mut() {
            *last = fire_box.end;
        }
        let last_read_end = target_unit_read_ends(gates, fire_box, total);
        let mut box_stats = ShufflingBoxStats {
            target: fire_box.target,
            source_start: fire_box.start,
            source_end: fire_box.end,
            output_start: out.len(),
            original_gates: fire_box.end - fire_box.start,
            ..ShufflingBoxStats::default()
        };
        let mut counts = vec![0usize; total];
        let mut original_counts = vec![0usize; total];
        let mut used = vec![false; total];
        let mut pending_masks: Vec<u16> = Vec::new();
        let mut next_cut = 0;
        let mut support_index = 0;
        for cut in fire_box.start..=fire_box.end {
            while support_index + 1 < fire_box.mask_support.len()
                && fire_box.mask_support[support_index + 1].0 <= cut
            {
                support_index += 1;
            }
            if cuts.get(next_cut) == Some(&cut) {
                let returning_home = return_home && next_cut + 1 == cuts.len();
                let partner = if returning_home {
                    pending_masks.first().copied()
                } else {
                    let mut excluded = used.clone();
                    for &wire in &fire_box.mask_support[support_index].1 {
                        excluded[wire as usize] = true;
                    }
                    let mut candidates = Vec::new();
                    let mut least_load = usize::MAX;
                    for role in np..total {
                        // Every control in a remaining target-writing unit is
                        // excluded, using roles rather than physical locations.
                        if excluded[role] || last_read_end[role] > cut {
                            continue;
                        }
                        let load = counts[layout[role] as usize];
                        if load < least_load {
                            candidates.clear();
                            least_load = load;
                        }
                        if load == least_load {
                            candidates.push(role as u16);
                        }
                    }
                    if candidates.is_empty() {
                        None
                    } else {
                        Some(candidates[rng.random_range(0..candidates.len())])
                    }
                };
                if let Some(partner) = partner {
                    let from = layout[target];
                    let to = layout[partner as usize];
                    out.push(XGate::cnot(to, from));
                    out.push(XGate::cnot(from, to));
                    counts[to as usize] += 1;
                    counts[from as usize] += 1;
                    layout.swap(target, partner as usize);
                    if returning_home {
                        // Home held the first temporary mask: the same two
                        // CNOTs remove it while returning the data role home.
                        assert_eq!(to, fire_box.target);
                        pending_masks.remove(0);
                    } else {
                        used[partner as usize] = true;
                        pending_masks.push(partner);
                    }
                    box_stats.transfers.push(ShufflingTransfer {
                        cut,
                        partner,
                        from,
                        to,
                        returning_home,
                    });
                } else {
                    box_stats.skipped_cut_count += 1;
                }
                next_cut += 1;
            }
            // A skipped transfer must never skip the corresponding source gate.
            if cut < fire_box.end {
                let gate = &gates[cut];
                let mapped = remap_gate(gate, &layout);
                counts[mapped.target as usize] += 1;
                if gate.target == fire_box.target {
                    original_counts[mapped.target as usize] += 1;
                }
                out.push(mapped);
            }
        }
        for partner in pending_masks {
            out.push(XGate::cnot(layout[target], layout[partner as usize]));
            counts[layout[target] as usize] += 1;
        }
        if return_home {
            assert_eq!(layout[target], fire_box.target);
        }
        box_stats.output_end = out.len();
        box_stats.added_gates =
            box_stats.output_end - box_stats.output_start - box_stats.original_gates;
        box_stats.all_target_counts = sparse_counts(&counts);
        box_stats.original_target_counts = sparse_counts(&original_counts);
        stats.transfer_count += box_stats.transfers.len();
        stats.skipped_cut_count += box_stats.skipped_cut_count;
        stats.boxes.push(box_stats);
        cursor = fire_box.end;
    }
    out.extend(gates[cursor..].iter().map(|gate| remap_gate(gate, &layout)));
    if return_home {
        assert!(
            layout[..np]
                .iter()
                .enumerate()
                .all(|(role, &wire)| role == wire as usize)
        );
    }
    stats.shuffled_gates = out.len();
    stats.added_gates = out.len() - gates.len();
    ShufflingOutput {
        gates: out,
        final_layout: layout,
        stats,
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/preprocessing/preprocessing_shuffling.rs"]
mod tests;
