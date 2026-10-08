//! Physical role transfers inside embedded-masking fire boxes.
//!
//! The emitter records exact unit boundaries and target-mask support in logical
//! coordinates. This pass moves a write-only target onto eligible band roles,
//! retaining each partner value as a temporary XOR mask until the box ends.
//! Gates outside boxes, including the final mask drain, use the same role map.
//! With data returned home, the last transfer removes the first temporary mask
//! after every original gate in the box, fencing the final carrier segment.
//! Band roles stay permuted between boxes. With `return_home` false the data
//! roles migrate PERSISTENTLY across boxes (value hopping, RC 2026-10-05) and
//! are swapped back to their physical ports only once, at the stream tail.

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
    /// Data roles swapped back to their home wires by the persistent-mode
    /// tail restoration (always 0 under per-box return home).
    pub restored_ports: usize,
    /// Fresh single-CNOT masks applied by just-in-time drain management to
    /// bridge idle data-role stretches (0 with eager per-box drains).
    pub jit_remasks: usize,
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

/// Reads since the last write, per PHYSICAL wire, over the emitted stream.
fn note_reads(rsw: &mut [usize], gate: &XGate) {
    for &(wire, _) in &gate.ctrls {
        rsw[wire as usize] += 1;
    }
    rsw[gate.target as usize] = 0;
}

/// Just-in-time drain management (punchlist #8, RC 2026-10-06). Transfer
/// masks survive their fire box; each standing mask drains at the last
/// admissible moment -- immediately before a gate READS its masked data role
/// (the reader needs the bare value) or WRITES its holder band role (the
/// drain must see the value the mask was applied with). Writes to the masked
/// role itself pass through the mask, so one mask protects the whole chain of
/// values until the next read. Whenever a data role goes bare and its next
/// touch is farther than `gap` source gates, a fresh single-CNOT mask bridges
/// the idle stretch, so bare role values exist only in short, read-adjacent
/// stretches that delivery windows can swallow -- instead of being reborn at
/// every fire-box end as the migration-drain celebrity class. Correctness is
/// local XOR algebra under the two triggers; every heuristic below only
/// affects cost and exposure, never function.
struct JitState {
    gap: usize,
    /// (masked data role, holder band role), in application order.
    pending: Vec<(usize, usize)>,
    /// Standing masks held per band role. A holder may serve as a partner
    /// for OTHER targets (its value is tracked through role moves), but
    /// never against a target it already masks: that reuse would silently
    /// cancel the standing mask while the ledger still counts it.
    holder: Vec<u32>,
    /// The data roles each band role currently masks (same-target reuse bar).
    holder_targets: Vec<Vec<usize>>,
    /// Standing masks per data role.
    masked: Vec<u32>,
    /// Original-stream read/write positions per role, for horizon queries.
    read_pos: Vec<Vec<u32>>,
    write_pos: Vec<Vec<u32>>,
    np: usize,
    source_len: usize,
    /// Data roles whose mask was drained by a holder-write trigger at the
    /// current gate; they get a re-mask evaluation even though the gate does
    /// not touch them.
    orphans: Vec<usize>,
    remasks: usize,
    rng: StdRng,
}

impl JitState {
    fn new(gates: &[XGate], np: usize, total: usize, gap: usize, seed: u64) -> Self {
        let mut read_pos = vec![Vec::new(); total];
        let mut write_pos = vec![Vec::new(); total];
        for (index, gate) in gates.iter().enumerate() {
            write_pos[gate.target as usize].push(index as u32);
            for &(wire, _) in &gate.ctrls {
                read_pos[wire as usize].push(index as u32);
            }
        }
        Self {
            gap,
            pending: Vec::new(),
            holder: vec![0; total],
            holder_targets: vec![Vec::new(); total],
            masked: vec![0; total],
            read_pos,
            write_pos,
            np,
            source_len: gates.len(),
            orphans: Vec::new(),
            remasks: 0,
            rng: StdRng::seed_from_u64(xxhash_rust::xxh3::xxh3_64_with_seed(
                b"embedded-masking/preprocessing-shuffling/jit/v1",
                seed,
            )),
        }
    }

    fn next_after(positions: &[u32], index: usize, default: usize) -> usize {
        let at = positions.partition_point(|&position| position as usize <= index);
        positions
            .get(at)
            .map_or(default, |&position| position as usize)
    }

    fn register_transfer_mask(&mut self, target: usize, partner: usize) {
        self.pending.push((target, partner));
        self.masked[target] += 1;
        self.holder[partner] += 1;
        self.holder_targets[partner].push(target);
    }

    /// True when `partner` already holds a standing mask for `target`.
    fn holds_for(&self, partner: usize, target: usize) -> bool {
        self.holder_targets[partner].contains(&target)
    }

    fn drain_at(
        &mut self,
        index: usize,
        layout: &[u16],
        out: &mut Vec<XGate>,
        rsw: &mut [usize],
        counts: &mut [usize],
    ) {
        let (target, partner) = self.pending.remove(index);
        let drain = XGate::cnot(layout[target], layout[partner]);
        note_reads(rsw, &drain);
        counts[drain.target as usize] += 1;
        out.push(drain);
        self.masked[target] -= 1;
        self.holder[partner] -= 1;
        let slot = self.holder_targets[partner]
            .iter()
            .position(|&held| held == target)
            .expect("holder ledger out of sync");
        self.holder_targets[partner].swap_remove(slot);
    }

    /// Drain every standing mask whose trigger fires at this source gate.
    fn before_gate(
        &mut self,
        gate: &XGate,
        layout: &[u16],
        out: &mut Vec<XGate>,
        rsw: &mut [usize],
        counts: &mut [usize],
    ) {
        let mut index = 0;
        while index < self.pending.len() {
            let (target, partner) = self.pending[index];
            if gate.reads(target as u16) {
                self.drain_at(index, layout, out, rsw, counts);
            } else if gate.target as usize == partner {
                self.orphans.push(target);
                self.drain_at(index, layout, out, rsw, counts);
            } else {
                index += 1;
            }
        }
    }

    /// Re-mask every data role this gate left bare with a far next touch.
    fn after_gate(
        &mut self,
        gate: &XGate,
        source_index: usize,
        layout: &[u16],
        out: &mut Vec<XGate>,
        rsw: &mut [usize],
        counts: &mut [usize],
    ) {
        let mut roles = std::mem::take(&mut self.orphans);
        if (gate.target as usize) < self.np {
            roles.push(gate.target as usize);
        }
        for &(wire, _) in &gate.ctrls {
            if (wire as usize) < self.np {
                roles.push(wire as usize);
            }
        }
        for role in roles {
            self.maybe_remask(role, source_index, layout, out, rsw, counts);
        }
    }

    fn maybe_remask(
        &mut self,
        role: usize,
        source_index: usize,
        layout: &[u16],
        out: &mut Vec<XGate>,
        rsw: &mut [usize],
        counts: &mut [usize],
    ) {
        if self.masked[role] > 0 {
            return;
        }
        let horizon = Self::next_after(&self.read_pos[role], source_index, self.source_len).min(
            Self::next_after(&self.write_pos[role], source_index, self.source_len),
        );
        if horizon - source_index <= self.gap {
            return;
        }
        // The holder must outlive the stretch it protects or it only trades
        // one bare rebirth for churn: require its next write beyond the gap,
        // and prefer the longest-lived candidates.
        // Prefer holders of nothing (distinct mask vocabulary); fall back to
        // roles masking other targets (safe -- only same-target reuse cancels).
        let mut best = Vec::new();
        let mut best_key = (0usize, 0usize);
        for candidate in self.np..self.holder.len() {
            if self.holds_for(candidate, role) {
                continue;
            }
            let next_write =
                Self::next_after(&self.write_pos[candidate], source_index, self.source_len);
            if next_write <= source_index + self.gap {
                continue;
            }
            let key = (usize::from(self.holder[candidate] == 0), next_write);
            if key > best_key {
                best_key = key;
                best.clear();
            }
            if key == best_key {
                best.push(candidate);
            }
        }
        if best.is_empty() {
            return;
        }
        let partner = best[self.rng.random_range(0..best.len())];
        let mask = XGate::cnot(layout[role], layout[partner]);
        note_reads(rsw, &mask);
        counts[mask.target as usize] += 1;
        out.push(mask);
        self.register_transfer_mask(role, partner);
        self.remasks += 1;
    }

    fn drain_all(
        &mut self,
        layout: &[u16],
        out: &mut Vec<XGate>,
        rsw: &mut [usize],
        counts: &mut [usize],
    ) {
        while !self.pending.is_empty() {
            self.drain_at(0, layout, out, rsw, counts);
        }
    }
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
    jit_gap: usize,
    seed: u64,
) -> ShufflingOutput {
    assert!(total <= u16::MAX as usize + 1);
    assert!(
        jit_gap == 0 || !return_home,
        "just-in-time drains require persistent migration (return_home=false)"
    );
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
        segments >= 1,
        "preprocessing shuffling requires at least one segment"
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
    // Reads since the last write, per PHYSICAL wire, maintained over the
    // emitted stream: partner selection prefers the highest-pressure wire
    // (punchlist #6), so every transfer doubles as a read-run cap.
    let mut rsw = vec![0usize; total];
    let mut jit = (jit_gap > 0).then(|| JitState::new(gates, np, total, jit_gap, route_seed));
    // Write counts of JIT gates emitted outside any box span; never reported.
    let mut loose_counts = vec![0usize; total];
    let mut cursor = 0;
    for fire_box in boxes {
        for source_index in cursor..fire_box.start {
            let gate = &gates[source_index];
            if let Some(jit) = &mut jit {
                jit.before_gate(gate, &layout, &mut out, &mut rsw, &mut loose_counts);
            }
            let mapped = remap_gate(gate, &layout);
            note_reads(&mut rsw, &mapped);
            out.push(mapped);
            if let Some(jit) = &mut jit {
                jit.after_gate(
                    gate,
                    source_index,
                    &layout,
                    &mut out,
                    &mut rsw,
                    &mut loose_counts,
                );
            }
        }
        let target = fire_box.target as usize;
        if return_home {
            assert_eq!(layout[target], fire_box.target);
        }
        let mut cuts = selected_cuts(gates, fire_box, segments);
        // Any band role WRITTEN inside this box may carry a value change at
        // some cut (restore-dropped drift ancillas since RC 2026-10-06; and
        // mid-bracket toggles are only epoch-stable at unit granularity), so
        // none of them may serve as a transfer partner here: a pending
        // transfer mask must stay constant from its cut to the drain.
        let written_in_box: Vec<bool> = {
            let mut wr = vec![false; total];
            for gate in &gates[fire_box.start..fire_box.end] {
                if (gate.target as usize) >= np {
                    wr[gate.target as usize] = true;
                }
            }
            wr
        };
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
                    let mut most_pressure = 0usize;
                    for role in np..total {
                        // Every control in a remaining target-writing unit is
                        // excluded, using roles rather than physical locations.
                        // A role already masking THIS target is excluded
                        // too: re-use would cancel its standing mask.
                        if excluded[role]
                            || last_read_end[role] > cut
                            || written_in_box[role]
                            || jit.as_ref().is_some_and(|jit| jit.holds_for(role, target))
                        {
                            continue;
                        }
                        // Highest reads-since-last-write first (punchlist #6):
                        // the heavy readers are exactly the wires that need a
                        // refreshing write.
                        let pressure = rsw[layout[role] as usize];
                        if pressure > most_pressure {
                            candidates.clear();
                            most_pressure = pressure;
                        }
                        if pressure == most_pressure {
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
                    let first = XGate::cnot(to, from);
                    note_reads(&mut rsw, &first);
                    out.push(first);
                    let second = XGate::cnot(from, to);
                    note_reads(&mut rsw, &second);
                    out.push(second);
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
                        if let Some(jit) = &mut jit {
                            // Deferred drains: the mask outlives the box and
                            // drains at its triggers instead of the box end.
                            jit.register_transfer_mask(target, partner as usize);
                        } else {
                            pending_masks.push(partner);
                        }
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
                if let Some(jit) = &mut jit {
                    jit.before_gate(gate, &layout, &mut out, &mut rsw, &mut counts);
                }
                let mapped = remap_gate(gate, &layout);
                counts[mapped.target as usize] += 1;
                if gate.target == fire_box.target {
                    original_counts[mapped.target as usize] += 1;
                }
                note_reads(&mut rsw, &mapped);
                out.push(mapped);
                if let Some(jit) = &mut jit {
                    jit.after_gate(gate, cut, &layout, &mut out, &mut rsw, &mut counts);
                }
            }
        }
        for partner in pending_masks {
            let drain = XGate::cnot(layout[target], layout[partner as usize]);
            note_reads(&mut rsw, &drain);
            out.push(drain);
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
    for source_index in cursor..gates.len() {
        let gate = &gates[source_index];
        if let Some(jit) = &mut jit {
            jit.before_gate(gate, &layout, &mut out, &mut rsw, &mut loose_counts);
        }
        let mapped = remap_gate(gate, &layout);
        note_reads(&mut rsw, &mapped);
        out.push(mapped);
        if let Some(jit) = &mut jit {
            jit.after_gate(
                gate,
                source_index,
                &layout,
                &mut out,
                &mut rsw,
                &mut loose_counts,
            );
        }
    }
    if let Some(jit) = &mut jit {
        // Public contract: every data role is bare at the stream end. Drain
        // the remaining standing masks before the port-restoration swaps.
        jit.drain_all(&layout, &mut out, &mut rsw, &mut loose_counts);
        stats.jit_remasks = jit.remasks;
    }
    if !return_home {
        // Persistent migration: data roles stayed wherever their last
        // transfer left them for the whole stream. Restore their physical
        // ports here, at the very tail, with full 3-CNOT swaps so every
        // downstream port contract (payload readout, verify, Z-side
        // material) holds unchanged. Band roles stay permuted, exactly as
        // they already do under per-box return home.
        let mut phys_role: Vec<usize> = vec![0; total];
        for (role, &wire) in layout.iter().enumerate() {
            phys_role[wire as usize] = role;
        }
        for role in 0..np {
            let wire = layout[role];
            if wire as usize != role {
                let home = role as u16;
                let displaced = phys_role[role];
                out.push(XGate::cnot(home, wire));
                out.push(XGate::cnot(wire, home));
                out.push(XGate::cnot(home, wire));
                layout[role] = home;
                layout[displaced] = wire;
                phys_role[role] = role;
                phys_role[wire as usize] = displaced;
                stats.restored_ports += 1;
            }
        }
    }
    assert!(
        layout[..np]
            .iter()
            .enumerate()
            .all(|(role, &wire)| role == wire as usize),
        "data roles must be physically home at the end of the stream"
    );
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
