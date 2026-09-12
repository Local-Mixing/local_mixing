//! Applying a separately selected quality-control repair to a live mixer.
//!
//! Selection and leakage testing belong to `db_mixing::quality`. This adapter
//! only validates and commits one plan while retaining the mixer's identity,
//! provenance, random stream, and ordinary sampling statistics.

use super::{DbMode, Dir, Mixer, NIL, XGate};
use crate::stages::db_mixing::leakage_repair::blocks::ConvexBlock;
use crate::stages::db_mixing::replacement::polys_equivalent;

impl Mixer {
    /// The original circuit retained by checkpoints and global verification.
    pub fn quality_reference(&self) -> &[XGate] {
        &self.original
    }

    pub fn quality_num_wires(&self) -> usize {
        self.num_wires
    }

    /// Commit a screened convex-block repair without reconstructing the mixer.
    ///
    /// `plan.permutation` gives old global positions in the new order for
    /// `plan.span` only. `plan.block` is the resulting contiguous replacement
    /// range, with an exclusive end. Plans must describe the current circuit;
    /// callers must replan after any intervening mutation.
    ///
    /// Both the gather's commuting swaps and exact replacement equivalence are
    /// checked before mutation, including when ordinary DB verification is off.
    /// The replacement stays at its screened position: birth advancement is
    /// disabled for this splice. QC counts belong to the caller's QC report,
    /// independently of the random sampler's attempt and hit statistics.
    pub fn apply_quality_repair(
        &mut self,
        plan: &ConvexBlock,
        replacement: Vec<XGate>,
    ) -> Result<(), String> {
        let span = &plan.span;
        let block = &plan.block;
        if span.start >= span.end
            || span.end > self.arena.len()
            || block.start < span.start
            || block.start >= block.end
            || block.end > span.end
            || plan.permutation.len() != span.len()
        {
            return Err("quality repair has an invalid span or block range".into());
        }
        if replacement.is_empty() && block.len() == self.arena.len() {
            return Err("quality repair would empty the checkpoint's circuit".into());
        }
        for gate in &replacement {
            if gate.target as usize >= self.num_wires
                || gate
                    .ctrls
                    .iter()
                    .any(|&(wire, _)| wire as usize >= self.num_wires || wire == gate.target)
                || gate.ctrls.windows(2).any(|pair| pair[0].0 >= pair[1].0)
            {
                return Err("quality replacement contains an invalid gate or wire".into());
            }
        }

        let mut seen = vec![false; span.len()];
        for &position in &plan.permutation {
            if !span.contains(&position) || seen[position - span.start] {
                return Err("quality repair order is not a permutation of its span".into());
            }
            seen[position - span.start] = true;
        }
        let relative_block = block.start - span.start..block.end - span.start;
        let mut selected = plan.selected.clone();
        selected.sort_unstable();
        let mut gathered = plan.permutation[relative_block.clone()].to_vec();
        gathered.sort_unstable();
        if selected != gathered {
            return Err("quality repair block does not contain its selected gates".into());
        }

        // Walk to the span without allocating an ID vector for the full circuit.
        let ids: Vec<u32> = self
            .arena
            .ids_in_order_iter()
            .skip(span.start)
            .take(span.len())
            .collect();
        let mut order: Vec<usize> = (0..ids.len()).collect();
        let mut positions = order.clone();
        // Stable extraction gives an explicit adjacent-swap proof. Only
        // inversions cost collision checks; an unchanged long span stays linear.
        for (destination, &global_position) in plan.permutation.iter().enumerate() {
            let old_position = global_position - span.start;
            let mut position = positions[old_position];
            while position > destination {
                let crossed = order[position - 1];
                if self.arena.collides_ids(ids[old_position], ids[crossed]) {
                    return Err("quality gather would cross noncommuting gates".into());
                }
                order.swap(position - 1, position);
                positions[crossed] = position;
                position -= 1;
                positions[old_position] = position;
            }
        }
        let reordered_ids: Vec<u32> = order.iter().map(|&position| ids[position]).collect();
        let block_ids = &reordered_ids[relative_block];
        let window: Vec<XGate> = block_ids
            .iter()
            .map(|&id| self.arena.gate(id).clone())
            .collect();
        // ANF comparison is exact and budgeted. If it cannot decide, retain
        // the existing exhaustive fallback for small enough support.
        let equivalent = polys_equivalent(&window, &replacement, self.db_budget).or_else(|| {
            let mut support: Vec<u16> = window
                .iter()
                .chain(&replacement)
                .flat_map(|gate| {
                    std::iter::once(gate.target).chain(gate.ctrls.iter().map(|&(wire, _)| wire))
                })
                .collect();
            support.sort_unstable();
            support.dedup();
            (support.len() <= 24).then(|| super::rules::verify_rewrite(&window, &replacement))
        });
        match equivalent {
            Some(true) => {}
            Some(false) => return Err("quality replacement is not functionally equivalent".into()),
            None => return Err("quality replacement equivalence exceeded the proof budget".into()),
        }

        // All fallible checks precede mutation. Relinking retains node IDs,
        // stamps, directions and metadata for every gate outside the block.
        let left = self.arena.neighbor(ids[0], Dir::L);
        let direction = self.meta_of(block_ids[0]).dir;
        for &id in &ids {
            self.arena.unlink(id);
        }
        let mut cursor = left;
        for &id in &reordered_ids {
            self.arena.link_after(id, cursor);
            cursor = id;
        }

        // Reuse the DB splice's generation, ancestry, litter, index and stamp
        // hooks. This is a QC event, so preserve the ordinary sampler's counters
        // and recorder. Its ledgers account only for their respective channels.
        let counters = std::mem::take(&mut self.counters);
        let recorder = self.db_record.take();
        let advance = std::mem::replace(&mut self.params.db_advance, false);
        let verify = std::mem::replace(&mut self.params.db_verify, false);
        let band_round = std::mem::replace(&mut self.band_led_round, false);
        let whole_circuit = block_ids.len() == self.arena.len();
        let applied = self.try_db_splice_curated(
            false,
            block_ids,
            direction,
            &window,
            replacement,
            1,
            DbMode::SizeAgnostic,
        );
        self.counters = counters;
        self.db_record = recorder;
        self.params.db_advance = advance;
        self.params.db_verify = verify;
        self.band_led_round = band_round;
        assert!(applied, "a preverified quality splice must succeed");

        // With no surviving old node, the standard death hook cannot move the
        // last tap anchor to a neighbor. Re-anchor those monitors to live material.
        if whole_circuit && !self.taps.is_empty() {
            let anchor = self.arena.head();
            debug_assert_ne!(anchor, NIL);
            self.tap_at.clear();
            for (index, tap) in self.taps.iter_mut().enumerate() {
                tap.anchor = anchor;
                self.tap_at.entry(anchor).or_default().push(index as u32);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/mixer/leakage_repair/tests.rs"]
mod tests;
