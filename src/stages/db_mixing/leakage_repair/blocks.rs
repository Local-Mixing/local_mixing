//! Smallest order-convex closure of a detected segment's endpoint gates.
//!
//! This planner is independent of the ordinary DB window sampler. A plan is
//! local to the endpoint span, and its permutation moves only commuting gates.

use std::ops::Range;

use crate::circuit::xgate::XGate;

#[derive(Clone, Copy, Debug)]
pub struct BlockLimits {
    /// Bounds the quadratic collision scan, even when the closure is tiny.
    pub max_span: usize,
    pub max_gates: usize,
    pub max_support: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockError {
    InvalidEndpoints,
    SpanCap,
    GateCap,
    SupportCap,
}

#[derive(Clone, Debug)]
pub struct ConvexBlock {
    /// The enclosing span in the original circuit.
    pub span: Range<usize>,
    /// Original global indices belonging to the closure, in circuit order.
    pub selected: Vec<usize>,
    /// Original global indices in their new order, for `span` only. Apply the
    /// same permutation to per-gate metadata when applying a successful plan.
    pub permutation: Vec<usize>,
    /// Global range of the gathered block after applying `permutation`.
    pub block: Range<usize>,
    pub support: usize,
}

impl ConvexBlock {
    pub fn reordered_span(&self, circuit: &[XGate]) -> Vec<XGate> {
        self.permutation
            .iter()
            .map(|&i| circuit[i].clone())
            .collect()
    }

    pub fn window(&self, circuit: &[XGate]) -> Vec<XGate> {
        self.selected.iter().map(|&i| circuit[i].clone()).collect()
    }
}

/// Compute the minimal order-convex set containing both endpoint gates in the
/// forward collision DAG. Endpoints may commute (including equal targets).
/// Zero limits are hard zero caps, not an instruction to do unlimited work.
pub fn plan_convex_block(
    circuit: &[XGate],
    start: usize,
    end: usize,
    limits: BlockLimits,
) -> Result<ConvexBlock, BlockError> {
    if start >= end || end >= circuit.len() {
        return Err(BlockError::InvalidEndpoints);
    }
    let span = start..end + 1;
    let n = span.len();
    if n > limits.max_span {
        return Err(BlockError::SpanCap);
    }
    let gates = &circuit[span.clone()];
    // Conv(S) = descendants(S) intersect ancestors(S), using reflexive
    // reachability. Starting BOTH scans from the entire seed set preserves
    // commuting endpoints, and includes every alternate path between them.
    let mut descendants = vec![false; n];
    descendants[0] = true;
    descendants[n - 1] = true;
    for j in 1..n {
        if !descendants[j] {
            descendants[j] =
                (0..j).any(|i| descendants[i] && XGate::collides(&gates[i], &gates[j]));
        }
    }
    let mut ancestors = vec![false; n];
    ancestors[0] = true;
    ancestors[n - 1] = true;
    for i in (0..n - 1).rev() {
        if !ancestors[i] {
            ancestors[i] =
                (i + 1..n).any(|j| ancestors[j] && XGate::collides(&gates[i], &gates[j]));
        }
    }
    let selected: Vec<usize> = (0..n)
        .filter(|&i| descendants[i] && ancestors[i])
        .map(|i| i + start)
        .collect();
    if selected.len() > limits.max_gates {
        return Err(BlockError::GateCap);
    }
    let mut wires: Vec<u16> = selected
        .iter()
        .flat_map(|&i| {
            std::iter::once(circuit[i].target).chain(circuit[i].ctrls.iter().map(|&(w, _)| w))
        })
        .collect();
    wires.sort_unstable();
    wires.dedup();
    if wires.len() > limits.max_support {
        return Err(BlockError::SupportCap);
    }
    // Put outside ancestors before the block, then the block, then everyone
    // else. Each group is stable. A reversed collision edge would imply that
    // an excluded gate lies in the closure, so every inversion commutes.
    let mut permutation: Vec<usize> = (0..n)
        .filter(|&i| ancestors[i] && !descendants[i])
        .map(|i| i + start)
        .collect();
    let block_start = start + permutation.len();
    permutation.extend(selected.iter().copied());
    permutation.extend((0..n).filter(|&i| !ancestors[i]).map(|i| i + start));
    Ok(ConvexBlock {
        span,
        block: block_start..block_start + selected.len(),
        selected,
        permutation,
        support: wires.len(),
    })
}

#[cfg(test)]
#[path = "../../../../tests/stages/db_mixing/leakage_repair/blocks/tests.rs"]
mod tests;
