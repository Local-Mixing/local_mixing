//! Final gather/reduce pipeline, with optional input-slice specialization and output liveness.
use crate::circuit::formats::PackedGate;
use crate::circuit::xgate::{Lits, XGate};
use crate::engine::mixer::{Merge, merge_result};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::OnceLock;

pub mod downhill;
pub mod packing;
pub mod reduce;
pub mod transport;
pub use packing::{compact, compact_gate, pack, pack_census};
use reduce::*;
use transport::*;

pub struct CompressParams {
    // None = equality required on every wire. Some(mask) = only on wires
    // with mask[w] == true; dead cones are pruned.
    pub live_out: Option<Vec<bool>>,
    // None = equality required on every input. Some(mask) = wires with
    // mask[w] == true are promised zero at circuit entry; gates are
    // specialized to that input subspace (literals on known wires fold,
    // never-firing gates drop). The output then equals the input circuit
    // ONLY on the promised subspace.
    pub zero_in: Option<Vec<bool>>,
    pub max_iters: usize,
    // Groups are closed proactively at this many members.
    pub group_cap: usize,
    // ANF rewrite attempted only when the group support fits (mask bits).
    pub anf_support_cap: usize,
    // Interleave one conjugation-descent pass (postprocessing::downhill) after each
    // gather/reduce iteration.
    pub downhill: bool,
    // Float groups across writers of their control wires by conjugation
    // (Toffoli sliding) instead of closing them, when the ESOP does not grow.
    pub transport: bool,
    // Extra cubes a transport may add (0 = neutral-or-better only). Growth
    // is speculative: the pass-level guard restores the previous circuit if
    // an iteration ends larger.
    pub transport_slack: usize,
    // A reader of the target separated from every member by an opposite
    // literal commutes with the group and does not close it.
    pub sep_reads: bool,
    // Also gather on the reversed gate list each iteration (leftward float).
    pub reverse_pass: bool,
    pub local_verify: bool,
    pub seed: u64,
}

impl Default for CompressParams {
    fn default() -> CompressParams {
        CompressParams {
            live_out: None,
            zero_in: None,
            max_iters: 10,
            group_cap: 64,
            anf_support_cap: 40,
            downhill: true,
            transport: true,
            transport_slack: 0,
            sep_reads: true,
            reverse_pass: true,
            local_verify: true,
            seed: 0,
        }
    }
}

#[derive(Default, Debug)]
pub struct CompressReport {
    pub iters: usize,
    pub liveness_dropped: usize,
    pub zero_killed: usize,
    pub zero_lits_dropped: u64,
    pub groups: u64,
    pub multi_groups: u64,
    pub max_group: usize,
    pub catalogue_merges: u64,
    pub anf_wins: u64,
    pub exact_wins: u64,
    pub downhill_swaps: u64,
    // Groups floated across a writer of one of their control wires (ESOP
    // changed / unchanged), refused on cost, refused to keep the frame
    // dependencies acyclic; readers that passed a group by separation.
    pub transports: u64,
    pub transport_noops: u64,
    pub transport_refused: u64,
    pub transport_cycle_refused: u64,
    pub sep_passes: u64,
    pub verifies_skipped: u64,
    pub gates_in: usize,
    pub gates_out: usize,
    pub lits_in: u64,
    pub lits_out: u64,
}

pub fn lits_of(gates: &[XGate]) -> u64 {
    gates.iter().map(|g| g.width() as u64).sum()
}

// Exact dead-cone elimination: keep a gate iff its target is live at its
// position; kept gates make their controls live. Returns dropped count.
pub fn liveness_prune(gates: Vec<XGate>, live_out: &[bool]) -> (Vec<XGate>, usize) {
    let (out, _, dropped) = liveness_prune_anc(gates, None, live_out);
    (out, dropped)
}

// Per-gate ancestor set, in the fmix sidecar's word layout. Threaded through
// compression so the compressed circuit keeps a meaningful sidecar: gathering
// is a permutation (sets follow gates), a reduced multi-member group stamps
// every survivor with the UNION of its members' sets (each emitted cube
// derives from the whole gathered ESOP), and pruned gates just drop out.
pub type AncBits = Vec<u64>;

pub(crate) fn or_anc(dst: &mut AncBits, src: &AncBits) {
    if dst.len() < src.len() {
        dst.resize(src.len(), 0);
    }
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d |= *s;
    }
}

// Full pass: [liveness] -> gather+reduce, iterated to a gate-count fixed
// point. Prints one [fcompress] line per iteration.
pub fn compress(
    gates: Vec<XGate>,
    wires: usize,
    p: &CompressParams,
) -> (Vec<XGate>, CompressReport) {
    let (out, _, rep) = compress_anc(gates, None, wires, p);
    (out, rep)
}

// Same pass with per-gate ancestor sets threaded through: sets follow gates
// under gathering, group survivors carry the member union, pruned gates drop.
// `anc` must be aligned with `gates`; the returned tags align with the output.
pub fn compress_anc(
    gates: Vec<XGate>,
    anc: Option<Vec<AncBits>>,
    wires: usize,
    p: &CompressParams,
) -> (Vec<XGate>, Option<Vec<AncBits>>, CompressReport) {
    if let Some(a) = &anc {
        assert_eq!(
            a.len(),
            gates.len(),
            "ancestry tags must align with the input gates"
        );
    }
    let mut rng = StdRng::seed_from_u64(p.seed);
    let mut rep = CompressReport::default();
    rep.gates_in = gates.len();
    rep.lits_in = lits_of(&gates);
    let mut cur = gates;
    let mut cur_anc = anc;
    let mut prev = (cur.len(), lits_of(&cur));
    for iter in 1..=p.max_iters {
        let before = cur.len();
        if let Some(z) = &p.zero_in {
            let (kept, kept_anc, killed, lits_dropped) =
                zero_specialize_anc(cur, cur_anc, z, wires);
            cur = kept;
            cur_anc = kept_anc;
            rep.zero_killed += killed;
            rep.zero_lits_dropped += lits_dropped;
        }
        if let Some(lv) = &p.live_out {
            let (kept, kept_anc, dropped) = liveness_prune_anc(cur, cur_anc, lv);
            cur = kept;
            cur_anc = kept_anc;
            rep.liveness_dropped += dropped;
        }
        let snapshot = (cur.clone(), cur_anc.clone());
        let (next, next_anc) =
            gather_reduce_pass(&cur, cur_anc.as_deref(), wires, p, &mut rng, &mut rep);
        cur = next;
        cur_anc = next_anc;
        if p.reverse_pass {
            // The reversed list is the inverse function (involutions), so a
            // forward gather of it is a leftward gather of the circuit.
            cur.reverse();
            if let Some(a) = cur_anc.as_mut() {
                a.reverse();
            }
            let (next, next_anc) =
                gather_reduce_pass(&cur, cur_anc.as_deref(), wires, p, &mut rng, &mut rep);
            cur = next;
            cur_anc = next_anc;
            cur.reverse();
            if let Some(a) = cur_anc.as_mut() {
                a.reverse();
            }
        }
        let mut dh_swaps = 0usize;
        if p.downhill {
            let (next, next_anc, swaps) =
                downhill::apply_pass(cur, cur_anc, &mut rng, p.local_verify);
            cur = next;
            cur_anc = next_anc;
            dh_swaps = swaps;
            rep.downhill_swaps += swaps as u64;
        }
        rep.iters = iter;
        println!(
            "[fcompress] iter={} gates {} -> {} | groups={} multi={} max={} | catalogue={} anf_wins={} exact={} downhill={} transport={} (noop={} refused={} cyc={}) sep={} live_dropped={} zero_killed={} vskip={}",
            iter,
            before,
            cur.len(),
            rep.groups,
            rep.multi_groups,
            rep.max_group,
            rep.catalogue_merges,
            rep.anf_wins,
            rep.exact_wins,
            dh_swaps,
            rep.transports,
            rep.transport_noops,
            rep.transport_refused,
            rep.transport_cycle_refused,
            rep.sep_passes,
            rep.liveness_dropped,
            rep.zero_killed,
            rep.verifies_skipped
        );
        // Progress = strictly smaller (gates, lits): downhill can shrink lits
        // at equal gate count, and that still enables later gather wins. The
        // pruners and downhill never regress the pair; gathering with
        // transport can in principle (a transported group may reduce worse
        // than its parts would have separately), so an iteration that ends
        // larger is discarded and the previous circuit kept.
        let now = (cur.len(), lits_of(&cur));
        if now > prev {
            println!(
                "[fcompress] iter={} regressed {:?} -> {:?}; keeping the previous circuit",
                iter, prev, now
            );
            cur = snapshot.0;
            cur_anc = snapshot.1;
            break;
        }
        if now == prev {
            break;
        }
        prev = now;
    }
    rep.gates_out = cur.len();
    rep.lits_out = lits_of(&cur);
    (cur, cur_anc, rep)
}

#[cfg(test)]
#[path = "../../../../tests/stages/post-processing/compression/compress_tests.rs"]
mod compress_tests;
