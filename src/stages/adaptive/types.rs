//! Stage 6 (adaptive resynthesis): the data shared by the analysis half
//! (census, boundary classification, planner) and the synthesis half
//! (block synthesis, rounds, driver). Library code here reads no
//! environment variables (the `src/stages` contract).
//!
//! Conventions (the ones of the server prototype `py6/stage6_run.py` and
//! the `stage6_driver` crate, kept so that results stay comparable):
//! * a CUT `t` of a circuit of `m` gates is the state after the first `t`
//!   gates, `0 <= t <= m`; a BLOCK `[a, b)` holds gates `a..b`;
//! * a LOCATION names a wire VALUE: the initial value of wire `w` is
//!   location `w` (`0..width`), the value written by gate `i` (0-based) is
//!   location `width + i`;
//! * a source BIRTH is a gate index `j` of C; the value it writes is
//!   `c_j(x)` on the honest slice (x on wires `0..n`, zeros elsewhere);
//! * the honest-slice samples of a census are `HonestSamples::draw(n,
//!   batches, seed)` from `stages::packing::cview`, 64 inputs per batch.
use crate::circuit::xgate::XGate;
use std::collections::BTreeSet;

/// A wire value: `0..width` the initial values, `width + i` gate `i`'s.
pub type Location = usize;

/// One exposure interval of a source birth: at every cut in
/// `start_cut..end_cut` the value `c_j(x) ^ constant` equals the XOR of the
/// listed locations (one location for the direct family). `end_cut` is
/// `m + 1` when the proxy is still available after the last gate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Interval {
    pub source_gate: usize,
    pub start_cut: usize,
    pub end_cut: usize,
    pub constant: bool,
    pub support: Vec<Location>,
    /// The birth is a final output of C (the last write of its wire).
    pub is_output: bool,
    /// The entry witness held on the audit's fresh and structured inputs.
    pub validated: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    Direct,
    Affine,
}

impl Family {
    pub fn name(self) -> &'static str {
        match self {
            Family::Direct => "direct",
            Family::Affine => "affine",
        }
    }
}

/// The independent re-test of every interval-entry witness on fresh random
/// and structured inputs (port of `audit_census.py`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AuditSummary {
    pub random_inputs: usize,
    pub structured_inputs: usize,
    pub failed_direct: usize,
    pub failed_affine: usize,
}

impl AuditSummary {
    pub fn all_passed(&self) -> bool {
        self.failed_direct == 0 && self.failed_affine == 0
    }
}

/// The every-cut direct/affine exposure census of one circuit against one
/// source (port of `proxy_intervals.rs` + `audit_census.py`).
#[derive(Clone, Debug, Default)]
pub struct Census {
    /// Gates of the scanned circuit (`m`).
    pub gates: usize,
    /// Wires of the scanned circuit.
    pub width: usize,
    /// Discovery inputs (64 per batch).
    pub samples: usize,
    pub affine: Vec<Interval>,
    pub direct: Vec<Interval>,
    pub audit: AuditSummary,
    pub seconds: f64,
}

impl Census {
    pub fn family(&self, f: Family) -> &[Interval] {
        match f {
            Family::Direct => &self.direct,
            Family::Affine => &self.affine,
        }
    }

    /// An empty census (what a C-blind run plans against): `m` gates, no
    /// intervals.
    pub fn empty(gates: usize, width: usize) -> Census {
        Census {
            gates,
            width,
            ..Default::default()
        }
    }
}

/// Per-family leakage of a circuit (`metrics()` of the prototype).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FamilyMetrics {
    /// Internal (non-output) births exposed somewhere.
    pub all_internal_births: usize,
    /// ... of which not in the excluded (public quadratic boundary) set.
    pub deeper_births: usize,
    /// Union of exposure cuts over internal births / (|J| * (m + 1)).
    pub cut_exposure_fraction: f64,
    pub ids: BTreeSet<usize>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct LeakageMetrics {
    pub direct: FamilyMetrics,
    pub affine: FamilyMetrics,
}

/// The source C (as XGates over wires `0..n`) and its birth classification.
#[derive(Clone, Debug)]
pub struct Source {
    pub gates: Vec<XGate>,
    pub n: usize,
    /// `J`: births that are not final outputs of C.
    pub internal: BTreeSet<usize>,
    /// The last write of every wire C writes.
    pub outputs: BTreeSet<usize>,
    /// Births classified `public_quadratic_boundary` (a polynomial of degree
    /// at most 2 in the public inputs, or in the public outputs); "deeper"
    /// = internal minus excluded.
    pub excluded: BTreeSet<usize>,
}

impl Source {
    pub fn new(gates: Vec<XGate>, n: usize, excluded: BTreeSet<usize>) -> Source {
        let mut last = std::collections::BTreeMap::new();
        for (i, g) in gates.iter().enumerate() {
            last.insert(g.target, i);
        }
        let outputs: BTreeSet<usize> = last.into_values().collect();
        let internal = (0..gates.len()).filter(|i| !outputs.contains(i)).collect();
        Source {
            gates,
            n,
            internal,
            outputs,
            excluded,
        }
    }

    /// `J \ excluded`.
    pub fn deeper(&self) -> BTreeSet<usize> {
        self.internal.difference(&self.excluded).copied().collect()
    }
}

/// The planner's frozen policy (`POLICY` + `VAR['v2']` of the prototype).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanParams {
    /// Total ANF terms at an eligible block endpoint.
    pub term_cap: usize,
    /// Longest block, in gates.
    pub max_block_gates: usize,
    /// Gates scanned beyond the latest eligible endpoint.
    pub lookahead: usize,
    /// Product-pair work limit per gate update.
    pub product_pairs: usize,
    /// Segments planned in parallel (1 = the sequential planner).
    pub segments: usize,
    /// Cuts within this distance of a previous round's boundary are
    /// penalized (0 = off).
    pub avoid_radius: usize,
}

impl Default for PlanParams {
    fn default() -> Self {
        PlanParams {
            term_cap: 131_072,
            max_block_gates: 65_536,
            lookahead: 256,
            product_pairs: 4_000_000,
            segments: 16,
            avoid_radius: 64,
        }
    }
}

/// One planned block (`blocks.tsv` of the prototype).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockReport {
    pub start: usize,
    pub end: usize,
    pub max_feasible_end: usize,
    pub stop_reason: &'static str,
    pub terms: usize,
    pub deep_affine_at_end: i32,
    pub all_internal_affine_at_end: i32,
    pub direct_locations_at_end: i32,
    pub signed_clearance: i32,
}

/// A partition of the circuit: `bounds` are the interior cuts (strictly
/// between 0 and `m`, ascending); the blocks are `[0, b0), [b0, b1), ...`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Plan {
    pub bounds: Vec<usize>,
    /// The parallel planner's segment seams (a subset of `bounds`).
    pub seams: Vec<usize>,
    pub blocks: Vec<BlockReport>,
    pub single_gate_exceptions: usize,
    pub seconds: f64,
}

impl Plan {
    /// `[0] + bounds + [m]`.
    pub fn edges(&self, m: usize) -> Vec<usize> {
        let mut e = Vec::with_capacity(self.bounds.len() + 2);
        e.push(0);
        e.extend_from_slice(&self.bounds);
        e.push(m);
        e
    }
}
