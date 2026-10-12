//! The delivery loop (docs/DELIVERY_STAGE.md §2): unpack every window of
//! the deliverable; a window that does not resynthesize is cut in two at an
//! exposure-aware cut, both halves are re-emitted into the deliverable and
//! unpacked in the next round, recursively, until every window unpacks or
//! `max_rounds` is reached.
//!
//! The Python reference is `scripts/windowed_delivery/waist/deliver.py`;
//! this module reproduces its loop, its numbering (a pending window is
//! synthesized with its index in the CURRENT lattice, which keys the
//! tie-break PRNG and the verify seed), its cut choice and its outputs byte
//! for byte (`tests/manual/delivery_parity.sh`):
//!
//! * lattice L_r = windows `[lo, hi)` in order, deliverable D_r = the
//!   container over the windows' blobs ([`container_bytes`]); D_{r+1}
//!   differs from D_r only in the split windows, whose blobs are replaced by
//!   the halves' blobs ([`emit_blob`]: the same `window_anf` emission as the
//!   packer, so a half's blob is a pure function of `gates[lo:hi]`);
//! * cut choice ([`choose_cuts`]): the packer's split candidates
//!   (`packing::repair::split::split_candidates`) minus the cuts at which
//!   some interior C value is exactly affine (one `CView::leaky` scan over
//!   the candidates of every failed window of the round), the candidate
//!   nearest the midpoint (smaller on a tie); none clean: the midpoint,
//!   counted in `unclean_cuts`;
//! * a failed window of one gate, and every window still failed at
//!   `max_rounds`, is emitted verbatim from the packed circuit (counted).
//!
//! One deliberate difference from the reference: a re-emitted half whose
//! ANF exceeds the packer's budget ([`RefineParams::budget`], 65,536
//! monomials) is NOT synthesized; it counts in `over_budget`, fails its
//! round without synthesis and is split again (the Python synthesizes it and
//! delivers it, which breaks the contract's final budget check).
use super::run::{PackedWindows, WinStatus, WindowResult, unpack_selected};
use super::spec::AnfStats;
use super::synth::NTAGS;
use super::types::{Stage7Error, UnpackParams};
use crate::circuit::xgate::XGate;
use crate::stages::packing::anf::{MulStats, window_anf};
use crate::stages::packing::cview::{CutOracle, CviewParams};
use crate::stages::packing::mono::{MonoWidthFn, with_mono_width};
use crate::stages::packing::repair::CView;
use crate::stages::packing::repair::split::{pick_candidate, split_candidates};
use crate::stages::packing::serialize::{
    ContainerHeader, Deliverable, EMIT_BUDGET, Format, HEADER_BYTES, IO_LAYOUT_CLASSIC_ZERO_SLICE,
    crc32_ieee, decode, encode_window,
};
use crate::stages::packing::types::CapSemantics;
use rustc_hash::FxHashSet;
use std::fmt::Write as _;
use std::time::Instant;

/// The packer's split budget (`tdp_packing --split-budget`): a delivered
/// window holds at most this many monomials.
pub const DEFAULT_BUDGET: usize = 65536;
/// A 1,024-gate window reaches 1-gate pieces in 10 rounds.
pub const DEFAULT_MAX_ROUNDS: usize = 16;

#[derive(Clone, Debug, PartialEq)]
pub struct RefineParams {
    /// Rounds that may split (round r splits iff `r + 1 <= max_rounds`, as
    /// the reference does: up to `max_rounds + 1` synthesis rounds).
    pub max_rounds: usize,
    /// Monomial budget of a delivered window.
    pub budget: usize,
    /// TEST ONLY: rounds `0..force_fail_rounds` synthesize nothing and fail
    /// every pending window, so the split, the re-emission and the next
    /// round run on any fixture.
    pub force_fail_rounds: usize,
    /// The C-view scan's sampling (the packer's defaults).
    pub cview: CviewParams,
}

impl Default for RefineParams {
    fn default() -> Self {
        RefineParams {
            max_rounds: DEFAULT_MAX_ROUNDS,
            budget: DEFAULT_BUDGET,
            force_fail_rounds: 0,
            cview: CviewParams::default(),
        }
    }
}

// ---------------------------------------------------------------- lattice and container

/// Windows `[lo, hi)` of a cut list `[0] + bounds + [m]`.
pub fn windows_of(cuts: &[usize]) -> Vec<(usize, usize)> {
    cuts.windows(2).map(|w| (w[0], w[1])).collect()
}

/// The bounds file of a lattice (`deliver.py write_bounds`): `lo\n` for
/// every window but the first (empty for a single window).
pub fn bounds_text(wins: &[(usize, usize)]) -> String {
    let mut s = String::new();
    for &(lo, _) in wins.iter().skip(1) {
        let _ = writeln!(s, "{lo}");
    }
    s
}

/// The body of a deliverable as its windows' blobs (`deliver.py
/// split_body`): `<I len>` + blob per window.
pub fn split_body(body: &[u8]) -> Result<Vec<Vec<u8>>, Stage7Error> {
    let mut blobs = Vec::new();
    let mut pos = 0usize;
    while pos < body.len() {
        if pos + 4 > body.len() {
            return Err(Stage7Error::Input(
                "deliverable body: truncated length prefix".into(),
            ));
        }
        let len = u32::from_le_bytes(body[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        if pos + len > body.len() {
            return Err(Stage7Error::Input(
                "deliverable body: last blob runs past the end".into(),
            ));
        }
        blobs.push(body[pos..pos + len].to_vec());
        pos += len;
    }
    Ok(blobs)
}

/// The blobs of a container file (header validated by the caller's
/// `decode`).
pub fn container_blobs(bytes: &[u8]) -> Result<Vec<Vec<u8>>, Stage7Error> {
    if bytes.len() < HEADER_BYTES as usize {
        return Err(Stage7Error::Input(
            "container shorter than its header".into(),
        ));
    }
    split_body(&bytes[HEADER_BYTES as usize..])
}

/// The body of the blobs: `<I len>` + blob per window.
pub fn body_bytes(blobs: &[&[u8]]) -> Vec<u8> {
    let total: usize = blobs.iter().map(|b| 4 + b.len()).sum();
    let mut body = Vec::with_capacity(total);
    for b in blobs {
        body.extend_from_slice(&(b.len() as u32).to_le_bytes());
        body.extend_from_slice(b);
    }
    body
}

/// Container v1 around the blobs (`deliver.py container`): the 48-byte
/// header (window count, body bytes, CRCs recomputed) and the body.
pub fn container_bytes(blobs: &[&[u8]], source_wires: u32, state_wires: u32) -> Vec<u8> {
    let body = body_bytes(blobs);
    let h = ContainerHeader {
        source_wires,
        state_wires,
        io_layout: IO_LAYOUT_CLASSIC_ZERO_SLICE,
        window_count: blobs.len() as u32,
        body_bytes: body.len() as u64,
        body_crc32: crc32_ieee(&body),
    };
    let mut out = Vec::with_capacity(HEADER_BYTES as usize + body.len());
    out.extend_from_slice(&h.to_bytes());
    out.extend_from_slice(&body);
    out
}

// ---------------------------------------------------------------- blob emission

/// One re-emitted window.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Emitted {
    pub blob: Vec<u8>,
    pub monomials: usize,
    /// The ANF fits `budget` (the composition at that budget succeeded).
    pub fits: bool,
}

struct Emit<'a> {
    gates: &'a [XGate],
    budget: usize,
}

impl MonoWidthFn<Result<Emitted, Stage7Error>> for Emit<'_> {
    fn call<const W: usize>(self) -> Result<Emitted, Stage7Error> {
        // The budget first: a cap only ever aborts, so when the composition
        // succeeds the ANF is the canonical one and the blob equals the
        // packer's (and the reference's budget-free) emission.
        let mut budgets = vec![(self.budget, true)];
        for b in [EMIT_BUDGET, 4 * EMIT_BUDGET] {
            if b > self.budget {
                budgets.push((b, false));
            }
        }
        for (budget, fits) in budgets {
            let mut st = MulStats::default();
            if let Ok(anf) = window_anf::<W>(self.gates, budget, CapSemantics::CPython39, &mut st) {
                let blob = encode_window(&anf)?;
                return Ok(Emitted {
                    blob,
                    monomials: anf.total,
                    fits,
                });
            }
        }
        Err(Stage7Error::Verify(format!(
            "a window of {} gates exceeds {} monomials: it cannot be delivered",
            self.gates.len(),
            4 * EMIT_BUDGET
        )))
    }
}

/// The blob of `gates[lo..hi]` (the packer's `emit_anf` + `encode_window`,
/// CPython 3.9 cap semantics) and whether its ANF fits `budget`.
/// `state_wires` sets the monomial width (≤ 1024).
pub fn emit_blob(
    gates: &[XGate],
    lo: usize,
    hi: usize,
    state_wires: usize,
    budget: usize,
) -> Result<Emitted, Stage7Error> {
    let maxw = gates[lo..hi]
        .iter()
        .map(|g| g.max_wire() as usize + 1)
        .max()
        .unwrap_or(0);
    with_mono_width(
        state_wires.max(maxw).max(1),
        Emit {
            gates: &gates[lo..hi],
            budget,
        },
    )
}

// ---------------------------------------------------------------- cut choice

/// `deliver.py choose_cuts`: for every window `[lo, hi)` (hi − lo ≥ 2) the
/// candidate cut nearest the midpoint at which no interior C value is
/// exactly affine (one C-view scan over the candidates of every window), the
/// smaller on a tie; `(midpoint, false)` when every candidate leaks. Results
/// are parallel to `wins`.
pub fn choose_cuts(
    cv: &CView,
    oracle: &mut CutOracle<'_>,
    wins: &[(usize, usize)],
) -> Vec<(usize, bool)> {
    choose_cuts_with(wins, &mut |cands| cv.leaky(oracle, cands))
}

/// [`choose_cuts`] over any leak scan: `leaky(cands)` (called once, with
/// the sorted, deduplicated union of every window's candidates) returns the
/// candidates at which some interior C value is exactly affine.
pub fn choose_cuts_with(
    wins: &[(usize, usize)],
    leaky: &mut dyn FnMut(&[usize]) -> FxHashSet<usize>,
) -> Vec<(usize, bool)> {
    let cands: Vec<Vec<usize>> = wins
        .iter()
        .map(|&(lo, hi)| split_candidates(lo, hi))
        .collect();
    let mut all: Vec<usize> = cands.iter().flatten().copied().collect();
    all.sort_unstable();
    all.dedup();
    let bad: FxHashSet<usize> = if all.is_empty() {
        FxHashSet::default()
    } else {
        leaky(&all)
    };
    wins.iter()
        .zip(&cands)
        .map(|(&(lo, hi), cs)| {
            let good: Vec<usize> = cs.iter().copied().filter(|c| !bad.contains(c)).collect();
            if good.is_empty() {
                ((lo + hi) / 2, false)
            } else {
                (pick_candidate(&good, lo, hi), true)
            }
        })
        .collect()
}

// ---------------------------------------------------------------- the loop

/// Everything the loop needs besides the deliverable.
pub struct RefineInputs<'a> {
    /// The packed circuit (`rep<G>.mpmct1`): the window gates, the verbatim
    /// source and the exact-check reference.
    pub packed: &'a [XGate],
    pub source_wires: u32,
    pub state_wires: u32,
    /// The C-view of the source circuit (`CView::new(C, n, &params.cview)`).
    pub cview: &'a CView,
}

/// One round of `rounds.tsv`.
#[derive(Clone, Debug, PartialEq)]
pub struct RoundStat {
    pub round: usize,
    pub pending: usize,
    pub converged: usize,
    pub failed: usize,
    pub split: usize,
    pub unclean: usize,
    pub seconds: f64,
}

/// A window of the final lattice.
#[derive(Clone, Debug)]
pub struct Piece {
    pub lo: usize,
    pub hi: usize,
    /// The round in which it converged; −1 when emitted verbatim.
    pub round: i64,
    /// `WinStatus::Ok` (synthesized in `round`, `idx` = its index in that
    /// round's lattice) or `WinStatus::Verbatim` (the packed gates; the
    /// reason in `note`).
    pub result: WindowResult,
    pub blob: Vec<u8>,
    /// The blob's ANF fits the budget (always, unless the loop stopped at
    /// `max_rounds` on an over-budget half).
    pub fits: bool,
}

/// What the loop did.
#[derive(Clone, Debug)]
pub struct RefineOutcome {
    pub pieces: Vec<Piece>,
    pub rounds: Vec<RoundStat>,
    pub windows_initial: usize,
    pub splits: usize,
    pub unclean_cuts: usize,
    pub over_budget: usize,
    pub verbatim: usize,
    /// Every cut the loop added, in the order it was added, with whether it
    /// was clean.
    pub new_cuts: Vec<(usize, bool)>,
}

impl RefineOutcome {
    pub fn wins(&self) -> Vec<(usize, usize)> {
        self.pieces.iter().map(|p| (p.lo, p.hi)).collect()
    }
    pub fn blobs(&self) -> Vec<&[u8]> {
        self.pieces.iter().map(|p| p.blob.as_slice()).collect()
    }
    /// The bounds of the final lattice (`0 < b < m`).
    pub fn bounds(&self) -> Vec<usize> {
        self.pieces.iter().skip(1).map(|p| p.lo).collect()
    }
    /// The final deliverable.
    pub fn container(&self, source_wires: u32, state_wires: u32) -> Vec<u8> {
        container_bytes(&self.blobs(), source_wires, state_wires)
    }
    /// The pieces' gates in lattice order and each piece's offset in it.
    pub fn concatenate(&self) -> (Vec<XGate>, Vec<usize>) {
        let total: usize = self.pieces.iter().map(|p| p.result.gates.len()).sum();
        let mut gates = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(self.pieces.len());
        for p in &self.pieces {
            offsets.push(gates.len());
            gates.extend(p.result.gates.iter().cloned());
        }
        (gates, offsets)
    }
    /// Final windows whose blob exceeds the budget.
    pub fn over_budget_final(&self) -> usize {
        self.pieces.iter().filter(|p| !p.fits).count()
    }
}

enum Slot {
    /// Waits for synthesis in the next round.
    Pending,
    /// Fails its round without synthesis (an over-budget half).
    PreFailed,
    Converged(i64, Box<WindowResult>),
    Verbatim(String),
}

struct Win {
    lo: usize,
    hi: usize,
    blob: Vec<u8>,
    fits: bool,
    slot: Slot,
}

/// Called at the start of every round with the round number, the lattice
/// and the round's deliverable (the reference writes `bounds.<r>` and
/// `deliverable.<r>.tdp4`).
pub type RoundHook<'a> = dyn FnMut(usize, &[(usize, usize)], &[u8]) -> Result<(), Stage7Error> + 'a;

/// The loop. `blobs` are the deliverable's windows (D_0, `len = wins_0`,
/// the lattice `[0] + bounds + [m]` of the packed circuit); `params.seed`
/// keys every round's synthesis. `progress` receives every synthesized
/// window (`idx` in its round's lattice); `say` the round lines.
#[allow(clippy::too_many_arguments)]
pub fn refine(
    blobs: Vec<Vec<u8>>,
    cuts0: &[usize],
    inp: &RefineInputs<'_>,
    params: &UnpackParams,
    rp: &RefineParams,
    on_round: &mut RoundHook<'_>,
    progress: &(dyn Fn(usize, usize, usize, &WindowResult) + Sync),
    say: &dyn Fn(&str),
) -> Result<RefineOutcome, Stage7Error> {
    let m = inp.packed.len();
    let wins0 = windows_of(cuts0);
    if wins0.len() != blobs.len() {
        return Err(Stage7Error::Input(format!(
            "the packed circuit has {} windows, the deliverable {}",
            wins0.len(),
            blobs.len()
        )));
    }
    if cuts0.first() != Some(&0) || cuts0.last() != Some(&m) {
        return Err(Stage7Error::Internal(format!(
            "lattice cuts must run from 0 to {m}"
        )));
    }
    let nw = inp.state_wires as usize;
    let mut wins: Vec<Win> = wins0
        .iter()
        .zip(blobs)
        .map(|(&(lo, hi), blob)| Win {
            lo,
            hi,
            blob,
            fits: true,
            slot: Slot::Pending,
        })
        .collect();
    let mut out = RefineOutcome {
        pieces: Vec::new(),
        rounds: Vec::new(),
        windows_initial: wins.len(),
        splits: 0,
        unclean_cuts: 0,
        over_budget: 0,
        verbatim: 0,
        new_cuts: Vec::new(),
    };
    let mut oracle = inp.cview.oracle(inp.packed);
    let mut round = 0usize;
    loop {
        let pending: Vec<usize> = wins
            .iter()
            .enumerate()
            .filter(|(_, w)| matches!(w.slot, Slot::Pending | Slot::PreFailed))
            .map(|(i, _)| i)
            .collect();
        if pending.is_empty() {
            break;
        }
        let t0 = Instant::now();
        // ---- D_r
        let lattice: Vec<(usize, usize)> = wins.iter().map(|w| (w.lo, w.hi)).collect();
        let blob_refs: Vec<&[u8]> = wins.iter().map(|w| w.blob.as_slice()).collect();
        let bytes = container_bytes(&blob_refs, inp.source_wires, inp.state_wires);
        on_round(round, &lattice, &bytes)?;
        let d = decode(&bytes, Format::Container)?;
        drop(bytes);
        let mut cuts: Vec<usize> = wins.iter().map(|w| w.lo).collect();
        cuts.push(m);
        let packed = PackedWindows {
            gates: inp.packed,
            cuts,
        };
        // ---- synthesis of the pending windows, with their indices in L_r
        let sel: Vec<usize> = if round < rp.force_fail_rounds {
            Vec::new()
        } else {
            pending
                .iter()
                .copied()
                .filter(|&i| matches!(wins[i].slot, Slot::Pending))
                .collect()
        };
        let total = sel.len();
        let results = unpack_selected(&d, Some(&packed), params, &sel, true, &|done, r| {
            progress(round, done, total, r)
        })?;
        drop(d);
        let mut failed: Vec<usize> = Vec::new();
        let mut converged = 0usize;
        for r in results {
            let i = r.idx;
            if r.status == WinStatus::Ok {
                wins[i].slot = Slot::Converged(round as i64, Box::new(r));
                converged += 1;
            } else {
                failed.push(i);
            }
        }
        failed.extend(pending.iter().copied().filter(|&i| !sel.contains(&i)));
        failed.sort_unstable();
        let nfailed = failed.len();
        // ---- refine
        let mut split = 0usize;
        let mut unclean = 0usize;
        // the reference: split iff `rnd + 1 <= max_rounds`
        if !failed.is_empty() && round < rp.max_rounds {
            let splittable: Vec<usize> = failed
                .iter()
                .copied()
                .filter(|&i| wins[i].hi - wins[i].lo >= 2)
                .collect();
            let sw: Vec<(usize, usize)> = splittable
                .iter()
                .map(|&i| (wins[i].lo, wins[i].hi))
                .collect();
            let choice = choose_cuts(inp.cview, &mut oracle, &sw);
            let mut next: Vec<Win> = Vec::with_capacity(wins.len() + splittable.len());
            let mut it = splittable.iter().zip(choice);
            let mut cur = it.next();
            for (i, w) in wins.into_iter().enumerate() {
                if cur.is_some_and(|(&j, _)| j == i) {
                    let (_, (c, clean)) = cur.take().unwrap();
                    split += 1;
                    unclean += (!clean) as usize;
                    out.new_cuts.push((c, clean));
                    for (x, y) in [(w.lo, c), (c, w.hi)] {
                        let e = emit_blob(inp.packed, x, y, nw, rp.budget)?;
                        if !e.fits {
                            out.over_budget += 1;
                        }
                        next.push(Win {
                            lo: x,
                            hi: y,
                            blob: e.blob,
                            fits: e.fits,
                            slot: if e.fits {
                                Slot::Pending
                            } else {
                                Slot::PreFailed
                            },
                        });
                    }
                    cur = it.next();
                } else if failed.contains(&i) {
                    next.push(Win {
                        slot: Slot::Verbatim("1-gate window".into()),
                        ..w
                    });
                } else {
                    next.push(w);
                }
            }
            wins = next;
        } else if !failed.is_empty() {
            for &i in &failed {
                wins[i].slot = Slot::Verbatim(format!("max_rounds {}", rp.max_rounds));
            }
        }
        out.splits += split;
        out.unclean_cuts += unclean;
        let secs = t0.elapsed().as_secs_f64();
        say(&format!(
            "round {round}: pending {}, converged {converged}, failed {nfailed}, split {split}{}, {secs:.1}s",
            pending.len(),
            if unclean > 0 {
                format!(" ({unclean} unclean)")
            } else {
                String::new()
            }
        ));
        out.rounds.push(RoundStat {
            round,
            pending: pending.len(),
            converged,
            failed: nfailed,
            split,
            unclean,
            seconds: secs,
        });
        round += 1;
    }
    // ---- the final lattice
    for (i, w) in wins.into_iter().enumerate() {
        let (round, result) = match w.slot {
            Slot::Converged(r, res) => (r, *res),
            Slot::Verbatim(reason) => {
                out.verbatim += 1;
                (
                    -1,
                    verbatim_result(i, w.lo, w.hi, &inp.packed[w.lo..w.hi], reason),
                )
            }
            Slot::Pending | Slot::PreFailed => {
                return Err(Stage7Error::Internal(format!(
                    "window [{}, {}) left pending after the loop",
                    w.lo, w.hi
                )));
            }
        };
        out.pieces.push(Piece {
            lo: w.lo,
            hi: w.hi,
            round,
            result,
            blob: w.blob,
            fits: w.fits,
        });
    }
    Ok(out)
}

/// A verbatim window's row (the reference reports zeros for its ANF
/// statistics and the reason as its synthesis status).
fn verbatim_result(
    idx: usize,
    lo: usize,
    hi: usize,
    gates: &[XGate],
    reason: String,
) -> WindowResult {
    WindowResult {
        idx,
        gates: gates.to_vec(),
        status: WinStatus::Verbatim,
        synth_status: "verbatim",
        secs: 0.0,
        stats: AnfStats::default(),
        universe: 0,
        width: 0,
        tags: [0; NTAGS],
        pre_postopt: 0,
        steps: 0,
        note: reason,
        restarts_used: 0,
        deadline_hit: false,
        exact_budget: 0,
        orig: Some((lo, hi)),
    }
}

/// `windows.tsv` of a refined run: the reference's columns `idx lo hi round
/// orig_gates syn_gates status synth_status seconds monomials changed_wires
/// input_vars max_degree` (a verbatim row: round −1, status `verbatim`, the
/// reason as `synth_status`, zeros), then the plain run's remaining columns
/// `syn_offset pre_postopt_gates steps note tag_* universe width exact_budget
/// restarts_used deadline_hit`; `syn_offset` locates the window in the
/// concatenation BEFORE canonical reordering.
pub fn refine_windows_tsv(pieces: &[Piece], offsets: &[usize]) -> String {
    let mut s = String::from(
        "idx\tlo\thi\tround\torig_gates\tsyn_gates\tstatus\tsynth_status\tseconds\tmonomials\tchanged_wires\tinput_vars\tmax_degree\tsyn_offset\tpre_postopt_gates\tsteps\tnote\ttag_greedy\ttag_lookahead\ttag_sink\ttag_helper\ttag_cycle\ttag_linear\ttag_other\ttag_back\tuniverse\twidth\texact_budget\trestarts_used\tdeadline_hit\n",
    );
    for (i, (p, &off)) in pieces.iter().zip(offsets).enumerate() {
        let r = &p.result;
        let verbatim = r.status == WinStatus::Verbatim;
        let (status, synth_status, note) = if verbatim {
            ("verbatim", r.note.as_str(), "")
        } else {
            (r.status.name(), r.synth_status, r.note.as_str())
        };
        let _ = write!(
            s,
            "{i}\t{}\t{}\t{}\t{}\t{}\t{status}\t{synth_status}\t{:.3}\t{}\t{}\t{}\t{}\t{off}\t{}\t{}\t{}",
            p.lo,
            p.hi,
            p.round,
            p.hi - p.lo,
            r.gates.len(),
            r.secs,
            r.stats.monomials,
            r.stats.changed_wires,
            r.stats.input_vars,
            r.stats.max_degree,
            r.pre_postopt,
            r.steps,
            note.replace(['\t', '\n'], " ")
        );
        for t in r.tags {
            let _ = write!(s, "\t{t}");
        }
        let _ = writeln!(
            s,
            "\t{}\t{}\t{}\t{}\t{}",
            r.universe, r.width, r.exact_budget, r.restarts_used, r.deadline_hit as u8
        );
    }
    s
}

/// `rounds.tsv`: `round pending converged failed split seconds`.
pub fn rounds_tsv(rounds: &[RoundStat]) -> String {
    let mut s = String::from("round\tpending\tconverged\tfailed\tsplit\tseconds\n");
    for r in rounds {
        let _ = writeln!(
            s,
            "{}\t{}\t{}\t{}\t{}\t{:.1}",
            r.round, r.pending, r.converged, r.failed, r.split, r.seconds
        );
    }
    s
}

/// A `--windows` selection: comma-separated indices and half-open ranges
/// `LO:HI`, ascending and deduplicated; every index below `count`.
pub fn parse_selection(spec: &str, count: usize) -> Result<Vec<usize>, Stage7Error> {
    let bad = |m: String| Stage7Error::Usage(format!("--windows {spec:?}: {m}"));
    let num = |t: &str| -> Result<usize, Stage7Error> {
        if t.is_empty() || t.len() > 9 || !t.bytes().all(|b| b.is_ascii_digit()) {
            return Err(bad(format!("{t:?} is not a window index")));
        }
        Ok(t.parse().expect("digits"))
    };
    let mut sel = Vec::new();
    for item in spec.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        match item.split_once(':') {
            Some((a, b)) => {
                let (lo, hi) = (num(a)?, num(b)?);
                if lo >= hi {
                    return Err(bad(format!("empty range {item}")));
                }
                sel.extend(lo..hi);
            }
            None => sel.push(num(item)?),
        }
    }
    sel.sort_unstable();
    sel.dedup();
    if sel.is_empty() {
        return Err(bad("no window selected".into()));
    }
    if let Some(&i) = sel.last()
        && i >= count
    {
        return Err(bad(format!(
            "window {i} selected, the deliverable has {count} windows"
        )));
    }
    Ok(sel)
}

/// `d`'s decoded form, for tests and the binary: a container from blobs.
pub fn decode_blobs(
    blobs: &[&[u8]],
    source_wires: u32,
    state_wires: u32,
) -> Result<Deliverable, Stage7Error> {
    Ok(decode(
        &container_bytes(blobs, source_wires, state_wires),
        Format::Container,
    )?)
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/unpacking/refine.rs"]
mod tests;
