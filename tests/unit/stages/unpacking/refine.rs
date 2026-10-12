use super::*;
use crate::stages::packing::cview::CviewParams;
use crate::stages::packing::oracle_io::window_cuts;
use crate::stages::packing::serialize::encode_body;
use crate::stages::packing::verify::check_windows_anf;
use crate::stages::unpacking::mono::Rng;
use crate::stages::unpacking::verify::check_equiv;
use smallvec::SmallVec;

fn g(t: u16, comp: bool, ctrls: &[(u16, bool)]) -> XGate {
    XGate {
        target: t,
        comp,
        ctrls: SmallVec::from_slice(ctrls),
    }
}

/// Random gates (k in {1, 2}) on `n` wires.
fn random_gates(n: u16, m: usize, rng: &mut Rng) -> Vec<XGate> {
    (0..m)
        .map(|_| {
            let t = rng.below(n as u64) as u16;
            let mut a = rng.below(n as u64) as u16;
            while a == t {
                a = rng.below(n as u64) as u16;
            }
            let mut lits = vec![(a, rng.below(2) == 1)];
            if rng.below(2) == 1 {
                let mut b = rng.below(n as u64) as u16;
                while b == t || b == a {
                    b = rng.below(n as u64) as u16;
                }
                lits.push((b, rng.below(2) == 1));
            }
            let mut x = XGate::conj(t, lits).unwrap();
            x.comp = rng.below(2) == 1;
            x
        })
        .collect()
}

/// `encode_body` of `(gates, bounds)` as a container.
fn packed_container(gates: &[XGate], bounds: &[i64], nw: u32) -> (Vec<u8>, Vec<u8>) {
    let mut body = Vec::new();
    encode_body(gates, bounds, gates.len(), nw as usize, 3, &mut body).unwrap();
    let blobs = split_body(&body).unwrap();
    let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
    (body, container_bytes(&refs, 0, nw))
}

// ---------------------------------------------------------------- lattice helpers

#[test]
fn bounds_and_windows() {
    assert_eq!(windows_of(&[0, 3, 7]), vec![(0, 3), (3, 7)]);
    assert_eq!(bounds_text(&[(0, 9)]), "");
    assert_eq!(bounds_text(&[(0, 3), (3, 7), (7, 9)]), "3\n7\n");
}

#[test]
fn selection_parsing() {
    assert_eq!(parse_selection("0,2,5", 6).unwrap(), vec![0, 2, 5]);
    assert_eq!(parse_selection("1:4,2, 0", 6).unwrap(), vec![0, 1, 2, 3]);
    assert_eq!(parse_selection("5", 6).unwrap(), vec![5]);
    for bad in ["", "6", "x", "3:3", "4:2", "-1", "0:7"] {
        assert!(
            matches!(parse_selection(bad, 6), Err(Stage7Error::Usage(_))),
            "{bad:?} accepted"
        );
    }
}

// ---------------------------------------------------------------- cut choice

#[test]
fn cut_choice_clean_unclean_and_tie() {
    // one scan over the sorted, deduplicated union of every window's candidates
    let mut scans: Vec<Vec<usize>> = Vec::new();
    let wins = [(0usize, 10usize), (10, 12), (20, 60)];
    let out = choose_cuts_with(&wins, &mut |c| {
        scans.push(c.to_vec());
        FxHashSet::default()
    });
    assert_eq!(scans.len(), 1);
    let mut want: Vec<usize> = wins
        .iter()
        .flat_map(|&(lo, hi)| split_candidates(lo, hi))
        .collect();
    want.sort_unstable();
    want.dedup();
    assert_eq!(scans[0], want);
    // clean: the candidate nearest the floor midpoint
    assert_eq!(out, vec![(5, true), (11, true), (40, true)]);
    // the midpoint leaks: its neighbours tie, the smaller wins
    let out = choose_cuts_with(&[(0, 10)], &mut |_| [5usize].into_iter().collect());
    assert_eq!(out, vec![(4, true)]);
    let out = choose_cuts_with(&[(0, 10)], &mut |_| [4usize, 5].into_iter().collect());
    assert_eq!(out, vec![(6, true)]);
    // every candidate leaks: the midpoint, flagged unclean
    let out = choose_cuts_with(&[(3, 10)], &mut |c| c.iter().copied().collect());
    assert_eq!(out, vec![(6, false)]);
    // a 2-gate window has exactly one candidate
    assert_eq!(split_candidates(7, 9), vec![8]);
    let out = choose_cuts_with(&[(7, 9)], &mut |_| FxHashSet::default());
    assert_eq!(out, vec![(8, true)]);
    // no window: no scan
    let mut called = false;
    assert!(
        choose_cuts_with(&[], &mut |_| {
            called = true;
            FxHashSet::default()
        })
        .is_empty()
    );
    assert!(!called);
}

// ---------------------------------------------------------------- blobs and the splice

#[test]
fn split_body_roundtrip_and_errors() {
    let mut rng = Rng::new(5);
    let gates = random_gates(6, 30, &mut rng);
    let (body, _) = packed_container(&gates, &[10, 20], 6);
    let blobs = split_body(&body).unwrap();
    assert_eq!(blobs.len(), 3);
    let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
    assert_eq!(body_bytes(&refs), body);
    assert!(split_body(&body[..body.len() - 1]).is_err());
    assert!(split_body(&[1, 0]).is_err());
    assert!(split_body(&[]).unwrap().is_empty());
}

#[test]
fn emit_blob_is_the_packers_emission_whatever_the_budget() {
    let mut rng = Rng::new(9);
    let gates = random_gates(7, 24, &mut rng);
    let (body, _) = packed_container(&gates, &[8, 16], 7);
    let blobs = split_body(&body).unwrap();
    for (i, (lo, hi)) in [(0, 8), (8, 16), (16, 24)].into_iter().enumerate() {
        let e = emit_blob(&gates, lo, hi, 7, DEFAULT_BUDGET).unwrap();
        assert_eq!(e.blob, blobs[i]);
        assert!(e.fits);
        // a budget the window exceeds: the same blob, flagged
        let tight = emit_blob(&gates, lo, hi, 7, 1).unwrap();
        assert_eq!(tight.blob, blobs[i]);
        assert_eq!(tight.monomials, e.monomials);
        assert!(!tight.fits || e.monomials <= 1);
    }
}

#[test]
fn spliced_container_equals_encode_body_of_the_refined_lattice() {
    let mut rng = Rng::new(21);
    let nw = 9u32;
    let gates = random_gates(nw as u16, 60, &mut rng);
    let bounds = [15i64, 30, 45];
    let (body, c0) = packed_container(&gates, &bounds, nw);
    let mut blobs = split_body(&body).unwrap();
    // split window 1 = [15, 30) at 22 and window 3 = [45, 60) at 50
    let mut wins = windows_of(&window_cuts(&bounds, gates.len()));
    for &(i, c) in [(3usize, 50usize), (1, 22)].iter() {
        let (lo, hi) = wins[i];
        let a = emit_blob(&gates, lo, c, nw as usize, DEFAULT_BUDGET).unwrap();
        let b = emit_blob(&gates, c, hi, nw as usize, DEFAULT_BUDGET).unwrap();
        wins.splice(i..i + 1, [(lo, c), (c, hi)]);
        blobs.splice(i..i + 1, [a.blob, b.blob]);
    }
    assert_eq!(
        wins,
        vec![(0, 15), (15, 22), (22, 30), (30, 45), (45, 50), (50, 60)]
    );
    let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
    let spliced = container_bytes(&refs, 0, nw);
    let new_bounds: Vec<i64> = wins.iter().skip(1).map(|w| w.0 as i64).collect();
    let (_, want) = packed_container(&gates, &new_bounds, nw);
    assert_eq!(
        spliced, want,
        "spliced container != encode_body of the refined lattice"
    );
    assert_ne!(spliced, c0);
    // the header is recomputed: it decodes, with the new window count
    let d = decode(&spliced, Format::Container).unwrap();
    assert_eq!(d.window_count(), 6);
    assert_eq!(d.header.unwrap().window_count, 6);
    check_windows_anf(&gates, &new_bounds, &d, 16, 1).unwrap();
    assert_eq!(bounds_text(&wins), "15\n22\n30\n45\n50\n");
}

// ---------------------------------------------------------------- the loop

fn tiny_params() -> UnpackParams {
    let mut p = UnpackParams::default();
    p.synth.time_limit = 20.0;
    p.synth.restarts = 2;
    p.allow_fallback = false;
    p.seed = 3;
    p
}

/// A C on 3 wires (n = 3) in an 8-wire state, a packed circuit of random
/// gates in 5 windows of 4 gates.
fn loop_fixture() -> (Vec<XGate>, Vec<XGate>, Vec<usize>, Vec<Vec<u8>>) {
    let c = vec![
        g(0, false, &[(1, true), (2, false)]),
        g(2, true, &[(0, true)]),
    ];
    let mut rng = Rng::new(77);
    let gates = random_gates(8, 20, &mut rng);
    let bounds = [4i64, 8, 12, 16];
    let (body, _) = packed_container(&gates, &bounds, 8);
    let blobs = split_body(&body).unwrap();
    (c, gates, window_cuts(&bounds, 20), blobs)
}

#[test]
fn forced_failure_round_splits_every_window_then_converges() {
    let (c, gates, cuts, blobs) = loop_fixture();
    let cv = CView::new(
        &c,
        3,
        &CviewParams {
            batches: 4,
            seed: 1,
        },
    );
    let inp = RefineInputs {
        packed: &gates,
        source_wires: 3,
        state_wires: 8,
        cview: &cv,
    };
    let rp = RefineParams {
        force_fail_rounds: 1,
        ..Default::default()
    };
    let mut rounds_seen: Vec<(usize, usize, usize)> = Vec::new();
    let lines = std::sync::Mutex::new(Vec::new());
    let out = refine(
        blobs.clone(),
        &cuts,
        &inp,
        &tiny_params(),
        &rp,
        &mut |r, wins, bytes| {
            let d = decode(bytes, Format::Container).unwrap();
            assert_eq!(d.window_count(), wins.len());
            rounds_seen.push((r, wins.len(), bytes.len()));
            Ok(())
        },
        &|_, _, _, _| {},
        &|s| lines.lock().unwrap().push(s.to_string()),
    )
    .unwrap();
    let lines = lines.into_inner().unwrap();
    assert_eq!(out.windows_initial, 5);
    assert_eq!(out.pieces.len(), 10, "every 4-gate window was split once");
    assert_eq!(out.rounds.len(), 2);
    assert_eq!(out.rounds[0].pending, 5);
    assert_eq!(out.rounds[0].converged, 0);
    assert_eq!(out.rounds[0].failed, 5);
    assert_eq!(out.rounds[0].split, 5);
    assert_eq!(out.rounds[1].pending, 10);
    assert_eq!(out.rounds[1].converged, 10);
    assert_eq!(out.rounds[1].failed, 0);
    assert_eq!(out.splits, 5);
    assert_eq!(out.verbatim, 0);
    assert_eq!(out.over_budget, 0);
    assert_eq!(out.new_cuts.len(), 5);
    assert_eq!(
        rounds_seen.iter().map(|r| (r.0, r.1)).collect::<Vec<_>>(),
        vec![(0, 5), (1, 10)]
    );
    assert_eq!(lines.len(), 2);
    assert!(lines[0].starts_with("round 0: pending 5, converged 0, failed 5, split 5"));
    // the lattice: every piece inside its parent, the new cuts strictly inside
    for (k, p) in out.pieces.iter().enumerate() {
        assert_eq!(p.round, 1);
        assert_eq!(p.result.status, WinStatus::Ok);
        assert_eq!(
            p.result.idx, k,
            "synthesized with its index in the round-1 lattice"
        );
        assert!(p.fits);
        let parent = (cuts[k / 2], cuts[k / 2 + 1]);
        assert!(parent.0 <= p.lo && p.hi <= parent.1);
        let e = emit_blob(&gates, p.lo, p.hi, 8, DEFAULT_BUDGET).unwrap();
        assert_eq!(p.blob, e.blob);
    }
    for (&(c, clean), k) in out.new_cuts.iter().zip(0..) {
        assert!(cuts[k] < c && c < cuts[k + 1]);
        assert!(clean, "random gates on wires >= n leak nothing");
    }
    // the final deliverable is the packed circuit's on the refined lattice,
    // and the assembled circuit computes it
    let final_bounds: Vec<i64> = out.bounds().iter().map(|&b| b as i64).collect();
    let d = decode(&out.container(3, 8), Format::Container).unwrap();
    assert_eq!(d.window_count(), 10);
    check_windows_anf(&gates, &final_bounds, &d, 16, 1).unwrap();
    let (concat, offsets) = out.concatenate();
    check_equiv(&concat, &d, 1, 1, "refined concatenation").unwrap();
    assert_eq!(offsets.len(), 10);
    let tsv = refine_windows_tsv(&out.pieces, &offsets);
    assert_eq!(tsv.lines().count(), 11);
    assert!(tsv.starts_with("idx\tlo\thi\tround\torig_gates\tsyn_gates\tstatus\tsynth_status\tseconds\tmonomials\tchanged_wires\tinput_vars\tmax_degree\tsyn_offset\t"));
    let row1: Vec<&str> = tsv.lines().nth(2).unwrap().split('\t').collect();
    assert_eq!(
        &row1[..4],
        &[
            "1",
            &cuts[0].max(out.pieces[1].lo).to_string(),
            &out.pieces[1].hi.to_string(),
            "1"
        ]
    );
    assert_eq!(row1[6], "ok");
    assert_eq!(row1[7], "converged");
    let rt = rounds_tsv(&out.rounds);
    assert!(rt.starts_with("round\tpending\tconverged\tfailed\tsplit\tseconds\n0\t5\t0\t5\t5\t"));
}

#[test]
fn without_failures_one_round_and_the_lattice_is_unchanged() {
    let (c, gates, cuts, blobs) = loop_fixture();
    let cv = CView::new(
        &c,
        3,
        &CviewParams {
            batches: 4,
            seed: 1,
        },
    );
    let inp = RefineInputs {
        packed: &gates,
        source_wires: 3,
        state_wires: 8,
        cview: &cv,
    };
    let out = refine(
        blobs.clone(),
        &cuts,
        &inp,
        &tiny_params(),
        &RefineParams::default(),
        &mut |_, _, _| Ok(()),
        &|_, _, _, _| {},
        &|_| {},
    )
    .unwrap();
    assert_eq!(out.rounds.len(), 1);
    assert_eq!(out.pieces.len(), 5);
    assert_eq!(out.wins(), windows_of(&cuts));
    assert!(out.new_cuts.is_empty());
    // the same blobs: the final deliverable is D_0
    let refs: Vec<&[u8]> = blobs.iter().map(|b| b.as_slice()).collect();
    assert_eq!(out.container(3, 8), container_bytes(&refs, 3, 8));
    for (k, p) in out.pieces.iter().enumerate() {
        assert_eq!(p.round, 0);
        assert_eq!(p.result.idx, k);
        assert_eq!(p.result.status, WinStatus::Ok);
    }
}

#[test]
fn max_rounds_zero_emits_forced_failures_verbatim() {
    let (c, gates, cuts, blobs) = loop_fixture();
    let cv = CView::new(
        &c,
        3,
        &CviewParams {
            batches: 4,
            seed: 1,
        },
    );
    let inp = RefineInputs {
        packed: &gates,
        source_wires: 3,
        state_wires: 8,
        cview: &cv,
    };
    let rp = RefineParams {
        force_fail_rounds: 1,
        max_rounds: 0,
        ..Default::default()
    };
    let out = refine(
        blobs,
        &cuts,
        &inp,
        &tiny_params(),
        &rp,
        &mut |_, _, _| Ok(()),
        &|_, _, _, _| {},
        &|_| {},
    )
    .unwrap();
    assert_eq!(out.rounds.len(), 1);
    assert_eq!(out.verbatim, 5);
    assert_eq!(out.splits, 0);
    for (k, p) in out.pieces.iter().enumerate() {
        assert_eq!(p.round, -1);
        assert_eq!(p.result.status, WinStatus::Verbatim);
        assert_eq!(p.result.note, "max_rounds 0");
        assert_eq!(p.result.gates, gates[cuts[k]..cuts[k + 1]].to_vec());
    }
    let (concat, _) = out.concatenate();
    assert_eq!(concat, gates);
    let tsv = refine_windows_tsv(&out.pieces, &[0, 4, 8, 12, 16]);
    let row: Vec<&str> = tsv.lines().nth(1).unwrap().split('\t').collect();
    assert_eq!(
        &row[3..9],
        &["-1", "4", "4", "verbatim", "max_rounds 0", "0.000"]
    );
}
