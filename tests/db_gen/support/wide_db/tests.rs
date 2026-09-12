use super::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn probe_round_trips_through_a_real_store() {
    // Build a one-entry wide store exactly the way the S7 pass does, then
    // probe it with a window computing the same function and demand a
    // proven-equivalent placement back.
    let budget = XPolyBudget::default();
    let window = vec![
        XGate::from_g57([0, 1, 2]),
        XGate::conj(3, [(0, true), (1, false), (2, true)]).unwrap(),
    ];
    // Min-dir keying, matching the builder.
    let canon_f = canonicalize_xgates_single(&window, false, budget).unwrap();
    let canon_r = canonicalize_xgates_single(&window, true, budget).unwrap();
    let blob_f = polys_repr_blob(&canon_f.polys);
    let blob_r = polys_repr_blob(&canon_r.polys);
    let (canon, blob, store_reversed) = if blob_r < blob_f {
        (canon_r, blob_r, true)
    } else {
        (canon_f, blob_f, false)
    };
    let key = xxh3_128(&blob).to_le_bytes();
    let window_for_store: Vec<XGate> = if store_reversed {
        window.iter().rev().cloned().collect()
    } else {
        window.clone()
    };

    // global -> canonical mapping (inverse of order over dense positions)
    let mut inv = vec![0u16; canon.order.data.len()];
    for (c, &d) in canon.order.data.iter().enumerate() {
        inv[d as usize] = c as u16;
    }
    let to_canonical = |w: u16| -> u16 {
        let dense = canon.used_wires.binary_search(&w).unwrap() as u16;
        inv[dense as usize]
    };
    let mapped: Vec<XGate> = window_for_store
        .iter()
        .map(|g| XGate {
            target: to_canonical(g.target),
            comp: g.comp,
            ctrls: g.ctrls.iter().map(|&(w, p)| (to_canonical(w), p)).collect(),
        })
        .collect();
    let chunk = mpx1::encode_circuit(&mapped).unwrap();

    let dir = std::env::temp_dir().join(format!("wide_probe_test_{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        opts.set_merge_operator_associative("append_merge_wide", append_merge_wide);
        let db = DB::open(&opts, &dir).unwrap();
        db.merge(key, &chunk).unwrap();
        db.flush().unwrap();
    }

    let wdb = WideDb::open(dir.to_str().unwrap());
    let mut rng = StdRng::seed_from_u64(11);
    let placed = wide_probe(&window, 8, &wdb, budget, &mut rng);
    assert!(
        !placed.is_empty(),
        "probe must return a proven-equivalent candidate"
    );
    drop(wdb);
    let _ = std::fs::remove_dir_all(&dir);
}
