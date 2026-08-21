//! End-to-end proof of the fmix frozen-DB contraction move over REAL store
//! files. Builds a synthetic sharded LMDB whose single entry reduces a
//! heterogeneous ("toffoli", i.e. conjunction-control) two-gate window to
//! nothing, converts it to a frozen-table directory with the production
//! builder, opens it as a FrozenDb, and drives an actual Mixer with the DB move
//! enabled. Asserts the move fires and never breaks equivalence.

use lmdb::Transaction;
use local_mixing::circuit::polys_repr_blob;
use local_mixing::circuit::xgate::{XGate, eval_lanes};
use local_mixing::db_generation::frozen_build::{
    LmdbShards, stage_tables, stage_validate, stage_write,
};
use local_mixing::db_mixing::db_replace::DbMode;
use local_mixing::db_mixing::frozen::FrozenDb;
use local_mixing::engine::mix::{MixParams, Mixer};
use local_mixing::engine::xpoly::{XPolyBudget, canonicalize_xgates_single};
use xxhash_rust::xxh3::xxh3_128;

fn key_of(window: &[XGate], reversed: bool) -> [u8; 16] {
    let c = canonicalize_xgates_single(window, reversed, XPolyBudget::default()).unwrap();
    xxh3_128(&polys_repr_blob(&c.polys)).to_le_bytes()
}

// A circuit computes identity iff every input maps to itself.
fn is_identity(gates: &[XGate], n: usize) -> bool {
    for input in 0..(1u64 << n) {
        let mut lanes: Vec<u64> = (0..n).map(|w| (input >> w) & 1).collect();
        eval_lanes(gates.iter(), &mut lanes);
        let out = (0..n).fold(0u64, |a, w| a | ((lanes[w] & 1) << w));
        if out != input {
            return false;
        }
    }
    true
}

#[test]
fn fmix_db_move_fires_and_preserves_equivalence() {
    let base = std::env::temp_dir().join(format!("fmix_dbmove_{}", std::process::id()));
    let lmdb_dir = base.join("db");
    let frz_dir = base.join("frz");
    let _ = std::fs::remove_dir_all(&base);
    std::fs::create_dir_all(&lmdb_dir).unwrap();
    std::fs::create_dir_all(&frz_dir).unwrap();

    // The reducible window: a conjunction-control gate applied twice. p;p is the
    // identity, so the store maps this window's key to a ZERO-gate friend — a
    // strict contraction the merge/undo catalogue is tried after, not before.
    let p = XGate::conj(0, [(1, true), (2, false)]).unwrap();
    let window = [p.clone(), p.clone()];

    // Value = one friend of zero gates: [len=0]. decode_value yields an empty
    // circuit, i.e. "replace the window with nothing".
    let value: Vec<u8> = vec![0u8];

    // --- synthetic sharded LMDB keyed exactly as the runtime will look it up ---
    let env = lmdb::Environment::new()
        .set_max_dbs(300)
        .set_map_size(1 << 30)
        .open(&lmdb_dir)
        .unwrap();
    let dbs: Vec<lmdb::Database> = (0u16..=255)
        .map(|s| {
            env.create_db(
                Some(format!("{s:02x}").as_str()),
                lmdb::DatabaseFlags::empty(),
            )
            .unwrap()
        })
        .collect();
    {
        let mut txn = env.begin_rw_txn().unwrap();
        for key in [key_of(&window, false), key_of(&window, true)] {
            // A shard db's keys must all start with its byte; put under the
            // matching shard, ignoring an accidental duplicate second key.
            let _ = txn.put(
                dbs[key[0] as usize],
                &key,
                &value,
                lmdb::WriteFlags::empty(),
            );
        }
        txn.commit().unwrap();
    }
    drop(env);

    // --- convert to the frozen store and byte-exact validate ---
    let ld = lmdb_dir.to_str().unwrap();
    let fd = frz_dir.to_str().unwrap();
    let source = LmdbShards::open(ld, "");
    stage_tables(&source, fd, 400_000);
    stage_write(&source, fd);
    stage_validate(&source, fd);

    // Sanity: the real store round-trips the key to the friend bytes.
    let db = FrozenDb::open(fd, None);
    assert_eq!(
        db.get_regular(&key_of(&window, false)).as_deref(),
        Some(&value[..])
    );

    // --- drive a real Mixer with the DB move as the only contraction that can
    // remove this material. Input = 3 anchor gates on disjoint wires {3,4,5}
    // (a non-identity the store can't shorten, so the circuit never collapses)
    // plus 30 [p,p] blocks on {0,1,2} that DB windows collapse to nothing. ---
    let n_wires = 6;
    let anchors = [
        XGate::from_g57([3, 4, 5]),
        XGate::from_g57([4, 5, 3]),
        XGate::from_g57([5, 3, 4]),
    ];
    let mut input: Vec<XGate> = anchors.to_vec();
    for _ in 0..30 {
        input.push(p.clone());
        input.push(p.clone());
    }
    // Reference function of the whole input, to compare the mixed output against.
    let input_truth: Vec<u64> = (0..(1u64 << n_wires))
        .map(|x| {
            let mut lanes: Vec<u64> = (0..n_wires).map(|w| (x >> w) & 1).collect();
            eval_lanes(input.iter(), &mut lanes);
            (0..n_wires).fold(0u64, |a, w| a | ((lanes[w] & 1) << w))
        })
        .collect();

    let base_params = || MixParams {
        target_size: 1,
        moves: 20_000,
        w_cross: 0.0,
        w_fresh: 0.0,
        w_unsub: 0.0,
        w_insert: 0.0,
        w_twist_neg: 0.0,
        w_twist_swap: 0.0,
        w_twist_cnot: 0.0,
        // Start every DB probe at exactly two contiguous gates. Prefix descent
        // may inspect a one-gate suffix only after a two-gate miss, so every
        // successful replacement in this synthetic store is the intended
        // [p, p] -> [] contraction.
        s_db: 2,
        p_convex: 0.0,
        db_prefixes: true,
        undo_frac: 0.0,
        verify_every: 1_000,
        report_every: 1_000_000,
        local_verify: true,
        seed: 7,
        ..MixParams::default()
    };
    let check_equiv = |out: &[XGate], label: &str| {
        for (x, &want) in input_truth.iter().enumerate() {
            let mut lanes: Vec<u64> = (0..n_wires).map(|w| ((x as u64) >> w) & 1).collect();
            eval_lanes(out.iter(), &mut lanes);
            let got = (0..n_wires).fold(0u64, |a, w| a | ((lanes[w] & 1) << w));
            assert_eq!(got, want, "{label} broke equivalence at input {x:06b}");
        }
    };

    // --- (1) COMPRESSING move, verification OFF, with attempt recording ---
    let rec_path = base.join("attempts.log");
    let params = MixParams {
        p_comp: 1.0, // every contraction tries the compressing DB channel first
        db_verify: false,
        ..base_params()
    };
    let mut mixer = Mixer::new_with_db(input.clone(), n_wires, params, FrozenDb::open(fd, None));
    mixer.enable_db_record(rec_path.to_str().unwrap());
    mixer.run(); // global_check runs internally; panics if equivalence ever breaks
    assert!(
        mixer.counters.db_comp_hits > 0,
        "compressing DB move never fired"
    );
    assert!(mixer.counters.db_gates_removed > 0, "no gates removed");
    let out = mixer.arena.to_vec();
    assert!(out.len() < input.len(), "circuit did not contract");
    check_equiv(&out, "compressing");
    drop(mixer); // flush the record writer
    let rec = std::fs::read_to_string(&rec_path).unwrap();
    assert!(rec.contains("matches="), "record missing match counts");
    assert!(
        rec.lines()
            .any(|l| l.starts_with("attempt") && l.contains("replaced=1")),
        "record shows no successful replacement"
    );
    assert!(
        rec.lines()
            .any(|l| l.starts_with("attempt") && (l.contains("smp=ctg") || l.contains("smp=cvx"))),
        "record missing the sampler tag"
    );
    assert!(
        rec.contains("  in  "),
        "record missing the replacing subcircuit line"
    );

    // --- (2) SIZE-AGNOSTIC move (top-level p_db), verification ON ---
    let params = MixParams {
        p_db: 1.0, // every round is a size-agnostic DB attempt
        db_mode: DbMode::SizeAgnostic,
        db_verify: true,
        ..base_params()
    };
    let mut mixer = Mixer::new_with_db(input.clone(), n_wires, params, FrozenDb::open(fd, None));
    mixer.run();
    assert!(
        mixer.counters.db_agn_hits > 0,
        "size-agnostic DB move never fired"
    );
    check_equiv(&mixer.arena.to_vec(), "size-agnostic");

    let _ = std::fs::remove_dir_all(&base);
}
