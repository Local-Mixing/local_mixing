//! DB-backed test for end-of-shuffle SAMF compression.
//!
//! Unlike `transforms.rs` (which runs DB-free, exercising only the verbatim undo path),
//! this opens the real curated+sharded LMDB `./db` so the per-SAMF compression actually
//! fires, and verifies every shuffle still preserves circuit semantics. Skips cleanly when
//! `./db` is not present (e.g. CI), so it never fails for lack of a database.

use lmdb::{Database, Environment};
use local_mixing::circuit::CircuitSeq;
use local_mixing::random::random_data::random_circuit;
use local_mixing::replace::main_mix::open_all_dbs;
use local_mixing::replace::transpositions::{
    END_SAMF_COMPRESSIONS_MADE, Transpositions, apply_unsamf, insert_wire_m_samfs_every_x,
    insert_wire_shuffles_knuth, insert_wire_shuffles_simple, insert_wire_shuffles_x,
    shuffled_shoot_then_samf, shuffled_shoot_then_samf_core, shuffled_shooting_game,
};
use std::path::Path;
use std::sync::atomic::Ordering;

// Shared, process-global DB handle. LMDB does not support opening the same env path twice in
// one process, and `cargo test` runs tests in parallel — so all DB-backed tests must share a
// single env (opened once via OnceLock) rather than each opening their own (which races on
// dbi-open and silently skips).
static DB: std::sync::OnceLock<Option<(Environment, Vec<Database>, Vec<Database>)>> =
    std::sync::OnceLock::new();

fn open_db() -> Option<&'static (Environment, Vec<Database>, Vec<Database>)> {
    DB.get_or_init(|| {
        if !Path::new("./db").exists() {
            return None;
        }
        let env = Environment::new()
            .set_max_dbs(556)
            .set_map_size(900 * 1024 * 1024 * 1024)
            .open(Path::new("./db"))
            .ok()?;
        // open_all_dbs panics if the named shard dbs are missing; treat that as "skip".
        let dbs =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| open_all_dbs(&env))).ok()?;
        Some((env, dbs.0, dbs.1))
    })
    .as_ref()
}

#[test]
fn db_backed_shuffles_preserve_equivalence() {
    let Some((env, shard_dbs, curated_shard_dbs)) = open_db() else {
        eprintln!("./db unavailable — skipping DB-backed SAMF compression test");
        return;
    };

    let before = END_SAMF_COMPRESSIONS_MADE.load(Ordering::Relaxed);

    // Scale is env-configurable for heavier stress runs (defaults keep the test quick).
    // Default n=10 (higher DB hit-rate) and enough rounds that the compression path
    // reliably fires; both overridable for heavier stress runs.
    let n: usize = std::env::var("LM_SAMF_N").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let rounds: usize = std::env::var("LM_SAMF_ROUNDS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let funcs: [&str; 6] = [
        "simple",
        "knuth",
        "x",
        "m_samfs_every_x",
        "shuffled_shooting_game",
        "shuffled_shoot_then_samf",
    ];

    // Optionally skip the shooting-game-based functions (k>=4); lets us stress the other
    // functions' end-compression in isolation.
    let skip_ssg = std::env::var("LM_SAMF_SKIP_SSG").is_ok();

    // Several rounds, rotating through every function that inserts SAMFs at the end.
    for round in 0..rounds {
        for (k, name) in funcs.iter().enumerate() {
            if k >= 4 && skip_ssg {
                continue;
            }
            let original = random_circuit(n, 40);
            let mut c = original.clone();
            match k {
                0 => insert_wire_shuffles_simple(&mut c, n, env, curated_shard_dbs, shard_dbs),
                1 => insert_wire_shuffles_knuth(&mut c, n, env, curated_shard_dbs, shard_dbs),
                2 => insert_wire_shuffles_x(&mut c, n, 3, env, curated_shard_dbs, shard_dbs),
                3 => insert_wire_m_samfs_every_x(
                    &mut c,
                    n,
                    4,
                    3,
                    &env,
                    &curated_shard_dbs,
                    &shard_dbs,
                ),
                4 => {
                    shuffled_shooting_game(&mut c, n, env, curated_shard_dbs, shard_dbs, 4);
                }
                _ => {
                    shuffled_shoot_then_samf(
                        &mut c,
                        n,
                        4,
                        3,
                        4,
                        &env,
                        &curated_shard_dbs,
                        &shard_dbs,
                    );
                }
            }
            assert!(
                c.probably_equal(&original, n, 400).is_ok(),
                "DB-backed shuffle '{name}' (round {round}) changed circuit semantics"
            );
            assert!(
                !c.gates.is_empty(),
                "shuffle '{name}' produced an empty circuit"
            );
        }
    }

    let fired = END_SAMF_COMPRESSIONS_MADE.load(Ordering::Relaxed) - before;
    eprintln!("end-of-shuffle SAMF compressions fired: {fired}");
    assert!(
        fired > 0,
        "expected the end-of-shuffle SAMF compression path to fire at least once against a real DB"
    );
}

// Mirrors the `--single-end` pipeline: accumulate SAMF state across several rounds (leaving
// the circuit non-equivalent in between) and undo it all in ONE pass after the last round.
// The final circuit must be equivalent to the original. (Compression between rounds is
// omitted here — it is function-preserving by construction — so this isolates the cross-round
// accumulate-and-single-undo logic.)
#[test]
fn single_end_multiround_preserves_equivalence() {
    let Some((env, shard_dbs, curated_shard_dbs)) = open_db() else {
        eprintln!("./db unavailable — skipping single-end multi-round test");
        return;
    };

    // Small params/rounds: this test omits the per-round compression the real pipeline does,
    // so the circuit would otherwise grow exponentially across rounds.
    let n = 10;
    let (m, x, gates_ahead, rounds) = (2usize, 8usize, 3usize, 3usize);

    for trial in 0..4 {
        let original = random_circuit(n, 30);

        let mut total_t = Transpositions {
            transpositions: Vec::new(),
        };
        let mut total_neg = vec![0u8; n];
        let mut gates = original.gates.clone();

        for round in 0..rounds {
            let (out, t_round, neg_round, _c) = shuffled_shoot_then_samf_core(
                &gates,
                n,
                m,
                x,
                gates_ahead,
                &env,
                &curated_shard_dbs,
                &shard_dbs,
            );
            gates = out;

            // Fold this round into the running accumulator (same as main_mix).
            let mut new_total_neg = neg_round;
            for w in 0..n {
                if total_neg[w] == 1 {
                    let cw = t_round.evaluate(w as u16) as usize;
                    new_total_neg[cw] ^= 1;
                }
            }
            total_neg = new_total_neg;
            total_t = total_t.concat(&t_round);

            // Mid-rounds: intentionally NOT equivalent. Final round: undo everything once.
            if round == rounds - 1 {
                apply_unsamf(
                    &mut gates,
                    &total_t,
                    &total_neg,
                    n,
                    &env,
                    &curated_shard_dbs,
                    &shard_dbs,
                );
            }
        }

        let result = CircuitSeq { gates };
        assert!(
            result.probably_equal(&original, n, 400).is_ok(),
            "single-end ({rounds} rounds, trial {trial}) did not restore equivalence"
        );
    }
}
