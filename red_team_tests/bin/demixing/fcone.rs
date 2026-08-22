//! Output-oriented demixing for `mpmct1` circuits.
//!
//! Given a live output half, orient the public fmix crossing rules so gates
//! targeting dead outputs move to the right of gates targeting live outputs.
//! Once exposed as a suffix, dead-target gates are removed by exact backward
//! liveness.  This uses only the released circuit and the published rewrite
//! calculus; no provenance or generator-run data is consulted.

use clap::Parser;
use local_mixing::engine::arena::{Arena, Dir, NIL};
use local_mixing::postprocessing::compress::liveness_prune;
use local_mixing::engine::format;
use local_mixing::engine::rules::{self, BlockReason, Outcome, RuleKind};
use local_mixing::circuit::xgate::{XGate, eval_lanes};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::VecDeque;

#[derive(Parser, Debug)]
#[command(name = "fcone")]
struct Args {
    #[arg(long)]
    input: String,
    #[arg(long)]
    output: Option<String>,
    /// First live target wire; wires below it are existential/dead outputs.
    #[arg(long)]
    live_start: usize,
    #[arg(long, default_value_t = 20)]
    k_max: usize,
    #[arg(long, default_value_t = 1_000_000)]
    max_moves: u64,
    #[arg(long, default_value_t = 4_000_000)]
    max_gates: usize,
    #[arg(long, default_value_t = 8)]
    verify_rounds: usize,
    #[arg(long, default_value_t = 1)]
    seed: u64,
}

#[derive(Default)]
struct Stats {
    swaps: u64,
    r1: u64,
    r2: u64,
    r3: u64,
    presplit_shot: u64,
    presplit_colliding: u64,
    width_block: u64,
    deadlock: u64,
    stale: u64,
}

fn inversion(arena: &Arena, left: u32, live_start: usize) -> Option<u32> {
    if left == NIL || !arena.is_linked(left) {
        return None;
    }
    let right = arena.neighbor(left, Dir::R);
    if right == NIL || !arena.is_linked(right) {
        return None;
    }
    let dead_left = (arena.gate(left).target as usize) < live_start;
    let live_right = (arena.gate(right).target as usize) >= live_start;
    (dead_left && live_right).then_some(right)
}

fn enqueue_local(arena: &Arena, ids: &[u32], live_start: usize, q: &mut VecDeque<(u32, u32)>) {
    for &id in ids {
        if let Some(r) = inversion(arena, id, live_start) {
            q.push_back((id, r));
        }
    }
}

fn replace_pair(arena: &mut Arena, left: u32, right: u32, seq: Vec<XGate>) -> Vec<u32> {
    let before = arena.neighbor(left, Dir::L);
    let after = arena.neighbor(right, Dir::R);
    arena.unlink(left);
    arena.unlink(right);
    arena.free_node(left);
    arena.free_node(right);
    let mut prev = before;
    let mut ids = Vec::with_capacity(seq.len() + 2);
    if before != NIL {
        ids.push(before);
    }
    for g in seq {
        let id = arena.insert_after(prev, g);
        ids.push(id);
        prev = id;
    }
    if after != NIL {
        ids.push(after);
    }
    ids
}

fn main() {
    let args = Args::parse();
    let (gates, wires) = format::read_mpmct(&args.input).expect("read mpmct1 circuit");
    assert!(args.live_start < wires);
    let original = gates.clone();
    let mut arena = Arena::from_gates(gates);
    let mut rng = StdRng::seed_from_u64(args.seed);
    let mut q = VecDeque::new();
    let ids = arena.ids_in_order();
    enqueue_local(&arena, &ids, args.live_start, &mut q);
    let mut stats = Stats::default();
    let mut moves = 0u64;

    while moves < args.max_moves && arena.len() < args.max_gates {
        let Some((g_id, h_id)) = q.pop_front() else {
            break;
        };
        if !arena.is_linked(g_id)
            || !arena.is_linked(h_id)
            || arena.neighbor(g_id, Dir::R) != h_id
            || inversion(&arena, g_id, args.live_start) != Some(h_id)
        {
            stats.stale += 1;
            continue;
        }
        let g = arena.gate(g_id).clone();
        let h = arena.gate(h_id).clone();
        let local_ids;
        if !XGate::collides(&g, &h) {
            // [g,h] -> [h,g]
            let before = arena.neighbor(g_id, Dir::L);
            let after = arena.neighbor(h_id, Dir::R);
            arena.unlink(g_id);
            arena.link_after(g_id, h_id);
            local_ids = [before, h_id, g_id, after]
                .into_iter()
                .filter(|&x| x != NIL)
                .collect();
            stats.swaps += 1;
        } else if g.comp {
            let pieces = rules::presplit(&g, &mut rng);
            local_ids = replace_pair(
                &mut arena,
                g_id,
                h_id,
                pieces.into_iter().chain([h]).collect(),
            );
            stats.presplit_shot += 1;
        } else {
            match rules::cross(&g, &h, args.k_max, &mut rng) {
                Outcome::R0Swap => {
                    let before = arena.neighbor(g_id, Dir::L);
                    let after = arena.neighbor(h_id, Dir::R);
                    arena.unlink(g_id);
                    arena.link_after(g_id, h_id);
                    local_ids = [before, h_id, g_id, after]
                        .into_iter()
                        .filter(|&x| x != NIL)
                        .collect();
                    stats.swaps += 1;
                }
                Outcome::PresplitColliding => {
                    let pieces = rules::presplit(&h, &mut rng);
                    local_ids = replace_pair(
                        &mut arena,
                        g_id,
                        h_id,
                        [g].into_iter().chain(pieces).collect(),
                    );
                    stats.presplit_colliding += 1;
                }
                Outcome::Rewrite { seq, kind, .. } => {
                    local_ids = replace_pair(
                        &mut arena,
                        g_id,
                        h_id,
                        seq.into_iter().map(|(gate, _)| gate).collect(),
                    );
                    match kind {
                        RuleKind::R1 => stats.r1 += 1,
                        RuleKind::R2 => stats.r2 += 1,
                        RuleKind::R3 => stats.r3 += 1,
                    }
                }
                Outcome::Blocked(reason) => {
                    match reason {
                        BlockReason::WidthCap => stats.width_block += 1,
                        BlockReason::Deadlock => stats.deadlock += 1,
                    }
                    continue;
                }
            }
        }
        moves += 1;
        enqueue_local(&arena, &local_ids, args.live_start, &mut q);
        if moves % 100_000 == 0 {
            eprintln!(
                "[fcone] moves={} gates={} queue={} swaps={} r1={} r2={} r3={} blocked={}/{}",
                moves,
                arena.len(),
                q.len(),
                stats.swaps,
                stats.r1,
                stats.r2,
                stats.r3,
                stats.width_block,
                stats.deadlock
            );
        }
    }

    let crossed = arena.to_vec();
    let mut live = vec![false; wires];
    live[args.live_start..].fill(true);
    let (out, dropped) = liveness_prune(crossed, &live);
    println!(
        "[fcone] moves={} gates_before={} gates_crossed={} gates_after_prune={} dropped={} queue={} swaps={} r1={} r2={} r3={} presplit={}/{} blocked={}/{} stale={}",
        moves,
        original.len(),
        arena.len(),
        out.len(),
        dropped,
        q.len(),
        stats.swaps,
        stats.r1,
        stats.r2,
        stats.r3,
        stats.presplit_shot,
        stats.presplit_colliding,
        stats.width_block,
        stats.deadlock,
        stats.stale
    );

    let mut verify_rng = StdRng::seed_from_u64(args.seed ^ 0xFC0E);
    for round in 0..args.verify_rounds {
        let state: Vec<u64> = (0..wires).map(|_| verify_rng.random()).collect();
        let mut a = state.clone();
        let mut b = state;
        eval_lanes(original.iter(), &mut a);
        eval_lanes(out.iter(), &mut b);
        assert_eq!(
            &a[args.live_start..],
            &b[args.live_start..],
            "live-output equivalence failed in round {round}"
        );
    }
    println!("[fcone] verified {} rounds x64 lanes", args.verify_rounds);
    if let Some(path) = &args.output {
        format::write_mpmct(path, &out, wires).expect("write output circuit");
        println!("[fcone] wrote {}", path);
    }
}
