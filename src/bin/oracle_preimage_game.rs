use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use clap::{Parser, ValueEnum};
use local_mixing::circuit::CircuitSeq;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum Policy {
    FormulaLinear,
    ForwardJacobian,
    InverseJacobian,
    InverseTarget,
    ForwardZero,
    LocalWalk,
    SliceSurrogate,
    UniformBoth,
}

#[derive(Parser, Debug)]
#[command(version, about = "Run the oracle-only smallest-preimage game.")]
struct Args {
    #[arg(short = 'n', long = "n", alias = "wires", default_value_t = 128)]
    wires: usize,

    #[arg(short = 'm', long = "m", alias = "gates", default_value_t = 900)]
    gates: usize,

    #[arg(
        long,
        value_delimiter = ',',
        default_value = "10,15,20",
        help = "Comma-separated k values to run."
    )]
    ks: Vec<usize>,

    #[arg(long, default_value_t = 1)]
    trials: usize,

    #[arg(long, value_enum, default_value_t = Policy::InverseTarget)]
    policy: Policy,

    #[arg(long, default_value_t = 0x0faca_de2026_u64)]
    seed: u64,

    #[arg(
        long,
        default_value_t = 0,
        help = "Stop a k run after this many oracle queries; 0 means no limit."
    )]
    max_queries: u64,

    #[arg(long, default_value_t = false)]
    parallel: bool,

    #[arg(
        long,
        default_value_t = 0,
        help = "Worker threads for parallel inverse-target search; 0 uses Rayon default."
    )]
    threads: usize,

    #[arg(long, default_value_t = 1 << 20)]
    batch_size: u64,

    #[arg(long, default_value = "work/oracle_preimage_game")]
    out_dir: PathBuf,

    #[arg(long, default_value_t = false)]
    save_private_circuit: bool,
}

#[derive(Clone)]
struct PublicChallenge {
    n: usize,
    m: usize,
    y: u128,
}

#[derive(Clone)]
struct OracleReply {
    forward: u128,
    inverse: u128,
}

struct Challenger {
    public: PublicChallenge,
    circuit: CircuitSeq,
    reverse_gates: Vec<[u16; 3]>,
}

struct Oracle<'a> {
    challenger: &'a Challenger,
}

#[derive(Debug)]
struct FinderResult {
    success: bool,
    queries: u64,
    wall_time_s: f64,
    cpu_time_s: f64,
    found_lz: usize,
    x: u128,
    queried_point: u128,
    forward_at_query: u128,
    inverse_at_query: u128,
}

#[derive(Clone, Copy)]
struct CandidateHit {
    found_lz: usize,
    x: u128,
    queried_point: u128,
    forward_at_query: u128,
    inverse_at_query: u128,
}

#[derive(Clone, Copy)]
struct ForwardPoint {
    hit: CandidateHit,
    residual: u128,
    residual_weight: u32,
}

#[derive(Clone)]
struct SliceSurrogateModel {
    vars: usize,
    k: usize,
    constant: u128,
    linear_coeffs: Vec<u128>,
    pairs: Vec<(usize, usize)>,
    pair_coeffs: Vec<u128>,
}

impl Challenger {
    fn new(n: usize, m: usize, circuit_seed: u64, y_seed: u64) -> Self {
        assert!(
            n > 0 && n <= 128,
            "this runner currently supports 1..=128 wires"
        );
        assert!(n % 2 == 0, "the challenge expects an even wire count");

        let mut circuit_rng = fastrand::Rng::with_seed(circuit_seed);
        let circuit = random_circuit_seeded(n, m, &mut circuit_rng);
        let mut reverse_gates = circuit.gates.clone();
        reverse_gates.reverse();

        let mut y_rng = fastrand::Rng::with_seed(y_seed);
        let y = random_bits(&mut y_rng, n / 2);

        Self {
            public: PublicChallenge { n, m, y },
            circuit,
            reverse_gates,
        }
    }

    fn oracle(&self) -> Oracle<'_> {
        Oracle { challenger: self }
    }

    fn evaluate(&self, input: u128) -> u128 {
        eval_gates_u128(
            mask_to_n(input, self.public.n),
            &self.circuit.gates,
            self.public.n,
        )
    }

    fn evaluate_inverse(&self, input: u128) -> u128 {
        eval_gates_u128(
            mask_to_n(input, self.public.n),
            &self.reverse_gates,
            self.public.n,
        )
    }

    fn is_solution(&self, x: u128, k: usize) -> bool {
        let output = self.evaluate(x);
        leading_zeros_n(x, self.public.n) >= k
            && low_bits(output, self.public.n / 2) == self.public.y
    }
}

impl Oracle<'_> {
    fn query(&self, q: u128) -> OracleReply {
        OracleReply {
            forward: self.challenger.evaluate(q),
            inverse: self.challenger.evaluate_inverse(q),
        }
    }
}

fn random_circuit_seeded(n: usize, m: usize, rng: &mut fastrand::Rng) -> CircuitSeq {
    let mut gates = Vec::with_capacity(m);

    for _ in 0..m {
        loop {
            let mut used = vec![false; n];
            let mut gate = [0u16; 3];
            for pin in &mut gate {
                loop {
                    let v = rng.usize(..n);
                    if !used[v] {
                        used[v] = true;
                        *pin = v as u16;
                        break;
                    }
                }
            }

            if gates.last() != Some(&gate) {
                gates.push(gate);
                break;
            }
        }
    }

    CircuitSeq { gates }
}

#[inline(always)]
fn eval_gates_u128(mut state: u128, gates: &[[u16; 3]], n: usize) -> u128 {
    for &[a, b, c] in gates {
        let c1 = (state >> b) & 1;
        let c2 = (state >> c) & 1;
        state ^= (c1 | (1 ^ c2)) << a;
    }
    mask_to_n(state, n)
}

#[inline(always)]
fn mask_to_n(x: u128, n: usize) -> u128 {
    if n == 128 { x } else { x & ((1u128 << n) - 1) }
}

#[inline(always)]
fn low_bits(x: u128, bits: usize) -> u128 {
    if bits == 128 {
        x
    } else if bits == 0 {
        0
    } else {
        x & ((1u128 << bits) - 1)
    }
}

#[inline(always)]
fn leading_zeros_n(x: u128, n: usize) -> usize {
    let extra = 128 - n;
    (mask_to_n(x, n).leading_zeros() as usize).saturating_sub(extra)
}

fn random_bits(rng: &mut fastrand::Rng, bits: usize) -> u128 {
    if bits == 0 {
        return 0;
    }
    let raw = ((rng.u64(..) as u128) << 64) | rng.u64(..) as u128;
    low_bits(raw, bits)
}

fn splitmix64(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

fn indexed_random_bits(seed: u64, index: u64, bits: usize) -> u128 {
    let lo = splitmix64(seed ^ index.wrapping_mul(0xd1b5_4a32_d192_ed03));
    let hi = splitmix64(seed ^ index.wrapping_mul(0xabc9_8355_88fb_8fac) ^ 0x632b_e59b_d9b4_e019);
    low_bits(((hi as u128) << 64) | lo as u128, bits)
}

fn make_target_point(z: u128, y: u128, half: usize, n: usize) -> u128 {
    mask_to_n((z << half) | low_bits(y, half), n)
}

fn is_forward_solution(public: &PublicChallenge, q: u128, forward: u128, k: usize) -> bool {
    leading_zeros_n(q, public.n) >= k && low_bits(forward, public.n / 2) == public.y
}

fn is_inverse_solution(public: &PublicChallenge, q: u128, inverse: u128, k: usize) -> bool {
    low_bits(q, public.n / 2) == public.y && leading_zeros_n(inverse, public.n) >= k
}

fn high_bits_value(x: u128, n: usize, k: usize) -> u128 {
    if k == 0 {
        0
    } else if k >= n {
        mask_to_n(x, n)
    } else {
        mask_to_n(x, n) >> (n - k)
    }
}

fn high_bits_popcount(x: u128, n: usize, k: usize) -> u32 {
    high_bits_value(x, n, k).count_ones()
}

fn better_inverse_candidate(
    public: &PublicChallenge,
    k: usize,
    candidate: u128,
    current: u128,
) -> bool {
    let candidate_lz = leading_zeros_n(candidate, public.n).min(k);
    let current_lz = leading_zeros_n(current, public.n).min(k);
    if candidate_lz != current_lz {
        return candidate_lz > current_lz;
    }

    let candidate_popcount = high_bits_popcount(candidate, public.n, k);
    let current_popcount = high_bits_popcount(current, public.n, k);
    if candidate_popcount != current_popcount {
        return candidate_popcount < current_popcount;
    }

    high_bits_value(candidate, public.n, k) < high_bits_value(current, public.n, k)
}

fn find_preimage(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    policy: Policy,
    finder_seed: u64,
    max_queries: u64,
) -> FinderResult {
    let mut rng = fastrand::Rng::with_seed(finder_seed);
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;

    loop {
        if max_queries != 0 && queries >= max_queries {
            return FinderResult {
                success: false,
                queries,
                wall_time_s: start_wall.elapsed().as_secs_f64(),
                cpu_time_s: cpu_seconds() - start_cpu,
                found_lz: 0,
                x: 0,
                queried_point: 0,
                forward_at_query: 0,
                inverse_at_query: 0,
            };
        }

        let q = match policy {
            Policy::InverseTarget => {
                let z = random_bits(&mut rng, public.n / 2);
                make_target_point(z, public.y, public.n / 2, public.n)
            }
            Policy::ForwardZero => {
                let suffix_bits = public.n.saturating_sub(k);
                random_bits(&mut rng, suffix_bits)
            }
            Policy::ForwardJacobian => random_forward_zero_point(public, k, &mut rng),
            Policy::InverseJacobian => random_target_point(public, &mut rng),
            Policy::FormulaLinear => random_target_point(public, &mut rng),
            Policy::LocalWalk => {
                let z = random_bits(&mut rng, public.n / 2);
                make_target_point(z, public.y, public.n / 2, public.n)
            }
            Policy::SliceSurrogate => random_target_point(public, &mut rng),
            Policy::UniformBoth => random_bits(&mut rng, public.n),
        };

        let reply = oracle.query(q);
        queries += 1;

        if is_inverse_solution(public, q, reply.inverse, k) {
            return FinderResult {
                success: true,
                queries,
                wall_time_s: start_wall.elapsed().as_secs_f64(),
                cpu_time_s: cpu_seconds() - start_cpu,
                found_lz: leading_zeros_n(reply.inverse, public.n),
                x: reply.inverse,
                queried_point: q,
                forward_at_query: reply.forward,
                inverse_at_query: reply.inverse,
            };
        }

        if is_forward_solution(public, q, reply.forward, k) {
            return FinderResult {
                success: true,
                queries,
                wall_time_s: start_wall.elapsed().as_secs_f64(),
                cpu_time_s: cpu_seconds() - start_cpu,
                found_lz: leading_zeros_n(q, public.n),
                x: q,
                queried_point: q,
                forward_at_query: reply.forward,
                inverse_at_query: reply.inverse,
            };
        }
    }
}

fn random_target_point(public: &PublicChallenge, rng: &mut fastrand::Rng) -> u128 {
    let z = random_bits(rng, public.n / 2);
    make_target_point(z, public.y, public.n / 2, public.n)
}

fn query_hit(public: &PublicChallenge, oracle: &Oracle<'_>, q: u128) -> CandidateHit {
    let reply = oracle.query(q);
    CandidateHit {
        found_lz: leading_zeros_n(reply.inverse, public.n),
        x: reply.inverse,
        queried_point: q,
        forward_at_query: reply.forward,
        inverse_at_query: reply.inverse,
    }
}

fn forward_query_point(public: &PublicChallenge, oracle: &Oracle<'_>, q: u128) -> ForwardPoint {
    let reply = oracle.query(q);
    let residual = low_bits(reply.forward, public.n / 2) ^ public.y;
    ForwardPoint {
        hit: CandidateHit {
            found_lz: leading_zeros_n(q, public.n),
            x: q,
            queried_point: q,
            forward_at_query: reply.forward,
            inverse_at_query: reply.inverse,
        },
        residual,
        residual_weight: residual.count_ones(),
    }
}

fn random_forward_zero_point(public: &PublicChallenge, k: usize, rng: &mut fastrand::Rng) -> u128 {
    random_bits(rng, public.n.saturating_sub(k))
}

fn better_forward_point(candidate: ForwardPoint, current: ForwardPoint) -> bool {
    if candidate.residual_weight != current.residual_weight {
        return candidate.residual_weight < current.residual_weight;
    }
    if candidate.hit.found_lz != current.hit.found_lz {
        return candidate.hit.found_lz > current.hit.found_lz;
    }
    candidate.hit.x < current.hit.x
}

fn shuffle_order(order: &mut [usize], rng: &mut fastrand::Rng) {
    for i in (1..order.len()).rev() {
        let j = rng.usize(0..(i + 1));
        order.swap(i, j);
    }
}

fn solve_gf2_columns(columns: &[u128], rhs: u128, eqs: usize, order: &[usize]) -> Option<u128> {
    let mut basis_col = [0u128; 128];
    let mut basis_solution = [0u128; 128];

    for &var in order {
        let mut col = low_bits(columns[var], eqs);
        if col == 0 {
            continue;
        }
        let mut solution = 1u128 << var;

        loop {
            let pivot = col.trailing_zeros() as usize;
            if basis_col[pivot] == 0 {
                basis_col[pivot] = col;
                basis_solution[pivot] = solution;
                break;
            }
            col ^= basis_col[pivot];
            solution ^= basis_solution[pivot];
            if col == 0 {
                break;
            }
        }
    }

    let mut remaining = low_bits(rhs, eqs);
    let mut solution = 0u128;
    while remaining != 0 {
        let pivot = remaining.trailing_zeros() as usize;
        if basis_col[pivot] == 0 {
            return None;
        }
        remaining ^= basis_col[pivot];
        solution ^= basis_solution[pivot];
    }

    Some(solution)
}

fn random_subdelta(delta: u128, max_flips: usize, rng: &mut fastrand::Rng) -> u128 {
    let mut bits = Vec::new();
    for bit in 0..128 {
        if ((delta >> bit) & 1) != 0 {
            bits.push(bit);
        }
    }

    if bits.len() <= max_flips {
        return delta;
    }

    shuffle_order(&mut bits, rng);
    let mut out = 0u128;
    for &bit in bits.iter().take(max_flips.max(1)) {
        out |= 1u128 << bit;
    }
    out
}

fn failed_result(queries: u64, start_wall: Instant, start_cpu: f64) -> FinderResult {
    FinderResult {
        success: false,
        queries,
        wall_time_s: start_wall.elapsed().as_secs_f64(),
        cpu_time_s: cpu_seconds() - start_cpu,
        found_lz: 0,
        x: 0,
        queried_point: 0,
        forward_at_query: 0,
        inverse_at_query: 0,
    }
}

fn successful_result(
    hit: CandidateHit,
    queries: u64,
    start_wall: Instant,
    start_cpu: f64,
) -> FinderResult {
    FinderResult {
        success: true,
        queries,
        wall_time_s: start_wall.elapsed().as_secs_f64(),
        cpu_time_s: cpu_seconds() - start_cpu,
        found_lz: hit.found_lz,
        x: hit.x,
        queried_point: hit.queried_point,
        forward_at_query: hit.forward_at_query,
        inverse_at_query: hit.inverse_at_query,
    }
}

fn find_preimage_local_walk(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    finder_seed: u64,
    max_queries: u64,
) -> FinderResult {
    let mut rng = fastrand::Rng::with_seed(finder_seed);
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;
    let half = public.n / 2;
    let gate_density = (public.m / public.n).max(1);
    let max_perturb_bits = gate_density.min(half).max(1);

    loop {
        if max_queries != 0 && queries >= max_queries {
            return failed_result(queries, start_wall, start_cpu);
        }

        let mut current = query_hit(public, oracle, random_target_point(public, &mut rng));
        queries += 1;
        if current.found_lz >= k {
            return successful_result(current, queries, start_wall, start_cpu);
        }

        let mut stale_rounds = 0usize;
        while stale_rounds < 4 {
            let mut best = CandidateHit { ..current };
            let start_bit = rng.usize(..half);

            for step in 0..half {
                if max_queries != 0 && queries >= max_queries {
                    return failed_result(queries, start_wall, start_cpu);
                }

                let z_bit = (start_bit + step) % half;
                let q = best.queried_point ^ (1u128 << (half + z_bit));
                let candidate = query_hit(public, oracle, q);
                queries += 1;

                if candidate.found_lz >= k {
                    return successful_result(candidate, queries, start_wall, start_cpu);
                }
                if better_inverse_candidate(public, k, candidate.x, best.x) {
                    best = candidate;
                }
            }

            if better_inverse_candidate(public, k, best.x, current.x) {
                current = best;
                stale_rounds = 0;
                continue;
            }

            stale_rounds += 1;
            let mut q = current.queried_point;
            let flips = 1 + rng.usize(..max_perturb_bits);
            for _ in 0..flips {
                q ^= 1u128 << (half + rng.usize(..half));
            }
            current = query_hit(public, oracle, q);
            queries += 1;
            if current.found_lz >= k {
                return successful_result(current, queries, start_wall, start_cpu);
            }
        }
    }
}

fn local_walk_episode(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    episode_seed: u64,
    query_budget: u64,
) -> (Option<CandidateHit>, u64) {
    let mut rng = fastrand::Rng::with_seed(episode_seed);
    let mut queries = 0u64;
    let half = public.n / 2;
    let gate_density = (public.m / public.n).max(1);
    let max_perturb_bits = gate_density.min(half).max(1);

    if query_budget == 0 {
        return (None, queries);
    }

    let mut current = query_hit(public, oracle, random_target_point(public, &mut rng));
    queries += 1;
    if current.found_lz >= k {
        return (Some(current), queries);
    }

    let mut stale_rounds = 0usize;
    while stale_rounds < 4 && queries < query_budget {
        let mut best = CandidateHit { ..current };
        let start_bit = rng.usize(..half);

        for step in 0..half {
            if queries >= query_budget {
                break;
            }

            let z_bit = (start_bit + step) % half;
            let q = best.queried_point ^ (1u128 << (half + z_bit));
            let candidate = query_hit(public, oracle, q);
            queries += 1;

            if candidate.found_lz >= k {
                return (Some(candidate), queries);
            }
            if better_inverse_candidate(public, k, candidate.x, best.x) {
                best = candidate;
            }
        }

        if better_inverse_candidate(public, k, best.x, current.x) {
            current = best;
            stale_rounds = 0;
            continue;
        }

        stale_rounds += 1;
        if queries >= query_budget {
            break;
        }
        let mut q = current.queried_point;
        let flips = 1 + rng.usize(..max_perturb_bits);
        for _ in 0..flips {
            q ^= 1u128 << (half + rng.usize(..half));
        }
        current = query_hit(public, oracle, q);
        queries += 1;
        if current.found_lz >= k {
            return (Some(current), queries);
        }
    }

    (None, queries)
}

fn find_preimage_parallel_local_walk(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    finder_seed: u64,
    max_queries: u64,
    batch_size: u64,
) -> FinderResult {
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;
    let mut next_episode = 0u64;
    let episode_budget = ((public.n as u64) * 16).max(1024);
    let batch_size = batch_size.max(episode_budget);

    loop {
        let remaining = if max_queries == 0 {
            batch_size
        } else {
            max_queries.saturating_sub(queries).min(batch_size)
        };

        if remaining == 0 {
            return failed_result(queries, start_wall, start_cpu);
        }

        let episodes = (remaining / episode_budget).max(1);
        let per_episode_budget = if episodes == 1 {
            remaining
        } else {
            episode_budget
        };
        let attempted = AtomicU64::new(0);
        let hit = (0..episodes).into_par_iter().find_map_any(|offset| {
            let seed = splitmix64(finder_seed ^ (next_episode + offset));
            let (hit, episode_queries) =
                local_walk_episode(public, oracle, k, seed, per_episode_budget);
            attempted.fetch_add(episode_queries, Ordering::Relaxed);
            hit
        });

        queries += attempted.load(Ordering::Relaxed);
        next_episode += episodes;

        if let Some(hit) = hit {
            return successful_result(hit, queries, start_wall, start_cpu);
        }
    }
}

fn try_forward_jacobian_candidates(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    current: ForwardPoint,
    columns: &[u128],
    rng: &mut fastrand::Rng,
    query_budget: u64,
    candidate_budget: usize,
) -> (Option<ForwardPoint>, u64) {
    let mut queries = 0u64;
    let free_bits = columns.len();
    let eqs = public.n / 2;
    let mut order: Vec<usize> = (0..free_bits).collect();
    let mut seen = Vec::<u128>::new();
    let mut best: Option<ForwardPoint> = None;

    for attempt in 0..candidate_budget {
        if queries >= query_budget {
            break;
        }

        match attempt {
            0 => {}
            1 => order.reverse(),
            _ => shuffle_order(&mut order, rng),
        }

        let Some(delta) = solve_gf2_columns(columns, current.residual, eqs, &order) else {
            continue;
        };
        if delta == 0 {
            continue;
        }

        let mut deltas = Vec::with_capacity(4);
        deltas.push(delta);
        if delta.count_ones() > 4 {
            deltas.push(random_subdelta(delta, 4, rng));
        }
        if delta.count_ones() > 8 {
            deltas.push(random_subdelta(delta, 8, rng));
        }
        if delta.count_ones() > 16 {
            deltas.push(random_subdelta(delta, 16, rng));
        }

        for proposed_delta in deltas {
            if queries >= query_budget {
                break;
            }
            if proposed_delta == 0 || seen.iter().any(|&old| old == proposed_delta) {
                continue;
            }
            seen.push(proposed_delta);

            let candidate = forward_query_point(public, oracle, current.hit.x ^ proposed_delta);
            queries += 1;
            if candidate.residual == 0 {
                return (Some(candidate), queries);
            }

            if best
                .map(|existing| better_forward_point(candidate, existing))
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }
    }

    (best, queries)
}

fn forward_jacobian_episode(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    episode_seed: u64,
    query_budget: u64,
) -> (Option<CandidateHit>, u64) {
    let mut rng = fastrand::Rng::with_seed(episode_seed);
    let mut queries = 0u64;
    let free_bits = public.n.saturating_sub(k);
    if query_budget == 0 {
        return (None, queries);
    }

    let mut current = forward_query_point(
        public,
        oracle,
        random_forward_zero_point(public, k, &mut rng),
    );
    queries += 1;
    if current.residual == 0 {
        return (Some(current.hit), queries);
    }
    if free_bits == 0 {
        return (None, queries);
    }

    let gate_density = (public.m / public.n).max(1);
    let max_perturb_bits = (gate_density * 2).min(free_bits).max(1);
    let candidate_budget = (gate_density * 4).clamp(16, 64);
    let mut stale_rounds = 0usize;

    while stale_rounds < 6 && queries < query_budget {
        let mut columns = vec![0u128; free_bits];
        let mut best = current;
        let start_bit = rng.usize(..free_bits);

        for step in 0..free_bits {
            if queries >= query_budget {
                break;
            }

            let bit = (start_bit + step) % free_bits;
            let candidate = forward_query_point(public, oracle, current.hit.x ^ (1u128 << bit));
            queries += 1;
            columns[bit] = current.residual ^ candidate.residual;

            if candidate.residual == 0 {
                return (Some(candidate.hit), queries);
            }
            if better_forward_point(candidate, best) {
                best = candidate;
            }
        }

        if queries < query_budget {
            let remaining = query_budget - queries;
            let (candidate, spent) = try_forward_jacobian_candidates(
                public,
                oracle,
                current,
                &columns,
                &mut rng,
                remaining,
                candidate_budget,
            );
            queries += spent;
            if let Some(candidate) = candidate {
                if candidate.residual == 0 {
                    return (Some(candidate.hit), queries);
                }
                if better_forward_point(candidate, best) {
                    best = candidate;
                }
            }
        }

        if better_forward_point(best, current) {
            current = best;
            stale_rounds = 0;
            continue;
        }

        stale_rounds += 1;
        if queries >= query_budget {
            break;
        }

        let mut q = current.hit.x;
        let flips = 1 + rng.usize(..max_perturb_bits);
        for _ in 0..flips {
            q ^= 1u128 << rng.usize(..free_bits);
        }
        current = forward_query_point(public, oracle, q);
        queries += 1;
        if current.residual == 0 {
            return (Some(current.hit), queries);
        }
    }

    (None, queries)
}

fn find_preimage_forward_jacobian(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    finder_seed: u64,
    max_queries: u64,
) -> FinderResult {
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;
    let free_bits = public.n.saturating_sub(k);
    let episode_budget = (((free_bits as u64) + 1) * 48).max(4096);
    let mut next_episode = 0u64;

    loop {
        let remaining = if max_queries == 0 {
            episode_budget
        } else {
            max_queries.saturating_sub(queries).min(episode_budget)
        };
        if remaining == 0 {
            return failed_result(queries, start_wall, start_cpu);
        }

        let seed = splitmix64(finder_seed ^ next_episode);
        let (hit, spent) = forward_jacobian_episode(public, oracle, k, seed, remaining);
        queries += spent;
        next_episode += 1;
        if let Some(hit) = hit {
            return successful_result(hit, queries, start_wall, start_cpu);
        }
    }
}

fn find_preimage_parallel_forward_jacobian(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    finder_seed: u64,
    max_queries: u64,
    batch_size: u64,
) -> FinderResult {
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;
    let mut next_episode = 0u64;
    let free_bits = public.n.saturating_sub(k);
    let episode_budget = (((free_bits as u64) + 1) * 48).max(4096);
    let batch_size = batch_size.max(episode_budget);

    loop {
        let remaining = if max_queries == 0 {
            batch_size
        } else {
            max_queries.saturating_sub(queries).min(batch_size)
        };
        if remaining == 0 {
            return failed_result(queries, start_wall, start_cpu);
        }

        let episodes = (remaining / episode_budget).max(1);
        let per_episode_budget = if episodes == 1 {
            remaining
        } else {
            episode_budget
        };
        let attempted = AtomicU64::new(0);
        let hit = (0..episodes).into_par_iter().find_map_any(|offset| {
            let seed = splitmix64(finder_seed ^ (next_episode + offset));
            let (hit, episode_queries) =
                forward_jacobian_episode(public, oracle, k, seed, per_episode_budget);
            attempted.fetch_add(episode_queries, Ordering::Relaxed);
            hit
        });

        queries += attempted.load(Ordering::Relaxed);
        next_episode += episodes;

        if let Some(hit) = hit {
            return successful_result(hit, queries, start_wall, start_cpu);
        }
    }
}

fn try_inverse_jacobian_candidates(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    current: CandidateHit,
    columns: &[u128],
    rng: &mut fastrand::Rng,
    query_budget: u64,
    candidate_budget: usize,
) -> (Option<CandidateHit>, u64) {
    let mut queries = 0u64;
    let vars = columns.len();
    let mut order: Vec<usize> = (0..vars).collect();
    let mut seen = Vec::<u128>::new();
    let mut best: Option<CandidateHit> = None;
    let residual = top_k_pack(current.x, public.n, k);

    for attempt in 0..candidate_budget {
        if queries >= query_budget {
            break;
        }

        match attempt {
            0 => {}
            1 => order.reverse(),
            _ => shuffle_order(&mut order, rng),
        }

        let Some(delta) = solve_gf2_columns(columns, residual, k, &order) else {
            continue;
        };
        if delta == 0 {
            continue;
        }

        let mut deltas = Vec::with_capacity(4);
        deltas.push(delta);
        if delta.count_ones() > 4 {
            deltas.push(random_subdelta(delta, 4, rng));
        }
        if delta.count_ones() > 8 {
            deltas.push(random_subdelta(delta, 8, rng));
        }
        if delta.count_ones() > 16 {
            deltas.push(random_subdelta(delta, 16, rng));
        }

        for proposed_delta in deltas {
            if queries >= query_budget {
                break;
            }
            if proposed_delta == 0 || seen.iter().any(|&old| old == proposed_delta) {
                continue;
            }
            seen.push(proposed_delta);

            let candidate_q = current.queried_point ^ (proposed_delta << (public.n / 2));
            let candidate = query_hit(public, oracle, candidate_q);
            queries += 1;
            if candidate.found_lz >= k {
                return (Some(candidate), queries);
            }

            if best
                .map(|existing| better_inverse_candidate(public, k, candidate.x, existing.x))
                .unwrap_or(true)
            {
                best = Some(candidate);
            }
        }
    }

    (best, queries)
}

fn inverse_jacobian_episode(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    episode_seed: u64,
    query_budget: u64,
) -> (Option<CandidateHit>, u64) {
    let mut rng = fastrand::Rng::with_seed(episode_seed);
    let mut queries = 0u64;
    let vars = public.n / 2;
    if query_budget == 0 {
        return (None, queries);
    }

    let mut current = query_hit(public, oracle, random_target_point(public, &mut rng));
    queries += 1;
    if current.found_lz >= k {
        return (Some(current), queries);
    }

    let gate_density = (public.m / public.n).max(1);
    let max_perturb_bits = (gate_density * 2).min(vars).max(1);
    let candidate_budget = (gate_density * 4).clamp(16, 64);
    let mut stale_rounds = 0usize;

    while stale_rounds < 6 && queries < query_budget {
        let mut columns = vec![0u128; vars];
        let mut best = current;
        let current_residual = top_k_pack(current.x, public.n, k);
        let start_bit = rng.usize(..vars);

        for step in 0..vars {
            if queries >= query_budget {
                break;
            }

            let bit = (start_bit + step) % vars;
            let candidate = query_hit(
                public,
                oracle,
                current.queried_point ^ (1u128 << (vars + bit)),
            );
            queries += 1;
            columns[bit] = current_residual ^ top_k_pack(candidate.x, public.n, k);

            if candidate.found_lz >= k {
                return (Some(candidate), queries);
            }
            if better_inverse_candidate(public, k, candidate.x, best.x) {
                best = candidate;
            }
        }

        if queries < query_budget {
            let remaining = query_budget - queries;
            let (candidate, spent) = try_inverse_jacobian_candidates(
                public,
                oracle,
                k,
                current,
                &columns,
                &mut rng,
                remaining,
                candidate_budget,
            );
            queries += spent;
            if let Some(candidate) = candidate {
                if candidate.found_lz >= k {
                    return (Some(candidate), queries);
                }
                if better_inverse_candidate(public, k, candidate.x, best.x) {
                    best = candidate;
                }
            }
        }

        if better_inverse_candidate(public, k, best.x, current.x) {
            current = best;
            stale_rounds = 0;
            continue;
        }

        stale_rounds += 1;
        if queries >= query_budget {
            break;
        }

        let mut q = current.queried_point;
        let flips = 1 + rng.usize(..max_perturb_bits);
        for _ in 0..flips {
            q ^= 1u128 << (vars + rng.usize(..vars));
        }
        current = query_hit(public, oracle, q);
        queries += 1;
        if current.found_lz >= k {
            return (Some(current), queries);
        }
    }

    (None, queries)
}

fn find_preimage_inverse_jacobian(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    finder_seed: u64,
    max_queries: u64,
) -> FinderResult {
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;
    let episode_budget = (((public.n / 2) as u64 + 1) * 48).max(4096);
    let mut next_episode = 0u64;

    loop {
        let remaining = if max_queries == 0 {
            episode_budget
        } else {
            max_queries.saturating_sub(queries).min(episode_budget)
        };
        if remaining == 0 {
            return failed_result(queries, start_wall, start_cpu);
        }

        let seed = splitmix64(finder_seed ^ next_episode);
        let (hit, spent) = inverse_jacobian_episode(public, oracle, k, seed, remaining);
        queries += spent;
        next_episode += 1;
        if let Some(hit) = hit {
            return successful_result(hit, queries, start_wall, start_cpu);
        }
    }
}

fn find_preimage_parallel_inverse_jacobian(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    finder_seed: u64,
    max_queries: u64,
    batch_size: u64,
) -> FinderResult {
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;
    let mut next_episode = 0u64;
    let episode_budget = (((public.n / 2) as u64 + 1) * 48).max(4096);
    let batch_size = batch_size.max(episode_budget);

    loop {
        let remaining = if max_queries == 0 {
            batch_size
        } else {
            max_queries.saturating_sub(queries).min(batch_size)
        };
        if remaining == 0 {
            return failed_result(queries, start_wall, start_cpu);
        }

        let episodes = (remaining / episode_budget).max(1);
        let per_episode_budget = if episodes == 1 {
            remaining
        } else {
            episode_budget
        };
        let attempted = AtomicU64::new(0);
        let hit = (0..episodes).into_par_iter().find_map_any(|offset| {
            let seed = splitmix64(finder_seed ^ (next_episode + offset));
            let (hit, episode_queries) =
                inverse_jacobian_episode(public, oracle, k, seed, per_episode_budget);
            attempted.fetch_add(episode_queries, Ordering::Relaxed);
            hit
        });

        queries += attempted.load(Ordering::Relaxed);
        next_episode += episodes;

        if let Some(hit) = hit {
            return successful_result(hit, queries, start_wall, start_cpu);
        }
    }
}

fn bit_words(bits: usize) -> usize {
    bits.div_ceil(64)
}

fn set_row_bit(row: &mut [u64], bit: usize) {
    row[bit / 64] |= 1u64 << (bit % 64);
}

fn get_row_bit(row: &[u64], bit: usize) -> bool {
    ((row[bit / 64] >> (bit % 64)) & 1) != 0
}

fn xor_bit_rows(dst: &mut [u64], src: &[u64]) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d ^= *s;
    }
}

fn first_row_bit(row: &[u64]) -> Option<usize> {
    for (word_idx, &word) in row.iter().enumerate() {
        if word != 0 {
            return Some(word_idx * 64 + word.trailing_zeros() as usize);
        }
    }
    None
}

fn random_quadratic_pairs(
    vars: usize,
    pair_count: usize,
    rng: &mut fastrand::Rng,
) -> Vec<(usize, usize)> {
    let mut pairs = Vec::with_capacity(vars.saturating_mul(vars.saturating_sub(1)) / 2);
    for a in 0..vars {
        for b in (a + 1)..vars {
            pairs.push((a, b));
        }
    }

    for i in (1..pairs.len()).rev() {
        let j = rng.usize(0..(i + 1));
        pairs.swap(i, j);
    }

    pairs.truncate(pair_count.min(pairs.len()));
    pairs
}

fn slice_feature_row(z: u128, vars: usize, pairs: &[(usize, usize)], features: usize) -> Vec<u64> {
    let mut row = vec![0u64; bit_words(features)];
    set_row_bit(&mut row, 0);

    for bit in 0..vars {
        if ((z >> bit) & 1) != 0 {
            set_row_bit(&mut row, 1 + bit);
        }
    }

    for (idx, &(a, b)) in pairs.iter().enumerate() {
        let feature = 1 + vars + idx;
        if feature >= features {
            break;
        }
        if ((z >> a) & 1) != 0 && ((z >> b) & 1) != 0 {
            set_row_bit(&mut row, feature);
        }
    }

    row
}

fn insert_independent_bit_row(
    mut row: Vec<u64>,
    mut rhs: u128,
    basis_rows: &mut [Option<Vec<u64>>],
    basis_rhs: &mut [u128],
) -> bool {
    while let Some(pivot) = first_row_bit(&row) {
        if pivot >= basis_rows.len() {
            return false;
        }
        if let Some(basis) = &basis_rows[pivot] {
            xor_bit_rows(&mut row, basis);
            rhs ^= basis_rhs[pivot];
        } else {
            basis_rows[pivot] = Some(row);
            basis_rhs[pivot] = rhs;
            return true;
        }
    }

    false
}

fn coefficients_from_full_basis(
    mut basis_rows: Vec<Option<Vec<u64>>>,
    mut basis_rhs: Vec<u128>,
    features: usize,
) -> Option<Vec<u128>> {
    let mut rows = Vec::with_capacity(features);
    for row in basis_rows.iter_mut().take(features) {
        rows.push(row.take()?);
    }

    for col in (0..features).rev() {
        let src = rows[col].clone();
        let src_rhs = basis_rhs[col];
        for row in 0..col {
            if get_row_bit(&rows[row], col) {
                xor_bit_rows(&mut rows[row], &src);
                basis_rhs[row] ^= src_rhs;
            }
        }
    }

    Some(basis_rhs)
}

fn build_slice_surrogate_model(
    vars: usize,
    k: usize,
    coeff_by_feature: Vec<u128>,
    pairs: Vec<(usize, usize)>,
) -> SliceSurrogateModel {
    let pair_offset = 1 + vars;
    SliceSurrogateModel {
        vars,
        k,
        constant: coeff_by_feature[0],
        linear_coeffs: coeff_by_feature[1..pair_offset].to_vec(),
        pair_coeffs: coeff_by_feature[pair_offset..].to_vec(),
        pairs,
    }
}

fn learn_slice_surrogate_model(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    rng: &mut fastrand::Rng,
    query_budget: u64,
) -> (Option<SliceSurrogateModel>, Option<CandidateHit>, u64) {
    let vars = public.n / 2;
    let all_quadratic_features = vars.saturating_mul(vars.saturating_sub(1)) / 2;
    let features = (1 + vars + all_quadratic_features).min(512);
    let pair_count = features.saturating_sub(1 + vars);
    let pairs = random_quadratic_pairs(vars, pair_count, rng);
    let mut basis_rows = vec![None; features];
    let mut basis_rhs = vec![0u128; features];
    let mut rank = 0usize;
    let mut queries = 0u64;
    let mut attempts = 0usize;
    let max_attempts = features * 64;

    while rank < features && attempts < max_attempts {
        if query_budget != 0 && queries >= query_budget {
            break;
        }
        attempts += 1;

        let q = random_target_point(public, rng);
        let hit = query_hit(public, oracle, q);
        queries += 1;
        if hit.found_lz >= k {
            return (None, Some(hit), queries);
        }

        let z = q >> vars;
        let row = slice_feature_row(z, vars, &pairs, features);
        let rhs = top_k_pack(hit.x, public.n, k);
        if insert_independent_bit_row(row, rhs, &mut basis_rows, &mut basis_rhs) {
            rank += 1;
        }
    }

    if rank != features {
        return (None, None, queries);
    }

    let Some(coeff_by_feature) = coefficients_from_full_basis(basis_rows, basis_rhs, features)
    else {
        return (None, None, queries);
    };

    (
        Some(build_slice_surrogate_model(
            vars,
            k,
            coeff_by_feature,
            pairs,
        )),
        None,
        queries,
    )
}

fn slice_surrogate_predict(model: &SliceSurrogateModel, z: u128) -> u128 {
    let z = low_bits(z, model.vars);
    let mut out = model.constant;

    for bit in 0..model.vars {
        if ((z >> bit) & 1) != 0 {
            out ^= model.linear_coeffs[bit];
        }
    }

    for (idx, &(a, b)) in model.pairs.iter().enumerate() {
        if ((z >> a) & 1) != 0 && ((z >> b) & 1) != 0 {
            out ^= model.pair_coeffs[idx];
        }
    }

    low_bits(out, model.k)
}

fn slice_surrogate_columns(model: &SliceSurrogateModel, z: u128) -> Vec<u128> {
    let z = low_bits(z, model.vars);
    let mut columns = model.linear_coeffs.clone();

    for (idx, &(a, b)) in model.pairs.iter().enumerate() {
        let coeff = model.pair_coeffs[idx];
        if ((z >> b) & 1) != 0 {
            columns[a] ^= coeff;
        }
        if ((z >> a) & 1) != 0 {
            columns[b] ^= coeff;
        }
    }

    for column in &mut columns {
        *column = low_bits(*column, model.k);
    }

    columns
}

fn slice_surrogate_optimized_z(model: &SliceSurrogateModel, rng: &mut fastrand::Rng) -> u128 {
    let mut z = random_bits(rng, model.vars);
    let mut residual = slice_surrogate_predict(model, z);
    let mut best_z = z;
    let mut best_weight = residual.count_ones();
    let mut order: Vec<usize> = (0..model.vars).collect();
    let max_steps = model.vars.clamp(16, 32);
    let max_perturb_bits = 8.min(model.vars).max(1);

    for step in 0..max_steps {
        if residual == 0 {
            return z;
        }

        let current_weight = residual.count_ones();
        if current_weight < best_weight {
            best_weight = current_weight;
            best_z = z;
        }

        let columns = slice_surrogate_columns(model, z);
        let mut chosen_z = z;
        let mut chosen_residual = residual;
        let mut chosen_weight = current_weight;

        if step % 4 == 0 {
            shuffle_order(&mut order, rng);
            if let Some(delta) = solve_gf2_columns(&columns, residual, model.k, &order) {
                let mut deltas = Vec::with_capacity(5);
                deltas.push(delta);
                if delta.count_ones() > 4 {
                    deltas.push(random_subdelta(delta, 4, rng));
                }
                if delta.count_ones() > 8 {
                    deltas.push(random_subdelta(delta, 8, rng));
                }
                if delta.count_ones() > 16 {
                    deltas.push(random_subdelta(delta, 16, rng));
                }
                if delta.count_ones() > 32 {
                    deltas.push(random_subdelta(delta, 32, rng));
                }

                for delta in deltas {
                    if delta == 0 {
                        continue;
                    }
                    let candidate_z = low_bits(z ^ delta, model.vars);
                    let candidate_residual = slice_surrogate_predict(model, candidate_z);
                    let candidate_weight = candidate_residual.count_ones();
                    if candidate_weight < chosen_weight {
                        chosen_z = candidate_z;
                        chosen_residual = candidate_residual;
                        chosen_weight = candidate_weight;
                    }
                }
            }
        }

        for (bit, &column) in columns.iter().enumerate() {
            let candidate_residual = residual ^ column;
            let candidate_weight = candidate_residual.count_ones();
            if candidate_weight < chosen_weight {
                chosen_z = z ^ (1u128 << bit);
                chosen_residual = candidate_residual;
                chosen_weight = candidate_weight;
            }
        }

        if chosen_weight < current_weight {
            z = low_bits(chosen_z, model.vars);
            residual = chosen_residual;
            continue;
        }

        let flips = 1 + rng.usize(..max_perturb_bits);
        for _ in 0..flips {
            z ^= 1u128 << rng.usize(..model.vars);
        }
        z = low_bits(z, model.vars);
        residual = slice_surrogate_predict(model, z);
    }

    best_z
}

fn slice_surrogate_candidate_hit(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    model: &SliceSurrogateModel,
    candidate_seed: u64,
) -> CandidateHit {
    let mut rng = fastrand::Rng::with_seed(candidate_seed);
    let z = slice_surrogate_optimized_z(model, &mut rng);
    let q = make_target_point(z, public.y, public.n / 2, public.n);
    query_hit(public, oracle, q)
}

fn find_preimage_slice_surrogate(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    finder_seed: u64,
    max_queries: u64,
) -> FinderResult {
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;
    let mut model_index = 0u64;

    loop {
        if max_queries != 0 && queries >= max_queries {
            return failed_result(queries, start_wall, start_cpu);
        }

        let remaining = if max_queries == 0 {
            0
        } else {
            max_queries - queries
        };
        let mut rng = fastrand::Rng::with_seed(splitmix64(finder_seed ^ model_index));
        let (model, hit, spent) =
            learn_slice_surrogate_model(public, oracle, k, &mut rng, remaining);
        queries += spent;
        if let Some(hit) = hit {
            return successful_result(hit, queries, start_wall, start_cpu);
        }

        let Some(model) = model else {
            model_index += 1;
            continue;
        };

        let candidate_checks = if max_queries == 0 {
            4096
        } else {
            (max_queries - queries).min(4096)
        };

        for idx in 0..candidate_checks {
            if max_queries != 0 && queries >= max_queries {
                return failed_result(queries, start_wall, start_cpu);
            }

            let seed = splitmix64(finder_seed ^ (model_index << 32) ^ idx);
            let hit = slice_surrogate_candidate_hit(public, oracle, &model, seed);
            queries += 1;
            if hit.found_lz >= k {
                return successful_result(hit, queries, start_wall, start_cpu);
            }
        }

        model_index += 1;
    }
}

fn find_preimage_parallel_slice_surrogate(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    finder_seed: u64,
    max_queries: u64,
    batch_size: u64,
) -> FinderResult {
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;
    let mut model_index = 0u64;
    let candidate_batch = batch_size.clamp(512, 4096);

    loop {
        if max_queries != 0 && queries >= max_queries {
            return failed_result(queries, start_wall, start_cpu);
        }

        let remaining = if max_queries == 0 {
            0
        } else {
            max_queries - queries
        };
        let mut rng = fastrand::Rng::with_seed(splitmix64(finder_seed ^ model_index));
        let (model, hit, spent) =
            learn_slice_surrogate_model(public, oracle, k, &mut rng, remaining);
        queries += spent;
        if let Some(hit) = hit {
            return successful_result(hit, queries, start_wall, start_cpu);
        }

        let Some(model) = model else {
            model_index += 1;
            continue;
        };

        let remaining = if max_queries == 0 {
            candidate_batch
        } else {
            max_queries.saturating_sub(queries).min(candidate_batch)
        };
        if remaining == 0 {
            return failed_result(queries, start_wall, start_cpu);
        }

        let attempted = AtomicU64::new(0);
        let hit = (0..remaining).into_par_iter().find_map_any(|idx| {
            let seed = splitmix64(finder_seed ^ (model_index << 32) ^ idx);
            let hit = slice_surrogate_candidate_hit(public, oracle, &model, seed);
            attempted.fetch_add(1, Ordering::Relaxed);
            (hit.found_lz >= k).then_some(hit)
        });

        queries += attempted.load(Ordering::Relaxed);
        if let Some(hit) = hit {
            return successful_result(hit, queries, start_wall, start_cpu);
        }

        model_index += 1;
    }
}

fn feature_row_affine(z: u128, vars: usize) -> u128 {
    low_bits(z, vars) | (1u128 << vars)
}

fn top_k_pack(x: u128, n: usize, k: usize) -> u128 {
    high_bits_value(x, n, k)
}

fn insert_independent_row(rows: &mut Vec<u128>, row: u128, features: usize) -> bool {
    let mut r = row;
    for &basis_row in rows.iter() {
        let pivot = basis_row.trailing_zeros() as usize;
        if ((r >> pivot) & 1) != 0 {
            r ^= basis_row;
        }
    }

    if r == 0 {
        return false;
    }

    let pivot = r.trailing_zeros() as usize;
    let mut insert_at = rows.len();
    for (idx, &basis_row) in rows.iter().enumerate() {
        if basis_row.trailing_zeros() as usize > pivot {
            insert_at = idx;
            break;
        }
    }
    rows.insert(insert_at, r & ((1u128 << features) - 1));
    true
}

fn affine_coefficients(
    mut rows: Vec<u128>,
    mut rhs_rows: Vec<u128>,
    features: usize,
) -> Option<Vec<u128>> {
    if rows.len() != features || rhs_rows.len() != features {
        return None;
    }

    for col in 0..features {
        let pivot = (col..features).find(|&row| ((rows[row] >> col) & 1) != 0)?;
        rows.swap(col, pivot);
        rhs_rows.swap(col, pivot);

        for row in 0..features {
            if row != col && ((rows[row] >> col) & 1) != 0 {
                rows[row] ^= rows[col];
                rhs_rows[row] ^= rhs_rows[col];
            }
        }
    }

    Some(rhs_rows)
}

fn random_solution_for_affine_predictions(
    coeff_by_feature: &[u128],
    vars: usize,
    k: usize,
    rng: &mut fastrand::Rng,
) -> Option<u128> {
    let mut rows = Vec::<u128>::new();
    let mut rhs = Vec::<bool>::new();

    for bit in 0..k {
        let mut row = 0u128;
        for var in 0..vars {
            if ((coeff_by_feature[var] >> bit) & 1) != 0 {
                row |= 1u128 << var;
            }
        }
        let b = ((coeff_by_feature[vars] >> bit) & 1) != 0;

        if row == 0 {
            if b {
                return None;
            }
            continue;
        }

        rows.push(row);
        rhs.push(b);
    }

    let mut pivot_cols = Vec::new();
    let mut row_idx = 0usize;
    for col in 0..vars {
        let Some(pivot) = (row_idx..rows.len()).find(|&r| ((rows[r] >> col) & 1) != 0) else {
            continue;
        };
        rows.swap(row_idx, pivot);
        rhs.swap(row_idx, pivot);

        for r in 0..rows.len() {
            if r != row_idx && ((rows[r] >> col) & 1) != 0 {
                rows[r] ^= rows[row_idx];
                rhs[r] ^= rhs[row_idx];
            }
        }

        pivot_cols.push(col);
        row_idx += 1;
        if row_idx == rows.len() {
            break;
        }
    }

    for r in row_idx..rows.len() {
        if rows[r] == 0 && rhs[r] {
            return None;
        }
    }

    let mut z = random_bits(rng, vars);
    for (r, &pivot_col) in pivot_cols.iter().enumerate() {
        let without_pivot = rows[r] & !(1u128 << pivot_col);
        let parity = ((without_pivot & z).count_ones() & 1) != 0;
        let pivot_value = rhs[r] ^ parity;
        if pivot_value {
            z |= 1u128 << pivot_col;
        } else {
            z &= !(1u128 << pivot_col);
        }
    }

    Some(low_bits(z, vars))
}

fn find_preimage_formula_linear(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    finder_seed: u64,
    max_queries: u64,
) -> FinderResult {
    let mut rng = fastrand::Rng::with_seed(finder_seed);
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;
    let vars = public.n / 2;
    let features = vars + 1;
    let candidate_checks_per_model = 4096u64;

    loop {
        if max_queries != 0 && queries >= max_queries {
            return failed_result(queries, start_wall, start_cpu);
        }

        let mut independent_rows = Vec::<u128>::new();
        let mut sample_rows = Vec::<u128>::new();
        let mut rhs_rows = Vec::<u128>::new();
        let mut attempts = 0usize;

        while sample_rows.len() < features && attempts < features * 16 {
            if max_queries != 0 && queries >= max_queries {
                return failed_result(queries, start_wall, start_cpu);
            }

            attempts += 1;
            let q = random_target_point(public, &mut rng);
            let hit = query_hit(public, oracle, q);
            queries += 1;
            if hit.found_lz >= k {
                return successful_result(hit, queries, start_wall, start_cpu);
            }

            let z = q >> vars;
            let row = feature_row_affine(z, vars);
            if insert_independent_row(&mut independent_rows, row, features) {
                sample_rows.push(row);
                rhs_rows.push(top_k_pack(hit.x, public.n, k));
            }
        }

        let Some(coeff_by_feature) = affine_coefficients(sample_rows, rhs_rows, features) else {
            continue;
        };

        for _ in 0..candidate_checks_per_model {
            if max_queries != 0 && queries >= max_queries {
                return failed_result(queries, start_wall, start_cpu);
            }

            let Some(z) =
                random_solution_for_affine_predictions(&coeff_by_feature, vars, k, &mut rng)
            else {
                break;
            };
            let q = make_target_point(z, public.y, vars, public.n);
            let hit = query_hit(public, oracle, q);
            queries += 1;

            if hit.found_lz >= k {
                return successful_result(hit, queries, start_wall, start_cpu);
            }
        }
    }
}

fn inverse_target_query(public: &PublicChallenge, finder_seed: u64, index: u64) -> u128 {
    let z = indexed_random_bits(finder_seed, index, public.n / 2);
    make_target_point(z, public.y, public.n / 2, public.n)
}

fn find_preimage_parallel_inverse_target(
    public: &PublicChallenge,
    oracle: &Oracle<'_>,
    k: usize,
    finder_seed: u64,
    max_queries: u64,
    batch_size: u64,
) -> FinderResult {
    let start_wall = Instant::now();
    let start_cpu = cpu_seconds();
    let mut queries = 0u64;
    let mut next_index = 0u64;
    let batch_size = batch_size.max(1);

    loop {
        let remaining = if max_queries == 0 {
            batch_size
        } else {
            max_queries.saturating_sub(queries).min(batch_size)
        };

        if remaining == 0 {
            return FinderResult {
                success: false,
                queries,
                wall_time_s: start_wall.elapsed().as_secs_f64(),
                cpu_time_s: cpu_seconds() - start_cpu,
                found_lz: 0,
                x: 0,
                queried_point: 0,
                forward_at_query: 0,
                inverse_at_query: 0,
            };
        }

        let attempted = AtomicU64::new(0);
        let hit = (0..remaining).into_par_iter().find_map_any(|offset| {
            let q = inverse_target_query(public, finder_seed, next_index + offset);
            let reply = oracle.query(q);
            attempted.fetch_add(1, Ordering::Relaxed);

            is_inverse_solution(public, q, reply.inverse, k).then(|| CandidateHit {
                found_lz: leading_zeros_n(reply.inverse, public.n),
                x: reply.inverse,
                queried_point: q,
                forward_at_query: reply.forward,
                inverse_at_query: reply.inverse,
            })
        });

        queries += attempted.load(Ordering::Relaxed);
        next_index += remaining;

        if let Some(hit) = hit {
            return FinderResult {
                success: true,
                queries,
                wall_time_s: start_wall.elapsed().as_secs_f64(),
                cpu_time_s: cpu_seconds() - start_cpu,
                found_lz: hit.found_lz,
                x: hit.x,
                queried_point: hit.queried_point,
                forward_at_query: hit.forward_at_query,
                inverse_at_query: hit.inverse_at_query,
            };
        }
    }
}

fn cpu_seconds() -> f64 {
    unsafe {
        let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0 {
            return 0.0;
        }
        let usage = usage.assume_init();
        let user = usage.ru_utime.tv_sec as f64 + (usage.ru_utime.tv_usec as f64 / 1_000_000.0);
        let sys = usage.ru_stime.tv_sec as f64 + (usage.ru_stime.tv_usec as f64 / 1_000_000.0);
        user + sys
    }
}

fn format_hex_n(x: u128, n: usize) -> String {
    let nybbles = n.div_ceil(4);
    format!("0x{:0width$x}", mask_to_n(x, n), width = nybbles)
}

fn expected_queries_baseline(policy: Policy, n: usize, k: usize) -> f64 {
    match policy {
        Policy::ForwardJacobian | Policy::ForwardZero => 2f64.powi((n / 2) as i32),
        _ => 2f64.powi(k as i32),
    }
}

fn write_metadata(path: &Path, challenger: &Challenger) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "n\t{}", challenger.public.n)?;
    writeln!(f, "m\t{}", challenger.public.m)?;
    writeln!(
        f,
        "y\t{}",
        format_hex_n(challenger.public.y, challenger.public.n / 2)
    )?;
    writeln!(
        f,
        "note\tfinder routines receive only n,m,k,y and oracle replies; they do not receive the circuit"
    )?;
    Ok(())
}

fn write_private_circuit(path: &Path, challenger: &Challenger) -> std::io::Result<()> {
    let mut f = BufWriter::new(File::create(path)?);
    writeln!(f, "{}", challenger.circuit.repr())?;
    Ok(())
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if args.threads != 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .expect("failed to initialize Rayon thread pool");
    }
    fs::create_dir_all(&args.out_dir)?;

    let summary_path = args.out_dir.join("oracle_preimage_timing.tsv");
    let mut summary = BufWriter::new(File::create(&summary_path)?);
    writeln!(
        summary,
        "trial\tpolicy\tn\tm\tk\ty\tparallel\tthreads\tbatch_size\tsuccess\tqueries\texpected_queries\twall_time_s\tcpu_time_s\tqueries_per_wall_s\tfound_lz\tx\tqueried_point\tforward_at_query\tinverse_at_query\tverified"
    )?;

    for trial in 0..args.trials {
        let circuit_seed = args.seed ^ 0x9e37_79b9_7f4a_7c15u64 ^ (trial as u64);
        let y_seed = args.seed ^ 0xbf58_476d_1ce4_e5b9u64 ^ ((trial as u64) << 17);
        let challenger = Challenger::new(args.wires, args.gates, circuit_seed, y_seed);

        let trial_dir = args.out_dir.join(format!("trial_{trial:03}"));
        fs::create_dir_all(&trial_dir)?;
        write_metadata(&trial_dir.join("challenge_public.tsv"), &challenger)?;
        if args.save_private_circuit {
            write_private_circuit(
                &trial_dir.join("challenge_private_circuit.txt"),
                &challenger,
            )?;
        }

        for &k in &args.ks {
            let finder_seed =
                args.seed ^ 0x94d0_49bb_1331_11ebu64 ^ ((trial as u64) << 32) ^ (k as u64);
            let result = if args.policy == Policy::FormulaLinear {
                find_preimage_formula_linear(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    finder_seed,
                    args.max_queries,
                )
            } else if args.parallel && args.policy == Policy::ForwardJacobian {
                find_preimage_parallel_forward_jacobian(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    finder_seed,
                    args.max_queries,
                    args.batch_size,
                )
            } else if args.policy == Policy::ForwardJacobian {
                find_preimage_forward_jacobian(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    finder_seed,
                    args.max_queries,
                )
            } else if args.parallel && args.policy == Policy::InverseJacobian {
                find_preimage_parallel_inverse_jacobian(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    finder_seed,
                    args.max_queries,
                    args.batch_size,
                )
            } else if args.policy == Policy::InverseJacobian {
                find_preimage_inverse_jacobian(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    finder_seed,
                    args.max_queries,
                )
            } else if args.parallel && args.policy == Policy::SliceSurrogate {
                find_preimage_parallel_slice_surrogate(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    finder_seed,
                    args.max_queries,
                    args.batch_size,
                )
            } else if args.policy == Policy::SliceSurrogate {
                find_preimage_slice_surrogate(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    finder_seed,
                    args.max_queries,
                )
            } else if args.parallel && args.policy == Policy::LocalWalk {
                find_preimage_parallel_local_walk(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    finder_seed,
                    args.max_queries,
                    args.batch_size,
                )
            } else if args.policy == Policy::LocalWalk {
                find_preimage_local_walk(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    finder_seed,
                    args.max_queries,
                )
            } else if args.parallel && args.policy == Policy::InverseTarget {
                find_preimage_parallel_inverse_target(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    finder_seed,
                    args.max_queries,
                    args.batch_size,
                )
            } else {
                find_preimage(
                    &challenger.public,
                    &challenger.oracle(),
                    k,
                    args.policy,
                    finder_seed,
                    args.max_queries,
                )
            };
            let verified = result.success && challenger.is_solution(result.x, k);
            let qps = if result.wall_time_s > 0.0 {
                result.queries as f64 / result.wall_time_s
            } else {
                0.0
            };

            writeln!(
                summary,
                "{trial}\t{:?}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.9}\t{:.9}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}",
                args.policy,
                args.wires,
                args.gates,
                k,
                format_hex_n(challenger.public.y, args.wires / 2),
                args.parallel,
                args.threads,
                args.batch_size,
                result.success,
                result.queries,
                expected_queries_baseline(args.policy, args.wires, k),
                result.wall_time_s,
                result.cpu_time_s,
                qps,
                result.found_lz,
                format_hex_n(result.x, args.wires),
                format_hex_n(result.queried_point, args.wires),
                format_hex_n(result.forward_at_query, args.wires),
                format_hex_n(result.inverse_at_query, args.wires),
                verified
            )?;
            summary.flush()?;

            println!(
                "trial={trial} policy={:?} k={k} success={} queries={} wall={:.6}s found_lz={} verified={verified}",
                args.policy, result.success, result.queries, result.wall_time_s, result.found_lz,
            );
        }
    }

    println!("summary\t{}", summary_path.display());
    Ok(())
}
