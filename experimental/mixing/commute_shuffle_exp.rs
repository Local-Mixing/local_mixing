// Experiment driver: random circuit C -> uniform commuting reshuffle C'.
//
// Generates a random circuit (--kind g57 = complemented-monomial 2-control
// gates a ^= x OR NOT y; --kind tof = mixed-polarity k-control Toffolis) and
// samples C' UNIFORMLY from the commutation class of C: the linear extensions
// of the partial order in which g_i < g_j iff i < j and
// XGate::collides(g_i, g_j). A linear order respects the transitive closure
// iff it respects every direct colliding pair, so legality checks stay local.
//
// Sampler: Gibbs remove-reinsert. Pick a gate uniformly, find its nearest
// colliding predecessor/successor in the current order, and relocate it
// uniformly among the slots strictly between them. That resamples the gate's
// exact conditional distribution given the relative order of the rest, so the
// stationary law is uniform over the class; the relocation is global, so the
// chain mixes in O(m log m)-ish moves instead of the adjacent-transposition
// chain's O(m^3 log m).
//
// Writes C and C' as mpmct1 and verifies functional equality on sampled
// inputs before exiting.
use clap::Parser;
use local_mixing::engine::format::write_mpmct;
use local_mixing::circuit::xgate::XGate;
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[command(name = "commute_shuffle_exp")]
struct Args {
    #[arg(long, default_value_t = 128)]
    wires: u16,
    #[arg(long, default_value_t = 6000)]
    gates: usize,
    /// g57 | tof
    #[arg(long)]
    kind: String,
    /// Control count for --kind tof
    #[arg(long, default_value_t = 2)]
    ctrls: usize,
    /// Seed for the circuit draw (same seed = same C)
    #[arg(long, default_value_t = 1)]
    seed: u64,
    /// Seed for the reshuffle chain (vary it to get independent C' of one C)
    #[arg(long, default_value_t = 0)]
    shuffle_seed: u64,
    /// Gibbs relocation moves (default ~100x m ln m for m=6000)
    #[arg(long, default_value_t = 5_000_000)]
    moves: u64,
    /// Batches of 64 random inputs for the equivalence check
    #[arg(long, default_value_t = 8)]
    check_batches: usize,
    #[arg(long)]
    out_c: String,
    #[arg(long)]
    out_cp: String,
}

fn distinct_wires(n: u16, k: usize, exclude: Option<u16>, rng: &mut StdRng) -> Vec<u16> {
    let mut out: Vec<u16> = Vec::with_capacity(k);
    while out.len() < k {
        let w = rng.random_range(0..n);
        if Some(w) != exclude && !out.contains(&w) {
            out.push(w);
        }
    }
    out
}

fn random_circuit(args: &Args, rng: &mut StdRng) -> Vec<XGate> {
    (0..args.gates)
        .map(|_| match args.kind.as_str() {
            "g57" => {
                let w = distinct_wires(args.wires, 3, None, rng);
                XGate::from_g57([w[0], w[1], w[2]])
            }
            "tof" => {
                let t = rng.random_range(0..args.wires);
                let cw = distinct_wires(args.wires, args.ctrls, Some(t), rng);
                let lits: Vec<(u16, bool)> =
                    cw.into_iter().map(|w| (w, rng.random::<bool>())).collect();
                XGate::conj(t, lits).expect("distinct control wires cannot contradict")
            }
            other => panic!("unknown --kind {other} (g57 | tof)"),
        })
        .collect()
}

fn spearman(x: &[f64], y: &[f64]) -> f64 {
    let rank = |v: &[f64]| -> Vec<f64> {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
        let mut r = vec![0f64; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (rx, ry) = (rank(x), rank(y));
    let n = x.len() as f64;
    let mx = rx.iter().sum::<f64>() / n;
    let my = ry.iter().sum::<f64>() / n;
    let mut num = 0f64;
    let mut dx = 0f64;
    let mut dy = 0f64;
    for i in 0..x.len() {
        num += (rx[i] - mx) * (ry[i] - my);
        dx += (rx[i] - mx).powi(2);
        dy += (ry[i] - my).powi(2);
    }
    num / (dx.sqrt() * dy.sqrt())
}

fn main() {
    let args = Args::parse();
    let mut rng = StdRng::seed_from_u64(args.seed);
    let gates = random_circuit(&args, &mut rng);
    let m = gates.len();
    // Everything downstream of the circuit draw (chain, checks) runs on its
    // own stream so one C can be reshuffled independently many times.
    let mut rng = StdRng::seed_from_u64(args.shuffle_seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(args.seed));

    // Collision density estimate over random unordered pairs.
    let mut coll = 0u64;
    let pairs = 200_000u64;
    for _ in 0..pairs {
        let a = rng.random_range(0..m);
        let b = rng.random_range(0..m);
        if a != b && XGate::collides(&gates[a], &gates[b]) {
            coll += 1;
        }
    }

    // Gibbs chain over the gate order.
    let mut order: Vec<u32> = (0..m as u32).collect();
    let mut accepted = 0u64;
    let mut span_sum = 0u64;
    for _ in 0..args.moves {
        let p = rng.random_range(0..m);
        let g = &gates[order[p] as usize];
        let mut lo = p;
        while lo > 0 && !XGate::collides(g, &gates[order[lo - 1] as usize]) {
            lo -= 1;
        }
        let mut hi = p;
        while hi + 1 < m && !XGate::collides(g, &gates[order[hi + 1] as usize]) {
            hi += 1;
        }
        span_sum += (hi - lo) as u64;
        let q = rng.random_range(lo..=hi);
        if q < p {
            order[q..=p].rotate_right(1);
            accepted += 1;
        } else if q > p {
            order[p..=q].rotate_left(1);
            accepted += 1;
        }
    }

    let shuffled: Vec<XGate> = order.iter().map(|&i| gates[i as usize].clone()).collect();

    // Mixing diagnostics: displacement of each original gate + rank correlation.
    let mut newpos = vec![0f64; m];
    for (pos, &gi) in order.iter().enumerate() {
        newpos[gi as usize] = pos as f64;
    }
    let orig: Vec<f64> = (0..m).map(|i| i as f64).collect();
    let mean_disp =
        orig.iter().zip(&newpos).map(|(a, b)| (a - b).abs()).sum::<f64>() / m as f64;
    let rho = spearman(&orig, &newpos);

    // Functional equivalence on random inputs (the shuffle is a product of
    // adjacent commuting swaps, but assert anyway).
    let nw = args.wires as usize;
    for b in 0..args.check_batches {
        let init: Vec<u64> = (0..nw).map(|_| rng.random::<u64>()).collect();
        let mut s1 = init.clone();
        let mut s2 = init;
        for g in &gates {
            g.apply_lanes(&mut s1);
        }
        for g in &shuffled {
            g.apply_lanes(&mut s2);
        }
        assert_eq!(s1, s2, "shuffle changed the function (batch {b})");
    }

    write_mpmct(&args.out_c, &gates, nw).expect("write C");
    write_mpmct(&args.out_cp, &shuffled, nw).expect("write C'");
    println!(
        "[shuffle-exp] kind={} ctrls={} m={} wires={} seed={} | collide-density {:.4} | \
         moves {} accept {:.3} mean-span {:.1} | mean|disp| {:.1} rho(orig,new) {:.4} | \
         equivalence OK ({} samples)",
        args.kind,
        if args.kind == "tof" { args.ctrls } else { 2 },
        m,
        nw,
        args.seed,
        coll as f64 / pairs as f64,
        args.moves,
        accepted as f64 / args.moves as f64,
        span_sum as f64 / args.moves as f64,
        mean_disp,
        rho,
        64 * args.check_batches
    );
}
