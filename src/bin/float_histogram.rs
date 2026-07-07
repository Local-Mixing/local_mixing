// Throwaway analysis: measure the NET per-gate displacement of one real float pass
// (float_all_gates, the exact production code) on a circuit file. Tags carry the original
// index through the float; net displacement = final position - original position.
// Usage: float_histogram <circuit.txt>
use local_mixing::circuit::CircuitSeq;
use local_mixing::replace::replace::{float_all_gates, Tag};
use std::env;
use std::fs;

fn bucket(mag: u64) -> usize {
    // 0:1, 1:2, 2:3-4, 3:5-8, 4:9-16, 5:17-32, 6:33-64, 7:65-128, 8:129-256,
    // 9:257-512, 10:513-1024, 11:1025+
    match mag {
        1 => 0,
        2 => 1,
        3..=4 => 2,
        5..=8 => 3,
        9..=16 => 4,
        17..=32 => 5,
        33..=64 => 6,
        65..=128 => 7,
        129..=256 => 8,
        257..=512 => 9,
        513..=1024 => 10,
        _ => 11,
    }
}

const BUCKET_LABELS: [&str; 12] = [
    "1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129-256", "257-512",
    "513-1024", ">1024",
];

fn main() {
    let path = env::args().nth(1).expect("usage: float_histogram <circuit.txt> [passes]");
    let passes: usize = env::args().nth(2).and_then(|a| a.parse().ok()).unwrap_or(1);
    let s = fs::read_to_string(&path).expect("read circuit");
    let mut c = CircuitSeq::from_string(&s);
    let n = c.gates.len();
    eprintln!("loaded {} gates from {}", n, path);
    let mut rng = rand::rng();

    for pass in 1..=passes {
        run_pass(&mut c, n, pass, &mut rng);
    }
}

fn run_pass(c: &mut CircuitSeq, n: usize, pass: usize, rng: &mut impl rand::Rng) {
    let mut tags: Vec<Tag> = (0..n).map(|j| Tag(j as u64)).collect();
    let (pass_moves, pass_disp) = float_all_gates(&mut c.gates, &mut tags, rng);

    let mut left = [0u64; 12];
    let mut right = [0u64; 12];
    let mut zero = 0u64;
    let mut mags: Vec<u64> = Vec::with_capacity(n);
    let (mut max_l, mut max_r) = (0i64, 0i64);
    let mut total_mag: u128 = 0;
    for (newpos, t) in tags.iter().enumerate() {
        let old = t.0 as i64;
        let d = newpos as i64 - old;
        let mag = d.unsigned_abs();
        mags.push(mag);
        total_mag += mag as u128;
        if d == 0 {
            zero += 1;
        } else if d < 0 {
            left[bucket(mag)] += 1;
            max_l = max_l.min(d);
        } else {
            right[bucket(mag)] += 1;
            max_r = max_r.max(d);
        }
    }
    mags.sort_unstable();
    let pct = |p: f64| mags[((n as f64 * p) as usize).min(n - 1)];
    let movers = n as u64 - zero;

    println!("SUMMARY pass={} gates={} in_pass_moves={} in_pass_summed_steps={}", pass, n, pass_moves, pass_disp);
    println!(
        "SUMMARY pass={} net_moved={} ({:.1}%) net_zero={} mean_abs={:.2} median_abs={} p90_abs={} p99_abs={} max_left={} max_right={}",
        pass,
        movers,
        100.0 * movers as f64 / n as f64,
        zero,
        total_mag as f64 / n as f64,
        pct(0.5),
        pct(0.9),
        pct(0.99),
        max_l,
        max_r
    );
    println!("HIST,pass,dir,steps,count");
    println!("HIST,{},zero,0,{}", pass, zero);
    for b in 0..12 {
        println!("HIST,{},left,{},{}", pass, BUCKET_LABELS[b], left[b]);
    }
    for b in 0..12 {
        println!("HIST,{},right,{},{}", pass, BUCKET_LABELS[b], right[b]);
    }
}
