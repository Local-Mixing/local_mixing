//! Exhaustive preimage search over N free input bits — the honest baseline the
//! SAT attack has to beat. Enumerates all 2^N candidates for the low N bits of X
//! (remaining bits pinned to a known X*), evaluating 64 candidates per pass with
//! the bit-sliced evaluator, and reports the first X whose payload block equals Y.
//!
//!   preimage_brute <circuit.mpmct1> <Yhex> --free-bits N --xstar <Xhex> [--threads T]
use local_mixing::postmix::format::read_mpmct;
use local_mixing::postmix::xgate::{eval_lanes, XGate};
use rayon::prelude::*;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

fn payloads(
    gates: &[XGate],
    num_wires: usize,
    xs: &[u128],
    xw0: usize,
    blk: usize,
    yw0: usize,
) -> Vec<u128> {
    let mut state = vec![0u64; num_wires];
    for (lane, &x) in xs.iter().enumerate() {
        for i in 0..blk {
            if (x >> i) & 1 == 1 {
                state[xw0 + i] |= 1u64 << lane;
            }
        }
    }
    eval_lanes(gates, &mut state);
    (0..xs.len())
        .map(|lane| {
            let mut y = 0u128;
            for j in 0..blk {
                if (state[yw0 + j] >> lane) & 1 == 1 {
                    y |= 1u128 << j;
                }
            }
            y
        })
        .collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let path = a.next().expect("usage: preimage_brute <circuit> <Yhex> --free-bits N --xstar X");
    let y = u128::from_str_radix(
        a.next().expect("need Y").trim_start_matches("0x"),
        16,
    )
    .expect("bad Y");
    let (mut free_bits, mut xstar, mut threads) = (16usize, 0u128, 0usize);
    let (mut xw0, mut yw0, mut blk) = (0usize, 128usize, 128usize);
    let mut it = a.peekable();
    while let Some(t) = it.next() {
        match t.as_str() {
            "--free-bits" => free_bits = it.next().unwrap().parse().unwrap(),
            "--xstar" => {
                xstar = u128::from_str_radix(it.next().unwrap().trim_start_matches("0x"), 16).unwrap()
            }
            "--threads" => threads = it.next().unwrap().parse().unwrap(),
            "--xw0" => xw0 = it.next().unwrap().parse().unwrap(),
            "--yw0" => yw0 = it.next().unwrap().parse().unwrap(),
            "--blk" => blk = it.next().unwrap().parse().unwrap(),
            o => panic!("unknown arg {o}"),
        }
    }
    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }

    let t0 = Instant::now();
    let (gates, num_wires) = read_mpmct(&path).expect("read mpmct");
    eprintln!("loaded {} gates / {num_wires} wires; searching 2^{free_bits} candidates", gates.len());

    let total = 1u64 << free_bits;
    let nbatch = total.div_ceil(64);
    let hi_mask = !((1u128 << free_bits) - 1);
    let base = xstar & hi_mask; // pinned high bits
    let found = AtomicBool::new(false);
    let answer = AtomicU64::new(0);
    let done = AtomicU64::new(0);

    (0..nbatch).into_par_iter().for_each(|b| {
        if found.load(Ordering::Relaxed) {
            return;
        }
        let lo = b * 64;
        let take = ((total - lo).min(64)) as usize;
        let xs: Vec<u128> = (0..take).map(|l| base | (lo + l as u64) as u128).collect();
        let ys = payloads(&gates, num_wires, &xs, xw0, blk, yw0);
        for (l, &got) in ys.iter().enumerate() {
            if got == y {
                answer.store(lo + l as u64, Ordering::SeqCst);
                found.store(true, Ordering::SeqCst);
            }
        }
        let d = done.fetch_add(take as u64, Ordering::Relaxed) + take as u64;
        if d % (1 << 20) < 64 {
            eprintln!("  {d}/{total} at {:.1}s", t0.elapsed().as_secs_f64());
        }
    });

    let el = t0.elapsed().as_secs_f64();
    if found.load(Ordering::SeqCst) {
        let x = base | answer.load(Ordering::SeqCst) as u128;
        println!("FOUND X = {x:032x}");
        println!("  matches X* : {}", if x == xstar { "YES" } else { "no (collision?)" });
    } else {
        println!("NOT FOUND in 2^{free_bits} candidates");
    }
    let searched = done.load(Ordering::Relaxed).max(1);
    println!("  searched {searched} candidates in {el:.2}s  ({:.0} evals/sec)", searched as f64 / el);
}
