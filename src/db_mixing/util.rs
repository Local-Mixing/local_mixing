use std::sync::atomic::{AtomicBool, Ordering};

use rand::{Rng, prelude::SliceRandom};

// Help with early stops without losing all data
pub static SHOULD_DUMP: AtomicBool = AtomicBool::new(false);
use signal_hook::consts::{SIGINT, SIGTERM};
use signal_hook::iterator::Signals;
use std::thread;

pub fn install_kill_handler() {
    let mut signals = Signals::new([SIGINT, SIGTERM]).expect("signals");

    thread::spawn(move || {
        // Block until the first SIGINT/SIGTERM, then flag a dump.
        if signals.forever().next().is_some() {
            eprintln!("Received termination signal, dumping acc...");
            SHOULD_DUMP.store(true, Ordering::SeqCst);
        }
    });
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Pair Replacement Methods
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

pub fn split_into_random_chunk_ranges(
    len: usize,
    k: usize,
    rng: &mut impl Rng,
) -> Vec<(usize, usize)> {
    if k == 1 {
        return vec![(0, len)];
    }

    let min_size = 100;
    assert!(k * min_size <= len);

    let slack = len - k * min_size;

    let mut cuts: Vec<usize> = (0..slack).collect();
    cuts.shuffle(rng);
    cuts.truncate(k - 1);
    cuts.sort_unstable();

    let mut sizes = Vec::with_capacity(k);
    let mut prev = 0;

    for &c in &cuts {
        sizes.push(c - prev + min_size);
        prev = c;
    }
    sizes.push(slack - prev + min_size);

    let mut ranges = Vec::with_capacity(k);
    let mut idx = 0;
    for size in sizes {
        let end = idx + size;
        ranges.push((idx, end));
        idx = end;
    }

    ranges
}
