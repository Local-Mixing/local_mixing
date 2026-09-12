//! Equivalence-class census over the FROZEN REGULAR store.
//!
//! Sizes the "enumerate every identity from the frozen db" pipeline before it
//! is built. Identities come from ordered pairs of distinct circuits within one
//! equivalence class (`a ++ reverse(b)`), so the cost is driven by
//! `sum |class| * (|class| - 1)`, not by the key count. A class of one circuit
//! yields nothing.
//!
//! `scan_shard` walks values without needing keys (the frozen format stores
//! only 76 key bits), and shards partition keys by hash, so any single shard is
//! an unbiased ~1/256 sample.
//!
//! ```text
//! frozen_class_census FROZEN_DIR [--shards N]
//! ```

use local_mixing::db_mixing::frozen::scan_shard;

const MAX_CLASS: usize = 4096;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(dir) = args.get(1) else {
        eprintln!("usage: frozen_class_census FROZEN_DIR [--shards N]");
        std::process::exit(2);
    };
    let shards: usize = args
        .iter()
        .position(|a| a == "--shards")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .clamp(1, 256);

    let mut hist = vec![0u64; MAX_CLASS + 1];
    let mut keys = 0u64;
    let mut circuits = 0u64;
    let mut qualifying = 0u64;
    // sum |c| * (|c| - 1): ordered pairs, which is what identity generation walks.
    let mut ordered_pairs = 0u128;
    let mut max_class = 0usize;
    let mut gate_hist = vec![0u64; 64];

    for shard in 0..shards {
        scan_shard(dir, shard, &mut |value| {
            // Legacy framing: [len][blob] repeated.
            let mut position = 0usize;
            let mut members = 0usize;
            while position < value.len() {
                let len = value[position] as usize;
                position += 1;
                if len == 0 || position + len > value.len() {
                    break;
                }
                gate_hist[(len / 3).min(63)] += 1;
                position += len;
                members += 1;
            }
            keys += 1;
            circuits += members as u64;
            hist[members.min(MAX_CLASS)] += 1;
            max_class = max_class.max(members);
            if members >= 2 {
                qualifying += 1;
                ordered_pairs += (members as u128) * (members as u128 - 1);
            }
        });
        eprintln!("[class-census] shard {shard} done, keys={keys}");
    }

    let scale = 256.0 / shards as f64;
    println!("scanned-shards={shards} (of 256)");
    println!("keys={keys}  circuits={circuits}  max-class={max_class}");
    println!(
        "mean-class={:.3}  qualifying(|c|>=2)={qualifying} ({:.4}%)",
        circuits as f64 / keys.max(1) as f64,
        qualifying as f64 * 100.0 / keys.max(1) as f64
    );
    println!("ordered-pairs(sum |c|*(|c|-1))={ordered_pairs}");

    println!("\nclass-size distribution (non-zero):");
    for (size, count) in hist.iter().enumerate() {
        if *count > 0 {
            println!("  |class|={size:<6} {count:>14}");
        }
    }

    println!("\ncircuit gate-count distribution (non-zero):");
    for (gates, count) in gate_hist.iter().enumerate() {
        if *count > 0 {
            println!("  gates={gates:<4} {count:>14}");
        }
    }

    // Each ordered pair yields 2 spellings; each identity of length n expands to
    // 4*n*(n-1) emitted candidates (2 directions x n rotations x (n-1) splits
    // x 2 emissions). The completed shortcut build averaged ~337 emitted per
    // identity, which is the multiplier used here.
    let est_keys = keys as f64 * scale;
    let est_pairs = ordered_pairs as f64 * scale;
    let est_identities = est_pairs * 2.0;
    let est_generated = est_identities * 337.0;
    println!("\n=== full-store projection (x{scale:.0}) ===");
    println!("keys                 {est_keys:>22.0}");
    println!("identities           {est_identities:>22.0}");
    println!("generated candidates {est_generated:>22.0}");
    println!(
        "  vs the completed shortcut build: {:.1}x the identities, {:.1}x the generated",
        est_identities / 39_469_712.0,
        est_generated / 13_325_475_432.0
    );
}
