//! V4 for the wide store: full round-trip verification. Every entry's MPX1
//! value must decode, and every decoded circuit — stored in canonical wire
//! space — must re-canonicalize to polynomials whose key equals the entry's
//! key. Any mismatch means the encode/canonical-mapping path is broken.
//!
//! Usage: wide_verify <wide_db_dir>
use local_mixing::circuit::polys_repr_blob;
use local_mixing::engine::mpx1;
use local_mixing::engine::xpoly::{XPolyBudget, canonicalize_xgates_single};
use rocksdb::{DB, IteratorMode, Options};
use xxhash_rust::xxh3::xxh3_128;

fn main() {
    let dir = std::env::args()
        .nth(1)
        .expect("usage: wide_verify <wide_db_dir>");
    let mut opts = Options::default();
    opts.create_if_missing(false);
    opts.set_max_open_files(-1);
    opts.set_merge_operator_associative(
        "append_merge_wide",
        local_mixing::db_generation::regular::append_merge_wide,
    );
    let db = DB::open_for_read_only(&opts, &dir, false).expect("open wide store");
    let budget = XPolyBudget::default();

    let mut entries = 0u64;
    let mut circuits = 0u64;
    let mut key_mismatches = 0u64;
    let mut decode_errors = 0u64;
    let mut noncanonical_order = 0u64;

    for item in db.iterator(IteratorMode::Start) {
        let (key, value) = item.expect("iterate");
        entries += 1;
        let key16: [u8; 16] = key.as_ref().try_into().expect("16-byte key");
        let decoded = match mpx1::decode_value(&value) {
            Ok(d) => d,
            Err(e) => {
                decode_errors += 1;
                if decode_errors <= 5 {
                    eprintln!("decode error at key {}: {e}", hex(&key16));
                }
                continue;
            }
        };
        for circ in decoded {
            circuits += 1;
            // Min-dir key: the store's convention keys each entry under the
            // smaller of the two directional canonical serializations.
            let (canon, blob) = {
                let f = canonicalize_xgates_single(&circ, false, budget);
                let r = canonicalize_xgates_single(&circ, true, budget);
                match (f, r) {
                    (Ok(f), Ok(r)) => {
                        let bf = polys_repr_blob(&f.polys);
                        let br = polys_repr_blob(&r.polys);
                        if br < bf { (r, br) } else { (f, bf) }
                    }
                    _ => {
                        key_mismatches += 1;
                        if key_mismatches <= 5 {
                            eprintln!("re-canon failed at key {}", hex(&key16));
                        }
                        continue;
                    }
                }
            };
            let recomputed = xxh3_128(&blob).to_le_bytes();
            if recomputed != key16 {
                key_mismatches += 1;
                if key_mismatches <= 5 {
                    eprintln!(
                        "KEY MISMATCH: stored {} recomputed {} (circuit {:?})",
                        hex(&key16),
                        hex(&recomputed),
                        circ
                    );
                }
            }
            // A truly canonical representative re-canonicalizes with an
            // identity permutation; count deviations (informational).
            if !canon
                .order
                .data
                .iter()
                .enumerate()
                .all(|(i, &d)| i == d as usize)
            {
                noncanonical_order += 1;
            }
        }
    }
    println!(
        "wide_verify {dir}: entries={entries} circuits={circuits} key_mismatches={key_mismatches} decode_errors={decode_errors} noncanonical_order={noncanonical_order}"
    );
    if key_mismatches == 0 && decode_errors == 0 {
        println!("V4: PASS");
    } else {
        println!("V4: FAIL");
        std::process::exit(1);
    }
}

fn hex(k: &[u8; 16]) -> String {
    k.iter().map(|b| format!("{b:02x}")).collect()
}
