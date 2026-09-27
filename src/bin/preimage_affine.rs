//! Oracle-only preimage attack against a mixed circuit A on the zero-slice
//! coset: given a target Y, find X with A(X, 0, 0, 0) = (*, Y, *, *), i.e. the
//! payload output block (wires 128..255) equals Y. Uses ONLY the mixed circuit
//! as a forward oracle — never the sandwich or any intermediate.
//!
//! Strategy: test whether the map  X (wires 0..127)  ->  output block Y
//! (wires 128..255)  is AFFINE over GF(2) on this coset (all other input wires
//! pinned 0). If it is, the mixing has leaked its full linear structure and the
//! preimage is a 128x128 GF(2) solve. If it is NOT affine, the linear attack
//! fails and we report the observed nonlinearity (evidence the mixing resists
//! this attack).
//!
//!   preimage_affine <circuit.mpmct1> <Yhex> [<Yhex> ...] [--samples N] [--xw0 W] [--yw0 W] [--blk B]
//!
//! Y is 32 hex digits (128 bit), same big-endian convention as eval_c output.
use local_mixing::postmix::format::read_mpmct;
use local_mixing::postmix::xgate::{eval_lanes, XGate};
use std::time::Instant;

const BLK_DEFAULT: usize = 128;

// Evaluate up to 64 inputs at once (bit-sliced). Each lane l gets X-block value
// xs[l] placed on wires [xw0, xw0+blk); every other wire is 0. Returns the full
// lane state (state[w] = one bit per lane) after running all gates.
fn eval_batch(gates: &[XGate], num_wires: usize, xs: &[u128], xw0: usize, blk: usize) -> Vec<u64> {
    assert!(xs.len() <= 64);
    let mut state = vec![0u64; num_wires];
    for (lane, &x) in xs.iter().enumerate() {
        for i in 0..blk {
            if (x >> i) & 1 == 1 {
                state[xw0 + i] |= 1u64 << lane;
            }
        }
    }
    eval_lanes(gates, &mut state);
    state
}

// Extract the y-block value for a given lane out of a lane state.
fn yblock_of_lane(state: &[u64], lane: usize, yw0: usize, blk: usize) -> u128 {
    let mut y = 0u128;
    for j in 0..blk {
        if (state[yw0 + j] >> lane) & 1 == 1 {
            y |= 1u128 << j;
        }
    }
    y
}

// Solve  rows . x = b  over GF(2). rows[j] has bit i = M[j][i]; b bit j = rhs of
// eqn j. Returns (solution, rank, free_columns). solution sets free vars to 0.
// None if inconsistent.
fn gf2_solve(rows: &[u128], b: u128, nvars: usize) -> Option<(u128, usize, usize)> {
    let mut a: Vec<u128> = rows.to_vec();
    let mut rhs: Vec<u8> = (0..a.len()).map(|j| ((b >> j) & 1) as u8).collect();
    let mut pivot_col_of_row: Vec<i32> = Vec::new();
    let mut where_pivot: Vec<i32> = vec![-1; nvars]; // col -> row
    let mut row = 0usize;
    for col in 0..nvars {
        // find a row >= `row` with a 1 in `col`
        let mut sel = None;
        for r in row..a.len() {
            if (a[r] >> col) & 1 == 1 {
                sel = Some(r);
                break;
            }
        }
        let Some(sel) = sel else { continue };
        a.swap(row, sel);
        rhs.swap(row, sel);
        for r in 0..a.len() {
            if r != row && (a[r] >> col) & 1 == 1 {
                a[r] ^= a[row];
                rhs[r] ^= rhs[row];
            }
        }
        where_pivot[col] = row as i32;
        pivot_col_of_row.push(col as i32);
        row += 1;
        if row == a.len() {
            break;
        }
    }
    let rank = row;
    // consistency: any all-zero row with rhs 1 => inconsistent
    for r in 0..a.len() {
        if a[r] == 0 && rhs[r] == 1 {
            return None;
        }
    }
    let mut x = 0u128;
    for col in 0..nvars {
        let pr = where_pivot[col];
        if pr >= 0 && rhs[pr as usize] == 1 {
            x |= 1u128 << col;
        }
    }
    let free = nvars - rank;
    Some((x, rank, free))
}

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: preimage_affine <circuit.mpmct1> <Yhex>...");
    let mut targets_hex: Vec<String> = Vec::new();
    let mut samples = 1024usize;
    let mut xw0 = 0usize;
    let mut yw0 = 128usize;
    let mut blk = BLK_DEFAULT;
    let mut it = args.peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--samples" => samples = it.next().unwrap().parse().unwrap(),
            "--xw0" => xw0 = it.next().unwrap().parse().unwrap(),
            "--yw0" => yw0 = it.next().unwrap().parse().unwrap(),
            "--blk" => blk = it.next().unwrap().parse().unwrap(),
            other => targets_hex.push(other.trim_start_matches("0x").to_string()),
        }
    }
    assert!(!targets_hex.is_empty(), "give at least one Y target (32 hex digits)");
    let targets: Vec<u128> =
        targets_hex.iter().map(|s| u128::from_str_radix(s, 16).expect("bad Y hex")).collect();

    let t0 = Instant::now();
    let (gates, num_wires) = read_mpmct(&path).expect("read mpmct");
    eprintln!(
        "loaded {} gates, {} wires in {:.1}s  (x wires {}..{}, y wires {}..{})",
        gates.len(),
        num_wires,
        t0.elapsed().as_secs_f64(),
        xw0,
        xw0 + blk,
        yw0,
        yw0 + blk
    );

    // ---- Build the affine model: c = A(0)|y ; col[i] = A(e_i)|y XOR c ----
    // 129 basis evaluations: index 0 = zero, index 1+i = unit vector e_i.
    let mut basis_out = vec![0u128; 1 + blk];
    let total = 1 + blk;
    let mut g = 0;
    while g < total {
        let take = (total - g).min(64);
        let xs: Vec<u128> = (0..take)
            .map(|l| {
                let idx = g + l;
                if idx == 0 {
                    0u128
                } else {
                    1u128 << (idx - 1)
                }
            })
            .collect();
        let state = eval_batch(&gates, num_wires, &xs, xw0, blk);
        for l in 0..take {
            basis_out[g + l] = yblock_of_lane(&state, l, yw0, blk);
        }
        g += take;
    }
    let c = basis_out[0];
    let col: Vec<u128> = (0..blk).map(|i| basis_out[1 + i] ^ c).collect();
    eprintln!("built affine model ({} basis evals) at {:.1}s", total, t0.elapsed().as_secs_f64());

    // ---- Verify affinity on random inputs ----
    // Deterministic LCG so the check is reproducible; these are just probes.
    let mut rng: u128 = 0x243F6A8885A308D3;
    let mut next_rand = || {
        // xorshift-ish over u128
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut mismatches = 0usize;
    let mut checked = 0usize;
    let mut per_bit_mismatch = vec![0u64; blk]; // per output bit: # of deviating inputs
    let mut resid_weight_sum = 0u64; // sum over inputs of #bits the nonlinear part flips
    // Online GF(2) row-basis of the nonlinear residual vectors r(x), pivot table
    // indexed by leading bit. If these span all `blk` dimensions, NO linear
    // combination of output bits is affine in X -> the SAT-invariant attack has
    // nothing to inject.
    let mut piv: Vec<u128> = vec![0u128; blk]; // piv[p] has leading bit p, or 0 if empty
    let blk_mask: u128 = if blk >= 128 { u128::MAX } else { (1u128 << blk) - 1 };
    let mut b = 0usize;
    while b < samples {
        let take = (samples - b).min(64);
        let xs: Vec<u128> = (0..take).map(|_| next_rand() & blk_mask).collect();
        let state = eval_batch(&gates, num_wires, &xs, xw0, blk);
        for l in 0..take {
            let actual = yblock_of_lane(&state, l, yw0, blk);
            // best affine prediction
            let mut pred = c;
            let x = xs[l];
            for i in 0..blk {
                if (x >> i) & 1 == 1 {
                    pred ^= col[i];
                }
            }
            let resid = actual ^ pred;
            if resid != 0 {
                mismatches += 1;
            }
            resid_weight_sum += resid.count_ones() as u64;
            for j in 0..blk {
                if (resid >> j) & 1 == 1 {
                    per_bit_mismatch[j] += 1;
                }
            }
            // fold residual into the GF(2) span (pivot table)
            let mut v = resid;
            while v != 0 {
                let p = 127 - v.leading_zeros() as usize;
                if piv[p] == 0 {
                    piv[p] = v;
                    break;
                }
                v ^= piv[p];
            }
            checked += 1;
        }
        b += take;
    }
    let affine_bits = per_bit_mismatch.iter().filter(|&&m| m == 0).count();
    let mean_resid = resid_weight_sum as f64 / checked as f64;
    let resid_rank = piv.iter().filter(|&&v| v != 0).count();
    let out_invariants = blk - resid_rank; // # output-side linear relations affine in X
    eprintln!(
        "affinity check: {}/{} random inputs matched the affine model  ({} mismatches) at {:.1}s",
        checked - mismatches,
        checked,
        mismatches,
        t0.elapsed().as_secs_f64()
    );
    eprintln!(
        "  per-bit: {}/{} output bits individually affine in X; mean nonlinear residual {:.1}/{} bits",
        affine_bits, blk, mean_resid, blk
    );
    eprintln!(
        "  residual span rank {}/{}  =>  {} output-side linear invariant(s) for a SAT solver to exploit",
        resid_rank, blk, out_invariants
    );

    // ---- Algebraic degree lower bound via order-k derivatives ----
    // A k-th order derivative of a degree-<k function is identically 0. For each
    // order k we take P random (base a, directions d_1..d_k) restricted to the
    // X-block, XOR the y-blocks over all 2^k corners, and if any is nonzero the
    // map has degree >= k. The 2^k corners fit in one lane batch (k <= 6).
    let mut deg_lb = 0usize;
    for k in 1..=6usize {
        let corners = 1usize << k;
        if corners > 64 {
            break;
        }
        let probes = 96usize;
        let mut nonzero_seen = false;
        'probe: for _ in 0..probes {
            let a = next_rand() & blk_mask;
            let dirs: Vec<u128> = (0..k).map(|_| next_rand() & blk_mask).collect();
            // corner c: a XOR (XOR of dirs[i] where bit i of c set)
            let xs: Vec<u128> = (0..corners)
                .map(|c| {
                    let mut x = a;
                    for i in 0..k {
                        if (c >> i) & 1 == 1 {
                            x ^= dirs[i];
                        }
                    }
                    x
                })
                .collect();
            let state = eval_batch(&gates, num_wires, &xs, xw0, blk);
            let mut deriv = 0u128;
            for c in 0..corners {
                deriv ^= yblock_of_lane(&state, c, yw0, blk);
            }
            if deriv != 0 {
                nonzero_seen = true;
                break 'probe;
            }
        }
        if nonzero_seen {
            deg_lb = k;
        } else {
            break; // no nonzero order-k derivative found; stop climbing
        }
    }
    eprintln!(
        "  algebraic degree lower bound (order-k derivative probe): deg(X -> Y) >= {}",
        deg_lb
    );

    if mismatches != 0 {
        println!("VERDICT: NOT affine on this coset — linear inversion does NOT apply.");
        println!(
            "  {} of {} random probes deviated from the best affine fit ({:.1}%).",
            mismatches,
            checked,
            100.0 * mismatches as f64 / checked as f64
        );
        println!(
            "  {}/{} output bits are individually affine in X; nonlinear part flips {:.1}/{} bits on average.",
            affine_bits, blk, mean_resid, blk
        );
        println!(
            "  residual span rank {}/{} => {} output-side linear invariant(s) available to a SAT-invariant attack.",
            resid_rank, blk, out_invariants
        );
        println!("  algebraic degree of X -> Y is >= {} (order-k derivative probe).", deg_lb);
        println!("  The mixed circuit resists the affine/linear preimage attack; no Xi recovered by this method.");
        return;
    }

    println!("VERDICT: map X -> output block Y is AFFINE over GF(2) on this coset.");
    println!("  => mixing leaked its linear structure; inverting each target by 128x128 GF(2) solve.\n");

    // Build M rows for the solver: rows[j] bit i = col[i] bit j.
    let mut rows = vec![0u128; blk];
    for i in 0..blk {
        let ci = col[i];
        for j in 0..blk {
            if (ci >> j) & 1 == 1 {
                rows[j] |= 1u128 << i;
            }
        }
    }

    for (idx, &y) in targets.iter().enumerate() {
        let rhs = y ^ c;
        match gf2_solve(&rows, rhs, blk) {
            None => println!("Y{}: no solution (target outside the affine image).", idx + 1),
            Some((xsol, rank, free)) => {
                // forward-check with the real circuit (single lane).
                let state = eval_batch(&gates, num_wires, &[xsol], xw0, blk);
                let ycheck = yblock_of_lane(&state, 0, yw0, blk);
                let ok = ycheck == y;
                println!(
                    "Y{} = {:032x}",
                    idx + 1,
                    y
                );
                println!(
                    "  X{} = {:032x}   [rank {}, {} free bits]  forward-check {}",
                    idx + 1,
                    xsol,
                    rank,
                    free,
                    if ok { "PASS ✓ (A(X,0,0,0) block2 == Y)" } else { "FAIL ✗" }
                );
            }
        }
    }
}
