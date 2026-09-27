//! collision_blocks — value collisions between gates and the contiguous
//! identity blocks they bracket.
//!
//! For C = g_1..g_m let C_i(x) be the value of g_i's active (target) wire in
//! the state right after g_i on input x. Gates i, j COLLIDE when C_i = C_j on
//! every (sampled legal) input. Values are compared as 64-bit hashes of their
//! vectors over `--samples` random legal inputs (random x on wires 0..n,
//! zeros elsewhere), so "for all x" means "for all sampled x".
//!
//! Identity blocks. A contiguous block g_a..g_b is the identity on the
//! reachable states iff every wire w written inside it is RESTORED: the value
//! after the block's last write to w equals the value before the block, i.e.
//! (k(a,w), l(b,w)) is a collision, where k(a,w) is the last write to w before
//! a and l(b,w) the last write to w at or before b. The sweep below tests that
//! bracket condition for every start a and every end b <= a + L (L =
//! `--max-len`), so it finds every contiguous identity block up to length L
//! (interleaved identities are not contiguous and are not found). For each
//! start it records the minimal end; a block is PRIME when no proper
//! sub-block is itself an identity.
//!
//! Outputs <prefix>.json (collision census + block census) and
//! <prefix>.primes.csv (start, end, length, wires written) for the prime
//! blocks, plus <prefix>.cover.csv (identity-cover density along the gate
//! list in 1024-gate bins).

use clap::Parser;
use local_mixing::engine::format::read_mpmct;
use rayon::prelude::*;
use std::io::Write;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "collision_blocks")]
struct Args {
    #[arg(long)]
    g: String,
    #[arg(long)]
    out: String,
    #[arg(long, default_value_t = 128)]
    n: usize,
    #[arg(long, default_value_t = 8192)]
    samples: usize,
    /// Maximal block length examined by the identity sweep.
    #[arg(long, default_value_t = 16384)]
    max_len: usize,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long, default_value_t = 0)]
    threads: usize,
}

const NONE: u32 = u32::MAX;

fn log2_bin(x: u64) -> usize {
    if x == 0 {
        0
    } else {
        (64 - x.leading_zeros()) as usize
    }
}
fn json_u(v: &[u64]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(","))
}
fn fnum(x: f64) -> String {
    if x.is_nan() { "null".into() } else { format!("{:.8}", x) }
}

fn main() -> std::io::Result<()> {
    let args = Args::parse();
    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(args.threads).build_global().ok();
    }
    let seed = args.seed.unwrap_or_else(|| fastrand::u64(..));
    let t0 = Instant::now();
    let (gates, w_cnt) = read_mpmct(&args.g)?;
    let m = gates.len();
    eprintln!("[collision_blocks] {} gates, {} wires ({:.1}s)", m, w_cnt, t0.elapsed().as_secs_f64());

    // previous write to the same wire (k(i, w_i)); input segment = NONE
    let mut last_write = vec![NONE; w_cnt];
    let mut prev_write = vec![NONE; m];
    let mut writes = vec![0u32; w_cnt];
    let mut fanout = vec![0u32; m]; // reads of the segment created by gate i
    for (i, g) in gates.iter().enumerate() {
        for &(c, _) in g.ctrls.iter() {
            let lw = last_write[c as usize];
            if lw != NONE {
                fanout[lw as usize] += 1;
            }
        }
        let t = g.target as usize;
        prev_write[i] = last_write[t];
        last_write[t] = i as u32;
        writes[t] += 1;
    }

    // per-wire sorted positions of writes and reads (for the anatomy of removable pairs)
    let mut writes_of: Vec<Vec<u32>> = vec![Vec::new(); w_cnt];
    let mut reads_of: Vec<Vec<u32>> = vec![Vec::new(); w_cnt];
    for (i, g) in gates.iter().enumerate() {
        writes_of[g.target as usize].push(i as u32);
        for &(c, _) in g.ctrls.iter() {
            reads_of[c as usize].push(i as u32);
        }
    }
    let count_in = |v: &Vec<u32>, lo: u32, hi: u32| -> usize {
        // number of positions strictly between lo and hi
        let a = v.partition_point(|&x| x <= lo);
        let b = v.partition_point(|&x| x < hi);
        b.saturating_sub(a)
    };

    // evaluate + hash every segment value vector
    let nw = (args.samples / 64).max(1);
    let nbits = nw * 64;
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut state = vec![0u64; w_cnt * nw];
    for w in 0..args.n.min(w_cnt) {
        for k in 0..nw {
            state[w * nw + k] = rng.u64(..);
        }
    }
    let hash_of = |v: &[u64]| -> u64 {
        let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 8) };
        xxhash_rust::xxh3::xxh3_64(bytes)
    };
    let zero_hash = hash_of(&vec![0u64; nw]);
    let mut input_hash = vec![0u64; w_cnt];
    for w in 0..w_cnt {
        input_hash[w] = hash_of(&state[w * nw..(w + 1) * nw]);
    }
    let mut gate_hash = vec![0u64; m]; // hash of C_i
    let mut fire_hash = vec![0u64; m]; // hash of the toggle vector of g_i
    let mut fired_pop = vec![0u32; m];
    let mut acc = vec![0u64; nw];
    for (i, g) in gates.iter().enumerate() {
        acc.fill(!0u64);
        for &(c, p) in g.ctrls.iter() {
            let xm = if p { 0u64 } else { !0u64 };
            let s = &state[c as usize * nw..(c as usize + 1) * nw];
            for k in 0..nw {
                acc[k] &= s[k] ^ xm;
            }
        }
        if g.comp {
            for k in 0..nw {
                acc[k] = !acc[k];
            }
        }
        let t = g.target as usize;
        let mut pop = 0u32;
        for k in 0..nw {
            state[t * nw + k] ^= acc[k];
            pop += acc[k].count_ones();
        }
        fired_pop[i] = pop;
        fire_hash[i] = hash_of(&acc);
        gate_hash[i] = hash_of(&state[t * nw..(t + 1) * nw]);
    }
    drop(state);
    eprintln!("[collision_blocks] evaluated {} inputs ({:.1}s)", nbits, t0.elapsed().as_secs_f64());
    let pre_hash = |i: usize| -> u64 {
        let p = prev_write[i];
        if p == NONE { input_hash[gates[i].target as usize] } else { gate_hash[p as usize] }
    };

    // ---------------- collision census ----------------
    let mut order: Vec<(u64, u32)> = (0..m).map(|i| (gate_hash[i], i as u32)).collect();
    order.par_sort_unstable();
    let mut n_classes = 0u64;
    let mut gates_in_collision = 0u64;
    let mut pairs_total: u128 = 0;
    let mut pairs_same_wire: u128 = 0;
    let mut pairs_zero_class: u128 = 0;
    let mut zero_class_size = 0u64;
    let mut class_size_hist = vec![0u64; 40]; // gates by log2 class size
    let mut top: Vec<u64> = Vec::new();
    let mut same_wire_gap_hist = vec![0u64; 40]; // consecutive same-wire members of a class: gap in gates
    let mut cross_wire_classes = 0u64;
    let mut i = 0;
    while i < order.len() {
        let mut j = i + 1;
        while j < order.len() && order[j].0 == order[i].0 {
            j += 1;
        }
        let s = (j - i) as u64;
        n_classes += 1;
        class_size_hist[log2_bin(s)] += s;
        if s >= 2 {
            gates_in_collision += s;
            pairs_total += (s as u128) * ((s - 1) as u128) / 2;
            top.push(s);
            // per-wire grouping inside the class
            let mut members: Vec<(u16, u32)> = order[i..j].iter().map(|&(_, g)| (gates[g as usize].target, g)).collect();
            members.sort_unstable();
            let mut k = 0;
            let mut nwires = 0;
            while k < members.len() {
                let mut l = k + 1;
                while l < members.len() && members[l].0 == members[k].0 {
                    l += 1;
                }
                let c = (l - k) as u128;
                pairs_same_wire += c * (c - 1) / 2;
                for q in k + 1..l {
                    same_wire_gap_hist[log2_bin((members[q].1 - members[q - 1].1) as u64)] += 1;
                }
                nwires += 1;
                k = l;
            }
            if nwires > 1 {
                cross_wire_classes += 1;
            }
            if order[i].0 == zero_hash {
                zero_class_size = s;
                pairs_zero_class = (s as u128) * ((s - 1) as u128) / 2;
            }
        }
        i = j;
    }
    top.sort_unstable_by(|a, b| b.cmp(a));
    top.truncate(12);
    // trivial collisions: dead gates (value unchanged from the previous write of the same wire)
    let dead: u64 = (0..m).filter(|&i| gate_hash[i] == pre_hash(i)).count() as u64;
    let never_fire: u64 = fired_pop.iter().filter(|&&p| p == 0).count() as u64;
    // same-wire restore pairs (a,b): b's value equals the value of an EARLIER write a to the same
    // wire that is not the immediately preceding one -> the wire was changed and later restored.
    let mut restore_pairs = 0u64;
    let mut restore_gap_hist = vec![0u64; 40];
    // what sits between the endpoints of a same-wire restore collision (a, b):
    // the writes to w in (a, b] have toggle vectors that XOR to zero.
    let mut inner_hist = vec![0u64; 40]; // number of writes strictly between a and b
    let mut inner1 = 0u64;
    let mut inner1_fire_twin = 0u64; // the single inner write c toggles exactly like b
    let mut inner1_identical_gate = 0u64; // c and b are the same gate (target, literals, comp)
    let mut inner1_gap_hist = vec![0u64; 40]; // b - c
    let mut inner_ge2 = 0u64;
    let mut inner_ge2_pairwise = 0u64; // toggle vectors of (a, b] pair up into identical pairs
    let mut twin_masked_reads_hist = vec![0u64; 40]; // reads of the masked segment inside a twin bracket
    // zero-read twin brackets (removable pairs): does anything in between write one of the
    // twin gate's control wires?  If not, the two gates commute to adjacency and cancel
    // syntactically; if so, the pair is only semantically removable.
    let mut removable = 0u64;
    let mut removable_ctrl_writes_hist = vec![0u64; 40]; // number of in-between writes to the gate's control wires
    let mut removable_len_hist = vec![0u64; 40]; // bracket length
    let mut removable_len_hist_noctrl = vec![0u64; 40]; // bracket length when no control write in between
    let mut twin_cover_diff = vec![0i32; m + 1]; // coverage by twin brackets [c, b]
    let mut restore_cover_diff = vec![0i32; m + 1]; // coverage by all restore brackets [first inner write, b]
    #[derive(Default)]
    struct Sem { total: u64, all_ctrls_restored: u64, written_ctrl_restored: u64, comp: u64, inner_twin: u64, inner_identical: u64, inner_twin_read: u64, inner_twin_unread: u64, csv_rows: usize,
                 n_written_hist: Vec<u64>, k_hist: Vec<u64>, w_class: Vec<u64>, u_class: Vec<u64>, inner_writes_hist: Vec<u64>, inner_reads_hist: Vec<u64> }
    let mut sem = Sem { n_written_hist: vec![0; 40], k_hist: vec![0; 40], w_class: vec![0; 3], u_class: vec![0; 3], inner_writes_hist: vec![0; 40], inner_reads_hist: vec![0; 40], ..Default::default() };
    let mut sem_csv = std::io::BufWriter::new(std::fs::File::create(format!("{}.semantic.csv", args.out))?);
    writeln!(sem_csv, "c,b,len,w,k,comp,literals,n_ctrls_written,all_ctrls_restored,written_ctrl,written_ctrl_restored,inner_writes,inner_twin,inner_identical,inner_reads,inner_first_write,inner_first_literals")?;
    let mut restore_csv = std::io::BufWriter::new(std::fs::File::create(format!("{}.restores.csv", args.out))?);
    writeln!(restore_csv, "a,b,wire,inner_writes,inner1_fire_twin,inner1_identical_gate")?;
    let mut csv_rows = 0usize;
    {
        // the input value of each wire counts as a (virtual) write at NONE
        let mut per_wire: Vec<std::collections::HashMap<u64, u32>> = (0..w_cnt).map(|w| { let mut h = std::collections::HashMap::new(); h.insert(input_hash[w], NONE); h }).collect();
        for i in 0..m {
            let t = gates[i].target as usize;
            let h = gate_hash[i];
            if let Some(&a) = per_wire[t].get(&h) {
                if prev_write[i] != a {
                    restore_pairs += 1;
                    let gap = if a == NONE { i as u64 + 1 } else { (i as u32 - a) as u64 };
                    restore_gap_hist[log2_bin(gap)] += 1;
                    let mut n_inner = 0u64;
                    let mut c = prev_write[i];
                    let mut first_inner = i as u32;
                    let mut fh: Vec<u64> = vec![fire_hash[i]];
                    while c != a && c != NONE {
                        n_inner += 1;
                        first_inner = c;
                        if fh.len() < 4096 {
                            fh.push(fire_hash[c as usize]);
                        }
                        c = prev_write[c as usize];
                    }
                    if n_inner >= 1 {
                        restore_cover_diff[first_inner as usize] += 1;
                        restore_cover_diff[i + 1] -= 1;
                    }
                    inner_hist[log2_bin(n_inner)] += 1;
                    let mut twin = false;
                    let mut ident = false;
                    if n_inner == 1 {
                        inner1 += 1;
                        let c = prev_write[i] as usize;
                        twin = fire_hash[c] == fire_hash[i];
                        ident = twin && gates[c].ctrls == gates[i].ctrls && gates[c].comp == gates[i].comp;
                        if twin { inner1_fire_twin += 1; }
                        if ident { inner1_identical_gate += 1; }
                        inner1_gap_hist[log2_bin((i - c) as u64)] += 1;
                        if twin {
                            twin_masked_reads_hist[log2_bin(fanout[c] as u64)] += 1;
                            twin_cover_diff[c] += 1;
                            twin_cover_diff[i + 1] -= 1;
                            if fanout[c] == 0 {
                                removable += 1;
                                let mut ctrl_writes = 0u64;
                                for j in c + 1..i {
                                    let tj = gates[j].target;
                                    if gates[i].ctrls.iter().any(|&(w, _)| w == tj) {
                                        ctrl_writes += 1;
                                    }
                                }
                                removable_ctrl_writes_hist[log2_bin(ctrl_writes)] += 1;
                                removable_len_hist[log2_bin((i - c) as u64)] += 1;
                                if ctrl_writes == 0 {
                                    removable_len_hist_noctrl[log2_bin((i - c) as u64)] += 1;
                                } else {
                                    // anatomy of a semantic-only pair
                                    let live = |u: usize, pos: u32| -> u64 {
                                        let v = &writes_of[u];
                                        let k = v.partition_point(|&x| x < pos);
                                        if k == 0 { input_hash[u] } else { gate_hash[v[k - 1] as usize] }
                                    };
                                    let g = &gates[i];
                                    let mut all_restored = true;
                                    let mut n_written = 0u64;
                                    let mut first_written: Option<usize> = None;
                                    for &(u, _) in g.ctrls.iter() {
                                        let u = u as usize;
                                        let nwr = count_in(&writes_of[u], c as u32, i as u32);
                                        if nwr > 0 {
                                            n_written += 1;
                                            if first_written.is_none() { first_written = Some(u); }
                                        }
                                        if live(u, c as u32) != live(u, i as u32) { all_restored = false; }
                                    }
                                    let u = first_written.unwrap();
                                    let v = &writes_of[u];
                                    let a0 = v.partition_point(|&x| x <= c as u32);
                                    let b0 = v.partition_point(|&x| x < i as u32);
                                    let inner_writes = b0 - a0;
                                    let f = v[a0] as usize;
                                    let l = v[b0 - 1] as usize;
                                    let inner_twin = inner_writes == 2 && fire_hash[f] == fire_hash[l];
                                    let inner_identical = inner_twin && gates[f].ctrls == gates[l].ctrls && gates[f].comp == gates[l].comp;
                                    let inner_reads = if inner_writes >= 2 { count_in(&reads_of[u], f as u32, l as u32) } else { 0 };
                                    let u_restored = live(u, c as u32) == live(u, i as u32);
                                    sem.total += 1;
                                    if all_restored { sem.all_ctrls_restored += 1; }
                                    if u_restored { sem.written_ctrl_restored += 1; }
                                    sem.n_written_hist[log2_bin(n_written)] += 1;
                                    sem.k_hist[g.ctrls.len().min(39)] += 1;
                                    if g.comp { sem.comp += 1; }
                                    let cls = |w: usize| -> usize { if w < w_cnt / 4 { 0 } else if w < w_cnt / 2 { 1 } else { 2 } };
                                    sem.w_class[cls(g.target as usize)] += 1;
                                    sem.u_class[cls(u)] += 1;
                                    sem.inner_writes_hist[log2_bin(inner_writes as u64)] += 1;
                                    if inner_twin { sem.inner_twin += 1; }
                                    if inner_identical { sem.inner_identical += 1; }
                                    sem.inner_reads_hist[log2_bin(inner_reads as u64)] += 1;
                                    if inner_twin && inner_reads > 0 { sem.inner_twin_read += 1; }
                                    if inner_twin && inner_reads == 0 { sem.inner_twin_unread += 1; }
                                    if sem.csv_rows < 200_000 {
                                        let lits = |gg: &local_mixing::circuit::xgate::XGate| gg.ctrls.iter().map(|&(w, p)| format!("{}{}", if p { "" } else { "!" }, w)).collect::<Vec<_>>().join(" ");
                                        writeln!(sem_csv, "{},{},{},{},{},{},\"{}\",{},{},{},{},{},{},{},{},{},\"{}\"", c, i, i - c, g.target, g.ctrls.len(), g.comp as u8, lits(g), n_written, all_restored as u8, u, u_restored as u8, inner_writes, inner_twin as u8, inner_identical as u8, inner_reads, f, lits(&gates[f]))?;
                                        sem.csv_rows += 1;
                                    }
                                }
                            }
                        }
                    } else if n_inner >= 2 {
                        inner_ge2 += 1;
                        if (n_inner as usize) + 1 == fh.len() {
                            fh.sort_unstable();
                            let mut ok = true;
                            let mut k = 0;
                            while k < fh.len() {
                                let mut l = k + 1;
                                while l < fh.len() && fh[l] == fh[k] { l += 1; }
                                if (l - k) % 2 == 1 { ok = false; break; }
                                k = l;
                            }
                            if ok { inner_ge2_pairwise += 1; }
                        }
                    }
                    if csv_rows < 400_000 {
                        writeln!(restore_csv, "{},{},{},{},{},{}", if a == NONE { -1i64 } else { a as i64 }, i, t, n_inner, twin as u8, ident as u8)?;
                        csv_rows += 1;
                    }
                }
            }
            per_wire[t].insert(h, i as u32);
        }
    }
    drop(restore_csv);
    drop(sem_csv);
    let cover_stats = |diff: &[i32]| -> (u64, f64, i64) {
        let mut run: i64 = 0;
        let mut covered = 0u64;
        let mut depth_sum: i64 = 0;
        let mut depth_max: i64 = 0;
        for i in 0..m {
            run += diff[i] as i64;
            if run > 0 {
                covered += 1;
                depth_sum += run;
                if run > depth_max { depth_max = run; }
            }
        }
        (covered, if covered > 0 { depth_sum as f64 / covered as f64 } else { 0.0 }, depth_max)
    };
    let (tc, tdm, tdx) = cover_stats(&twin_cover_diff);
    let (rc, rdm, rdx) = cover_stats(&restore_cover_diff);
    let restore_json = format!(
        "{{\"pairs\":{},\"gap_log2_hist\":{},\"inner_writes_log2_hist\":{},\"inner1\":{},\"inner1_fire_twin\":{},\"inner1_identical_gate\":{},\"inner1_gap_log2_hist\":{},\"inner_ge2\":{},\"inner_ge2_pairwise_cancelling\":{},\
          \"twin_masked_reads_log2_hist\":{},\"twin_cover_gates\":{},\"twin_cover_frac\":{},\"twin_depth_mean\":{},\"twin_depth_max\":{},\
          \"restore_cover_gates\":{},\"restore_cover_frac\":{},\"restore_depth_mean\":{},\"restore_depth_max\":{},\
          \"removable_pairs\":{},\"removable_ctrl_writes_log2_hist\":{},\"removable_len_log2_hist\":{},\"removable_len_log2_hist_no_ctrl_write\":{},\
          \"semantic\":{{\"total\":{},\"all_ctrls_restored\":{},\"written_ctrl_restored\":{},\"comp\":{},\"inner_twin\":{},\"inner_identical\":{},\"inner_twin_read\":{},\"inner_twin_unread\":{},\
            \"n_ctrls_written_log2_hist\":{},\"k_hist\":{},\"w_class_x_y_anc\":{},\"written_ctrl_class_x_y_anc\":{},\"inner_writes_log2_hist\":{},\"inner_reads_log2_hist\":{}}}}}",
        restore_pairs, json_u(&restore_gap_hist), json_u(&inner_hist), inner1, inner1_fire_twin, inner1_identical_gate, json_u(&inner1_gap_hist), inner_ge2, inner_ge2_pairwise,
        json_u(&twin_masked_reads_hist), tc, fnum(tc as f64 / m as f64), fnum(tdm), tdx, rc, fnum(rc as f64 / m as f64), fnum(rdm), rdx,
        removable, json_u(&removable_ctrl_writes_hist), json_u(&removable_len_hist), json_u(&removable_len_hist_noctrl),
        sem.total, sem.all_ctrls_restored, sem.written_ctrl_restored, sem.comp, sem.inner_twin, sem.inner_identical, sem.inner_twin_read, sem.inner_twin_unread,
        json_u(&sem.n_written_hist), json_u(&sem.k_hist), json_u(&sem.w_class), json_u(&sem.u_class), json_u(&sem.inner_writes_hist), json_u(&sem.inner_reads_hist)
    );
    eprintln!("[collision_blocks] collision census ({:.1}s)", t0.elapsed().as_secs_f64());

    // ---------------- identity sweep ----------------
    let l_max = args.max_len;
    let targets: Vec<u16> = gates.iter().map(|g| g.target).collect();
    let pre_all: Vec<u64> = (0..m).map(|i| pre_hash(i)).collect();
    struct Sweep {
        min_end: Vec<u32>,
        min_end_wires: Vec<u16>,
        all_pairs: u64,
        hit_cutoff: u64,
    }
    let chunk = 4096usize;
    let sweeps: Vec<Sweep> = (0..(m + chunk - 1) / chunk)
        .into_par_iter()
        .map(|c| {
            let lo = c * chunk;
            let hi = (lo + chunk).min(m);
            let mut min_end = vec![NONE; hi - lo];
            let mut min_end_wires = vec![0u16; hi - lo];
            let mut all_pairs = 0u64;
            let mut hit_cutoff = 0u64;
            let mut stamp = vec![u32::MAX; w_cnt];
            let mut pre = vec![0u64; w_cnt];
            let mut restored = vec![false; w_cnt];
            for a in lo..hi {
                let mut unrestored = 0i64;
                let mut nwires = 0u16;
                let end = (a + l_max).min(m);
                let mut b = a;
                let mut found = false;
                while b < end {
                    let w = targets[b] as usize;
                    let cur = gate_hash[b];
                    if stamp[w] != a as u32 {
                        stamp[w] = a as u32;
                        pre[w] = pre_all[b];
                        nwires += 1;
                        let r = cur == pre[w];
                        restored[w] = r;
                        if !r {
                            unrestored += 1;
                        }
                    } else {
                        let r = cur == pre[w];
                        if restored[w] && !r {
                            unrestored += 1;
                        } else if !restored[w] && r {
                            unrestored -= 1;
                        }
                        restored[w] = r;
                    }
                    if unrestored == 0 {
                        all_pairs += 1;
                        if !found {
                            found = true;
                            min_end[a - lo] = b as u32;
                            min_end_wires[a - lo] = nwires;
                        }
                    }
                    b += 1;
                }
                if !found && end == a + l_max {
                    hit_cutoff += 1;
                }
            }
            Sweep { min_end, min_end_wires, all_pairs, hit_cutoff }
        })
        .collect();
    let mut min_end = Vec::with_capacity(m);
    let mut min_end_wires = Vec::with_capacity(m);
    let mut all_pairs = 0u64;
    let mut hit_cutoff = 0u64;
    for s in sweeps {
        min_end.extend(s.min_end);
        min_end_wires.extend(s.min_end_wires);
        all_pairs += s.all_pairs;
        hit_cutoff += s.hit_cutoff;
    }
    eprintln!("[collision_blocks] identity sweep L={} ({:.1}s)", l_max, t0.elapsed().as_secs_f64());

    // prime blocks: [a, e] with e = min_end[a] and no start a' in (a, e] with min_end[a'] <= e.
    // sparse table for range-min over min_end
    let logm = (usize::BITS - m.leading_zeros()) as usize;
    let mut sp: Vec<Vec<u32>> = vec![min_end.clone()];
    for k in 1..logm {
        let prev = &sp[k - 1];
        let half = 1 << (k - 1);
        let len = m.saturating_sub((1 << k) - 1);
        let row: Vec<u32> = (0..len).map(|i| prev[i].min(prev[i + half])).collect();
        sp.push(row);
    }
    let rmq = |l: usize, r: usize| -> u32 {
        // min over [l, r], inclusive; requires l <= r
        let len = r - l + 1;
        let k = (usize::BITS - len.leading_zeros() - 1) as usize;
        sp[k][l].min(sp[k][r + 1 - (1 << k)])
    };
    let mut primes: Vec<(u32, u32, u16)> = Vec::new();
    let mut n_min_blocks = 0u64;
    let mut min_len_hist = vec![0u64; 40];
    for a in 0..m {
        let e = min_end[a];
        if e == NONE {
            continue;
        }
        n_min_blocks += 1;
        min_len_hist[log2_bin((e - a as u32 + 1) as u64)] += 1;
        let e = e as usize;
        let inner_ok = if e > a { rmq(a + 1, e) as usize > e } else { true };
        if inner_ok {
            primes.push((a as u32, e as u32, min_end_wires[a]));
        }
    }
    // coverage by the union of all minimal blocks, and by primes
    let mut cover = vec![0u8; m];
    for a in 0..m {
        let e = min_end[a];
        if e != NONE {
            // mark via difference array
            cover[a] = cover[a].wrapping_add(1);
            if (e as usize) + 1 < m {
                cover[e as usize + 1] = cover[e as usize + 1].wrapping_sub(1);
            }
        }
    }
    let mut covered = 0u64;
    let mut run: i64 = 0;
    let mut cover_bins = vec![0u64; (m + 1023) / 1024];
    for i in 0..m {
        run += cover[i] as i8 as i64;
        if run > 0 {
            covered += 1;
            cover_bins[i / 1024] += 1;
        }
    }
    let mut prime_len_hist = vec![0u64; 40];
    let mut prime_wires_hist = vec![0u64; 40];
    let mut prime_gates = 0u64;
    let mut prime_len1 = 0u64;
    let mut prime_len2 = 0u64;
    for &(a, e, p) in &primes {
        let len = (e - a + 1) as u64;
        prime_len_hist[log2_bin(len)] += 1;
        prime_wires_hist[log2_bin(p as u64)] += 1;
        prime_gates += len;
        if len == 1 { prime_len1 += 1; }
        if len == 2 { prime_len2 += 1; }
    }
    eprintln!("[collision_blocks] primes ({:.1}s)", t0.elapsed().as_secs_f64());

    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(format!("{}.primes.csv", args.out))?);
        writeln!(f, "start,end,length,wires_written")?;
        for &(a, e, p) in &primes {
            writeln!(f, "{},{},{},{}", a, e, e - a + 1, p)?;
        }
        let mut c = std::io::BufWriter::new(std::fs::File::create(format!("{}.cover.csv", args.out))?);
        writeln!(c, "bin_start,covered_gates")?;
        for (b, &v) in cover_bins.iter().enumerate() {
            writeln!(c, "{},{}", b * 1024, v)?;
        }
    }
    let mut j = std::fs::File::create(format!("{}.json", args.out))?;
    write!(
        j,
        "{{\"circuit\":\"{}\",\"gates\":{},\"wires\":{},\"samples\":{},\"seed\":{},\"max_len\":{},\
         \"collisions\":{{\"classes\":{},\"gates_in_collision\":{},\"frac_gates_in_collision\":{},\"pairs\":{},\"pairs_same_wire\":{},\"pairs_cross_wire\":{},\
           \"zero_class_size\":{},\"pairs_in_zero_class\":{},\"top_class_sizes\":{},\"gates_by_log2_class_size\":{},\"classes_spanning_several_wires\":{},\
           \"same_wire_gap_log2_hist\":{},\"dead_gates_value_unchanged\":{},\"never_fire\":{},\"restore_pairs_same_wire\":{},\"restore_gap_log2_hist\":{}}},\
         \"blocks\":{{\"identity_start_end_pairs_up_to_L\":{},\"starts_with_an_identity\":{},\"frac_starts_with_an_identity\":{},\"starts_hitting_cutoff\":{},\
           \"min_block_len_log2_hist\":{},\"gates_covered_by_identity_blocks\":{},\"frac_gates_covered\":{},\
           \"prime_blocks\":{},\"prime_len1\":{},\"prime_len2\":{},\"prime_gates_total\":{},\"prime_len_log2_hist\":{},\"prime_wires_log2_hist\":{}}},\"restore\":{}}}\n",
        args.g, m, w_cnt, nbits, seed, l_max,
        n_classes, gates_in_collision, fnum(gates_in_collision as f64 / m as f64), pairs_total, pairs_same_wire, pairs_total - pairs_same_wire,
        zero_class_size, pairs_zero_class, json_u(&top), json_u(&class_size_hist), cross_wire_classes,
        json_u(&same_wire_gap_hist), dead, never_fire, restore_pairs, json_u(&restore_gap_hist),
        all_pairs, n_min_blocks, fnum(n_min_blocks as f64 / m as f64), hit_cutoff,
        json_u(&min_len_hist), covered, fnum(covered as f64 / m as f64),
        primes.len(), prime_len1, prime_len2, prime_gates, json_u(&prime_len_hist), json_u(&prime_wires_hist), restore_json
    )?;
    println!(
        "gates={} collisions: classes>=2 gates={} ({:.3}) pairs={} (same-wire {}) zero-class={} dead={} restore-pairs={} (inner1 {} twin {} identical {}; inner>=2 {} pairwise {}) | blocks: starts-with-identity={} ({:.3}) covered={:.3} primes={} (len1 {} len2 {}) cutoff-hits={}",
        m, gates_in_collision, gates_in_collision as f64 / m as f64, pairs_total, pairs_same_wire, zero_class_size, dead, restore_pairs, inner1, inner1_fire_twin, inner1_identical_gate, inner_ge2, inner_ge2_pairwise,
        n_min_blocks, n_min_blocks as f64 / m as f64, covered as f64 / m as f64, primes.len(), prime_len1, prime_len2, hit_cutoff
    );
    Ok(())
}
