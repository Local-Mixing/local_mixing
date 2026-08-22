//! Gauntlet driver for the Rust gadgetizers — one arm of the systematic
//! local-mixing gadget test pipeline.
//!
//! Reads a pre-generated serial r57 chain C (mpmct1), gadgetizes it with the
//! chosen gadgetizer, samples inputs bit-sliced, and dumps the uniform audit
//! bundle consumed by `gauntlet_audit`:
//!
//!   <prefix>.meta         line-based metadata (key<TAB>value; target list)
//!   <prefix>.trace.bin    bit-packed columns: init wires, then per gate
//!                         (flip, newval) interleaved
//!   <prefix>.targets.bin  bit-packed columns: per source gate (a, b, c_old,
//!                         f, c_new), plus a trailing NULL random column
//!
//! Gadget arms (the unified ladder, weakest to strongest):
//!   none  — G = C verbatim (positive control: every attack must fire)
//!   ss    — paired secret-share  w = s ⊕ r   (gadgetize_xgates, masks/prod off)
//!   semi  — single-carrier product-share  V = C ⊕ M(B) ⊕ κ, Gray fold
//!           (gadgetize_xgates_single, ProdConfig::production_single(); "band"
//!           accepted as an alias)
//!   file  — load a pre-gadgetized circuit from --g-in (mpmct1); this is how
//!           the Python-built nonlinear193/nonlinear291 gadgets enter the SAME
//!           trace/audit pipeline. The checked builder sidecar is validated
//!           against the loaded files, and the mixer re-verifies after mixing.
//!
//! Mixing (`--mix MOVES`): after gadgetization, run the engine Mixer (their
//! local-mixing walk: crossings, unsubsume/copy splits, conjugation twists,
//! thermostat at the input size) in-process, then trace the MIXED circuit.
//! With mixing on, this is the same pipeline as gen_*_gadget -> fmix, with a
//! generic gadgetization front-end.
//!
//! Input policy: source wires 0..n get fresh random bits per sample; the rest
//! are zero (`--aux zero`, the hmap_affine convention) or random (`--aux random`).

use clap::Parser;
use local_mixing::circuit::xgate::{XGate, max_wire};
use local_mixing::engine::format::{read_mpmct, write_mpmct};
use local_mixing::engine::mix::{MixParams, MixStop, Mixer};
use local_mixing::preprocessing::gadgets::{
    CnotCircuit, MaskConfig, ProdConfig, gadgetize_xgates, gadgetize_xgates_single,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::collections::HashMap;

#[derive(Parser)]
struct Args {
    /// none | ss | semi | band | file
    #[arg(long)]
    gadget: String,
    /// source chain, mpmct1
    #[arg(long)]
    c_in: String,
    /// pre-gadgetized circuit, mpmct1 (only for --gadget file)
    #[arg(long)]
    g_in: Option<String>,
    /// builder-provided init columns (only for --gadget file); the matching
    /// <init-in prefix>.buildmeta supplies x_holders + decode recipes
    #[arg(long)]
    init_in: Option<String>,
    #[arg(long)]
    out_prefix: String,
    /// logical wires of C
    #[arg(long)]
    n: usize,
    #[arg(long)]
    seed: u64,
    #[arg(long)]
    gadget_seed: u64,
    /// samples reserved at the tail for the correlation audit
    #[arg(long)]
    corr_samples: usize,
    /// zero | random
    #[arg(long)]
    aux: String,
    /// mixing moves (0 = mixing off)
    #[arg(long, default_value_t = 0)]
    mix: u64,
    /// seed for the mixing walk
    #[arg(long, default_value_t = 777)]
    mix_seed: u64,
    /// Report the post-gadgetization/post-mix dimensions without sampling or
    /// reading builder initial columns. Used by the orchestrator's mix probe.
    #[arg(long, default_value_t = false)]
    size_only: bool,
}

/// One bit-sliced column: `w` words, only the low `s_bits` of the tail word live.
type Col = Vec<u64>;

fn words_for(samples: usize) -> usize {
    samples.div_ceil(64)
}

fn tail_mask(samples: usize) -> u64 {
    let r = samples % 64;
    if r == 0 { u64::MAX } else { (1u64 << r) - 1 }
}

fn random_col(samples: usize, rng: &mut StdRng) -> Col {
    let w = words_for(samples);
    let mut c: Col = (0..w).map(|_| rng.random::<u64>()).collect();
    *c.last_mut().unwrap() &= tail_mask(samples);
    c
}

/// Simulate an XGate circuit bit-sliced; returns (final state, flips, newvals).
fn simulate(gates: &[XGate], state: &mut [Col], samples: usize) -> (Vec<Col>, Vec<Col>) {
    let tm = tail_mask(samples);
    let w = words_for(samples);
    let mut flips = Vec::with_capacity(gates.len());
    let mut newvals = Vec::with_capacity(gates.len());
    for g in gates {
        // flip = comp XOR AND(literals)
        let mut f: Col = vec![u64::MAX; w];
        for &(cw, pol) in &g.ctrls {
            let s = &state[cw as usize];
            if pol {
                for i in 0..w {
                    f[i] &= s[i];
                }
            } else {
                for i in 0..w {
                    f[i] &= !s[i];
                }
            }
        }
        *f.last_mut().unwrap() &= tm;
        if g.comp {
            for i in 0..w - 1 {
                f[i] = !f[i];
            }
            let last = f.last_mut().unwrap();
            *last = !*last & tm;
        }
        let t = &mut state[g.target as usize];
        for i in 0..w {
            t[i] ^= f[i];
        }
        flips.push(f);
        newvals.push(t.clone());
    }
    (flips, newvals)
}

/// Extract (a, b) of the r57 gate `t ^= a OR NOT b` from its XGate encoding.
/// from_g57([t, x, y]) = `t ^= x OR NOT y` stores x as the NEGATIVE-polarity
/// control and y as the POSITIVE-polarity control (fires = 1 ^ (!x & y)), so
/// a is the negative-polarity control wire and b the positive-polarity one.
fn r57_pins(g: &XGate) -> (u16, u16) {
    assert!(g.comp, "chain gates must be r57 (comp=1)");
    assert_eq!(g.ctrls.len(), 2, "chain gates must have 2 controls");
    let mut pos = None;
    let mut neg = None;
    for &(w, p) in &g.ctrls {
        if p { pos = Some(w) } else { neg = Some(w) }
    }
    (
        neg.expect("need a negative control (a)"),
        pos.expect("need a positive control (b)"),
    )
}

fn pack_col(c: &Col, out: &mut Vec<u8>) {
    for &word in c.iter() {
        out.extend_from_slice(&word.to_le_bytes());
    }
}

/// Read a columns file (u64-LE bit-packed, `samples` bits per column).
fn read_cols(path: &str, samples: usize) -> Vec<Col> {
    let bytes = std::fs::read(path).expect("read columns file");
    let w = words_for(samples);
    let col_bytes = w * 8;
    assert!(bytes.len() % col_bytes == 0, "columns file size mismatch");
    bytes
        .chunks(col_bytes)
        .map(|ch| {
            ch.chunks(8)
                .map(|w8| u64::from_le_bytes(w8.try_into().unwrap()))
                .collect()
        })
        .collect()
}

/// The Python gadgets' wire decode: blocks of 5 with
/// E(x0..x4) = x0 ^ x1 ^ maj(x2,x3,x4); a wire's value is the XOR of E over
/// its two 5-block shares (10 physical wires total).
fn decode_e(wires: &[usize], state: &[Col], samples: usize) -> Col {
    assert!(wires.len() % 5 == 0, "decode blocks are 5 wires");
    let w = words_for(samples);
    let tm = tail_mask(samples);
    let mut acc = vec![0u64; w];
    for blk in wires.chunks(5) {
        let (x0, x1) = (&state[blk[0]], &state[blk[1]]);
        let (x2, x3, x4) = (&state[blk[2]], &state[blk[3]], &state[blk[4]]);
        for i in 0..w {
            acc[i] ^= x0[i] ^ x1[i] ^ (x2[i] & x3[i]) ^ (x2[i] & x4[i]) ^ (x3[i] & x4[i]);
        }
    }
    *acc.last_mut().unwrap() &= tm;
    acc
}

fn main() {
    let args = Args::parse();
    let (source, cn) = read_mpmct(&args.c_in).expect("read source chain");
    assert_eq!(cn, args.n, "--n must match the chain's wire count");

    // ---------------- gadgetize ----------------
    let mut grng = StdRng::seed_from_u64(args.gadget_seed);
    let mut file_header: Option<(usize, usize)> = None;
    let gcirc = match args.gadget.as_str() {
        "none" => CnotCircuit {
            gates: source.clone(),
            num_wires: args.n,
        },
        "ss" => gadgetize_xgates(
            &source,
            args.n,
            2,
            &MaskConfig::off(),
            &ProdConfig::off(),
            &mut grng,
        ),
        "semi" | "band" => gadgetize_xgates_single(
            &source,
            args.n,
            2,
            &ProdConfig {
                // band=8 (the band=n default at n=8) exhausts its slot space at
                // ~13 gates (560 slots vs 45 masks/gate); band=16 gives headroom
                // for the k=16 chain. Everything else = production_single().
                band: 16,
                ..ProdConfig::production_single()
            },
            &mut grng,
        ),
        "file" => {
            let path = args.g_in.as_deref().unwrap_or_else(|| {
                eprintln!("--gadget file needs --g-in <mpmct1>");
                std::process::exit(2);
            });
            let (gates, gw) = read_mpmct(path).expect("read gadgetized circuit");
            file_header = Some((gw, gates.len()));
            CnotCircuit {
                gates,
                num_wires: gw,
            }
        }
        other => {
            eprintln!("unknown gadget {other}");
            std::process::exit(2);
        }
    };
    let mut nw = gcirc.num_wires;
    let mut gates = gcirc.gates;
    let k = source.len();

    // ---------------- optional mixing (their local-mixing walk, in-process) ---
    let mut mixed = false;
    if args.mix > 0 {
        let params = MixParams {
            moves: args.mix,
            seed: args.mix_seed,
            // churn profile: crossings + splits as per default, plus the
            // conjugation twists (the SAMF state/progress-mixing mechanism)
            // at moderate weight. Store-free (no FROZEN_DB_DIR needed).
            w_twist_neg: 0.08,
            w_twist_swap: 0.08,
            p_twist: 0.16,
            ..MixParams::default()
        };
        let mut mixer = Mixer::new(gates, nw, params);
        let stop = mixer.run();
        let (moved, disp) = mixer.final_float();
        mixer.global_check();
        gates = mixer.arena.to_vec();
        mixed = true;
        let stop_name = match stop {
            MixStop::MovesBudget => "moves-budget",
            MixStop::StopFlag => "stop-flag",
            MixStop::DoseReached => "dose-reached",
            MixStop::CanaryFired => "canary",
            MixStop::ProfileDone => "profile-done",
            MixStop::SplitDone => "split-done",
        };
        println!(
            "[mix] moves={} stop={} gates={} float=({moved},{disp}) verified",
            args.mix,
            stop_name,
            gates.len()
        );
    }
    let ng = gates.len();
    nw = nw.max(max_wire(&gates) as usize + 1);
    let nfeat = nw + 2 * ng;
    if args.size_only {
        println!("[size] nw={nw} gates={ng} features={nfeat} mixed={mixed}");
        return;
    }

    // ---------------- sample ----------------
    // layout: [0, s_fit) exact-attack fit | [s_fit, +2048) held-out CV |
    // [s_fit+2048, samples) correlation tail. All boundaries 64-aligned.
    assert!(
        args.corr_samples > 0 && args.corr_samples % 64 == 0,
        "corr_samples must be positive and 64-aligned"
    );
    let s_fit = (nfeat + 256).next_multiple_of(64);
    let samples = s_fit + 2048 + args.corr_samples;
    let mut srng = StdRng::seed_from_u64(args.seed);
    let aux_random = match args.aux.as_str() {
        "zero" => false,
        "random" => true,
        _other => {
            eprintln!("--aux must be zero|random");
            std::process::exit(2);
        }
    };
    // Builder-provided init (file mode): columns from --init-in, plus the
    // buildmeta's x_holders (input x verbatim) and decode[v] recipes.
    let mut x_holders: Vec<usize> = (0..args.n).collect();
    let mut decode: Vec<Vec<usize>> = Vec::new();
    let mut holder_x: Vec<Col> = Vec::new();
    let mut null_holder: Option<Col> = None; // builder-shipped band-class NULL (pool=band)
    let mut builder_gadget = String::new();
    let mut state: Vec<Col>;
    if args.gadget == "file" {
        let init_path = args.init_in.as_deref().unwrap_or_else(|| {
            eprintln!("--gadget file needs --init-in <init.bin>");
            std::process::exit(2);
        });
        // buildmeta sits next to the init file
        let bm_path = init_path
            .strip_suffix(".init.bin")
            .unwrap_or(init_path)
            .to_string()
            + ".buildmeta";
        let bm = std::fs::read_to_string(&bm_path).expect("read buildmeta");
        let mut fields: HashMap<&str, &str> = HashMap::new();
        for line in bm.lines() {
            let (key, value) = line
                .split_once('\t')
                .unwrap_or_else(|| panic!("malformed buildmeta line: {line:?}"));
            assert!(
                fields.insert(key, value).is_none(),
                "duplicate buildmeta key {key}"
            );
        }
        let field_usize = |key: &str| -> usize {
            fields
                .get(key)
                .unwrap_or_else(|| panic!("buildmeta missing {key}"))
                .parse()
                .unwrap_or_else(|_| panic!("buildmeta has invalid {key}"))
        };
        let field_u64 = |key: &str| -> u64 {
            fields
                .get(key)
                .unwrap_or_else(|| panic!("buildmeta missing {key}"))
                .parse()
                .unwrap_or_else(|_| panic!("buildmeta has invalid {key}"))
        };
        assert_eq!(
            fields.get("builder_checked").copied(),
            Some("true"),
            "buildmeta must attest builder_checked=true"
        );
        assert_eq!(
            field_usize("builder_schema"),
            2,
            "unsupported buildmeta schema"
        );
        let (file_nw, file_ng) = file_header.expect("file header shape");
        assert_eq!(field_usize("n"), args.n, "buildmeta n mismatch");
        assert_eq!(field_usize("k"), k, "buildmeta k mismatch");
        assert_eq!(field_u64("seed"), args.seed, "buildmeta seed mismatch");
        assert_eq!(
            field_usize("n_wires"),
            file_nw,
            "buildmeta n_wires mismatch"
        );
        assert_eq!(
            field_usize("n_gates"),
            file_ng,
            "buildmeta n_gates mismatch"
        );
        assert_eq!(
            field_usize("samples"),
            samples,
            "buildmeta samples mismatch"
        );
        assert_eq!(nw, file_nw, "mixing changed the file circuit's wire span");
        builder_gadget = fields
            .get("gadget")
            .expect("buildmeta missing gadget")
            .to_string();
        let expected_period = match builder_gadget.as_str() {
            "nonlinear193" => 193usize,
            "nonlinear291" => 291usize,
            other => panic!("unsupported builder gadget {other}"),
        };
        assert_eq!(
            file_ng,
            expected_period
                .checked_mul(k)
                .expect("expected builder gate count overflow"),
            "builder gadget gate-period mismatch"
        );
        let builder_pool = fields.get("pool").copied().expect("buildmeta missing pool");
        assert!(
            matches!(builder_pool, "ideal" | "band"),
            "unsupported buildmeta pool {builder_pool}"
        );
        let has_null_holder = match fields.get("null_holder").copied() {
            Some("true") => true,
            Some("false") => false,
            Some(other) => panic!("invalid buildmeta null_holder {other}"),
            None => panic!("buildmeta missing null_holder"),
        };
        assert_eq!(
            has_null_holder,
            builder_pool == "band",
            "buildmeta pool/null_holder mismatch"
        );
        let xh: Vec<usize> = fields
            .get("x_holders")
            .expect("buildmeta missing x_holders")
            .split(',')
            .map(|s| s.parse().expect("invalid x_holders entry"))
            .collect();
        let expected_xh: Vec<usize> = (file_nw..file_nw + args.n).collect();
        assert_eq!(xh, expected_xh, "buildmeta x_holders mismatch");
        state = read_cols(init_path, samples);
        let expected_init_cols = file_nw + args.n + usize::from(has_null_holder);
        assert_eq!(
            field_usize("init_cols"),
            expected_init_cols,
            "buildmeta init_cols mismatch"
        );
        assert_eq!(
            state.len(),
            expected_init_cols,
            "init.bin column count mismatch"
        );
        // The trailing n columns are the input x (out-of-band disclosure to
        // the tracer only; they are NOT wires of G and never enter the trace);
        // an optional n+1-th holder carries the builder's band-class NULL.
        let mut holders: Vec<Col> = state.split_off(file_nw);
        if has_null_holder {
            null_holder = holders.pop();
        }
        holder_x = holders;
        let mut dec: Vec<Option<Vec<usize>>> = vec![None; args.n];
        for (v, slot) in dec.iter_mut().enumerate() {
            let key = format!("decode[{v}]");
            let value = fields
                .get(key.as_str())
                .unwrap_or_else(|| panic!("buildmeta missing {key}"));
            let wires: Vec<usize> = value
                .split(',')
                .map(|s| s.parse().expect("invalid decode wire"))
                .collect();
            assert_eq!(
                wires.len(),
                10,
                "decode recipes must contain exactly two 5-wire blocks"
            );
            assert!(
                wires.iter().all(|&wire| wire < file_nw),
                "decode wire out of range"
            );
            let mut unique_wires = wires.clone();
            unique_wires.sort_unstable();
            unique_wires.dedup();
            assert_eq!(
                unique_wires.len(),
                wires.len(),
                "decode recipe contains duplicate wires"
            );
            *slot = Some(wires);
        }
        x_holders = xh;
        decode = dec
            .into_iter()
            .map(|d| d.expect("buildmeta missing decode[v]"))
            .collect();
        // decode recipes reference circuit wires only (< nw): fine
    } else {
        state = (0..nw)
            .map(|w| {
                if w < args.n || aux_random {
                    random_col(samples, &mut srng)
                } else {
                    vec![0; words_for(samples)]
                }
            })
            .collect();
    }
    // x columns: for file mode the holders were split off `state` above;
    // re-derive from x_holders against the PRE-split init copy.
    let xcols: Vec<Col> = if args.gadget == "file" {
        // holders were split off; x_holders index into the pre-split vector,
        // which is exactly (state ++ holders)
        x_holders
            .iter()
            .map(|&h| {
                state
                    .get(h)
                    .cloned()
                    .unwrap_or_else(|| holder_x[h - nw].clone())
            })
            .collect()
    } else {
        (0..args.n).map(|w| state[w].clone()).collect()
    };
    let init = state.clone();
    let mut cstate: Vec<Col> = xcols.clone();
    let mut targets: Vec<Col> = Vec::with_capacity(5 * k + 1);
    let mut target_names: Vec<String> = Vec::new();
    let mut trivial_feats: Vec<i64> = Vec::new();
    let mut written = vec![false; args.n]; // wires written by an earlier source gate
    for (i, g) in source.iter().enumerate() {
        let (aw, bw) = r57_pins(g);
        let t = g.target as usize;
        let a = cstate[aw as usize].clone();
        let b = cstate[bw as usize].clone();
        let cold = cstate[t].clone();
        // f = a OR NOT b
        let w = words_for(samples);
        let tm = tail_mask(samples);
        let mut f: Col = (0..w).map(|i| a[i] | !b[i]).collect();
        *f.last_mut().unwrap() &= tm;
        let cnew: Col = (0..w).map(|i| cold[i] ^ f[i]).collect();
        cstate[t] = cnew.clone();
        targets.extend([a, b, cold, f, cnew]);
        // A target is TRIVIAL when it equals a raw input wire of C: a, b, cold
        // are the value of some wire at gate i, which is init[wire] iff that
        // wire was never written before gate i.
        let triv = if args.gadget == "file" {
            // File-mode raw x holders were removed from `state` and therefore
            // are not adversarial trace features. A physical share at the same
            // small numeric index is not a trivial disclosure.
            [-1; 5]
        } else {
            [
                (!written[aw as usize]).then_some(aw as i64).unwrap_or(-1),
                (!written[bw as usize]).then_some(bw as i64).unwrap_or(-1),
                (!written[t]).then_some(t as i64).unwrap_or(-1),
                -1,
                -1,
            ]
        };
        trivial_feats.extend(triv);
        for kind in ["a", "b", "cold", "f", "cnew"] {
            target_names.push(format!("g{i}:{kind}"));
        }
        written[t] = true;
    }
    // NULL target: correlation baseline.  Default: independent random column.
    // In band-pool mode every trace value is a deterministic function of the
    // full pipeline input, so the baseline must come from the same class: the
    // builder ships one extra band column as a holder, never consumed by any
    // gadget gate.
    let pool_band = null_holder.is_some();
    let null_col: Col = match null_holder {
        Some(c) => c,
        None => random_col(samples, &mut srng),
    };
    targets.push(null_col);
    target_names.push("NULL".to_string());

    // ---------------- run G ----------------
    let (flips, newvals) = simulate(&gates, &mut state, samples);

    // ---------------- behavioral check ----------------
    // Rust-native gadgets: the logical value sits on wire v at the end (ss:
    // w = s⊕r verbatim; semi retires clean) -- compare directly.  File mode:
    // decode each logical wire from the buildmeta recipe (10 wires =
    // E(P1)^E(P2)) on the FINAL state -- a genuine end-to-end check of the
    // (possibly mixed) circuit.
    let mut behavioral_ok = true;
    if args.gadget == "file" {
        for v in 0..args.n {
            let dec = decode_e(&decode[v], &state, samples);
            if dec != cstate[v] {
                behavioral_ok = false;
            }
        }
    } else {
        for v in 0..args.n {
            if state[v] != cstate[v] {
                behavioral_ok = false;
            }
        }
    }

    // ---------------- dump ----------------
    let mut trace = Vec::with_capacity(nfeat * samples / 8 + 64);
    for c in &init {
        pack_col(c, &mut trace);
    }
    for i in 0..ng {
        pack_col(&flips[i], &mut trace);
        pack_col(&newvals[i], &mut trace);
    }
    std::fs::write(format!("{}.trace.bin", args.out_prefix), &trace).unwrap();

    let mut tb = Vec::with_capacity(targets.len() * samples / 8 + 64);
    for c in &targets {
        pack_col(c, &mut tb);
    }
    std::fs::write(format!("{}.targets.bin", args.out_prefix), &tb).unwrap();

    write_mpmct(&format!("{}.g.mpmct1", args.out_prefix), &gates, nw).unwrap();

    let mut meta = String::new();
    let push = |m: &mut String, key: &str, val: String| {
        m.push_str(&format!("{key}\t{val}\n"));
    };
    push(&mut meta, "gadget", args.gadget.clone());
    push(&mut meta, "mixed", mixed.to_string());
    push(&mut meta, "mix_moves", args.mix.to_string());
    push(&mut meta, "k", k.to_string());
    push(&mut meta, "n", args.n.to_string());
    push(&mut meta, "seed", args.seed.to_string());
    push(&mut meta, "gadget_seed", args.gadget_seed.to_string());
    push(&mut meta, "aux", args.aux.clone());
    push(
        &mut meta,
        "pool",
        if pool_band {
            "band".to_string()
        } else {
            "ideal".to_string()
        },
    );
    if args.gadget == "file" {
        push(&mut meta, "builder_gadget", builder_gadget.clone());
    }
    push(&mut meta, "samples", samples.to_string());
    push(&mut meta, "corr_samples", args.corr_samples.to_string());
    push(&mut meta, "n_wires", nw.to_string());
    push(&mut meta, "n_gates", ng.to_string());
    push(&mut meta, "n_features", nfeat.to_string());
    push(&mut meta, "n_targets", targets.len().to_string());
    push(&mut meta, "behavioral_ok", behavioral_ok.to_string());
    let gt: Vec<String> = gates.iter().map(|g| g.target.to_string()).collect();
    push(&mut meta, "gate_targets", gt.join(","));
    // per-target metadata: name + trivial raw-input-wire feature (or -1)
    trivial_feats.push(-1); // NULL
    for (i, name) in target_names.iter().enumerate() {
        push(
            &mut meta,
            &format!("target[{i}]"),
            format!("{name}\t{}", trivial_feats[i]),
        );
    }
    std::fs::write(format!("{}.meta", args.out_prefix), meta).unwrap();

    println!(
        "[{}] k={k} n={} nw={nw} gates={ng} features={nfeat} samples={samples} mixed={mixed} behavioral_ok={behavioral_ok}",
        args.gadget, args.n
    );
    if !behavioral_ok {
        eprintln!("behavioral equivalence check failed; audit bundle is marked invalid");
        std::process::exit(1);
    }
}
