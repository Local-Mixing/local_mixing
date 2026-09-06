// Final compression pass for fmix/fsplit output (see postprocessing/compress.rs):
// gather same-target gates that can float together (transporting groups
// across writers of their controls by conjugation, floating past separated
// readers, forward and on the reversed list), XOR-reduce each group in
// ESOP/ANF space, re-emit surviving cubes as consecutive XGates, iterate to a
// fixed point, then PACK: each maximal same-target run becomes one
// generalized gate whose activation function is spelled in algebraic normal
// form (the anf1 format, engine/format.rs) -- the unique representation, so
// the artifact carries nothing of how its cubes were spelled upstream.
// Deterministic and attacker-computable, so it cannot weaken the hiding; the
// compressed size is the honest "effective size" of the artifact.
//
// The ANF is then COMPACTED into a mixed-polarity ESOP by the deterministic
// reducer strategies, from the ANF alone, so the result is still one
// spelling per function (the esop1 format, ~2.3x fewer terms than the ANF;
// the ANF is regenerated from it whenever needed). --no-pack writes the
// mpmct1 cube circuit instead. Every mpmct1 reader in the tree loads esop1
// files transparently (as the term expansion).
//
// Optional dead-cone pruning for gadgetized circuits, where equality is
// required only on designated output wires: --live-wires upper-half (or
// lower-half, or an explicit list "0-255,300"). Default all wires live.
//
// Example:
//   fcompress --input crossB.mpmct1 --output final.esop1
//   fcompress --input crossB.mpmct1 --output final.mpmct1 --no-pack
//   fcompress --input gadget.txt --output gadget_fcmp.esop1 --live-wires upper-half
use clap::Parser;
use local_mixing::circuit::xgate::{XGate, eval_lanes, max_wire};
use local_mixing::engine::format;
use local_mixing::engine::mix::Mixer;
use local_mixing::engine::format::expand_packed;
use local_mixing::postprocessing::compress::{
    CompressParams, compact, compress_anc, lits_of, pack, pack_census,
};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[derive(Parser, Debug)]
#[command(name = "fcompress")]
struct Args {
    /// Input circuit file
    #[arg(long)]
    input: String,
    /// Input format: mpmct1 | g57
    #[arg(long, default_value = "mpmct1")]
    input_format: String,
    /// Output file (mpmct1); omit for a dry run (verify + report only)
    #[arg(long)]
    output: Option<String>,
    /// Wires whose final value must be preserved: all | upper-half |
    /// lower-half | explicit list like "0-255,300,510"
    #[arg(long, default_value = "all")]
    live_wires: String,
    /// Wires promised ZERO at circuit entry (upper-half | lower-half |
    /// explicit list). Enables input-side specialization; the output then
    /// equals the input only on that subspace, and the global check samples
    /// only such entries. Omit for whole-function equality.
    #[arg(long)]
    zero_wires: Option<String>,
    #[arg(long, default_value_t = 10)]
    max_iters: usize,
    #[arg(long, default_value_t = 64)]
    group_cap: usize,
    #[arg(long, default_value_t = 40)]
    anf_support_cap: usize,
    /// Disable the interleaved conjugation-descent (downhill) pass
    #[arg(long, default_value_t = false)]
    no_downhill: bool,
    /// Disable in-gather transport (floating a group across a writer of one
    /// of its control wires by conjugation instead of closing it)
    #[arg(long, default_value_t = false)]
    no_transport: bool,
    /// Extra cubes a transport may add (0 = neutral-or-better only)
    #[arg(long, default_value_t = 0)]
    transport_slack: usize,
    /// Disable separation-aware reads (a reader separated from every member
    /// of a group by an opposite literal no longer floats past it)
    #[arg(long, default_value_t = false)]
    no_sep_reads: bool,
    /// Disable the reversed-list (leftward) gather each iteration
    #[arg(long, default_value_t = false)]
    no_reverse: bool,
    /// Write the output as an mpmct1 cube circuit instead of the packed
    /// canonical esop1 form (one generalized gate per maximal same-target
    /// run, its ANF compacted by the deterministic reducer)
    #[arg(long, default_value_t = false)]
    no_pack: bool,
    /// Report the packing statistics (runs, monomials, support, canonical
    /// ESOP size) of the compressed circuit
    #[arg(long, default_value_t = false)]
    pack_census: bool,
    /// Rounds of the 64-lane sampled global check against the input
    #[arg(long, default_value_t = 64)]
    verify_rounds: usize,
    /// Disable the per-group exhaustive/sampled verification
    #[arg(long, default_value_t = false)]
    no_local_verify: bool,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    /// Per-gate ancestry sidecar for the INPUT circuit (written by fmix
    /// --anc-out; gate count must match). Sets are threaded through
    /// compression: gathering permutes them, every survivor of a reduced
    /// group carries the members' UNION, pruned gates drop out.
    #[arg(long)]
    anc_in: Option<String>,
    /// Write the compressed circuit's ancestry sidecar (same universe header
    /// as the input sidecar). Requires --anc-in. Works in dry-run mode too.
    #[arg(long)]
    anc_out: Option<String>,
}

fn parse_live(spec: &str, wires: usize) -> Option<Vec<bool>> {
    match spec {
        "all" => None,
        "upper-half" => {
            let mut v = vec![false; wires];
            v[wires / 2..].iter_mut().for_each(|b| *b = true);
            Some(v)
        }
        "lower-half" => {
            let mut v = vec![false; wires];
            v[..wires / 2].iter_mut().for_each(|b| *b = true);
            Some(v)
        }
        list => {
            let mut v = vec![false; wires];
            for part in list.split(',') {
                let part = part.trim();
                if let Some((a, b)) = part.split_once('-') {
                    let (a, b): (usize, usize) = (
                        a.parse().expect("live range"),
                        b.parse().expect("live range"),
                    );
                    v[a..=b].iter_mut().for_each(|x| *x = true);
                } else {
                    v[part.parse::<usize>().expect("live wire")] = true;
                }
            }
            Some(v)
        }
    }
}

fn main() {
    let args = Args::parse();
    let (gates, file_wires): (Vec<XGate>, usize) = match args.input_format.as_str() {
        "mpmct1" => format::read_mpmct(&args.input).expect("read mpmct1 circuit"),
        "g57" => {
            let g = format::read_g57_file(&args.input).expect("read g57 circuit");
            let w = max_wire(&g) as usize + 1;
            (g, w)
        }
        other => panic!("unknown --input-format {other}"),
    };
    let wires = file_wires.max(max_wire(&gates) as usize + 1);
    let live = parse_live(&args.live_wires, wires);
    let n_live = live
        .as_ref()
        .map_or(wires, |v| v.iter().filter(|&&b| b).count());
    let zero = args.zero_wires.as_ref().map(|spec| {
        parse_live(spec, wires).expect("--zero-wires needs an explicit wire set, not 'all'")
    });
    let n_zero = zero.as_ref().map_or(0, |v| v.iter().filter(|&&b| b).count());
    println!(
        "[fcompress] input: {} gates ({} lits), {} wires ({} live, {} zero-in); max_iters={} group_cap={} anf_cap={} downhill={} transport={} slack={} sep_reads={} reverse={} seed={}",
        gates.len(),
        lits_of(&gates),
        wires,
        n_live,
        n_zero,
        args.max_iters,
        args.group_cap,
        args.anf_support_cap,
        !args.no_downhill,
        !args.no_transport,
        args.transport_slack,
        !args.no_sep_reads,
        !args.no_reverse,
        args.seed
    );

    let params = CompressParams {
        live_out: live.clone(),
        zero_in: zero.clone(),
        max_iters: args.max_iters,
        group_cap: args.group_cap,
        anf_support_cap: args.anf_support_cap,
        downhill: !args.no_downhill,
        transport: !args.no_transport,
        transport_slack: args.transport_slack,
        sep_reads: !args.no_sep_reads,
        reverse_pass: !args.no_reverse,
        local_verify: !args.no_local_verify,
        seed: args.seed,
    };
    assert!(
        args.anc_out.is_none() || args.anc_in.is_some(),
        "--anc-out needs --anc-in: there is no ancestry to thread otherwise"
    );
    let sidecar = args.anc_in.as_ref().map(|p| {
        let sc = Mixer::read_anc_sidecar(p).expect("read ancestry sidecar");
        assert_eq!(
            sc.sets.len(),
            gates.len(),
            "--anc-in: sidecar has {} sets but the input circuit has {} gates",
            sc.sets.len(),
            gates.len()
        );
        println!(
            "[fcompress] ancestry threaded: {} mode, universe m={}, K={}",
            if sc.sampled { "sampled" } else { "exact" },
            sc.m,
            sc.tracers.len()
        );
        sc
    });
    let original = gates.clone();
    let t0 = std::time::Instant::now();
    let anc_sets = sidecar.as_ref().map(|sc| sc.sets.clone());
    let (out, out_anc, rep) = compress_anc(gates, anc_sets, wires, &params);
    let secs = t0.elapsed().as_secs_f64();

    // Pack (the default): one canonical ANF gate per maximal same-target
    // run. The written artifact is what gets verified below, so the packed
    // circuit is expanded back to cubes and checked alongside.
    let packed = if args.no_pack {
        None
    } else {
        let anf = pack(&out);
        let monos: usize = anf.iter().map(|g| g.term_count()).sum();
        let p = compact(&anf);
        let terms: usize = p.iter().map(|g| g.term_count()).sum();
        println!(
            "[fcompress] packed: {} cubes -> {} canonical gates ({:.1}%); ANF {} monomials -> compacted ESOP {} terms",
            out.len(),
            p.len(),
            100.0 * p.len() as f64 / out.len().max(1) as f64,
            monos,
            terms
        );
        Some(p)
    };
    let packed_expanded: Option<Vec<XGate>> = packed.as_ref().map(|p| expand_packed(p));

    // Sampled global check against the untouched input, on live wires only,
    // over entry states inside the promised zero slice (when one is given).
    let mut rng = StdRng::seed_from_u64(args.seed ^ 0xC0FFEE);
    for round in 0..args.verify_rounds {
        let sa: Vec<u64> = (0..wires)
            .map(|w| {
                if zero.as_ref().is_some_and(|z| z[w]) {
                    0
                } else {
                    rng.random::<u64>()
                }
            })
            .collect();
        let mut sb = sa.clone();
        let mut sp = packed_expanded.as_ref().map(|_| sa.clone());
        let mut sa = sa;
        eval_lanes(original.iter(), &mut sa);
        eval_lanes(out.iter(), &mut sb);
        if let (Some(px), Some(sp)) = (&packed_expanded, sp.as_mut()) {
            eval_lanes(px.iter(), sp);
        }
        for w in 0..wires {
            if live.as_ref().is_none_or(|v| v[w]) {
                assert_eq!(
                    sa[w], sb[w],
                    "global check failed on wire {w} (round {round})"
                );
                if let Some(sp) = &sp {
                    assert_eq!(
                        sa[w], sp[w],
                        "packed circuit check failed on wire {w} (round {round})"
                    );
                }
            }
        }
    }
    println!(
        "[fcompress] done in {:.1}s (verified {} rounds x64 lanes): gates {} -> {} ({:.1}%), lits {} -> {} ({:.1}%), live_dropped={}",
        secs,
        args.verify_rounds,
        rep.gates_in,
        rep.gates_out,
        100.0 * rep.gates_out as f64 / rep.gates_in.max(1) as f64,
        rep.lits_in,
        rep.lits_out,
        100.0 * rep.lits_out as f64 / rep.lits_in.max(1) as f64,
        rep.liveness_dropped
    );

    if args.pack_census {
        pack_census(&out);
    }

    if let Some(path) = &args.output {
        match &packed {
            Some(p) => {
                format::write_esop1(path, p, wires).expect("write packed output");
                println!("[fcompress] wrote {} packed gates to {} (esop1)", p.len(), path);
            }
            None => {
                format::write_mpmct(path, &out, wires).expect("write output");
                println!("[fcompress] wrote {} gates to {} (mpmct1)", out.len(), path);
            }
        }
        if zero.is_some() {
            println!(
                "[fcompress] WARNING: --zero-wires output equals the input ONLY on the \
                 promised zero slice; the mpmct1 header does not record this — do not \
                 substitute it for the whole-function circuit"
            );
        }
    } else {
        println!("[fcompress] no --output given; result discarded after verification");
    }

    if let Some(path) = &args.anc_out {
        use std::fmt::Write as _;
        let sc = sidecar.expect("asserted above");
        let tags = out_anc.expect("threaded when anc_in is given");
        let words = sc.words;
        let mut o = String::with_capacity(tags.len() * words * 8);
        let _ = writeln!(
            o,
            "fmix-anc 1 {} m={} words={} gates={}",
            if sc.sampled { "sampled" } else { "exact" },
            sc.m,
            words,
            out.len()
        );
        if sc.sampled {
            let _ = write!(o, "tracers {}", sc.tracers.len());
            for t in &sc.tracers {
                let _ = write!(o, " {t}");
            }
            o.push('\n');
        }
        for row in &tags {
            let mut first = true;
            for wi in 0..words {
                if !first {
                    o.push(' ');
                }
                let _ = write!(o, "{}", row.get(wi).copied().unwrap_or(0));
                first = false;
            }
            o.push('\n');
        }
        std::fs::write(path, o).expect("write ancestry sidecar");
        println!(
            "[fcompress] wrote compressed ancestry sidecar to {path} ({} sets)",
            tags.len()
        );
    }
}
