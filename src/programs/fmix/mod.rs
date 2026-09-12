//! fmix application adapter: resolved CLI controls, stage execution, resume and artifacts.
use crate::circuit::formats as format;
use crate::circuit::xgate::{XGate, max_wire};
use crate::engine::mixer::{
    GEN_FRESH, MixParams, MixStop, Mixer, ORIGIN_SYNTH, PieceCfg, run_piecewise,
};
use crate::stages::db_mixing::replacement::DbMode;

#[cfg(test)]
use clap::Parser;
mod cli;
pub use cli::Args;
use cli::*;

pub fn run(matches: clap::ArgMatches) {
    // Keep the raw matches so the preset below can tell "the user asked for
    // this value" from "the user said nothing" — `--p-twist 0` is a real
    // request and must not be overwritten just because 0 is also the default.
    let given =
        |name: &str| matches.value_source(name) == Some(clap::parser::ValueSource::CommandLine);
    let mut args = <Args as clap::FromArgMatches>::from_arg_matches(&matches).expect("parse args");

    if args.quality.qc {
        if args.split || args.resume.is_some() {
            eprintln!(
                "[fmix] QC belongs to DB mixing; run it on the db_mixing circuit before split/resume"
            );
            std::process::exit(2);
        }
        if let Err(error) = args.quality.config().validate() {
            eprintln!("[fmix] invalid QC configuration: {error}");
            std::process::exit(2);
        }
    }

    // Parse the reference before a potentially long mixing run. A narrower
    // original shares the low input wires; additional mixed wires stay random.
    let qc_reference = if args.quality.qc {
        args.quality.qc_reference.as_ref().map(|path| {
            if args.quality.qc_reference_format == "g57" {
                format::read_g57_file(path).expect("read QC original reference")
            } else {
                format::read_mpmct(path)
                    .expect("read QC original reference")
                    .0
            }
        })
    } else {
        None
    };

    // Off switches for the default-on DB knobs (2026-08-03 defaults). Applied
    // first, so everything below -- the db_mixing preset included -- reads the
    // settled values.
    if args.no_db_prefixes {
        args.db_prefixes = false;
    }
    if args.no_curated {
        args.curated = false;
    }
    if args.no_curated_exhaust {
        args.curated_exhaust = false;
    }
    if args.no_curated_in_comp {
        args.curated_in_comp = false;
    }

    // LAYER-2 db_mixing preset: fill the db_mixing default block for any knob the
    // user did not pass explicitly. Explicit flags always win. (curated and
    // p_convex used to be set here; both are covered by the 2026-08-03
    // shipped defaults -- curated ON, p_convex 0.4 -- so db_mixing now just
    // inherits them.)
    if args.db_mixing {
        if !given("twist_g57") {
            args.twist_g57 = true;
        }
        if !given("p_twist") {
            args.p_twist = 0.0005;
        }
        if !given("db_advance") {
            args.db_advance = true;
        }
        if !given("mix_pay_random") {
            args.mix_pay_random = true;
        }
        // Its DB opinions (p_mingen 0.6 / p_mingen_comp 0) live in
        // DbLayer::db_mixing(), applied with the rest of the stack below.
        println!(
            "[fmix] db_mixing preset ON: twist-g57={} p_twist={} db-advance={} curated={} mix-pay-random={} (DB knobs settled below)",
            args.twist_g57, args.p_twist, args.db_advance, args.curated, args.mix_pay_random
        );
    }

    // GSS profile: the non-DB half (its DB block is DbLayer::gss()).
    if args.gss {
        if !given("curated") {
            args.curated = true;
        }
        if !given("curated_in_comp") {
            args.curated_in_comp = true;
        }
        if !given("db_advance") {
            args.db_advance = true;
        }
        println!(
            "[fmix] GSS profile ON: curated={} db-advance={} | p_mix NOT set (layer-2 owns it) | DB knobs settled below",
            args.curated, args.db_advance
        );
    }

    // ---- DB knob precedence, in one place ----
    //
    //   explicit CLI  >  preset (--gss over --db-mixing)  >  shipped defaults
    //
    // and within each layer, most specific wins at USE time (mode+geometry ->
    // mode -> base, in Mixer::active_*). The rule that matters, and that this
    // replaced: an EXPLICIT base knob now beats a merely-DEFAULTED mode knob.
    // Before, `--db-mode comp --s-db 20` silently ran at 12 because
    // `s_db_comp`'s default of 12 was indistinguishable from a real request.
    let cli = DbLayer {
        s_db: given("s_db").then_some(args.s_db),
        p_convex: given("p_convex").then_some(args.p_convex),
        p_mingen: given("p_mingen").then_some(args.p_mingen),
        // `--no-db-prefixes` is just "explicitly false" -- folding it in here
        // retires it as a separate mechanism.
        prefixes: if args.no_db_prefixes {
            Some(false)
        } else {
            given("db_prefixes").then_some(args.db_prefixes)
        },
        // These are Option at the CLI, so Some IS "the user asked".
        s_db_ctg: args.s_db_ctg,
        s_db_comp: args.s_db_comp,
        s_db_comp_ctg: args.s_db_comp_ctg,
        p_convex_comp: args.p_convex_comp,
        p_mingen_comp: args.p_mingen_comp,
        prefixes_mix: args.db_prefixes_mix,
        prefixes_comp: args.db_prefixes_comp,
    };
    let mut preset = DbLayer::default();
    if args.db_mixing {
        preset = preset.over(DbLayer::db_mixing());
    }
    if args.gss {
        // GSS is the more specific profile, so it sits above db_mixing.
        preset = DbLayer::gss().over(preset);
    }
    let base_given = BaseGiven {
        s_db: given("s_db"),
        p_convex: given("p_convex"),
    };
    let db = cli.over(preset).over(DbLayer::shipped(base_given));
    // Settle the args once, so every consumer below reads the same values.
    let must = |o: Option<f64>| o.expect("DbLayer::shipped() sets every base knob");
    args.s_db = db.s_db.expect("shipped sets s_db");
    args.p_convex = must(db.p_convex);
    args.p_mingen = must(db.p_mingen);
    args.db_prefixes = db.prefixes.expect("shipped sets prefixes");
    args.s_db_ctg = db.s_db_ctg;
    args.s_db_comp = db.s_db_comp;
    args.s_db_comp_ctg = db.s_db_comp_ctg;
    args.p_convex_comp = db.p_convex_comp;
    args.p_mingen_comp = db.p_mingen_comp;
    args.db_prefixes_mix = db.prefixes_mix;
    args.db_prefixes_comp = db.prefixes_comp;

    // A knob that cannot possibly fire is a bug in the command line, not a
    // no-op to shrug at -- silently-inert flags are exactly how the shadowing
    // bug hid for two days. With the overlay off (p_mix < 0) only one mode
    // ever runs, so the other mode's overrides can never be read.
    if args.p_db > 0.0 && args.p_mix < 0.0 {
        let comp_only = args.db_mode == "comp";
        // Read `cli`, NOT the settled args: after the layers are applied every
        // override holds a value, so blaming `args` would name flags the user
        // never typed.
        let inert: Vec<&str> = if comp_only {
            [
                ("--s-db-ctg", cli.s_db_ctg.is_some()),
                ("--db-prefixes-mix", cli.prefixes_mix.is_some()),
            ]
            .iter()
            .filter(|(_, set)| *set)
            .map(|(n, _)| *n)
            .collect()
        } else {
            [
                ("--s-db-comp", cli.s_db_comp.is_some()),
                ("--s-db-comp-ctg", cli.s_db_comp_ctg.is_some()),
                ("--p-convex-comp", cli.p_convex_comp.is_some()),
                ("--p-mingen-comp", cli.p_mingen_comp.is_some()),
                ("--db-prefixes-comp", cli.prefixes_comp.is_some()),
            ]
            .iter()
            .filter(|(_, set)| *set)
            .map(|(n, _)| *n)
            .collect()
        };
        if !inert.is_empty() {
            eprintln!(
                "[fmix] ERROR: {} can never take effect: --db-mode {} with no --p-mix overlay means {}-DB rounds never happen. Drop the flag, or arm the overlay with --p-mix.",
                inert.join(", "),
                args.db_mode,
                if comp_only { "MIX" } else { "COMP" }
            );
            std::process::exit(2);
        }
    }

    // Resolve the curated store BEFORE any banner mentions it, so the
    // printouts below describe what will actually run.
    if let Some(d) = &args.frozen_db_dir {
        unsafe { std::env::set_var("FROZEN_DB_DIR", d) };
    }
    if let Some(d) = &args.frozen_curated_dir {
        unsafe { std::env::set_var("FROZEN_CURATED_DIR", d) };
    }
    if args.curated && std::env::var("FROZEN_CURATED_DIR").is_err() {
        assert!(
            !given("curated"),
            "--curated needs FROZEN_CURATED_DIR; refusing to run, because \
             degrading silently to regular-only would look like a measurement"
        );
        // curated is only a DEFAULT here, so degrade -- but say so loudly.
        println!(
            "[fmix] WARNING: curated-first is the shipped default but FROZEN_CURATED_DIR is \
             unset -> curated OFF for this run (regular store only). Export FROZEN_CURATED_DIR \
             (or pass --frozen-curated-dir); pass --curated explicitly to make this an error."
        );
        args.curated = false;
    }
    if args.curated {
        assert!(
            !args.no_db_verify,
            "--curated with --no-db-verify is refused: the curated store has already been \
             observed returning a non-equivalent replacement (forward/reverse key confusion), \
             and the per-splice check is what caught it"
        );
    }

    assert!(
        !(args.split && !args.profile.is_empty()),
        "--split and --profile both steer the round; run the split stage as its own invocation"
    );
    // Parse the size profile and enforce single size authority.
    let profile: Option<([f64; 3], [f64; 2])> = if args.profile.is_empty() {
        None
    } else {
        let v: Vec<f64> = args
            .profile
            .split(',')
            .map(|x| {
                x.trim()
                    .parse()
                    .expect("--profile wants N0,N1,N2,R1,R2 reals")
            })
            .collect();
        assert_eq!(
            v.len(),
            5,
            "--profile wants exactly 5 comma-separated values N0,N1,N2,R1,R2"
        );
        // N2 may be given either as an ABSOLUTE effective-work mark (N2 >= N1)
        // or, as the original spec phrased it ("or N3 moves per gate"), as the
        // compression leg's BUDGET. A value below N1 can only mean the latter,
        // so read it that way and say so rather than rejecting the profile.
        let mut n = [v[0], v[1], v[2]];
        if n[2] < n[1] {
            let budget = n[2];
            n[2] = n[1] + budget;
            println!(
                "[fmix] profile: N2={budget} < N1={} read as the COMPRESSION BUDGET -> absolute end mark {}",
                n[1], n[2]
            );
        }
        let r = [v[3], v[4]];
        assert!(
            n[0] > 0.0 && n[1] >= n[0] && n[2] >= n[1],
            "--profile needs 0 < N0 <= N1 and a positive compression leg"
        );
        // R1 = 1 (no expand leg, pure hold) and R2 < 1 (compress below the
        // input size) are both valid schedules: prof_target is plain linear
        // interpolation with no (R1-1) divisions, and the controller tracks
        // err against S* symmetrically. The compress leg below x1 is only as
        // feasible as the material allows — best-effort, same as any leg.
        assert!(
            r[0] >= 1.0 && r[1] > 0.0 && r[0] >= r[1],
            "--profile needs R1 >= 1 and 0 < R2 <= R1"
        );
        assert!(
            args.target_size.is_none() && args.size_hi == 0 && args.size_lo == 0,
            "--profile is the sole size authority: remove --target-size / --size-hi / --size-lo (make up your mind)"
        );
        assert!(
            args.p_mix < 0.0,
            "--profile drives p_mix; do not also pass --p-mix"
        );
        Some((n, r))
    };

    // A resume carries its own circuit, so there is nothing to read here.
    let (gates, file_wires): (Vec<XGate>, usize) = match (&args.resume, &args.input) {
        (Some(_), _) => (Vec::new(), 0),
        (None, Some(path)) => match args.input_format.as_str() {
            "mpmct1" => format::read_mpmct(path).expect("read mpmct1 circuit"),
            "g57" => {
                let g = format::read_g57_file(path).expect("read g57 circuit");
                let w = max_wire(&g) as usize + 1;
                (g, w)
            }
            other => panic!("unknown --input-format {other}"),
        },
        (None, None) => unreachable!("clap requires --input unless --resume"),
    };
    let num_wires = file_wires.max(max_wire(&gates) as usize + 1);
    let input_len = gates.len();
    let comp0 = gates.iter().filter(|g| g.comp).count();
    let target = args.target_size.unwrap_or(input_len);
    if args.resume.is_none() {
        println!(
            "[fmix] input: {} gates ({} g57 fossils), {} wires; k_max={} split_damp={} split_base={} dir_p={} dir_q={} target={} temp={} moves={} seed={}",
            input_len,
            comp0,
            num_wires,
            args.k_max,
            args.split_damp,
            args.split_base,
            args.dir_p,
            args.dir_q,
            target,
            args.temp.unwrap_or((target as f64 / 100.0).max(64.0)),
            args.moves,
            args.seed
        );
    }
    if args.p_pair > 0.0 {
        println!(
            "[fmix] pair geometry ON: p_pair={} scan_cap={} pick={} (far-pair fusion, docs/NONLOCAL_PHASE_A.md)",
            args.p_pair,
            args.pair_scan_cap,
            if args.pair_pick_uniform {
                "uniform"
            } else {
                "far"
            }
        );
    }
    if args.p_bridge > 0.0 {
        println!(
            "[fmix] bridge fusion ON: p_bridge={} span=[{},{}] max_colliders={} — wake corrections are non-g57 (polf > 0 expected; docs/NONLOCAL_PHASE_A.md)",
            args.p_bridge, args.bridge_min_span, args.bridge_max_span, args.bridge_max_colliders
        );
    }
    if args.p_twist > 0.0 {
        println!(
            "[fmix] first-class twist rounds ON: p_twist={} (w-twist-* weights serve as type ratios)",
            args.p_twist
        );
    }
    if args.curated && args.curated_in_comp {
        println!(
            "[fmix] curated-in-comp ON: COMPRESSION probes the curated store too (the size rule keeps only curated spellings shorter than the window, i.e. the shorter halves of identity splits)"
        );
    }
    if args.curated_exhaust {
        if args.curated {
            println!(
                "[fmix] curated-exhaust ON: the prefix descent runs CURATED-ONLY over every window length, and falls back to the regular store only if that whole pass missed ({})",
                if args.curated_in_comp {
                    "MIX and COMP alike"
                } else {
                    "expansion only; compression stays regular-only"
                }
            );
        } else if given("curated_exhaust") {
            println!(
                "[fmix] WARNING: --curated-exhaust without a curated store -- it has no effect"
            );
        }
    }
    if args.shuffle_rate > 0.0 {
        println!(
            "[fmix] global re-randomisation ON: shuffle_rate={} (per-round p = {}/|circuit|, i.e. one expected whole-circuit reshuffle per |circuit|/{} rounds; semantics- and size-preserving)",
            args.shuffle_rate, args.shuffle_rate, args.shuffle_rate
        );
    } else {
        println!("[fmix] global re-randomisation OFF (--shuffle-rate 0)");
    }
    if args.p_twist > 0.0 {
        println!(
            "[fmix] twists ON (swap family): p_twist={} twist_min_len={} twist_neg_p={} -- each swapped wire negated w.p. twist_neg_p (0.5 => swap 1/4, negate-one 1/2, negate-both 1/4; 0 => pure swap, no polarity flips) (w_twist_* retired/ignored)",
            args.p_twist, args.twist_min_len, args.twist_neg_p
        );
    }
    if args.twist_g57 {
        // Force-build the engine now so its cost is paid (and printed) at
        // startup rather than silently inside the first twist round.
        let eng = crate::engine::moves::swap_words::engine();
        println!(
            "[fmix] twist-g57 ON: brackets are adaptive all-g57 words (pure swap; twist_neg_p ignored), inserted gates take the birth-advance; engine ball {} perms, built in {} ms",
            eng.back_len, eng.build_ms
        );
    }
    if args.anc_samples > 0 {
        println!(
            "[fmix] SAMPLED ancestry ON: tracing {} input gates (sample_seed={}); reports per-tracer descendant count, positional coverage and entropy. Scales to any input size; anc=/ancspan= stay 0 (see the tracers line).",
            args.anc_samples, args.anc_sample_seed
        );
    }
    if args.p_mix >= 0.0 {
        println!(
            "[fmix] mode overlay ON (slot 0): p_mix={} -> MIX-DB w.p. p_mix else COMP-DB, per round (each mode's settled knobs are on the 'DB effective per mode' line below)",
            args.p_mix
        );
    }
    if args.p_comp > 0.0 || args.p_db > 0.0 || args.p_any > 0.0 {
        println!(
            "[fmix] DB ON: p_db(slot2)={} db_mode={} p_comp(contract)={} p_any(expand)={} w_window={} w_pool={} verify={} curated={} (FROZEN_DB_DIR required)",
            args.p_db,
            args.db_mode,
            args.p_comp,
            args.p_any,
            args.w_window,
            args.w_pool,
            !args.no_db_verify,
            args.curated
        );
        // Print what each mode will ACTUALLY use, resolved by the SAME code the
        // mixer runs (MixParams::db_knobs) rather than a second copy of the
        // fall-through rules -- the old banner re-derived them itself and could
        // therefore drift from reality, which is how the COMP shadowing hid.
        let probe = MixParams {
            s_db: args.s_db,
            db_min_window: args.db_min_window,
            s_db_ctg: args.s_db_ctg,
            s_db_comp: args.s_db_comp,
            s_db_comp_ctg: args.s_db_comp_ctg,
            p_convex: args.p_convex,
            p_convex_comp: args.p_convex_comp,
            p_mingen: args.p_mingen,
            p_mingen_comp: args.p_mingen_comp,
            db_prefixes: args.db_prefixes,
            db_prefixes_mix: args.db_prefixes_mix,
            db_prefixes_comp: args.db_prefixes_comp,
            ..MixParams::default()
        };
        let km = probe.db_knobs(DbMode::Mix);
        let kc = probe.db_knobs(DbMode::Compressing);
        println!(
            "[fmix] DB effective per mode: MIX p_convex={} s_db(cvx)={} s_db(ctg)={} p_mingen={} descent={} | COMP p_convex={} s_db(cvx)={} s_db(ctg)={} p_mingen={} descent={}",
            km.p_convex,
            km.s_db_cvx,
            km.s_db_ctg,
            km.p_mingen,
            km.prefixes,
            kc.p_convex,
            kc.s_db_cvx,
            kc.s_db_ctg,
            kc.p_mingen,
            kc.prefixes
        );
        if args.p_comp_g57 > 0.0 {
            println!(
                "[fmix] g57-only COMP attempts: p={} starting at s_db_g57={}",
                args.p_comp_g57, args.s_db_g57
            );
        }
    }
    if args.curated {
        println!(
            "[fmix] curated ON: ordinary expansion probes the CURATED store only (forward key, any size); compression {}",
            if args.curated_in_comp {
                "probes it too (--curated-in-comp)"
            } else {
                "stays regular-only"
            }
        );
    }
    if args.db_advance {
        println!(
            "[fmix] db-advance ON: splice products take the ballistic birth-advance (dir_q={})",
            args.dir_q
        );
    }
    if args.gen_target > 0 {
        println!(
            "[fmix] generation targeting ON: gen_target={} p_mingen={} pool_k={} gen_rescan={} gen_stop_frac={} twist_cov_stop={} split_rule={}",
            args.gen_target,
            args.p_mingen,
            args.pool_k,
            args.gen_rescan,
            args.gen_stop_frac,
            args.twist_cov_stop,
            if args.gen_split_inherit {
                "inherit"
            } else {
                "ratchet(+1)"
            }
        );
        if args.canary_theta > 0.0 {
            println!(
                "[fmix] canary ON: theta={} window={} qualifying rounds",
                args.canary_theta, args.canary_window
            );
        }
    }

    assert!(
        args.db_max_degree == 0 || args.db_max_degree <= 11,
        "--db-max-degree {} unusable: the degree probe caps subcube dimension \
         at 12, so caps above 11 would silently disable the guard",
        args.db_max_degree
    );
    let params = MixParams {
        k_max: args.k_max,
        split_damp: args.split_damp,
        split_base: args.split_base,
        dir_p: args.dir_p,
        dir_q: args.dir_q,
        target_size: target,
        mix_pay_random: args.mix_pay_random,
        prof_n: profile.map(|(n, _)| n).unwrap_or([0.0; 3]),
        prof_r: profile.map(|(_, r)| r).unwrap_or([0.0; 2]),
        prof_cadence_eff: args.prof_cadence_eff,
        prof_deadband: args.prof_deadband,
        prof_dp_max: args.prof_dp_max,
        prof_ewma: args.prof_ewma,
        prof_ki: args.prof_ki,
        temp: args.temp.unwrap_or(0.0),
        moves: args.moves,
        merge_reach: args.merge_reach,
        journal_len: args.journal_len,
        // Piecewise-round knobs are set per piece by the driver, never here.
        eff_budget: 0.0,
        span_norm: 0,
        rank_base: 0,
        rank_total: 0,
        undo_frac: args.undo_frac,
        tabu_moves: args.tabu_moves,
        w_cross: args.w_cross,
        w_fresh: args.w_fresh,
        w_unsub: args.w_unsub,
        w_insert: args.w_insert,
        w_twist_neg: args.w_twist_neg,
        w_twist_swap: args.w_twist_swap,
        w_twist_cnot: args.w_twist_cnot,
        twist_neg_p: args.twist_neg_p,
        twist_g57: args.twist_g57,
        twist_min_len: args.twist_min_len,
        p_comp: args.p_comp,
        p_any: args.p_any,
        db_mode: crate::stages::db_mixing::replacement::DbMode::parse(&args.db_mode)
            .unwrap_or_else(|| panic!("unknown --db-mode {} (mix|comp|any|stable|stable-grow|stable-ledger|same|band-ledger)", args.db_mode)),
        p_db: args.p_db,
        p_twist: args.p_twist,
        shuffle_rate: args.shuffle_rate,
        curated_exhaust: args.curated_exhaust,
        curated_in_comp: args.curated_in_comp,
        s_db: args.s_db,
        db_min_window: args.db_min_window,
        w_window: args.w_window,
        w_pool: args.w_pool,
        p_convex: args.p_convex,
        db_convex_p: args.db_convex_p,
        db_verify: !args.no_db_verify,
        db_dry_run: args.db_dry_run,
        db_max_degree: args.db_max_degree,
        db_degree_probes: args.db_degree_probes,
        db_max_span: args.db_max_span,
        db_wire_terms: args.db_wire_terms,
        db_total_terms: args.db_total_terms,
        db_prefixes: args.db_prefixes,
        db_advance: args.db_advance,
        p_pair: args.p_pair,
        pair_scan_cap: args.pair_scan_cap,
        pair_pick_uniform: args.pair_pick_uniform,
        p_bridge: args.p_bridge,
        bridge_min_span: args.bridge_min_span,
        bridge_max_span: args.bridge_max_span,
        bridge_max_colliders: args.bridge_max_colliders,
        p_mix: args.p_mix,
        s_db_comp: args.s_db_comp,
        p_convex_comp: args.p_convex_comp,
        s_db_ctg: args.s_db_ctg,
        s_db_comp_ctg: args.s_db_comp_ctg,
        db_prefixes_mix: args.db_prefixes_mix,
        db_prefixes_comp: args.db_prefixes_comp,
        p_mingen_comp: args.p_mingen_comp,
        curated: args.curated,
        ancestors: args.ancestors,
        anc_samples: args.anc_samples,
        anc_sample_seed: args.anc_sample_seed,
        p_comp_g57: args.p_comp_g57,
        contract_ceiling: args.contract_ceiling,
        size_hi: args.size_hi,
        size_lo: args.size_lo,
        comp_release_eps: args.comp_release_eps,
        comp_release_window: args.comp_release_window,
        s_db_g57: args.s_db_g57,
        gen_target: args.gen_target,
        p_mingen: args.p_mingen,
        pool_k: args.pool_k,
        canary_theta: args.canary_theta,
        canary_window: args.canary_window,
        litter_ban: args.litter_ban,
        litter_samples: args.litter_samples,
        twist_place_tries: args.twist_place_tries,
        gen_rescan: args.gen_rescan,
        gen_split_inherit: args.gen_split_inherit,
        gen_median_low: args.gen_median_low,
        gen_stop_frac: args.gen_stop_frac,
        twist_cov_stop: args.twist_cov_stop,
        gen_snap_every: args.gen_snap_every,
        snap_every_moves: args.snap_every_moves,
        split: args.split,
        split_stop: args.split_stop,
        p_split_twist: args.p_split_twist,
        p_join: args.p_join,
        split_fail_limit: args.split_fail_limit,
        split_canaries: args.split_canaries,
        split_reach_k: args.split_reach_k,
        p_mincross: args.p_mincross,
        cross_pool_k: args.cross_pool_k,
        cross_rescan: args.cross_rescan,
        verify_every: args.verify_every,
        report_every: args.report_every,
        local_verify: !args.no_local_verify,
        seed: args.seed,
    };
    assert!(
        args.anc_in.is_none() || args.resume.is_none(),
        "--anc-in seeds a FRESH run's ancestry; a --resume carries its own (drop one)"
    );
    assert!(
        args.anc_in.is_none() || (args.anc_samples == 0 && !args.ancestors),
        "--anc-in defines the ancestry universe; drop --ancestors / --anc-samples"
    );
    assert!(
        args.anc_out.is_none()
            || args.ancestors
            || args.anc_samples > 0
            || args.anc_in.is_some()
            || args.resume.is_some(),
        "--anc-out needs ancestry armed (--ancestors, --anc-samples or --anc-in): refusing now \
         rather than at the end of the run"
    );
    // Piecewise-parallel rounds (fixed or automatic): refuse the combinations the
    // driver does not carry, and cuts too short to mix, BEFORE the store is
    // opened. A silent clamp of --parallel-pieces is never acceptable: the value is a
    // recipe setting the GSS_MIX manifest locks.
    if args.piecewise_enabled() {
        let refuse = |why: &str| {
            let mode = match args.min_block_size {
                Some(size) => format!("--parallel-target-piece-gates {size}"),
                None => format!("--parallel-pieces {}", args.pieces),
            };
            eprintln!("[fmix] ERROR: {mode} {why}");
            std::process::exit(2);
        };
        if args.resume.is_some() {
            refuse(
                "cannot resume a state file: the crossing walk stays serial, drop --parallel-pieces/--parallel-target-piece-gates",
            );
        }
        if matches!(params.db_mode, DbMode::StableLedger | DbMode::BandLedger) {
            refuse(
                "does not carry stable-ledger/band-ledger size accounting between rounds: use serial mode or another --db-mode",
            );
        }
        if args.anc_in.is_some() || args.anc_out.is_some() || args.ancestors || args.anc_samples > 0
        {
            refuse(
                "does not carry ancestry yet: drop --ancestors/--anc-samples/--anc-in/--anc-out",
            );
        }
        if args.db_record.is_some() {
            refuse("cannot write one --db-record from many pieces");
        }
        if args.gen_snap_every > 0 || args.snap_every_moves > 0 {
            refuse("takes no mid-run snapshots: drop --gen-snap-every/--snap-every-moves");
        }
        if args.split && !args.split_stop {
            refuse("with --split needs --split-stop (the stage boundary ends the run)");
        }
        if !(0.0..0.25).contains(&args.piece_jitter) {
            refuse("needs --piece-jitter in [0, 0.25)");
        }
        if args.min_block_size.is_none() {
            let shortest =
                ((input_len / args.pieces) as f64 * (0.5 - args.piece_jitter)).floor() as usize;
            if shortest < args.piece_min_len.max(2) {
                eprintln!(
                    "[fmix] ERROR: --parallel-pieces {} on {} gates makes pieces as short as {} gates (< --piece-min-len {}): use fewer pieces or lower the floor",
                    args.pieces, input_len, shortest, args.piece_min_len
                );
                std::process::exit(2);
            }
        }
    }

    if args.quality.qc {
        if let Err(error) = args
            .quality
            .config()
            .detector
            .validate_input(qc_reference.as_deref().unwrap_or(&gates), num_wires)
        {
            eprintln!("[fmix] invalid QC reference or storage budget: {error}");
            std::process::exit(2);
        }
    }
    let make_mixer = |gates, num_wires, params| {
        if args.quality.qc {
            // QC has its own DB channel, even when all random DB move coins
            // are zero (e.g. --moves 0 to repair an existing db_mixing artifact).
            Mixer::new_with_db(
                gates,
                num_wires,
                params,
                crate::database::frozen::FrozenDb::from_env(),
            )
        } else {
            Mixer::new(gates, num_wires, params)
        }
    };
    let mut mixer = match &args.resume {
        Some(path) => {
            let db = if params.p_comp > 0.0 || params.p_db > 0.0 || params.p_any > 0.0 {
                crate::database::frozen::FrozenDb::from_env()
            } else {
                crate::database::frozen::FrozenDb::empty()
            };
            let mx = Mixer::resume_state(path, params, db).expect("resume from state file");
            println!(
                "[fmix] RESUMED from {path}: {} gates at move {}, verifying against the original",
                mx.arena.len(),
                mx.moves_done
            );
            mx
        }
        None => match &args.anc_in {
            Some(p) => {
                let sc = Mixer::read_anc_sidecar(p).expect("read ancestry sidecar");
                let (mode_s, sc_m, sc_k, sc_n) = (
                    if sc.sampled { "sampled" } else { "exact" },
                    sc.m,
                    sc.tracers.len(),
                    sc.sets.len(),
                );
                // Construct with ancestry OFF: the sidecar defines the
                // universe, and the constructor would otherwise pick its own
                // tracers against the wrong input population.
                let mut mx = make_mixer(
                    gates,
                    num_wires,
                    MixParams {
                        ancestors: false,
                        anc_samples: 0,
                        ..params
                    },
                );
                mx.import_ancestry(sc);
                println!(
                    "[fmix] ancestry IMPORTED from {p}: {mode_s} mode, universe m={sc_m}, K={sc_k}, {sc_n} per-gate sets; anc meters continue the PRODUCING run's clock"
                );
                mx
            }
            None => make_mixer(gates, num_wires, params),
        },
    };
    // External litter assignment (e.g. the SGDB substitution's sidecar: every
    // replaced gate's block is one litter, so --litter-ban covers the INITIAL
    // replacements too, not only the walk's own splices).
    if let Some(p) = &args.litter_in {
        let ids: Vec<u64> = std::fs::read_to_string(p)
            .unwrap_or_else(|e| panic!("read --litter-in {p}: {e}"))
            .lines()
            .skip(1) // header "litter1 N"
            .filter(|l| !l.trim().is_empty())
            .map(|l| l.trim().parse().expect("bad litter id"))
            .collect();
        mixer.load_litters(&ids);
    }
    // A resumed run has no input file, so the summary's "before" figures must
    // come from the resumed circuit or it reports 0 -> N and reads as if the
    // chain started from nothing.
    let (input_len, comp0) = match &args.resume {
        Some(_) => (mixer.arena.len(), mixer.remaining_g57()),
        None => (input_len, comp0),
    };
    if let Some(path) = &args.db_record {
        mixer.enable_db_record(path);
        println!("[fmix] recording DB attempts to {path}");
    }
    if args.gen_snap_every > 0 || args.snap_every_moves > 0 {
        let base = args
            .output
            .clone()
            .unwrap_or_else(|| "fmix_out".to_string());
        if args.gen_snap_every > 0 {
            println!(
                "[fmix] gen snapshots armed: every {} circuit generations -> {base}.gen<m>.mpmct1",
                args.gen_snap_every
            );
        }
        if args.snap_every_moves > 0 {
            println!(
                "[fmix] move snapshots armed: every {} moves -> {base}.mv<m>.mpmct1",
                args.snap_every_moves
            );
        }
        mixer.set_gen_snap_base(base);
    }

    let stop = std::env::var("FMIX_STOP_FLAG")
        .ok()
        .filter(|s| !s.is_empty());
    let dump = std::env::var("FMIX_DUMP_FLAG")
        .ok()
        .filter(|s| !s.is_empty());
    if stop.is_some() || dump.is_some() {
        let dump_out = std::env::var("FMIX_DUMP_OUT")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                args.output
                    .as_deref()
                    .map(|o| format!("{o}.snapshot.txt"))
                    .unwrap_or_else(|| "fmix_snapshot.txt".to_string())
            });
        if let Some(f) = &stop {
            println!("[fmix] stop flag armed: touch {f} -> clean finish");
        }
        if let Some(f) = &dump {
            println!("[fmix] dump signal armed: touch {f} -> snapshot to {dump_out}");
        }
        mixer.enable_flags(stop, dump, dump_out);
    }

    let t0 = std::time::Instant::now();
    let stop_reason = if args.piecewise_enabled() {
        let cfg = PieceCfg {
            pieces: args.pieces,
            min_block_size: args.min_block_size,
            jitter: args.piece_jitter,
            round_eff: args.piece_round_eff,
            threads: args.piece_threads,
            split_rounds_max: args.split_rounds_max,
            verbose: args.piece_verbose,
            sequential: false,
        };
        run_piecewise(&mut mixer, &cfg)
    } else {
        mixer.run()
    };
    let secs = t0.elapsed().as_secs_f64();
    mixer.report();
    // Canary dump for runs that stopped BEFORE the stage boundary (budget,
    // stop flag); a no-op when the boundary already reported them.
    mixer.split_tap_summary();
    {
        use std::sync::atomic::Ordering;
        let rl = crate::circuit::CANON_RULE_L_SKIPS.load(Ordering::Relaxed);
        let mc = crate::circuit::CANON_CAP_SKIPS.load(Ordering::Relaxed);
        let rlb = crate::circuit::CANON4_RULE_L_BRANCHES.load(Ordering::Relaxed);
        let rlc = crate::circuit::CANON4_RULE_L_CALLS.load(Ordering::Relaxed);
        println!(
            "[fmix] canon caps: rule_l_skips={rl} monomial_skips={mc} rule_l_calls={rlc} rule_l_branches={rlb}"
        );
    }
    println!(
        "[fmix] chain done in {:.1}s: {} ({} -> {} gates, {} -> {} g57 fossils)",
        secs,
        match stop_reason {
            MixStop::MovesBudget => "moves budget spent",
            MixStop::StopFlag => "stop flag",
            MixStop::DoseReached => "dose reached (gen + twist coverage targets met)",
            MixStop::CanaryFired => "canary fired (pool is unspellable by the store)",
            MixStop::ProfileDone => "profile complete (size schedule finished)",
            MixStop::SplitDone => "split stage complete (stopped at the stage boundary)",
            MixStop::RoundDone => "piece round budget spent",
            MixStop::CircuitEmpty => "circuit empty (no gates remain to mix)",
        },
        input_len,
        mixer.arena.len(),
        comp0,
        mixer.remaining_g57(),
    );

    if !args.skip_final_float {
        let t1 = std::time::Instant::now();
        let (moved, disp) = mixer.final_float();
        mixer.global_check();
        println!(
            "[fmix] final float: {} gates moved, {} total displacement, {:.1}s (verified)",
            moved,
            disp,
            t1.elapsed().as_secs_f64()
        );
    }

    if args.quality.qc {
        let report = crate::stages::db_mixing::leakage_repair::run_quality_control(
            &mut mixer,
            qc_reference.as_deref(),
            &args.quality.config(),
        )
        .unwrap_or_else(|error| {
            eprintln!("[fmix] QC failed: {error}");
            std::process::exit(2);
        });
        mixer.global_check();
        let text = format!(
            "reference: {}\n{}",
            args.quality
                .qc_reference
                .as_deref()
                .unwrap_or("mixer input (same input/wire coordinates)"),
            report.to_text()
        );
        println!(
            "[fmix] QC: {} repairs, {} attempts, budget_exhausted={}",
            report.repairs,
            report.events.len(),
            report.budget_exhausted
        );
        if let Some(path) = args
            .quality
            .qc_report
            .clone()
            .or_else(|| args.output.as_ref().map(|p| format!("{p}.qc.txt")))
        {
            std::fs::write(&path, text).expect("write QC report");
            println!("[fmix] wrote QC report to {path}");
        } else {
            println!("{text}");
        }
    }

    if let Some(sp) = &args.state_out {
        mixer.save_state(sp).expect("write state file");
        println!("[fmix] wrote resume state to {sp}");
    }
    if let Some(out) = &args.output {
        let final_gates = mixer.arena.to_vec();
        format::write_mpmct(out, &final_gates, num_wires).expect("write output");
        println!("[fmix] wrote {} gates to {}", final_gates.len(), out);
    } else {
        println!("[fmix] no --output given; result discarded after verification");
    }

    if let Some(path) = &args.origins_out {
        let origins = mixer.origins_in_order();
        let mut s = String::with_capacity(origins.len() * 8);
        for o in origins {
            s.push_str(&format!("{o}\n"));
        }
        std::fs::write(path, s).expect("write origins");
        println!("[fmix] wrote origins sidecar to {path} (synthetic = {ORIGIN_SYNTH})");
    }

    if let Some(path) = &args.gens_out {
        let gens = mixer.gens_in_order();
        let mut s = String::with_capacity(gens.len() * 4);
        for g in gens {
            s.push_str(&format!("{g}\n"));
        }
        std::fs::write(path, s).expect("write gens");
        println!("[fmix] wrote gens sidecar to {path} (born-random = {GEN_FRESH})");
    }

    if let Some(path) = &args.anc_out {
        // After the final float, so line i is gate i of the written circuit.
        mixer
            .write_anc_sidecar(path)
            .expect("write ancestry sidecar");
        println!(
            "[fmix] wrote ancestry sidecar to {path} ({} per-gate sets; import with --anc-in)",
            mixer.arena.len()
        );
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/programs/fmix/tests.rs"]
mod tests;
