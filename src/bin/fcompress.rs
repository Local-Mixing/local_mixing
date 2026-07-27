// Final compression pass for fmix/fsplit output (see postmix/compress.rs):
// gather same-target gates that can float together, XOR-reduce each group in
// ESOP/ANF space, re-emit surviving cubes as consecutive XGates, iterate to a
// fixed point. Deterministic and attacker-computable, so it cannot weaken the
// hiding; the compressed size is the honest "effective size" of the artifact.
//
// Optional dead-cone pruning for gadgetized circuits, where equality is
// required only on designated output wires: --live-wires upper-half (or
// lower-half, or an explicit list "0-255,300"). Default all wires live.
//
// Example:
//   fcompress --input mixed_fmix.txt --output mixed_fcmp.txt
//   fcompress --input gadget.txt --output gadget_fcmp.txt --live-wires upper-half
use clap::Parser;
use local_mixing::postmix::compress::{
    CompressParams, RecoverySummary, compress_traced_with_sources, lits_of,
};
use local_mixing::postmix::db_compress::{
    DbCompressParams, DbCompressReport, compress_frozen_contiguous_traced,
};
use local_mixing::postmix::format;
use local_mixing::postmix::lineage::{FragmentCoverage, ResolvedCoverage};
use local_mixing::postmix::reassemble::{ReassemblyStats, analyze_barrier_free, is_structural_g57};
use local_mixing::postmix::source::{self, SourceClassCounts, UNKNOWN_SOURCE};
use local_mixing::postmix::xgate::{XGate, eval_lanes, max_wire};
use local_mixing::postmix::xpoly::XPolyBudget;
use local_mixing::replace::frozen::FrozenDb;
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
    /// fsource1 sidecar emitted by fsplit, aligned with --input.
    #[arg(long, requires = "parent_g57")]
    sources_in: Option<String>,
    /// Original pre-split g57 tape indexed by --sources-in parent ids.
    #[arg(long, requires = "sources_in")]
    parent_g57: Option<String>,
    /// Optional fsource1 sidecar aligned with the final compressed output.
    #[arg(long, requires = "sources_in")]
    sources_out: Option<String>,
    /// Wires whose final value must be preserved: all | upper-half |
    /// lower-half | explicit list like "0-255,300,510"
    #[arg(long, default_value = "all")]
    live_wires: String,
    #[arg(long, default_value_t = 256)]
    max_iters: usize,
    #[arg(long, default_value_t = 64)]
    group_cap: usize,
    #[arg(long, default_value_t = 24)]
    anf_support_cap: usize,
    /// Rounds of the 64-lane sampled global check against the input
    #[arg(long, default_value_t = 64)]
    verify_rounds: usize,
    /// Disable the per-group exhaustive/sampled verification
    #[arg(long, default_value_t = false)]
    no_local_verify: bool,
    /// Also run the legacy replacement-table compressor through the new
    /// heterogeneous polynomial/key path. Zero disables database compression.
    #[arg(long, default_value_t = 0)]
    db_trials: usize,
    #[arg(long, default_value_t = 2)]
    db_min_window: usize,
    #[arg(long, default_value_t = 12)]
    db_max_window: usize,
    /// Only canonicalize/probe the forward window function.
    #[arg(long, default_value_t = false)]
    db_forward_only: bool,
    /// Maximum raw monomial products in one heterogeneous polynomial multiply.
    #[arg(long, default_value_t = 1 << 20)]
    poly_max_mul_terms: usize,
    /// Maximum reduced terms in one live/intermediate polynomial.
    #[arg(long, default_value_t = 1 << 18)]
    poly_max_terms: usize,
    /// Maximum reduced terms across every wire polynomial in a window.
    #[arg(long, default_value_t = 1 << 20)]
    poly_max_total_terms: usize,
    /// Optional mpmct1 snapshot after ESOP/reassembly but before DB compression.
    #[arg(long)]
    esop_output: Option<String>,
    #[arg(long, default_value_t = 0)]
    seed: u64,
}

fn percent(numer: u64, denom: usize) -> f64 {
    if denom == 0 {
        0.0
    } else {
        100.0 * numer as f64 / denom as f64
    }
}

fn width_bucket(coverage: &FragmentCoverage, width: usize) -> u64 {
    coverage.by_initial_width.get(width).copied().unwrap_or(0)
}

fn print_coverage_row(
    stage: &str,
    route: &str,
    attribution: &str,
    coverage: &FragmentCoverage,
    denominator: usize,
) {
    let over_15: u64 = coverage.by_initial_width.iter().skip(16).sum();
    let mut fields = vec![
        stage.to_string(),
        route.to_string(),
        attribution.to_string(),
        coverage.total.to_string(),
        format!("{:.9}", percent(coverage.total, denominator)),
    ];
    fields.extend((0..=15).map(|width| width_bucket(coverage, width).to_string()));
    fields.push(over_15.to_string());
    println!("lineage_recovery_csv,{}", fields.join(","));
}

fn print_resolved(stage: &str, route: &str, coverage: &ResolvedCoverage, denominator: usize) {
    print_coverage_row(stage, route, "exact", &coverage.exact, denominator);
    print_coverage_row(
        stage,
        route,
        "group_attributed",
        &coverage.inclusive,
        denominator,
    );
}

fn print_recovery_summary(stage: &str, summary: &RecoverySummary) {
    let denominator = summary.input_plain_fragments;
    for (route, coverage) in [
        ("direct_pass1", &summary.direct_pass1),
        ("direct_later", &summary.direct_later),
        ("anf", &summary.anf),
        ("database", &summary.database),
        ("ever", &summary.ever),
        ("final_structural_g57", &summary.final_structural_g57),
    ] {
        print_resolved(stage, route, coverage, denominator);
    }
    println!(
        "[fcompress] lineage {stage}: exact ever={}/{} ({:.6}%), inclusive ever={}/{} ({:.6}%), exact final={}/{} ({:.6}%), inclusive final={}/{} ({:.6}%)",
        summary.ever.exact.total,
        denominator,
        percent(summary.ever.exact.total, denominator),
        summary.ever.inclusive.total,
        denominator,
        percent(summary.ever.inclusive.total, denominator),
        summary.final_structural_g57.exact.total,
        denominator,
        percent(summary.final_structural_g57.exact.total, denominator),
        summary.final_structural_g57.inclusive.total,
        denominator,
        percent(summary.final_structural_g57.inclusive.total, denominator),
    );
}

fn print_input_widths(summary: &RecoverySummary) {
    for (width, &count) in summary.input_width_histogram.iter().enumerate() {
        println!(
            "lineage_input_width_csv,{width},{count},{:.9}",
            percent(count, summary.input_plain_fragments),
        );
    }
}

fn other_complemented(gates: &[XGate]) -> usize {
    gates
        .iter()
        .filter(|g| g.comp && !is_structural_g57(g))
        .count()
}

fn classify_structural_sources(
    gates: &[XGate],
    marks: &[u32],
    parents: &[XGate],
) -> SourceClassCounts {
    assert_eq!(gates.len(), marks.len());
    let mut counts = SourceClassCounts::default();
    for (gate, &mark) in gates.iter().zip(marks) {
        if is_structural_g57(gate) {
            counts.record(mark, gate, parents);
        }
    }
    counts
}

fn print_source_counts(stage: &str, route: &str, counts: SourceClassCounts) {
    let known = counts.known_total();
    let new = counts.new_total();
    let new_percent = if known == 0 {
        0.0
    } else {
        100.0 * new as f64 / known as f64
    };
    let mixed_percent = if known == 0 {
        0.0
    } else {
        100.0 * counts.new_mixed_parents as f64 / known as f64
    };
    println!(
        "source_g57_csv,{stage},{route},{},{},{},{},{},{},{:.9},{:.9}",
        known + counts.unknown,
        counts.returned_to_parent,
        counts.new_same_parent,
        counts.new_mixed_parents,
        new,
        counts.unknown,
        new_percent,
        mixed_percent,
    );
}

fn print_reassembly(label: &str, stats: &ReassemblyStats, other_comp: usize) {
    println!(
        "[fcompress] reassembly {label}: total={} plain={} structural_g57={} other_comp={} singleton_1cc={} homogeneous_2cc={} pairable_pairs={} pairable_fragments={} plain_coverage={:.6}% candidate_coverage={:.6}% all_gate_coverage={:.6}%",
        stats.total_gates,
        stats.plain_fragments,
        stats.structural_g57,
        other_comp,
        stats.singleton_1cc,
        stats.homogeneous_2cc,
        stats.reassemblable_pairs,
        stats.reassemblable_fragments,
        stats.percent_reassemblable_of_plain(),
        stats.percent_of_candidate_fragments_reassemblable(),
        stats.percent_of_all_gates_reassemblable(),
    );
    println!(
        "reassembly_summary_csv,{label},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:.9},{:.9}",
        stats.total_gates,
        stats.plain_fragments,
        stats.structural_g57,
        other_comp,
        stats.singleton_1cc,
        stats.homogeneous_2cc,
        stats.positive_1cc,
        stats.negative_1cc,
        stats.positive_2cc,
        stats.negative_2cc,
        stats.compatible_singleton_1cc,
        stats.compatible_homogeneous_2cc,
        stats.reassemblable_pairs,
        stats.reassemblable_fragments,
        stats.percent_reassemblable_of_plain(),
        stats.percent_of_candidate_fragments_reassemblable(),
        stats.percent_of_all_gates_reassemblable(),
    );
}

fn print_db_report(rep: &DbCompressReport, seconds: f64) {
    println!(
        "[fcompress-db] done in {:.3}s: gates {} -> {} ({:.6}%), lits {} -> {} ({:.6}%), trials={}/{} lookups={} hits={} shorter_windows={} replacements={} removed={} adjacent_cancelled={} poly_skips={} budget_skips={} heterogeneous_windows={} heterogeneous_hits={} heterogeneous_replacements={} attributed_g57_windows={} attributed_g57_outputs={} source_windows(single={},mixed={},unknown={})",
        seconds,
        rep.gates_in,
        rep.gates_out,
        100.0 * rep.gates_out as f64 / rep.gates_in.max(1) as f64,
        rep.lits_in,
        rep.lits_out,
        100.0 * rep.lits_out as f64 / rep.lits_in.max(1) as f64,
        rep.trials_attempted,
        rep.trials_requested,
        rep.lookups,
        rep.lookup_hits,
        rep.windows_with_shorter_candidate,
        rep.replacements,
        rep.gates_removed_by_db,
        rep.adjacent_gates_cancelled,
        rep.polynomial_skips,
        rep.budget_skips,
        rep.heterogeneous_windows_attempted,
        rep.heterogeneous_lookup_hits,
        rep.heterogeneous_replacements,
        rep.attributed_g57_replacement_windows,
        rep.attributed_structural_g57_outputs,
        rep.single_parent_replacement_windows,
        rep.mixed_parent_replacement_windows,
        rep.unknown_source_replacement_windows,
    );
    let fields = [
        rep.gates_in.to_string(),
        rep.gates_out.to_string(),
        rep.lits_in.to_string(),
        rep.lits_out.to_string(),
        rep.trials_requested.to_string(),
        rep.trials_attempted.to_string(),
        rep.heterogeneous_windows_attempted.to_string(),
        rep.all_g57_windows_attempted.to_string(),
        rep.canonical_directions_attempted.to_string(),
        rep.polynomial_skips.to_string(),
        rep.budget_skips.to_string(),
        rep.windows_without_key.to_string(),
        rep.lookups.to_string(),
        rep.lookup_hits.to_string(),
        rep.heterogeneous_lookup_hits.to_string(),
        rep.all_g57_lookup_hits.to_string(),
        rep.malformed_value_entries.to_string(),
        rep.candidates_decoded.to_string(),
        rep.shorter_candidates.to_string(),
        rep.windows_with_shorter_candidate.to_string(),
        rep.replacements.to_string(),
        rep.heterogeneous_replacements.to_string(),
        rep.all_g57_replacements.to_string(),
        rep.gates_removed_by_db.to_string(),
        rep.adjacent_gates_cancelled.to_string(),
        rep.attributed_g57_replacement_windows.to_string(),
        rep.attributed_structural_g57_outputs.to_string(),
        rep.single_parent_replacement_windows.to_string(),
        rep.mixed_parent_replacement_windows.to_string(),
        rep.unknown_source_replacement_windows.to_string(),
        format!("{seconds:.9}"),
    ];
    println!("db_compress_summary_csv,{}", fields.join(","));
}

fn verify_equivalent(
    original: &[XGate],
    candidate: &[XGate],
    wires: usize,
    live: Option<&[bool]>,
    rounds: usize,
    seed: u64,
    label: &str,
) {
    let mut rng = StdRng::seed_from_u64(seed);
    for round in 0..rounds {
        let before: Vec<u64> = (0..wires).map(|_| rng.random::<u64>()).collect();
        let mut after = before.clone();
        let mut before = before;
        eval_lanes(original.iter(), &mut before);
        eval_lanes(candidate.iter(), &mut after);
        for wire in 0..wires {
            if live.is_none_or(|mask| mask[wire]) {
                assert_eq!(
                    before[wire], after[wire],
                    "{label} global check failed on wire {wire} (round {round})"
                );
            }
        }
    }
    println!(
        "[fcompress] verified {label}: {} rounds x 64 lanes on {} live wires",
        rounds,
        live.map_or(wires, |mask| mask.iter().filter(|&&bit| bit).count()),
    );
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
    assert!(args.verify_rounds > 0, "--verify-rounds must be positive");
    assert!(args.max_iters > 0, "--max-iters must be positive");
    assert!(args.group_cap >= 2, "--group-cap must be at least 2");
    if args.db_trials > 0 {
        assert!(
            args.db_min_window >= 2 && args.db_min_window <= args.db_max_window,
            "database window bounds must satisfy 2 <= min <= max"
        );
        assert!(
            args.poly_max_mul_terms > 0 && args.poly_max_terms > 0 && args.poly_max_total_terms > 0,
            "polynomial budgets must be positive"
        );
    }
    let (gates, file_wires): (Vec<XGate>, usize) = match args.input_format.as_str() {
        "mpmct1" => format::read_mpmct(&args.input).expect("read mpmct1 circuit"),
        "g57" => {
            let g = format::read_g57_file(&args.input).expect("read g57 circuit");
            let w = max_wire(&g) as usize + 1;
            (g, w)
        }
        other => panic!("unknown --input-format {other}"),
    };
    let (source_marks, source_parents) =
        if let (Some(sidecar), Some(parent_path)) = (&args.sources_in, &args.parent_g57) {
            let (marks, declared_parents) =
                source::read_source_marks(sidecar, gates.len()).expect("read fsource1 sidecar");
            let parents = format::read_g57_file(parent_path).expect("read parent g57 circuit");
            assert_eq!(
                declared_parents,
                parents.len(),
                "fsource parent count does not match --parent-g57"
            );
            (marks, parents)
        } else {
            (vec![UNKNOWN_SOURCE; gates.len()], Vec::new())
        };
    let wires = file_wires.max(max_wire(&gates) as usize + 1);
    let live = parse_live(&args.live_wires, wires);
    let n_live = live
        .as_ref()
        .map_or(wires, |v| v.iter().filter(|&&b| b).count());
    println!(
        "[fcompress] input: {} gates ({} lits), {} wires ({} live); max_iters={} group_cap={} anf_cap={} seed={}",
        gates.len(),
        lits_of(&gates),
        wires,
        n_live,
        args.max_iters,
        args.group_cap,
        args.anf_support_cap,
        args.seed
    );
    println!(
        "reassembly_summary_csv,label,total_gates,plain_fragments,structural_g57,other_complemented,singleton_1cc,homogeneous_2cc,positive_1cc,negative_1cc,positive_2cc,negative_2cc,compatible_singleton_1cc,compatible_homogeneous_2cc,reassemblable_pairs,reassemblable_fragments,plain_percent,candidate_percent,all_gates_percent"
    );
    println!("lineage_input_width_csv,width,count,plain_percent");
    println!(
        "lineage_recovery_csv,stage,route,attribution,unique_input_fragments,plain_percent,width0,width1,width2,width3,width4,width5,width6,width7,width8,width9,width10,width11,width12,width13,width14,width15,width_gt15"
    );
    println!(
        "source_g57_csv,stage,route,total,returned_to_parent,new_same_parent,new_mixed_parents,new_total,unknown,new_percent_known,mixed_percent_known"
    );

    // The whole-tape maximum ignores intervening barriers and is therefore an
    // upper bound. The compressor report below counts the exact pairs actually
    // fused inside legally gathered same-target groups on its first pass.
    let input_analysis = analyze_barrier_free(&gates);
    print_reassembly(
        "input_upper_bound",
        &input_analysis.stats,
        other_complemented(&gates),
    );

    let params = CompressParams {
        live_out: live.clone(),
        max_iters: args.max_iters,
        group_cap: args.group_cap,
        anf_support_cap: args.anf_support_cap,
        local_verify: !args.no_local_verify,
        seed: args.seed,
    };
    let original = gates.clone();
    let t0 = std::time::Instant::now();
    let (mut traced, rep) =
        compress_traced_with_sources(gates, wires, &params, source_marks, source_parents);
    let secs = t0.elapsed().as_secs_f64();

    verify_equivalent(
        &original,
        &traced.gates,
        wires,
        live.as_deref(),
        args.verify_rounds,
        args.seed ^ 0xC0FFEE,
        "post-ESOP",
    );
    println!(
        "[fcompress] done in {:.1}s (verified {} rounds x64 lanes): gates {} -> {} ({:.1}%), lits {} -> {} ({:.1}%), live_dropped={} noop_dropped={}/{} fixed_point={}",
        secs,
        args.verify_rounds,
        rep.gates_in,
        rep.gates_out,
        100.0 * rep.gates_out as f64 / rep.gates_in.max(1) as f64,
        rep.lits_in,
        rep.lits_out,
        100.0 * rep.lits_out as f64 / rep.lits_in.max(1) as f64,
        rep.liveness_dropped,
        rep.identity_noops_dropped_input,
        rep.identity_noops_dropped_synthesized,
        rep.reached_fixed_point,
    );
    print_source_counts("after_esop", "direct_pass1", rep.direct_pass1_sources);
    print_source_counts("after_esop", "direct_later", rep.direct_later_sources);
    print_source_counts("after_esop", "esop_outputs", rep.esop_sources);
    print_source_counts("after_esop", "anf_outputs", rep.anf_sources);
    print_source_counts(
        "after_esop",
        "final_structural_g57",
        classify_structural_sources(&traced.gates, &traced.source_marks, &traced.source_parents),
    );
    println!(
        "[fcompress] first-pass reachable g57 recovery: {} pairs / {} fragments = {:.6}% of {} plain input fragments ({:.6}% of barrier-free upper bound)",
        rep.reassembled_pairs,
        rep.reassembled_fragments,
        percent(
            rep.reassembled_fragments,
            input_analysis.stats.plain_fragments
        ),
        input_analysis.stats.plain_fragments,
        percent(
            rep.reassembled_fragments,
            input_analysis.stats.reassemblable_fragments,
        ),
    );
    println!(
        "fcompress_summary_csv,input_gates,output_gates,input_lits,output_lits,iters,reached_fixed_point,groups,multi_groups,max_group,catalogue_merges,anf_wins,anf_structural_g57,esop_structural_g57,direct_pass1_pairs,direct_pass1_operands,direct_total_pairs,direct_later_pairs,direct_total_operands,pass1_plain_percent,pass1_candidate_percent,pass1_all_gates_percent,pass1_upper_bound_percent,liveness_dropped,identity_noops_input,identity_noops_synthesized,seconds,verify_rounds"
    );
    println!(
        "fcompress_summary_csv,{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{:.9},{:.9},{:.9},{:.9},{},{},{},{:.9},{}",
        rep.gates_in,
        rep.gates_out,
        rep.lits_in,
        rep.lits_out,
        rep.iters,
        rep.reached_fixed_point,
        rep.groups,
        rep.multi_groups,
        rep.max_group,
        rep.catalogue_merges,
        rep.anf_wins,
        rep.anf_structural_g57,
        rep.esop_structural_g57,
        rep.reassembled_pairs,
        rep.reassembled_fragments,
        rep.direct_pairs_total,
        rep.direct_pairs_later,
        rep.direct_fragments_total,
        percent(
            rep.reassembled_fragments,
            input_analysis.stats.plain_fragments,
        ),
        percent(
            rep.reassembled_fragments,
            input_analysis.stats.singleton_1cc + input_analysis.stats.homogeneous_2cc,
        ),
        percent(rep.reassembled_fragments, input_analysis.stats.total_gates),
        percent(
            rep.reassembled_fragments,
            input_analysis.stats.reassemblable_fragments,
        ),
        rep.liveness_dropped,
        rep.identity_noops_dropped_input,
        rep.identity_noops_dropped_synthesized,
        secs,
        args.verify_rounds,
    );

    let esop_recovery = traced.recovery_summary();
    print_input_widths(&esop_recovery);
    print_recovery_summary("after_esop", &esop_recovery);
    let esop_analysis = analyze_barrier_free(&traced.gates);
    print_reassembly(
        "after_esop",
        &esop_analysis.stats,
        other_complemented(&traced.gates),
    );
    if let Some(path) = &args.esop_output {
        format::write_mpmct(path, &traced.gates, wires).expect("write ESOP output");
        println!("[fcompress] wrote post-ESOP snapshot to {path}");
    }

    if args.db_trials > 0 {
        println!(
            "db_compress_summary_csv,input_gates,output_gates,input_lits,output_lits,trials_requested,trials_attempted,heterogeneous_windows,all_g57_windows,canonical_directions,polynomial_skips,budget_skips,windows_without_key,lookups,lookup_hits,heterogeneous_lookup_hits,all_g57_lookup_hits,malformed_values,candidates_decoded,shorter_candidates,windows_with_shorter_candidate,replacements,heterogeneous_replacements,all_g57_replacements,gates_removed,adjacent_cancelled,attributed_g57_windows,attributed_g57_outputs,single_parent_replacement_windows,mixed_parent_replacement_windows,unknown_source_replacement_windows,seconds"
        );
        let db = FrozenDb::from_env();
        let db_params = DbCompressParams {
            trials: args.db_trials,
            min_window: args.db_min_window,
            max_window: args.db_max_window,
            probe_reverse: !args.db_forward_only,
            poly_budget: XPolyBudget {
                max_mul_terms: args.poly_max_mul_terms,
                max_poly_terms: args.poly_max_terms,
                max_total_terms: args.poly_max_total_terms,
            },
            seed: args.seed ^ 0xDB_C0_57,
        };
        let db_started = std::time::Instant::now();
        let (db_out, db_report) = compress_frozen_contiguous_traced(traced, wires, &db, &db_params);
        let db_seconds = db_started.elapsed().as_secs_f64();
        traced = db_out;
        print_db_report(&db_report, db_seconds);
        print_source_counts(
            "database",
            "emitted_structural_g57",
            db_report.structural_sources,
        );
        print_source_counts(
            "after_database",
            "final_structural_g57",
            classify_structural_sources(
                &traced.gates,
                &traced.source_marks,
                &traced.source_parents,
            ),
        );
        let db_recovery = traced.recovery_summary();
        print_recovery_summary("after_database", &db_recovery);
        let db_analysis = analyze_barrier_free(&traced.gates);
        print_reassembly(
            "after_database",
            &db_analysis.stats,
            other_complemented(&traced.gates),
        );
        verify_equivalent(
            &original,
            &traced.gates,
            wires,
            live.as_deref(),
            args.verify_rounds,
            args.seed ^ 0xDB_C0_57 ^ 0xC0FFEE,
            "post-database",
        );
    }

    if let Some(path) = &args.output {
        format::write_mpmct(path, &traced.gates, wires).expect("write output");
        println!("[fcompress] wrote {} gates to {}", traced.gates.len(), path);
    } else {
        println!("[fcompress] no --output given; result discarded after verification");
    }
    if let Some(path) = &args.sources_out {
        source::write_source_marks(path, &traced.source_marks, traced.source_parents.len())
            .expect("write compressed fsource1 sidecar");
        println!("[fcompress] wrote source sidecar to {path}");
    }
}
