//! `tdp_packing run`: the production stage 4 (docs/STAGE4_RUST_DESIGN.md §5).
//!
//! One process does what `val_chain.sh` does after stage 3:
//!
//! ```text
//! window_pack3.py G2 G3 512 512 24576 3000 TAG-G512 packed_G512.mpmct1   > pack_G512.txt
//! repair.sh TAG 512 N                                                    > repair512.log
//!   (incl. anf_compose.py rep512 65536 TAG-rep512 500 > anf_rep512.txt,
//!    link_census.py G2 rep512 bounds TAG-rep512       > link_rep512.txt)
//! anf_dump_eval.py rep512.mpmct1 rep512.bounds deliverable.bin TAG
//! ```
//!
//! Every intermediate is handed on in memory exactly as the next oracle
//! script would read it back from disk (`OracleCircuit` widths, bounds as
//! integers). The deliverable is written last, via `*.tmp`, only after the
//! final identity and budget tests (exit 5/6 unless `--allow-*`) and every
//! check passed: the exact pack and repair certificates, both sampled
//! equivalence checks, decode == window ANF, window ANF == gates, DE's
//! evaluation check and the zero slice against C.
//!
//! Once the inputs have loaded, every output name of an earlier run in a
//! reused `--out-dir` is removed first, so a failed run never leaves a
//! deliverable, report or intermediate that belongs to another run. On
//! success the commit order is deliverable.bin, deliverable.tdp4, then
//! packing.report: the report is the completion record, and a directory
//! without one (or with a status other than `ok`) holds no recommended
//! deliverable.
use super::*;
use crate::stages::packing::census::{LinkCensus, link_census};
use crate::stages::packing::oracle_io::OracleCircuit;
use crate::stages::packing::scheduler::Packed;
use std::fmt::Write as _;

/// `repair.sh`'s `anf_compose.py $P $B 65536 TAG-rep$G 500` (report only):
/// both literals, independent of the split budget.
const AC_BUDGET: usize = 65536;
const AC_CHECK_EVERY: usize = 500;

/// The production defaults, which are also the `[packing]` defaults of the
/// managed driver (`docs/STAGE4_RUST_DESIGN.md` §11.2).
pub(super) const RUN_OPTIONS: [&str; 18] = [
    "stage2",
    "stage3",
    "source-c",
    "n",
    "out-dir",
    "label",
    "window-gates",
    "interval-cap",
    "monomial-budget",
    "span-cap",
    "split-budget",
    "fix-iterations",
    "fallback-iterations",
    "anf-cap-semantics",
    "allow-floor-miss",
    "allow-over-budget",
    "no-oracle-files",
    "threads",
];

/// `OracleCircuit` as `oracle_io::parse_oracle_mpmct` would return it for a
/// file written with header `header_n` and these gates.
pub(super) fn as_read_back(header_n: usize, gates: Vec<XGate>) -> OracleCircuit {
    let computed_n = if gates.is_empty() {
        header_n
    } else {
        gates
            .iter()
            .map(|g| g.max_wire() as usize + 1)
            .max()
            .unwrap_or(0)
            .max(header_n)
    };
    OracleCircuit {
        header_n,
        header_m: gates.len(),
        computed_n,
        gates,
    }
}

fn pack_params(a: &Args) -> Result<PackParams, Stage4Error> {
    let mut p = PackParams::PRODUCTION;
    p.cap_semantics = cap_semantics(a)?;
    p.g = opt_usize(a, "window-gates", p.g)?;
    p.w_cap = opt_usize(a, "interval-cap", p.w_cap)?;
    p.b_mono = opt_usize(a, "monomial-budget", p.b_mono)?;
    p.span_cap = opt_usize(a, "span-cap", p.span_cap)?;
    p.validate().map_err(Stage4Error::Usage)?;
    Ok(p)
}

/// `grep -oE "size: ANF [0-9.]+ MB"` over the report text.
fn grep_size(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let mut rest = line;
        while let Some(i) = rest.find("size: ANF ") {
            let tail = &rest[i + 10..];
            let num: String = tail
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if !num.is_empty() && tail[num.len()..].starts_with(" MB") {
                out.push(format!("size: ANF {num} MB"));
                rest = &tail[num.len() + 3..];
            } else {
                rest = &rest[i + 1..];
            }
        }
    }
    out
}

/// The report part `repair.sh` prints after the loops.
fn report_part(label: &str, g: usize, anf: &str, link: &LinkCensus, cview2: &str) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "== {label} rep{g} final");
    if let Some(l) = anf.lines().next() {
        let _ = writeln!(s, "{l}");
    }
    for l in grep_size(anf) {
        let _ = writeln!(s, "{l}");
    }
    for l in anf.lines().filter(|l| l.contains("spot")) {
        let _ = writeln!(s, "{l}");
    }
    for l in link.text.lines().skip(1).take(2) {
        let _ = writeln!(s, "{l}");
    }
    let _ = writeln!(s, "{cview2}");
    s
}

struct Ser {
    body: Staged,
    container: Staged,
    rep: crate::stages::packing::serialize::EmitReport,
    exhaustive: usize,
    sampled: usize,
    zero_slice: usize,
}

/// `anf_dump_eval.py` + every check; both files left as `*.tmp`.
fn serialize_staged(
    rep: &OracleCircuit,
    bounds: &[i64],
    c: &[XGate],
    n: usize,
    body_path: &Path,
    container_path: &Path,
) -> Result<Ser, Stage4Error> {
    let (body, mut w) = Staged::create(body_path)?;
    let er = match encode_body(
        &rep.gates,
        bounds,
        rep.header_m,
        rep.computed_n,
        MAX_CONCURRENT_ANF.max(rayon::current_num_threads()),
        &mut w,
    ) {
        Ok(er) => er,
        Err(e) => {
            body.abandon();
            return Err(e);
        }
    };
    Staged::finish(w, &body.tmp)?;
    let checked = (|| -> Result<(usize, usize, usize), Stage4Error> {
        let bytes = fs::read(&body.tmp).map_err(|e| Stage4Error::io(&body.tmp, e))?;
        let d = decode(
            &bytes,
            Format::Raw {
                source_wires: n as u32,
                state_wires: rep.computed_n as u32,
            },
        )?;
        let wc = check_windows_anf(
            &rep.gates,
            bounds,
            &d,
            EXHAUSTIVE_SUPPORT_MAX,
            WINDOW_CHECK_SEED,
        )?;
        check_deliverable(&rep.gates, rep.computed_n, &d, DE_EVAL_SEED)?;
        let zs = check_zero_slice(c, &d, n, ZERO_SLICE_SAMPLES, ZERO_SLICE_SEED)?;
        Ok((wc.exhaustive, wc.sampled, zs))
    })();
    let (exhaustive, sampled, zero_slice) = match checked {
        Ok(x) => x,
        Err(e) => {
            body.abandon();
            return Err(e);
        }
    };
    let h = ContainerHeader {
        source_wires: n as u32,
        state_wires: rep.computed_n as u32,
        io_layout: IO_LAYOUT_CLASSIC_ZERO_SLICE,
        window_count: er.windows as u32,
        body_bytes: er.body_bytes,
        body_crc32: er.body_crc32,
    };
    // the container goes to its own *.tmp (write_container commits; stage it by hand)
    let (ct, mut cw) = Staged::create(container_path)?;
    let r = (|| -> Result<(), Stage4Error> {
        cw.write_all(&h.to_bytes())
            .map_err(|e| Stage4Error::io(&ct.tmp, e))?;
        let mut f = File::open(&body.tmp).map_err(|e| Stage4Error::io(&body.tmp, e))?;
        io::copy(&mut f, &mut cw).map_err(|e| Stage4Error::io(&ct.tmp, e))?;
        Ok(())
    })();
    if let Err(e) = r {
        ct.abandon();
        body.abandon();
        return Err(e);
    }
    if let Err(e) = Staged::finish(cw, &ct.tmp) {
        ct.abandon();
        body.abandon();
        return Err(e);
    }
    Ok(Ser {
        body,
        container: ct,
        rep: er,
        exhaustive,
        sampled,
        zero_slice,
    })
}

fn xxh3_file(path: &Path) -> Option<u128> {
    let bytes = fs::read(path).ok()?;
    Some(xxhash_rust::xxh3::xxh3_128(&bytes))
}

#[allow(clippy::too_many_arguments)]
fn packing_report(
    status: &str,
    n: usize,
    pp: &PackParams,
    rp: &RepairParams,
    packed: &Packed,
    r: &crate::stages::packing::repair::RepairReport,
    anf: &crate::stages::packing::census::AnfSizeReport,
    link: &LinkCensus,
    ser: Option<&Ser>,
    times: [f64; 4],
) -> String {
    let join = |v: &[usize]| {
        if v.is_empty() {
            "-".to_string()
        } else {
            v.iter()
                .map(|x| x.to_string())
                .collect::<Vec<_>>()
                .join(",")
        }
    };
    let pr = &packed.report;
    let mut s = String::new();
    let mut kv = |k: &str, v: String| {
        let _ = writeln!(s, "{k}={v}");
    };
    kv("packing_report", "2".into());
    kv("status", status.into());
    kv("tool", format!("tdp_packing {}", env!("CARGO_PKG_VERSION")));
    kv(
        "binary_xxh3",
        std::env::current_exe()
            .ok()
            .and_then(|p| xxh3_file(&p))
            .map(|h| format!("{h:032x}"))
            .unwrap_or_else(|| "unknown".into()),
    );
    kv("n", n.to_string());
    kv("state_wires", pr.state_wires.to_string());
    kv("param.window_gates", pp.g.to_string());
    kv("param.interval_cap", pp.w_cap.to_string());
    kv("param.monomial_budget", pp.b_mono.to_string());
    kv("param.mul_cap", pp.mul_cap().to_string());
    kv("param.span_cap", pp.span_cap.to_string());
    kv("param.pad", pp.pad.to_string());
    kv("param.cmax", pp.cmax.to_string());
    kv("param.fp_seed", pp.fp_seed.to_string());
    kv("param.fp_samples", "1024".into());
    kv("param.split_budget", rp.split_budget.to_string());
    kv("param.fix_iterations", rp.fix_iterations.to_string());
    kv(
        "param.fallback_iterations",
        rp.fallback_iterations.to_string(),
    );
    kv("param.cview_batches", rp.cview.batches.to_string());
    kv("param.cview_seed", rp.cview.seed.to_string());
    kv("param.verify_seed", "77".into());
    kv("param.anf_cap_semantics", rp.cap_semantics.name().into());
    kv("param.repair_mode", rp.mode.name().into());
    kv(
        "param.exhaustive_support_max",
        EXHAUSTIVE_SUPPORT_MAX.to_string(),
    );
    kv("pack.gates", pr.gates.to_string());
    kv("pack.windows", pr.windows.to_string());
    kv("pack.bounds", pr.bounds.to_string());
    kv("pack.duplicate_bounds", pr.duplicate_bounds.to_string());
    kv("pack.spans", pr.spans.to_string());
    kv("pack.carried", pr.carried.to_string());
    kv("pack.forced", pr.forced.to_string());
    kv("pack.winnable", pr.winnable.winnable.to_string());
    kv("pack.disq_long", pr.winnable.disq_long.to_string());
    kv("pack.disq_big", pr.winnable.disq_big.to_string());
    kv("pack.rescued", pr.winnable.rescued.to_string());
    kv("pack.intervals", pr.intervals.to_string());
    kv("pack.hidden_spans", pr.hidden_spans.to_string());
    kv("pack.hidden_values", pr.hidden_values.to_string());
    for (name, v) in &pr.deaths {
        kv(&format!("pack.death.{name}"), v.to_string());
    }
    kv("pack.exposed", pr.exposed.to_string());
    kv("pack.mul_calls", pr.mul.calls.to_string());
    kv("pack.mul_ambiguous", pr.mul.ambiguous.to_string());
    kv("pack.compose_aborts", pr.compose_aborts.to_string());
    kv(
        "pack.compose_aborts_order_dependent",
        pr.mul.aborts_order_dependent.to_string(),
    );
    kv("repair.cancel_passes", join(&r.cancel_passes));
    kv(
        "repair.cancel_pairs",
        r.cancel_passes.iter().sum::<usize>().to_string(),
    );
    kv("repair.fix_trajectory", join(&r.fix_trajectory));
    kv("repair.fix_moved", join(&r.fix_moved));
    kv(
        "repair.fix_exhausted",
        u8::from(r.fix_exhausted).to_string(),
    );
    kv("repair.fallback_trajectory", join(&r.fallback_trajectory));
    kv("repair.merged", join(&r.merged));
    kv("repair.splits", join(&r.splits));
    kv("repair.forced_splits", r.forced_splits.len().to_string());
    kv(
        "repair.single_window_scans",
        r.single_window_scans.to_string(),
    );
    kv("repair.final_gates", r.final_gates.to_string());
    kv("repair.final_bounds", r.final_bounds.to_string());
    kv("repair.final_visible", r.final_visible.to_string());
    kv(
        "repair.final_nonoutput_visible",
        r.final_nonoutput_visible.len().to_string(),
    );
    kv(
        "repair.final_over_budget",
        r.final_over_budget.len().to_string(),
    );
    kv(
        "repair.split_mul_ambiguous",
        r.split_stats.ambiguous.to_string(),
    );
    kv(
        "repair.split_order_dependent",
        r.split_stats.aborts_order_dependent.to_string(),
    );
    kv("anf.windows", anf.windows.to_string());
    kv("anf.overflow", anf.overflow.to_string());
    kv("anf.total_monomials", anf.total_monomials.to_string());
    kv("anf.spot_checks", anf.checked.to_string());
    match ser {
        Some(se) => {
            kv("emit.windows", se.rep.windows.to_string());
            kv("emit.monomials", se.rep.monomials.to_string());
            kv(
                "emit.max_window_monomials",
                se.rep.max_window_monomials.to_string(),
            );
            kv("emit.body_bytes", se.rep.body_bytes.to_string());
            kv("emit.container_bytes", (se.rep.body_bytes + 48).to_string());
            kv(
                "emit.deliverable_xxh3",
                format!("{:032x}", se.rep.body_xxh3),
            );
        }
        None => kv("emit.deliverable", "not_written".into()),
    }
    kv("census.proxy_values", link.proxy.0.to_string());
    kv("census.proxy_L2", link.proxy.1.to_string());
    kv("census.proxy_M2", link.proxy.2.to_string());
    kv("census.froth_values", link.froth.0.to_string());
    kv("census.froth_L2", link.froth.1.to_string());
    kv("census.froth_M2", link.froth.2.to_string());
    kv("check.reorder_cert", "ok".into());
    kv("check.pack_verify", "ok".into());
    kv("check.repair_cert", "ok".into());
    kv("check.equiv", "ok".into());
    if let Some(se) = ser {
        kv(
            "check.windows_anf",
            format!("ok({}/{})", se.exhaustive, se.sampled),
        );
        kv("check.decode", "ok".into());
        kv("check.eval", "ok".into());
        kv("check.zero_slice", format!("ok({})", se.zero_slice));
    }
    kv("time.pack_s", format!("{:.1}", times[0]));
    kv("time.repair_s", format!("{:.1}", times[1]));
    kv("time.reports_s", format!("{:.1}", times[2]));
    kv("time.emit_s", format!("{:.1}", times[3]));
    s
}

pub(super) fn cmd_run(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&RUN_OPTIONS)?;
    if !a.pos.is_empty() {
        return Err(Stage4Error::Usage(
            "run --stage2 G2 --stage3 G3 --source-c C --n N --out-dir DIR [...]".into(),
        ));
    }
    setup_threads(a)?;
    let pp = pack_params(a)?;
    let mut rp = repair_params(a)?;
    rp.cap_semantics = pp.cap_semantics;
    let label = a.get("label").unwrap_or("tdp").to_string();
    if label.is_empty() || label.contains(['/', '\n']) {
        return Err(Stage4Error::Usage(format!("--label {label:?}")));
    }
    let n = a
        .num("n")?
        .ok_or_else(|| Stage4Error::Usage("--n is required".into()))?;
    let out_dir = PathBuf::from(req(a, "out-dir")?);
    let g = pp.g;
    let t0 = Instant::now();
    let g2 = read_oracle_mpmct(Path::new(req(a, "stage2")?))?;
    let g3 = read_oracle_mpmct(Path::new(req(a, "stage3")?))?;
    let c = load_source_c(
        req(a, "source-c")?,
        n,
        crate::stages::packing::scheduler::wp_width(&g3),
    )?;
    let n = n as usize;
    private_dir(&out_dir)?;
    remove_stale_outputs(&out_dir)?;

    // ---- pack (window_pack3.py)
    let mut tee = Tee { buf: Vec::new() };
    let packed = pack(&g2, &g3, &pp, &format!("{label}-G{g}"), &mut tee)?;
    let pack_log = std::mem::take(&mut tee.buf);
    let packed_gates: Vec<XGate> = packed.gates(&g3.gates).cloned().collect();
    drop(g3);
    let packed_path = out_dir.join(format!("packed_G{g}.mpmct1"));
    let mut packed_bounds_path = packed_path.as_os_str().to_owned();
    packed_bounds_path.push(".bounds");
    write_all_staged(vec![
        (
            packed_path.clone(),
            Box::new(|w| write_pack_mpmct(w, packed.n_out, &packed_gates)),
        ),
        (
            PathBuf::from(packed_bounds_path),
            Box::new(|w| write_bounds(w, &packed.bounds)),
        ),
        (
            out_dir.join(format!("pack_G{g}.txt")),
            Box::new(|w| w.write_all(&pack_log)),
        ),
    ])?;
    let t_pack = t0.elapsed().as_secs_f64();

    // ---- repair (repair.sh), on the packed circuit as read back
    let bounds0: Vec<i64> = packed.bounds.iter().map(|&b| b as i64).collect();
    let rep_path = out_dir.join(format!("rep{g}.mpmct1"));
    let cv = CView::new(&c, n, &rp.cview);
    let r = repair(
        &packed_gates,
        packed.n_out,
        &bounds0,
        &cv,
        &rp,
        &rep_path.display().to_string(),
        &mut tee,
    )?;
    drop(packed_gates);
    let t_repair = t0.elapsed().as_secs_f64() - t_pack;
    let status = r
        .report
        .status(a.flag("allow-floor-miss"), a.flag("allow-over-budget"));
    // every violation the final tests found, allowed or not (the exit code
    // is `status`'s: Floor first, then OverBudget)
    let status_name = match (
        !r.report.final_nonoutput_visible.is_empty(),
        !r.report.final_over_budget.is_empty(),
    ) {
        (false, false) => "ok",
        (true, false) => "floor_miss",
        (false, true) => "over_budget",
        (true, true) => "floor_miss+over_budget",
    };

    // ---- reports (anf_compose.py, link_census.py) and repair.sh's tail
    let rep_bounds: Vec<i64> = r.bounds.iter().map(|&b| b as i64).collect();
    let rep_label = format!("{label}-rep{g}");
    let rep = as_read_back(r.n_header, r.gates.clone());
    let anf = anf_size_report(
        &rep.gates,
        rep.computed_n,
        &rep_bounds,
        AC_BUDGET,
        &rep_label,
        AC_CHECK_EVERY,
        rp.cap_semantics,
    );
    if anf.failed > 0 {
        return Err(Stage4Error::Verify(format!(
            "anf_compose spot checks: {} failed",
            anf.failed
        )));
    }
    let link = link_census(&g2, &rep, &rep_bounds, &rep_label);
    drop(g2);
    let path = |suffix: &str| out_dir.join(format!("rep{g}.{suffix}"));
    let (pe, pa, pw) = (path("exposure.tsv"), path("allcuts.tsv"), path("wit.tsv"));
    let cview_log = cv.scan_log(
        &r.last_scan,
        &pe.display().to_string(),
        &pa.display().to_string(),
        &pw.display().to_string(),
    );
    let tail = report_part(
        &label,
        g,
        &anf.text,
        &link,
        cview_log.get(1).map(String::as_str).unwrap_or(""),
    );
    print!("{tail}");
    tee.buf.extend_from_slice(tail.as_bytes());
    let t_reports = t0.elapsed().as_secs_f64() - t_pack - t_repair;

    let mut files: Vec<(PathBuf, FileBody)> = vec![
        (
            rep_path.clone(),
            Box::new(|w| write_repair_mpmct(w, r.n_header, &r.gates)),
        ),
        (path("bounds"), Box::new(|w| write_bounds(w, &r.bounds))),
        (
            out_dir.join(format!("repair{g}.log")),
            Box::new(|w| w.write_all(&tee.buf)),
        ),
        (
            out_dir.join(format!("anf_rep{g}.txt")),
            Box::new(|w| w.write_all(anf.text.as_bytes())),
        ),
        (
            out_dir.join(format!("link_rep{g}.txt")),
            Box::new(|w| w.write_all(link.text.as_bytes())),
        ),
    ];
    if !a.flag("no-oracle-files") {
        let s = &r.last_scan.report;
        files.push((pe, Box::new(|w| s.write_exposure_tsv(w))));
        files.push((pa, Box::new(|w| s.write_allcuts_tsv(w))));
        files.push((pw, Box::new(|w| s.write_witness_tsv(w))));
        files.push((
            path("cview.log"),
            Box::new(|w| {
                for l in &cview_log {
                    writeln!(w, "{l}")?;
                }
                Ok(())
            }),
        ));
    }
    write_all_staged(files)?;
    let fr = &r.report;
    let summary = format!(
        "run: pack {} gates in {} windows; repair -> {} gates, {} boundaries; final scan {} visible, \
         {} non-output visible; {} windows over budget; status={status_name}",
        fr.packed_gates,
        packed.report.windows,
        fr.final_gates,
        fr.final_bounds,
        fr.final_visible,
        fr.final_nonoutput_visible.len(),
        fr.final_over_budget.len(),
    );
    let report_path = out_dir.join("packing.report");
    if let Err(e) = status {
        // exit 5 or 6: the intermediates stay, no deliverable is written
        let text = packing_report(
            status_name,
            n,
            &pp,
            &rp,
            &packed,
            fr,
            &anf,
            &link,
            None,
            [t_pack, t_repair, t_reports, 0.0],
        );
        write_all_staged(vec![(
            report_path,
            Box::new(|w| w.write_all(text.as_bytes())),
        )])?;
        eprintln!("{summary}");
        return Err(e);
    }

    // ---- deliverable (anf_dump_eval.py) + checks; committed last
    let t_e = Instant::now();
    let body_path = out_dir.join("deliverable.bin");
    let container_path = out_dir.join("deliverable.tdp4");
    let ser = serialize_staged(&rep, &rep_bounds, &c, n, &body_path, &container_path)?;
    println!(
        "{label}: serialized {:.1} MB, {} windows; checks: decode ok; windows == window_anf and == gates \
         ({} exhaustive, {} sampled); eval vs gates (1024 lanes, seed {DE_EVAL_SEED}) match=True; \
         zero slice C(x) on wires {n}..{} ok ({} samples)",
        ser.rep.body_bytes as f64 / 1e6,
        ser.rep.windows,
        ser.exhaustive,
        ser.sampled,
        2 * n,
        ser.zero_slice
    );
    let t_emit = t_e.elapsed().as_secs_f64();
    let text = packing_report(
        status_name,
        n,
        &pp,
        &rp,
        &packed,
        fr,
        &anf,
        &link,
        Some(&ser),
        [t_pack, t_repair, t_reports, t_emit],
    );
    // the deliverables first, the report (the completion record) last
    if let Err(e) = ser.body.commit() {
        ser.container.abandon();
        return Err(e);
    }
    ser.container.commit()?;
    write_all_staged(vec![(
        report_path,
        Box::new(|w| w.write_all(text.as_bytes())),
    )])?;
    eprintln!("{summary} ({:.1}s)", t0.elapsed().as_secs_f64());
    Ok(())
}

/// Whether `name` is one of the files `run` writes into `--out-dir` for SOME
/// window size (or its `*.tmp`): deliverable.bin, deliverable.tdp4,
/// packing.report, packed_G{g}.mpmct1[.bounds], pack_G{g}.txt,
/// repair{g}.log, anf_rep{g}.txt, link_rep{g}.txt and
/// rep{g}.{mpmct1,bounds,exposure.tsv,allcuts.tsv,wit.tsv,cview.log}.
pub(super) fn is_run_output(name: &str) -> bool {
    let name = name.strip_suffix(".tmp").unwrap_or(name);
    if matches!(
        name,
        "deliverable.bin" | "deliverable.tdp4" | "packing.report"
    ) {
        return true;
    }
    // PREFIX <digits> SUFFIX
    let shapes: [(&str, &[&str]); 6] = [
        ("packed_G", &[".mpmct1", ".mpmct1.bounds"]),
        ("pack_G", &[".txt"]),
        ("repair", &[".log"]),
        ("anf_rep", &[".txt"]),
        ("link_rep", &[".txt"]),
        (
            "rep",
            &[
                ".mpmct1",
                ".bounds",
                ".exposure.tsv",
                ".allcuts.tsv",
                ".wit.tsv",
                ".cview.log",
            ],
        ),
    ];
    shapes.iter().any(|(prefix, suffixes)| {
        name.strip_prefix(prefix).is_some_and(|rest| {
            let digits = rest.bytes().take_while(u8::is_ascii_digit).count();
            digits > 0 && suffixes.contains(&&rest[digits..])
        })
    })
}

/// Removes every output of an earlier run from `out_dir`, whatever window
/// size it used, so a reused directory never mixes runs.
fn remove_stale_outputs(out_dir: &Path) -> Result<(), Stage4Error> {
    let entries = fs::read_dir(out_dir).map_err(|e| Stage4Error::io(out_dir, e))?;
    for entry in entries {
        let entry = entry.map_err(|e| Stage4Error::io(out_dir, e))?;
        let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
            continue;
        };
        if !is_run_output(&name) {
            continue;
        }
        let path = entry.path();
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(Stage4Error::io(&path, e)),
        }
    }
    Ok(())
}

/// `tdp_packing mult-census <A> <label>`: mult_census.py's stdout
/// (the driver's report-only stage-2 side report).
pub(super) fn cmd_mult_census(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&[])?;
    if a.pos.len() != 2 {
        return Err(Stage4Error::Usage("mult-census <A.mpmct1> <label>".into()));
    }
    let ca = read_oracle_mpmct(Path::new(&a.pos[0]))?;
    check_census_width(&ca, &a.pos[0])?;
    if ca.gates.is_empty() {
        return Err(Stage4Error::Input("mult-census: no gates".into()));
    }
    print!(
        "{}",
        crate::stages::packing::census::mult_census(&ca, &a.pos[1])
    );
    Ok(())
}

/// `tdp_packing census <A> <P> <B> <label>`: link_census.py's stdout.
pub(super) fn cmd_census(a: &Args) -> Result<(), Stage4Error> {
    a.check_known(&[])?;
    if a.pos.len() != 4 {
        return Err(Stage4Error::Usage(
            "census <A.mpmct1> <packed.mpmct1> <bounds> <label>".into(),
        ));
    }
    let ca = read_oracle_mpmct(Path::new(&a.pos[0]))?;
    check_census_width(&ca, &a.pos[0])?;
    let cp = read_oracle_mpmct(Path::new(&a.pos[1]))?;
    check_census_width(&cp, &a.pos[1])?;
    let b = read_bounds(Path::new(&a.pos[2]))?;
    print!("{}", link_census(&ca, &cp, &b, &a.pos[3]).text);
    Ok(())
}
