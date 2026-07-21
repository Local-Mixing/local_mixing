use std::{fs::File, io::Write};

use rand::{Rng, RngCore};

use crate::{
    circuit::circuit::{CircuitSeq, Gate, U1024},
    replace::{
        frozen::FrozenDb,
        gadgets::{
            feistalize, feistalize_with_slice_zero, feistalize_with_slice_zero_hardcoded,
            feistalize_with_slice_zero_random, gadgetize, packed_bit,
        },
        pairs::interleave,
        replace::compress_loop,
        transpositions::{
            insert_wire_m_samfs_every_x, insert_wire_m_samfs_every_x_tagged,
            shuffled_shoot_then_samf_stage_b_pass,
        },
    },
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

fn feistal_middle_matches_original(
    original: &CircuitSeq,
    transformed: &CircuitSeq,
    original_n: usize,
    total_wires: usize,
    num_inputs: usize,
    fixed_slice: Option<(U1024, U1024)>,
) -> Result<(), String> {
    assert!(
        total_wires <= 1024,
        "feistalized equality check supports up to 1024 wires"
    );
    let mask = if original_n < 1024 {
        (U1024::one() << original_n) - U1024::one()
    } else {
        U1024::MAX
    };
    for _ in 0..num_inputs {
        let mut bytes = [0u8; 128];
        rand::rng().fill_bytes(&mut bytes);
        let random = U1024::from_little_endian(&bytes);
        let x = random & mask;
        let (y, z) = if let Some((fixed_y, fixed_z)) = fixed_slice {
            (fixed_y, fixed_z)
        } else {
            (
                (random >> original_n) & mask,
                (random >> (2 * original_n)) & mask,
            )
        };
        let extra = if total_wires > 3 * original_n {
            let extra_mask = (U1024::one() << (total_wires - 3 * original_n)) - U1024::one();
            (random >> (3 * original_n)) & extra_mask
        } else {
            U1024::zero()
        };
        let input = x | (y << original_n) | (z << (2 * original_n)) | (extra << (3 * original_n));
        let original_output = Gate::evaluate_index_list_1024(x, &original.gates) & mask;
        let transformed_output = Gate::evaluate_index_list_1024(input, &transformed.gates);
        let middle = (transformed_output >> original_n) & mask;
        if middle != (y ^ original_output) {
            let scope = if fixed_slice.is_some() {
                " on the fixed (y,z) slice"
            } else {
                ""
            };
            return Err(format!(
                "Feistalized circuit middle block is not y ^ C(x){scope}"
            ));
        }
    }
    Ok(())
}

fn packed_words_to_u1024(words: &[u64], n: usize) -> U1024 {
    let mut out = U1024::zero();
    for bit in 0..n.min(1024) {
        if packed_bit(words, bit) {
            out = out | (U1024::one() << bit);
        }
    }
    out
}

fn packed_words_to_hex(words: &[u64], n: usize) -> String {
    let nibbles = n.div_ceil(4).max(1);
    let mut out = String::with_capacity(2 + nibbles);
    out.push_str("0x");
    for nibble in (0..nibbles).rev() {
        let mut value = 0u8;
        for offset in 0..4 {
            let bit = nibble * 4 + offset;
            if bit < n && packed_bit(words, bit) {
                value |= 1 << offset;
            }
        }
        out.push(char::from_digit(value as u32, 16).unwrap());
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct GenerationProgress {
    total: usize,
    reached: usize,
    fraction: f64,
    min: u32,
    median: u32,
    max: u32,
}

fn generation_progress(tags: &[u32], min_gen: usize) -> GenerationProgress {
    if tags.is_empty() {
        return GenerationProgress {
            total: 0,
            reached: 0,
            fraction: 1.0,
            min: 0,
            median: 0,
            max: 0,
        };
    }
    let threshold = min_gen.min(u32::MAX as usize) as u32;
    let reached = tags.iter().filter(|&&tag| tag >= threshold).count();
    let mut sorted = tags.to_vec();
    sorted.sort_unstable();
    GenerationProgress {
        total: tags.len(),
        reached,
        fraction: reached as f64 / tags.len() as f64,
        min: sorted[0],
        median: sorted[sorted.len() / 2],
        max: *sorted.last().unwrap(),
    }
}

fn generation_goal_met(tags: &[u32], min_gen: usize, min_gen_fraction: f64) -> bool {
    let target = min_gen_fraction.clamp(0.0, 1.0);
    generation_progress(tags, min_gen).fraction >= target
}

fn new_sss_stage_size_cap(
    last_compressed: usize,
    target_size: usize,
    grow_permille: usize,
) -> Option<usize> {
    if target_size > 0 {
        // An absolute target is a hard ceiling. In particular, do not raise it when a prior
        // compression stalls above the requested target.
        Some(target_size)
    } else if grow_permille > 0 {
        Some(
            (((last_compressed as u128 * (1000 + grow_permille as u128)) / 1000) as usize)
                .max(last_compressed.saturating_add(1)),
        )
    } else {
        None
    }
}

fn choose_low_generation_anchor(tags: &[u32], min_gen: usize, rng: &mut impl Rng) -> Option<usize> {
    let threshold = min_gen.min(u32::MAX as usize) as u32;
    let mut best_tag = u32::MAX;
    let mut seen = 0usize;
    let mut chosen = None;
    for (idx, &tag) in tags.iter().enumerate() {
        if tag >= threshold {
            continue;
        }
        if tag < best_tag {
            best_tag = tag;
            seen = 1;
            chosen = Some(idx);
        } else if tag == best_tag {
            seen += 1;
            if rng.random_range(0..seen) == 0 {
                chosen = Some(idx);
            }
        }
    }
    chosen
}

fn write_slice_zero_random_metadata(
    path: &str,
    original_n: usize,
    gate_count: usize,
    public_y: &[u64],
    public_z: &[u64],
) {
    let y_hex = packed_words_to_hex(public_y, original_n);
    let z_hex = packed_words_to_hex(public_z, original_n);
    let meta_path = format!("{path}.slice_zero_random");
    let contents = format!(
        "mode=slice_zero_random\n\
         n={original_n}\n\
         gates={gate_count}\n\
         y_hex={y_hex}\n\
         z_hex={z_hex}\n\
         bit_order=bit i is wire n+i for y and wire 2n+i for z\n"
    );
    File::create(&meta_path)
        .and_then(|mut f| f.write_all(contents.as_bytes()))
        .expect("Failed to write slice_zero_random metadata");
    println!(
        "slice_zero_random public slice: y={} z={} (metadata: {})",
        y_hex, z_hex, meta_path
    );
}

fn print_slice_zero_random_public_slice(original_n: usize, public_y: &[u64], public_z: &[u64]) {
    println!(
        "slice_zero_random public slice: Y={} Z={}",
        packed_words_to_hex(public_y, original_n),
        packed_words_to_hex(public_z, original_n)
    );
}

fn env_truthy(name: &str) -> bool {
    match std::env::var(name) {
        Ok(value) => {
            let value = value.trim();
            !value.is_empty()
                && !matches!(
                    value.to_ascii_lowercase().as_str(),
                    "0" | "false" | "off" | "no"
                )
        }
        Err(_) => false,
    }
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
}

fn sat_cone_log_enabled() -> bool {
    env_truthy("SAT_CONE_LOG") || env_truthy("SAT_HARDEN")
}

fn sat_cone_range(total_wires: usize, feistal_original_n: Option<usize>) -> Option<(usize, usize)> {
    if !sat_cone_log_enabled() {
        return None;
    }

    let default_start = feistal_original_n.unwrap_or(0);
    let default_bits = feistal_original_n.unwrap_or(total_wires);
    let start = env_usize("SAT_CONE_START").unwrap_or(default_start);
    let bits = env_usize("SAT_CONE_BITS").unwrap_or(default_bits);
    if bits == 0 || start.saturating_add(bits) > total_wires {
        eprintln!(
            "[sat-cone] disabled invalid range start={} bits={} total_wires={}",
            start, bits, total_wires
        );
        return None;
    }
    Some((start, bits))
}

fn print_sat_cone(label: &str, gates: &[[u16; 3]], cone_range: Option<(usize, usize)>) {
    let Some((output_start, output_bits)) = cone_range else {
        return;
    };
    if let Some(stats) =
        crate::replace::sat_score::output_cone_stats(gates, output_start, output_bits)
    {
        println!(
            "[sat-cone] label={} output_start={} output_bits={} cone_gates={} cone_gate_fraction={:.6} cone_wires={} cone_input_wires={} cone_max_live_wires={}",
            label,
            stats.output_start,
            stats.output_bits,
            stats.cone_gates,
            stats.cone_gate_fraction,
            stats.cone_wires,
            stats.cone_input_wires,
            stats.cone_max_live_wires
        );
    }
}

fn sat_global_mix_enabled() -> bool {
    env_truthy("SAT_GLOBAL_MIX") || env_truthy("SAT_HARDEN")
}

fn sat_global_mix_every() -> usize {
    env_usize("SAT_GLOBAL_MIX_EVERY").unwrap_or(1).max(1)
}

fn sat_global_mix_m(total_wires: usize) -> usize {
    env_usize("SAT_GLOBAL_MIX_M").unwrap_or(total_wires)
}

fn sat_hard_cores_enabled() -> bool {
    env_truthy("SAT_HARD_CORES")
}

fn sat_hard_core_count(total_wires: usize) -> usize {
    env_usize("SAT_HARD_CORE_COUNT").unwrap_or((total_wires / 8).max(1))
}

fn insert_sat_hard_identity_cores(
    circuit: &mut CircuitSeq,
    total_wires: usize,
    count: usize,
    cone_range: Option<(usize, usize)>,
    rng: &mut impl Rng,
) {
    if total_wires < 6 || count == 0 {
        return;
    }

    let mut insertions: Vec<(usize, [[u16; 3]; 4])> = Vec::with_capacity(count);
    for _ in 0..count {
        let gate_a = random_core_gate(total_wires, &[], cone_range, rng);
        let forbidden = [gate_a[0], gate_a[1], gate_a[2]];
        let gate_b = random_core_gate(total_wires, &forbidden, cone_range, rng);
        let pos = rng.random_range(0..=circuit.gates.len());
        // Disjoint self-inverse gates commute, so A B A B is an identity without adjacent
        // duplicate gates for the simple cleanup pass to erase immediately.
        insertions.push((pos, [gate_a, gate_b, gate_a, gate_b]));
    }

    insertions.sort_by(|a, b| b.0.cmp(&a.0));
    for (pos, core) in insertions {
        circuit.gates.splice(pos..pos, core);
    }
    println!(
        "[sat-hard-core] inserted {} commuting identity cores ({} gates)",
        count,
        count * 4
    );
}

fn random_core_gate(
    total_wires: usize,
    forbidden: &[u16],
    cone_range: Option<(usize, usize)>,
    rng: &mut impl Rng,
) -> [u16; 3] {
    let active = random_core_wire(total_wires, forbidden, cone_range, true, rng);
    let control_a = random_core_wire(
        total_wires,
        &[forbidden, &[active]].concat(),
        None,
        false,
        rng,
    );
    let control_b = random_core_wire(
        total_wires,
        &[forbidden, &[active, control_a]].concat(),
        None,
        false,
        rng,
    );
    [active, control_a, control_b]
}

fn random_core_wire(
    total_wires: usize,
    forbidden: &[u16],
    cone_range: Option<(usize, usize)>,
    prefer_cone: bool,
    rng: &mut impl Rng,
) -> u16 {
    for _ in 0..128 {
        let wire = if prefer_cone && rng.random_bool(0.75) {
            if let Some((start, bits)) = cone_range {
                rng.random_range(start..start + bits) as u16
            } else {
                rng.random_range(0..total_wires) as u16
            }
        } else {
            rng.random_range(0..total_wires) as u16
        };
        if !forbidden.contains(&wire) {
            return wire;
        }
    }
    (0..total_wires as u16)
        .find(|wire| !forbidden.contains(wire))
        .unwrap_or(0)
}

pub fn main_shuffle_shoot_shuffle(
    c: &CircuitSeq,
    rounds: usize,
    n: usize,
    m: usize,
    x: usize,
    save: &str,
    source: &str,
    db: &FrozenDb,
    leave: bool,
    do_gadgetize: bool,
    do_feistalize: bool,
    slice_zero: bool,
    slice_zero_random: bool,
    slice_zero_random_gates: usize,
    slice_zero_hardcoded: bool,
    slice_zero_hardcoded_rounds: usize,
    gadget_path: Option<&str>,
    full_shuffle: bool,
    full_shuffle_early: bool,
    gates_ahead_expand: usize,
    gates_ahead_samf: usize,
    type_attempts: usize,
    shooting_times: usize,
    collision_rounds: usize,
    stable_compressions: usize,
    expansion_game: bool,
    equality_check: bool,
    rg_freq: usize,
    single_end: bool,
    min_gen: usize,
    min_gen_fraction: f64,
    pass_length: usize,
    max_passes: usize,
    grow_threshold: f64,
    compress_fraction: f64,
    target_size: usize,
) {
    // Start with the input circuit
    let save_base = save.strip_suffix(".txt").unwrap_or(save);
    println!("Starting len: {}", c.gates.len());
    let mut circuit = c.clone();
    // Repeat `rounds` times
    let mut post_len = 0;
    let mut count = 0;
    let mut fixed_feistal_slice =
        (slice_zero || slice_zero_hardcoded).then_some((U1024::zero(), U1024::zero()));
    let mut slice_zero_random_public: Option<(Vec<u64>, Vec<u64>)> = None;
    if do_gadgetize || do_feistalize {
        let mut rng = rand::rng();
        let before = circuit.gates.len();
        let (label, transformed_n) = if do_feistalize {
            if slice_zero_random {
                let transformed = feistalize_with_slice_zero_random(
                    &circuit,
                    n,
                    rg_freq,
                    slice_zero_random_gates,
                    &mut rng,
                );
                fixed_feistal_slice = Some((
                    packed_words_to_u1024(&transformed.public_y, n),
                    packed_words_to_u1024(&transformed.public_z, n),
                ));
                slice_zero_random_public = Some((transformed.public_y, transformed.public_z));
                circuit = transformed.circuit;
                ("Slice-zero-random feistalized", 3 * n)
            } else if slice_zero_hardcoded {
                circuit = feistalize_with_slice_zero_hardcoded(
                    &circuit,
                    n,
                    rg_freq,
                    slice_zero_hardcoded_rounds,
                    &mut rng,
                );
                ("Slice-zero-hardcoded feistalized", 3 * n)
            } else if slice_zero {
                circuit = feistalize_with_slice_zero(&circuit, n, rg_freq, &mut rng);
                ("Slice-zero feistalized", 3 * n)
            } else {
                circuit = feistalize(&circuit, n, rg_freq, &mut rng);
                ("Feistalized", 3 * n)
            }
        } else {
            circuit = gadgetize(&circuit, n, rg_freq, &mut rng);
            ("Gadgetized", 2 * n)
        };
        println!(
            "{}: {} gates → {} gates, {} wires",
            label,
            before,
            circuit.gates.len(),
            transformed_n
        );
        // Save the transformed circuit to --gadget_path, or ./gadgetized/{final path
        // component of source} when none was supplied.
        let gadget_path = match gadget_path {
            Some(p) => p.to_string(),
            None => {
                let file_name = std::path::Path::new(source)
                    .file_name()
                    .expect("Source path has no final component")
                    .to_str()
                    .expect("Source file name is not valid UTF-8");
                std::fs::create_dir_all("./gadgetized").expect("Failed to create ./gadgetized");
                format!("./gadgetized/{}", file_name)
            }
        };
        if let Some(parent) = std::path::Path::new(&gadget_path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).expect("Failed to create gadget output directory");
            }
        }
        File::create(&gadget_path)
            .and_then(|mut f| f.write_all(circuit.repr().as_bytes()))
            .expect("Failed to write transformed circuit");
        println!("{} circuit written to {}", label, gadget_path);
        if let Some((public_y, public_z)) = &slice_zero_random_public {
            write_slice_zero_random_metadata(
                &gadget_path,
                n,
                slice_zero_random_gates,
                public_y,
                public_z,
            );
        }
    }
    let n = if do_feistalize {
        3 * n
    } else if do_gadgetize {
        2 * n
    } else {
        n
    };
    if leave {
        circuit = interleave(&circuit, n, db);
    }
    let n = if leave { 2 * n } else { n };
    let feistal_original_n = do_feistalize.then_some(if leave { n / 6 } else { n / 3 });
    let sat_cone_range = sat_cone_range(n, feistal_original_n);
    print_sat_cone("initial", &circuit.gates, sat_cone_range);
    if full_shuffle_early {
        // SAMF insertion is equivalence-preserving by construction, so no retry guard.
        insert_wire_m_samfs_every_x(&mut circuit, n, n, 1, db);
        println!("After early full shuffle: {} gates", circuit.gates.len());
        print_sat_cone("after-full-shuffle-early", &circuit.gates, sat_cone_range);
    }
    // --single-end accumulator: SAMF state carried across ALL rounds,
    // undone in one pass after the last round. `total_t` is the composed wire permutation
    // (round order), `total_neg` the pending negation in the current wire space.
    let mut total_t = crate::replace::transpositions::Transpositions {
        transpositions: Vec::new(),
    };
    let mut total_neg = vec![0u8; n];
    let track = crate::replace::replace::track_survivors();
    let mut survivor_tags: Vec<u32> = if track {
        vec![0u32; circuit.gates.len()]
    } else {
        Vec::new()
    };
    let samf_target =
        crate::replace::replace::SAMF_TARGET.load(std::sync::atomic::Ordering::Relaxed);
    let mut m_eff = m;
    let stage_b_enabled = min_gen > 0;
    let stage_b_target_fraction = min_gen_fraction.clamp(0.0, 1.0);
    let grow_permille = (grow_threshold.max(0.0) * 10.0).round() as usize;
    let compress_fraction_permille = (compress_fraction.clamp(0.0, 1.0) * 1000.0).round() as usize;
    let stage_d_enabled = grow_permille > 0 || target_size > 0;
    const STAGE_D_MAX_STAGES: usize = 100_000;
    const STAGE_D_STALL_LIMIT: usize = 8;
    let mut last_compressed = circuit.gates.len();
    let mut best_floor_gen = 0u32;
    let mut best_below_fraction = f64::INFINITY;
    let mut stage_d_stall = 0usize;
    if stage_b_enabled && single_end {
        println!(
            "[stage-B] --single-end ignored while --min-gen is active; each low-gen pass resolves SAMF state immediately"
        );
    }
    if stage_d_enabled {
        println!(
            "[new-sss] bounded Stage B: target_size={} grow_threshold={:.1}% compress_fraction={:.1}% pass_length={} collision_rounds={}",
            target_size,
            grow_permille as f64 / 10.0,
            compress_fraction_permille as f64 / 10.0,
            pass_length,
            collision_rounds
        );
    }

    // Per-round SAMF stats (deltas): inserted / hidden / hide-failed / curated expansions.
    use crate::replace::transpositions::{
        CURATED_REPLACEMENTS_MADE, SAMF_COMPRESSIONS_FAILED, SAMF_COMPRESSIONS_MADE,
        SAMF_HIDE_ATTEMPTS, SAMF_HIDE_ELIGIBLE_EXPANSIONS, SAMF_HIDE_LOOKUP_MISSES,
        SAMF_HIDE_REJECTED_EXPOSED, SAMF_HIDE_SKIPPED_MATERIALIZED, SAMF_INSERTIONS_MADE,
        UNCLEAN_EXPANSIONS,
    };
    use std::sync::atomic::Ordering::Relaxed;
    #[derive(Clone, Copy)]
    struct RoundSamfStats {
        inserted: usize,
        hidden: usize,
        failed: usize,
        curated: usize,
        eligible: usize,
        skipped_materialized: usize,
        attempts: usize,
        lookup_misses: usize,
        rejected_exposed: usize,
        unclean: usize,
    }

    let mut per_round_samf: Vec<RoundSamfStats> = Vec::new();
    let mut prev_ins = SAMF_INSERTIONS_MADE.load(Relaxed);
    let mut prev_made = SAMF_COMPRESSIONS_MADE.load(Relaxed);
    let mut prev_failed = SAMF_COMPRESSIONS_FAILED.load(Relaxed);
    let mut prev_cur = CURATED_REPLACEMENTS_MADE.load(Relaxed);
    let mut prev_eligible = SAMF_HIDE_ELIGIBLE_EXPANSIONS.load(Relaxed);
    let mut prev_skipped = SAMF_HIDE_SKIPPED_MATERIALIZED.load(Relaxed);
    let mut prev_attempts = SAMF_HIDE_ATTEMPTS.load(Relaxed);
    let mut prev_misses = SAMF_HIDE_LOOKUP_MISSES.load(Relaxed);
    let mut prev_rejected = SAMF_HIDE_REJECTED_EXPOSED.load(Relaxed);
    let mut prev_unclean = UNCLEAN_EXPANSIONS.load(Relaxed);
    // A single-end shuffle cannot be reversed back after each round because its SAMF state is
    // intentionally left live. Choose one direction for the complete accumulated shuffle and
    // reverse it back only after the final unsamf.
    let single_end_reversed = single_end && !stage_b_enabled && rand::rng().random_bool(0.5);
    if single_end_reversed {
        println!("Collision-game direction: reversed (complete single-end shuffle)");
        circuit.gates.reverse();
        if track {
            survivor_tags.reverse();
        }
    }
    let mut hardening_rng = rand::rng();
    let mut i = 0usize;
    loop {
        if !stage_d_enabled && i >= rounds {
            break;
        }
        if stage_d_enabled && i >= STAGE_D_MAX_STAGES {
            println!(
                "[new-sss] reached the {}-stage runaway guard; stopping",
                STAGE_D_MAX_STAGES
            );
            break;
        }
        if target_size > 0 && last_compressed >= target_size {
            println!(
                "[new-sss] absolute target-size ceiling reached after compression ({} >= {}); stopping instead of raising the cap",
                last_compressed, target_size
            );
            break;
        }
        let stage_size_cap = new_sss_stage_size_cap(last_compressed, target_size, grow_permille);
        if let Some(cap) = stage_size_cap {
            println!(
                "[new-sss] stage {}: grow from {} to at most {} gates before compression",
                i + 1,
                last_compressed,
                cap
            );
        }
        crate::replace::replace::REC_ROUND.store(i + 1, Relaxed);
        if sat_global_mix_enabled() && i % sat_global_mix_every() == 0 {
            let mix_m = sat_global_mix_m(n);
            if mix_m > 0 {
                let mix_x = circuit.gates.len().saturating_add(1).max(1);
                if track {
                    insert_wire_m_samfs_every_x_tagged(
                        &mut circuit,
                        n,
                        mix_m,
                        mix_x,
                        db,
                        &mut survivor_tags,
                    );
                } else {
                    insert_wire_m_samfs_every_x(&mut circuit, n, mix_m, mix_x, db);
                }
                println!(
                    "[sat-global-mix] round={} swaps={} gates={}",
                    i + 1,
                    mix_m,
                    circuit.gates.len()
                );
                let cone_label = format!("round{}-after-global-mix", i + 1);
                print_sat_cone(&cone_label, &circuit.gates, sat_cone_range);
            }
        }
        if sat_hard_cores_enabled() {
            let count = sat_hard_core_count(n);
            insert_sat_hard_identity_cores(
                &mut circuit,
                n,
                count,
                sat_cone_range,
                &mut hardening_rng,
            );
            let cone_label = format!("round{}-after-hard-cores", i + 1);
            print_sat_cone(&cone_label, &circuit.gates, sat_cone_range);
        }
        if single_end && !stage_b_enabled {
            // Accumulate this round's SAMFs WITHOUT undoing — functionality is intentionally
            // broken between rounds; we undo everything once after the last round (below).
            use crate::replace::transpositions::shuffled_shoot_then_samf_core;
            let (out, t_round, neg_round, _c, out_tags, _) = shuffled_shoot_then_samf_core(
                &circuit.gates,
                n,
                m_eff,
                x,
                gates_ahead_expand,
                gates_ahead_samf,
                type_attempts,
                shooting_times,
                collision_rounds,
                expansion_game,
                db,
                &survivor_tags,
            );
            circuit.gates = out;
            if track {
                survivor_tags = out_tags;
            }
            // Fold this round into the running accumulator: transport the existing pending
            // negation through this round's permutation, then add this round's negation;
            // compose the permutations (existing first, then this round).
            let mut new_total_neg = neg_round;
            for w in 0..n {
                if total_neg[w] == 1 {
                    let cw = t_round.evaluate(w as u16) as usize;
                    new_total_neg[cw] ^= 1;
                }
            }
            total_neg = new_total_neg;
            total_t = total_t.concat(&t_round);
        } else if stage_b_enabled {
            let stage_start = std::time::Instant::now();
            let mut stage_rng = rand::rng();
            let mut passes = 0usize;
            let mut total_replacements = 0usize;
            let mut total_hidden_samfs = 0usize;
            let no_replacement_limit = std::env::var("STAGEB_NO_REPLACEMENT_LIMIT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(128)
                .max(1);
            let mut no_replacement_passes = 0usize;
            let start_progress = generation_progress(&survivor_tags, min_gen);
            println!(
                "[stage-B] round {} start: {}/{} ({:.2}%) at gen>={} | min={} median={} max={}",
                i + 1,
                start_progress.reached,
                start_progress.total,
                100.0 * start_progress.fraction,
                min_gen,
                start_progress.min,
                start_progress.median,
                start_progress.max
            );

            while !generation_goal_met(&survivor_tags, min_gen, stage_b_target_fraction)
                && passes < max_passes.max(1)
                && stage_size_cap.is_none_or(|cap| circuit.gates.len() < cap)
            {
                let reversed = stage_rng.random_bool(0.5);
                if reversed {
                    circuit.gates.reverse();
                    survivor_tags.reverse();
                }
                let Some(anchor) =
                    choose_low_generation_anchor(&survivor_tags, min_gen, &mut stage_rng)
                else {
                    if reversed {
                        circuit.gates.reverse();
                        survivor_tags.reverse();
                    }
                    break;
                };
                crate::replace::replace::REC_PASS.store(passes + 1, Relaxed);
                let pass_result = shuffled_shoot_then_samf_stage_b_pass(
                    &mut circuit,
                    n,
                    m_eff,
                    x,
                    gates_ahead_expand,
                    gates_ahead_samf,
                    type_attempts,
                    collision_rounds,
                    expansion_game,
                    db,
                    &mut survivor_tags,
                    anchor,
                    pass_length,
                    stage_size_cap,
                );
                if reversed {
                    circuit.gates.reverse();
                    survivor_tags.reverse();
                }
                passes += 1;
                if pass_result.cap_rejected {
                    println!(
                        "[new-sss] round {} pass {} stopped at the hard size ceiling: rejected {}-gate candidate (cap {})",
                        i + 1,
                        passes,
                        pass_result.candidate_gates,
                        stage_size_cap.unwrap()
                    );
                    break;
                }
                total_replacements += pass_result.replacements;
                total_hidden_samfs += pass_result.hidden_samfs;
                if pass_result.replacements == 0 {
                    no_replacement_passes += 1;
                } else {
                    no_replacement_passes = 0;
                }

                if passes <= 3 || passes % 25 == 0 {
                    let progress = generation_progress(&survivor_tags, min_gen);
                    println!(
                        "[stage-B] round {} pass {}: repl={} hidden_samfs={} gates={} progress={}/{} ({:.2}%) min={} median={} max={}",
                        i + 1,
                        passes,
                        pass_result.replacements,
                        pass_result.hidden_samfs,
                        circuit.gates.len(),
                        progress.reached,
                        progress.total,
                        100.0 * progress.fraction,
                        progress.min,
                        progress.median,
                        progress.max
                    );
                }

                if no_replacement_passes >= no_replacement_limit {
                    println!(
                        "[stage-B] round {} stopping after {} consecutive passes with no replacements",
                        i + 1,
                        no_replacement_passes
                    );
                    break;
                }
            }

            if let Some(cap) = stage_size_cap {
                if circuit.gates.len() >= cap
                    && !generation_goal_met(&survivor_tags, min_gen, stage_b_target_fraction)
                {
                    println!(
                        "[new-sss] stage {} size cap reached at {} gates (cap {}); pausing Stage B for compression",
                        i + 1,
                        circuit.gates.len(),
                        cap
                    );
                }
            }

            let end_progress = generation_progress(&survivor_tags, min_gen);
            println!(
                "[stage-B] round {} done: passes={} replacements={} hidden_samfs={} elapsed_ms={} progress={}/{} ({:.2}%) target={:.2}%",
                i + 1,
                passes,
                total_replacements,
                total_hidden_samfs,
                stage_start.elapsed().as_millis(),
                end_progress.reached,
                end_progress.total,
                100.0 * end_progress.fraction,
                100.0 * stage_b_target_fraction
            );
        } else {
            // Choose the shooting direction outside the collision game. Reversal surrounds the
            // complete shooting + plain-SAMF + unsamf operation so no pending SAMF state is
            // reversed as though it were an ordinary gate sequence.
            use crate::replace::transpositions::shuffled_shoot_then_samf;
            let reversed = rand::rng().random_bool(0.5);
            println!(
                "Collision-game direction: {}",
                if reversed { "reversed" } else { "forward" }
            );
            if reversed {
                circuit.gates.reverse();
                if track {
                    survivor_tags.reverse();
                }
            }
            shuffled_shoot_then_samf(
                &mut circuit,
                n,
                m_eff,
                x,
                gates_ahead_expand,
                gates_ahead_samf,
                type_attempts,
                shooting_times,
                collision_rounds,
                expansion_game,
                db,
                &mut survivor_tags,
            );
            if reversed {
                circuit.gates.reverse();
                if track {
                    survivor_tags.reverse();
                }
            }
        }
        println!("After collision game: {} gates", circuit.gates.len());
        let cone_label = format!("round{}-after-collision-game", i + 1);
        print_sat_cone(&cone_label, &circuit.gates, sat_cone_range);
        if full_shuffle {
            // SAMF insertion is equivalence-preserving by construction, so no retry guard.
            if track {
                insert_wire_m_samfs_every_x_tagged(&mut circuit, n, n, 1, db, &mut survivor_tags);
            } else {
                insert_wire_m_samfs_every_x(&mut circuit, n, n, 1, db);
            }
            println!("After full shuffle: {} gates", circuit.gates.len());
            let cone_label = format!("round{}-after-full-shuffle", i + 1);
            print_sat_cone(&cone_label, &circuit.gates, sat_cone_range);
        }
        // --single-end: after the FINAL round's shuffle, before its compression, undo ALL
        // accumulated SAMFs/NOTs in one pass — restoring equivalence to the original input.
        if single_end && !stage_b_enabled && i == rounds - 1 {
            if track {
                use crate::replace::transpositions::apply_unsamf_tagged;
                apply_unsamf_tagged(
                    &mut circuit.gates,
                    &total_t,
                    &total_neg,
                    n,
                    db,
                    &mut survivor_tags,
                );
            } else {
                use crate::replace::transpositions::apply_unsamf;
                apply_unsamf(&mut circuit.gates, &total_t, &total_neg, n, db);
            }
            println!("After single-end unsamf: {} gates", circuit.gates.len());
            if single_end_reversed {
                circuit.gates.reverse();
                if track {
                    survivor_tags.reverse();
                }
                println!("Restored forward circuit direction");
            }
        }
        let stage_compress_target = if stage_d_enabled && compress_fraction_permille > 0 {
            let basis = if target_size > 0 {
                target_size
            } else {
                circuit.gates.len()
            };
            let target = ((basis as u128 * compress_fraction_permille as u128) / 1000) as usize;
            println!(
                "[new-sss] stage {} compression target: {:.1}% of {} = {} gates",
                i + 1,
                compress_fraction_permille as f64 / 10.0,
                basis,
                target
            );
            Some(target.max(1))
        } else {
            None
        };
        circuit = compress_loop(
            &circuit,
            n,
            db,
            stable_compressions,
            i + 1,
            rounds,
            "temp_compression.txt",
            stage_compress_target,
            &mut survivor_tags,
        );
        println!("After compression: {} gates", circuit.gates.len());
        let cone_label = format!("round{}-after-compression", i + 1);
        print_sat_cone(&cone_label, &circuit.gates, sat_cone_range);
        // Record + print this round's SAMF stats (deltas from the previous round).
        {
            let ins = SAMF_INSERTIONS_MADE.load(Relaxed);
            let made = SAMF_COMPRESSIONS_MADE.load(Relaxed);
            let failed = SAMF_COMPRESSIONS_FAILED.load(Relaxed);
            let cur = CURATED_REPLACEMENTS_MADE.load(Relaxed);
            let eligible = SAMF_HIDE_ELIGIBLE_EXPANSIONS.load(Relaxed);
            let skipped = SAMF_HIDE_SKIPPED_MATERIALIZED.load(Relaxed);
            let attempts = SAMF_HIDE_ATTEMPTS.load(Relaxed);
            let misses = SAMF_HIDE_LOOKUP_MISSES.load(Relaxed);
            let rejected = SAMF_HIDE_REJECTED_EXPOSED.load(Relaxed);
            let unclean = UNCLEAN_EXPANSIONS.load(Relaxed);
            let d_ins = ins - prev_ins;
            let d_made = made - prev_made;
            let d_failed = failed - prev_failed;
            let d_cur = cur - prev_cur;
            let d_eligible = eligible - prev_eligible;
            let d_skipped = skipped - prev_skipped;
            let d_attempts = attempts - prev_attempts;
            let d_misses = misses - prev_misses;
            let d_rejected = rejected - prev_rejected;
            let d_unclean = unclean - prev_unclean;
            println!(
                "  Round {}/{} SAMFs inserted: {} (hidden {}, plain {}) | curated expansions: {} | hide-fails: {}",
                i + 1,
                rounds,
                d_ins,
                d_made,
                d_ins.saturating_sub(d_made),
                d_cur,
                d_failed
            );
            println!(
                "    hide diagnostics: eligible {} | skipped-materialized {} | attempts {} | lookup-misses {} | rejected-exposed {} | unclean-absorbed {}",
                d_eligible, d_skipped, d_attempts, d_misses, d_rejected, d_unclean
            );
            if samf_target > 0 && m_eff > 0 && d_made >= samf_target {
                println!(
                    "    [samf-target] round hid {} >= target {}; disabling plain-SAMF insertion (m {} -> 0)",
                    d_made, samf_target, m_eff
                );
                m_eff = 0;
            }
            per_round_samf.push(RoundSamfStats {
                inserted: d_ins,
                hidden: d_made,
                failed: d_failed,
                curated: d_cur,
                eligible: d_eligible,
                skipped_materialized: d_skipped,
                attempts: d_attempts,
                lookup_misses: d_misses,
                rejected_exposed: d_rejected,
                unclean: d_unclean,
            });
            prev_ins = ins;
            prev_made = made;
            prev_failed = failed;
            prev_cur = cur;
            prev_eligible = eligible;
            prev_skipped = skipped;
            prev_attempts = attempts;
            prev_misses = misses;
            prev_rejected = rejected;
            prev_unclean = unclean;
        }
        if circuit.gates.len() == 0 {
            break;
        }

        if !stage_d_enabled {
            if circuit.gates.len() == post_len {
                count += 1;
            } else {
                post_len = circuit.gates.len();
                count = 0;
            }

            if count > 2 {
                break;
            }
        }
        let mut j = 0;
        while j < circuit.gates.len().saturating_sub(1) {
            if circuit.gates[j] == circuit.gates[j + 1] {
                // remove elements at i and i+1
                circuit.gates.drain(j..=j + 1);
                if track {
                    survivor_tags.drain(j..=j + 1);
                }

                // step back up to 2 indices, but not below 0
                j = j.saturating_sub(2);
            } else {
                j += 1;
            }
        }
        // Optional dispersal pass (DISPERSE=1): reorder into a random linear extension of
        // the dependency DAG so replacement windows stop sitting as dense contiguous blocks.
        // Function-preserving; runs before the equality check so checked runs verify it.
        if crate::replace::disperse::disperse_enabled() && !circuit.gates.is_empty() {
            use crate::replace::disperse::{disperse_random_topo, fanout_stats, leeway_stats};
            let stride = (circuit.gates.len() / 100_000).max(1);
            let before = leeway_stats(&circuit.gates, stride);
            let seed = crate::replace::disperse::disperse_seed() ^ ((i as u64 + 1) << 32);
            let chunk = crate::replace::disperse::disperse_chunk_size();
            if track {
                disperse_random_topo(&mut circuit.gates, Some(&mut survivor_tags), chunk, seed);
            } else {
                disperse_random_topo(&mut circuit.gates, None, chunk, seed);
            }
            let after = leeway_stats(&circuit.gates, stride);
            let (fanout, fanout_zero) = fanout_stats(&circuit.gates, stride);
            // Usage entropy near 1.0 means the remaining leeway/fanout gap is schedule-side;
            // low entropy means it is structural (uneven wire usage) and reordering alone
            // cannot close it.
            let (active_entropy, control_entropy) =
                crate::replace::disperse::wire_usage_entropy(&circuit.gates, n);
            println!(
                "[disperse] round {} leeway median {}->{} avg {:.1}->{:.1} p99 {}->{} | fanout p99 {} max {} zero {:.1}% | usage entropy active {:.3} control {:.3}",
                i + 1,
                before.median,
                after.median,
                before.avg,
                after.avg,
                before.p99,
                after.p99,
                fanout.p99,
                fanout.max,
                100.0 * fanout_zero,
                active_entropy,
                control_entropy
            );
        }
        if equality_check {
            let original_n = if leave { n / 2 } else { n };
            let original_n = if do_feistalize {
                original_n / 3
            } else if do_gadgetize {
                original_n / 2
            } else {
                original_n
            };
            let functionality_ok = if do_feistalize {
                feistal_middle_matches_original(
                    &c,
                    &circuit,
                    original_n,
                    n,
                    1_000,
                    fixed_feistal_slice,
                )
            } else {
                c.probably_equal(&circuit, original_n, 1_000)
            };
            if functionality_ok.is_err() {
                panic!("The functionality has changed");
            }
        }
        {
            // Stage D keeps one checkpoint per bounded shoot/compress cadence; fixed-round
            // mode retains the historical round names.
            let checkpoint_label = if stage_d_enabled { "stage" } else { "round" };
            let round_path = format!("{}{}{}.txt", save_base, checkpoint_label, i + 1);
            println!("Writing {} {} to {}", checkpoint_label, i + 1, round_path);
            File::create(&round_path)
                .and_then(|mut f| f.write_all(circuit.repr().as_bytes()))
                .expect("Failed to write round circuit");
        }
        i += 1;
        if stage_d_enabled {
            last_compressed = circuit.gates.len();
            let progress = generation_progress(&survivor_tags, min_gen);
            let below_fraction = 1.0 - progress.fraction;
            let skip = (((1.0 - stage_b_target_fraction) * survivor_tags.len() as f64).floor()
                as usize)
                .min(survivor_tags.len().saturating_sub(1));
            let floor_gen = if survivor_tags.is_empty() {
                0
            } else {
                let mut sorted = survivor_tags.clone();
                *sorted.select_nth_unstable(skip).1
            };
            println!(
                "[new-sss] stage {} progress: {}/{} ({:.2}%) at gen>={} target={:.2}% floor_gen={} abs_min={} compressed_gates={}",
                i,
                progress.reached,
                progress.total,
                100.0 * progress.fraction,
                min_gen,
                100.0 * stage_b_target_fraction,
                floor_gen,
                progress.min,
                circuit.gates.len()
            );
            if generation_goal_met(&survivor_tags, min_gen, stage_b_target_fraction) {
                println!(
                    "[new-sss] min-generation condition met after {} stage(s); stopping",
                    i
                );
                break;
            }

            let progressed =
                floor_gen > best_floor_gen || below_fraction + f64::EPSILON < best_below_fraction;
            if progressed {
                best_floor_gen = best_floor_gen.max(floor_gen);
                best_below_fraction = best_below_fraction.min(below_fraction);
                stage_d_stall = 0;
            } else {
                stage_d_stall += 1;
                if stage_d_stall >= STAGE_D_STALL_LIMIT {
                    println!(
                        "[new-sss] no generation progress for {} stages (best floor {}, best below target {:.2}%); stopping",
                        stage_d_stall,
                        best_floor_gen,
                        100.0 * best_below_fraction
                    );
                    break;
                }
            }
        }
    }

    println!("Final len: {}", circuit.gates.len());
    print_sat_cone("final", &circuit.gates, sat_cone_range);
    // Compare against the original circuit. Gadgetize/interleave preserve the low original_n
    // outputs; feistalize preserves C(x) in the middle original_n-wire block as y ^ C(x).
    let original_n = if leave { n / 2 } else { n };
    let original_n = if do_feistalize {
        original_n / 3
    } else if do_gadgetize {
        original_n / 2
    } else {
        original_n
    };
    if equality_check {
        if do_feistalize {
            feistal_middle_matches_original(
                &c,
                &circuit,
                original_n,
                n,
                10_000,
                fixed_feistal_slice,
            )
            .expect("The circuits differ somewhere!");
        } else {
            circuit
                .probably_equal(&c, original_n, 10_000)
                .expect("The circuits differ somewhere!");
        }
    }

    // Write to file
    let circuit_str = circuit.repr();
    File::create(save)
        .and_then(|mut f| f.write_all(circuit_str.as_bytes()))
        .expect("Failed to write recent_circuit.txt");
    if let Some((public_y, public_z)) = &slice_zero_random_public {
        write_slice_zero_random_metadata(
            save,
            original_n,
            slice_zero_random_gates,
            public_y,
            public_z,
        );
    }

    println!("Final circuit written to {}", save);
    if track && crate::replace::replace::gen_mode() {
        let mut hist = std::collections::BTreeMap::<u32, usize>::new();
        for &tag in &survivor_tags {
            *hist.entry(tag).or_insert(0) += 1;
        }
        let total = survivor_tags.len().max(1);
        let min_gen = survivor_tags.iter().copied().min().unwrap_or(0);
        let max_gen = survivor_tags.iter().copied().max().unwrap_or(0);
        let mut sorted = survivor_tags.clone();
        sorted.sort_unstable();
        let median_gen = sorted.get(sorted.len() / 2).copied().unwrap_or(0);
        let path = format!("{}.generations", save);
        match File::create(&path) {
            Ok(mut f) => {
                let _ = writeln!(
                    f,
                    "# generation histogram of final circuit. n_gates={} min_gen={} median_gen={} max_gen={}\n\
                     # generation count fraction",
                    total, min_gen, median_gen, max_gen
                );
                for (generation, count) in hist {
                    let _ = writeln!(
                        f,
                        "{} {} {:.4}",
                        generation,
                        count,
                        count as f64 / total as f64
                    );
                }
                println!(
                    "Generations: n_gates={} min={} median={} max={} -> {}",
                    total, min_gen, median_gen, max_gen, path
                );
            }
            Err(e) => eprintln!("Failed to write generations file {}: {}", path, e),
        }
    }

    {
        println!("--- SAMF stats per round ---");
        let (
            mut t_ins,
            mut t_made,
            mut t_failed,
            mut t_cur,
            mut t_eligible,
            mut t_skipped,
            mut t_attempts,
            mut t_misses,
            mut t_rejected,
            mut t_unclean,
        ) = (
            0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize,
        );
        for (r, stats) in per_round_samf.iter().enumerate() {
            println!(
                "Round {}: inserted {} (hidden {}, plain {}) | curated expansions {} | hide-fails {}",
                r + 1,
                stats.inserted,
                stats.hidden,
                stats.inserted.saturating_sub(stats.hidden),
                stats.curated,
                stats.failed
            );
            println!(
                "  hide diagnostics: eligible {} | skipped-materialized {} | attempts {} | lookup-misses {} | rejected-exposed {} | unclean-absorbed {}",
                stats.eligible,
                stats.skipped_materialized,
                stats.attempts,
                stats.lookup_misses,
                stats.rejected_exposed,
                stats.unclean
            );
            t_ins += stats.inserted;
            t_made += stats.hidden;
            t_failed += stats.failed;
            t_cur += stats.curated;
            t_eligible += stats.eligible;
            t_skipped += stats.skipped_materialized;
            t_attempts += stats.attempts;
            t_misses += stats.lookup_misses;
            t_rejected += stats.rejected_exposed;
            t_unclean += stats.unclean;
        }
        println!(
            "Total (this run): SAMFs inserted {} (hidden {}, plain {}) | curated expansions {} | hide-fails {}",
            t_ins,
            t_made,
            t_ins.saturating_sub(t_made),
            t_cur,
            t_failed
        );
        println!(
            "Total hide diagnostics: eligible {} | skipped-materialized {} | attempts {} | lookup-misses {} | rejected-exposed {} | unclean-absorbed {}",
            t_eligible, t_skipped, t_attempts, t_misses, t_rejected, t_unclean
        );
    }
    if let Some((public_y, public_z)) = &slice_zero_random_public {
        print_slice_zero_random_public_slice(original_n, public_y, public_z);
    }
}

#[cfg(test)]
mod new_sss_size_cap_tests {
    use super::new_sss_stage_size_cap;

    #[test]
    fn absolute_target_never_drifts_above_requested_ceiling() {
        assert_eq!(new_sss_stage_size_cap(1_196_261, 500_000, 0), Some(500_000));
    }

    #[test]
    fn relative_growth_keeps_rounding_progress() {
        assert_eq!(new_sss_stage_size_cap(7, 0, 1), Some(8));
        assert_eq!(new_sss_stage_size_cap(1_000, 0, 150), Some(1_150));
    }

    #[test]
    fn legacy_fixed_round_mode_has_no_size_cap() {
        assert_eq!(new_sss_stage_size_cap(42_000, 0, 0), None);
    }
}
