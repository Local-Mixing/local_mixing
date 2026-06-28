use std::{fs::File, io::Write};

use rand::{Rng, RngCore};

use crate::{
    circuit::circuit::{CircuitSeq, Gate, U1024},
    replace::{
        gadgets::{
            feistalize, feistalize_with_slice_zero, feistalize_with_slice_zero_hardcoded,
            feistalize_with_slice_zero_random, gadgetize, packed_bit,
        },
        pairs::interleave,
        replace::{ExpandPairMode, compress_loop, expand_once},
        transpositions::insert_wire_m_samfs_every_x,
    },
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// Open all dbs ahead of time in the LMDB
// LMDB used for fast reads
// nXmY store the canonicalized (up to gate ordering and wire relabeling) version of all the circuits
// perms_tables_nX store a list of tables that share a permutation. Legacy use for building random identities
// nXmYperms stores all circuits canonicalized only up to gate ordering
// ids_nXgK stores identities on X wires with gate pair taxonomy K on the first two gates. See Taxonomies to_int to see
// Last row of tables is used for swapping wires, CNOTS, NOTS
pub fn open_curated_shard_dbs(env: &lmdb::Environment) -> Vec<lmdb::Database> {
    (0u16..=255)
        .map(|s| {
            let name = format!("curated_{:02x}", s);
            env.open_db(Some(name.as_str()))
                .unwrap_or_else(|e| panic!("Failed to open curated shard db {:02x}: {:?}", s, e))
        })
        .collect()
}

pub fn open_shard_dbs(env: &lmdb::Environment) -> Vec<lmdb::Database> {
    (0u16..=255)
        .map(|s| {
            let name = format!("{:02x}", s);
            env.open_db(Some(name.as_str()))
                .unwrap_or_else(|e| panic!("Failed to open shard db {:02x}: {:?}", s, e))
        })
        .collect()
}

pub fn open_all_dbs(env: &lmdb::Environment) -> (Vec<lmdb::Database>, Vec<lmdb::Database>) {
    let shard_dbs = open_shard_dbs(env);
    let curated_shard_dbs = open_curated_shard_dbs(env);
    (shard_dbs, curated_shard_dbs)
}

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
    env: &lmdb::Environment,
    shard_dbs: &[lmdb::Database],
    curated_shard_dbs: &[lmdb::Database],
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
    egg: bool,
    equality_check: bool,
    rg_freq: usize,
    single_end: bool,
    light_compression: bool,
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
        circuit = interleave(&circuit, n, env);
    }
    let n = if leave { 2 * n } else { n };
    let feistal_original_n = do_feistalize.then_some(if leave { n / 6 } else { n / 3 });
    let sat_cone_range = sat_cone_range(n, feistal_original_n);
    print_sat_cone("initial", &circuit.gates, sat_cone_range);
    if full_shuffle_early {
        // SAMF insertion is equivalence-preserving by construction, so no retry guard.
        insert_wire_m_samfs_every_x(&mut circuit, n, n, 1, env, curated_shard_dbs, shard_dbs);
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

    // Per-round SAMF stats (deltas): inserted / hidden / hide-failed / curated expansions.
    use crate::replace::transpositions::{
        CURATED_REPLACEMENTS_MADE, SAMF_COMPRESSIONS_FAILED, SAMF_COMPRESSIONS_MADE,
        SAMF_HIDE_ATTEMPTS, SAMF_HIDE_ELIGIBLE_EXPANSIONS, SAMF_HIDE_LOOKUP_MISSES,
        SAMF_HIDE_REJECTED_EXPOSED, SAMF_HIDE_SKIPPED_MATERIALIZED, SAMF_INSERTIONS_MADE,
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
    // A single-end shuffle cannot be reversed back after each round because its SAMF state is
    // intentionally left live. Choose one direction for the complete accumulated shuffle and
    // reverse it back only after the final unsamf.
    let single_end_reversed = single_end && rand::rng().random_bool(0.5);
    if single_end_reversed {
        println!("Collision-game direction: reversed (complete single-end shuffle)");
        circuit.gates.reverse();
    }
    let mut hardening_rng = rand::rng();
    // --track-survivors: tag every gate present right now (after feistelization, before local
    // mixing) with its index; maintain the tag vector in lockstep through the rounds. Originals
    // still present at the end were never part of any replacement.
    let track = crate::replace::replace::track_survivors();
    let orig_gates: Vec<[u16; 3]> = if track {
        circuit.gates.clone()
    } else {
        Vec::new()
    };
    let n_orig = orig_gates.len();
    // Tag vector: gen mode -> all generation 0; survivor mode -> origin index 0..n_orig.
    let mut survivor_tags: Vec<u32> = if !track {
        Vec::new()
    } else if crate::replace::replace::gen_mode() {
        vec![0u32; n_orig]
    } else {
        (0..n_orig as u32).collect()
    };
    // Adaptive plain-SAMF reduction (ssg --samf-target): the plain-SAMF count actually used each
    // round. Starts at the requested `m`; once a round hides >= SAMF_TARGET SAMFs in the shooting
    // game, this is dropped to 0 (enough scrambling SAMFs are already woven in).
    let samf_target = crate::replace::replace::SAMF_TARGET.load(Relaxed);
    let mut m_eff = m;
    // ---- Stage D (size-threshold compression cadence) ----
    // When GROW_THRESHOLD_PERMILLE > 0 the fixed `-r` round count is ignored: instead we run
    // shoot/compress "stages" until the min-gen condition is met. Each stage shoots until the
    // working circuit reaches (1 + grow) * (size at the end of the previous compression), then
    // compresses all the way down and saves the result. The min-gen condition is the stop rule.
    let grow_permille = crate::replace::replace::GROW_THRESHOLD_PERMILLE.load(Relaxed);
    // Stage D per-stage compression target: stop compressing once the circuit is this fraction
    // (permille) of its post-shooting size. 0 = compress fully each stage.
    let compress_fraction_permille = crate::replace::replace::COMPRESS_FRACTION_PERMILLE.load(Relaxed);
    // TARGET_SIZE: absolute steady-state size = the held/final size. When > 0, each stage shoots
    // until the circuit reaches target_size (the cap), then compresses back down to
    // compress_fraction * target_size, instead of the relative grow-threshold cadence. At the
    // incompressibility ceiling the circuit pins at target_size. Stage D is on if either mechanism
    // is requested.
    let target_size = crate::replace::replace::TARGET_SIZE.load(Relaxed);
    let stage_d = grow_permille > 0 || target_size > 0;
    // Safety bound on the number of stages (a runaway guard if the min-gen target is unreachable).
    const STAGE_D_MAX_STAGES: usize = 100_000;
    // Last post-compression size; the per-stage shoot grows from here. Seed with the current size
    // (after gadgetization / early shuffles).
    let mut last_compressed = circuit.gates.len();
    // No-progress guard: progress is the minimum generation (the floor) reaching a new high — a
    // monotonic signal — OR the per-mille of gates still below min-gen reaching a new best. Stop
    // only if NEITHER improves for several stages.
    let mut best_min_gen: u32 = 0;
    let mut best_below_permille = usize::MAX;
    let mut stall = 0usize;
    // Last stage whose per-stage equality check PASSED (path + number). On a later equality failure
    // we deliver this verified-equal stage instead of panicking and losing the whole run.
    let mut last_good_path: Option<String> = None;
    let mut last_good_stage = 0usize;
    if stage_d {
        if target_size > 0 {
            println!(
                "[stage-D] target-size cadence ON | hold final size ≈ {} (shoot to {}, compress to {:.0}% = {}) | stopping on min-gen (-r ignored)",
                target_size,
                target_size,
                compress_fraction_permille as f64 / 10.0,
                (target_size as u128 * compress_fraction_permille as u128 / 1000) as usize
            );
        } else {
            println!(
                "[stage-D] size-threshold cadence ON | grow {:.1}% per stage | stopping on min-gen (-r ignored)",
                grow_permille as f64 / 10.0
            );
        }
    }
    let mut i = 0usize;
    loop {
        if !stage_d && i >= rounds {
            break;
        }
        if stage_d && i >= STAGE_D_MAX_STAGES {
            println!("[stage-D] reached safety cap of {} stages; stopping", STAGE_D_MAX_STAGES);
            break;
        }
        if stage_d {
            // Target-size mode: shoot until target_size (the cap = held/final size). Otherwise the
            // relative grow-threshold cadence: shoot until `grow` larger than the last compressed size.
            let cap = if target_size > 0 {
                target_size
            } else {
                ((last_compressed as u128 * (1000 + grow_permille as u128)) / 1000) as usize
            };
            let cap = cap.max(last_compressed + 1);
            crate::replace::replace::SHOOT_SIZE_CAP
                .store(cap, std::sync::atomic::Ordering::Relaxed);
            println!(
                "[stage-D] stage {} | last_compressed {} | shoot size cap {}",
                i + 1,
                last_compressed,
                cap
            );
        }
        crate::replace::replace::REC_ROUND.store(i + 1, std::sync::atomic::Ordering::Relaxed);
        if sat_global_mix_enabled() && i % sat_global_mix_every() == 0 {
            let mix_m = sat_global_mix_m(n);
            if mix_m > 0 {
                let mix_x = circuit.gates.len().saturating_add(1).max(1);
                insert_wire_m_samfs_every_x(
                    &mut circuit,
                    n,
                    mix_m,
                    mix_x,
                    env,
                    curated_shard_dbs,
                    shard_dbs,
                );
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
        if egg {
            let pair_mode = ExpandPairMode::Curated {
                curated_shard_dbs: &curated_shard_dbs,
            };
            circuit = expand_once(&circuit, n, env, shard_dbs, &pair_mode);
        } else if single_end {
            // Accumulate this round's SAMFs WITHOUT undoing — functionality is intentionally
            // broken between rounds; we undo everything once after the last round (below).
            use crate::replace::transpositions::shuffled_shoot_then_samf_core;
            let (out, t_round, neg_round, _c, _tags) = shuffled_shoot_then_samf_core(
                &circuit.gates,
                n,
                m_eff,
                x,
                gates_ahead_expand,
                gates_ahead_samf,
                type_attempts,
                shooting_times,
                env,
                curated_shard_dbs,
                shard_dbs,
                &[],
            );
            circuit.gates = out;
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
                env,
                curated_shard_dbs,
                shard_dbs,
                &mut survivor_tags,
            );
            if reversed {
                circuit.gates.reverse();
                if track {
                    survivor_tags.reverse();
                }
            }
        }
        println!("After shooting game: {} gates", circuit.gates.len());
        let cone_label = format!("round{}-after-shooting", i + 1);
        print_sat_cone(&cone_label, &circuit.gates, sat_cone_range);
        // The normal shooting path already inserts plain SAMFs as part of
        // shuffled_shoot_then_samf[_core]. Egg mode does not use that path, so it performs its
        // one plain-SAMF insertion here.
        if egg {
            insert_wire_m_samfs_every_x(&mut circuit, n, m, x, env, curated_shard_dbs, shard_dbs);
            println!("After inserting samfs: {} gates", circuit.gates.len());
        }
        if full_shuffle {
            // SAMF insertion is equivalence-preserving by construction, so no retry guard.
            insert_wire_m_samfs_every_x(&mut circuit, n, n, 1, env, curated_shard_dbs, shard_dbs);
            println!("After full shuffle: {} gates", circuit.gates.len());
            let cone_label = format!("round{}-after-full-shuffle", i + 1);
            print_sat_cone(&cone_label, &circuit.gates, sat_cone_range);
        }
        // --single-end: after the FINAL round's shuffle, before its compression, undo ALL
        // accumulated SAMFs/NOTs in one pass — restoring equivalence to the original input.
        if single_end && i == rounds - 1 {
            use crate::replace::transpositions::apply_unsamf;
            apply_unsamf(
                &mut circuit.gates,
                &total_t,
                &total_neg,
                n,
                env,
                curated_shard_dbs,
                shard_dbs,
                &mut Vec::new(),
            );
            println!("After single-end unsamf: {} gates", circuit.gates.len());
            if single_end_reversed {
                circuit.gates.reverse();
                println!("Restored forward circuit direction");
            }
        }
        // Stage D compression target. Target-size mode compresses to compress_fraction * target_size
        // (so the circuit oscillates between that and the target_size cap, pinning at target_size at
        // the ceiling); otherwise to `compress_fraction` of the POST-shooting size (0 => full).
        let stage_d_compress_target = if stage_d && target_size > 0 && compress_fraction_permille > 0 {
            let t = ((target_size as u128 * compress_fraction_permille as u128) / 1000) as usize;
            println!(
                "[stage-D] compress target: {:.1}% of target-size {} = {} gates",
                compress_fraction_permille as f64 / 10.0,
                target_size,
                t
            );
            Some(t)
        } else if stage_d && target_size > 0 {
            // target-size with no compress-fraction: compress fully each stage (max amplitude).
            None
        } else if stage_d && compress_fraction_permille > 0 {
            let post_shoot = circuit.gates.len();
            let t = ((post_shoot as u128 * compress_fraction_permille as u128) / 1000) as usize;
            println!(
                "[stage-D] compress target: {:.1}% of {} = {} gates",
                compress_fraction_permille as f64 / 10.0,
                post_shoot,
                t
            );
            Some(t)
        } else {
            None
        };
        // Compression convergence window (stop when total reduction over the last STABLE_MAX
        // iterations is < 50 gates). Overridable via env STABLE_MAX; default 6.
        let stable_max = std::env::var("STABLE_MAX")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .filter(|&v| v >= 1)
            .unwrap_or(6);
        circuit = compress_loop(
            &circuit,
            n,
            env,
            shard_dbs,
            stable_max,
            i + 1,
            rounds,
            "temp_compression.txt",
            // Light compression applies only between rounds; the final round always compresses all
            // the way to the end. Stage D uses an explicit per-stage target instead (below).
            light_compression && !stage_d && (i + 1 < rounds),
            stage_d_compress_target,
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
            let d_ins = ins - prev_ins;
            let d_made = made - prev_made;
            let d_failed = failed - prev_failed;
            let d_cur = cur - prev_cur;
            let d_eligible = eligible - prev_eligible;
            let d_skipped = skipped - prev_skipped;
            let d_attempts = attempts - prev_attempts;
            let d_misses = misses - prev_misses;
            let d_rejected = rejected - prev_rejected;
            let unclean = crate::replace::transpositions::UNCLEAN_EXPANSIONS.load(Relaxed);
            println!(
                "  Round {}/{} SAMFs inserted: {} (hidden {}, plain {}) | curated expansions: {} (unclean/#10: {}) | hide-fails: {}",
                i + 1,
                rounds,
                d_ins,
                d_made,
                d_ins.saturating_sub(d_made),
                d_cur,
                unclean,
                d_failed
            );
            // Hidden-SAMF success rate: of the curated expansions where a hide was attempted, the
            // fraction where a SAMF was actually absorbed (made / (made + failed)); also relative
            // to all hide-eligible expansions.
            let hide_denom = (d_made + d_failed).max(1);
            let elig_denom = d_eligible.max(1);
            println!(
                "    hidden-SAMF success: {:.1}% of attempts ({}/{}), {:.1}% of eligible expansions ({}/{})",
                100.0 * d_made as f64 / hide_denom as f64,
                d_made,
                d_made + d_failed,
                100.0 * d_made as f64 / elig_denom as f64,
                d_made,
                d_eligible
            );
            println!(
                "    hide diagnostics: eligible {} | skipped-materialized {} | attempts {} | lookup-misses {} | rejected-exposed {}",
                d_eligible, d_skipped, d_attempts, d_misses, d_rejected
            );
            // Adaptive plain-SAMF reduction: once a round hides enough SAMFs on its own, stop the
            // explicit plain-SAMF insertion for the remaining rounds.
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
        }
        if circuit.gates.len() == 0 {
            break;
        }

        if circuit.gates.len() == post_len {
            count += 1;
        } else {
            post_len = circuit.gates.len();
            count = 0;
        }

        if count > 2 {
            break;
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
        if track {
            // Per-round survivor count (after this round's shooting + compression).
            let alive = survivor_tags
                .iter()
                .filter(|&&t| (t as usize) < n_orig)
                .count();
            println!(
                "[survivors] round {} alive {} circuit_gates {}",
                i + 1,
                alive,
                circuit.gates.len()
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
                // Graceful stop instead of panic: this stage's circuit is NOT equivalent, but the
                // previous stage WAS verified-equal and is on disk. Deliver it to --destination and
                // stop, so a rare break does not throw away the whole (often multi-hour) run.
                eprintln!(
                    "[equality] FAILURE at stage {}: functionality changed. Stopping cadence.",
                    i + 1
                );
                match &last_good_path {
                    Some(p) => {
                        std::fs::copy(p, save).expect("Failed to copy last verified-equal stage");
                        println!(
                            "[equality] delivered last verified-equal stage {} -> {} (stage {} broke; its circuit discarded)",
                            last_good_stage, save, i + 1
                        );
                    }
                    None => {
                        eprintln!(
                            "[equality] no earlier verified-equal stage exists (stage 1 broke); nothing written to {}.",
                            save
                        );
                    }
                }
                print_sat_cone("final-after-equality-break", &circuit.gates, sat_cone_range);
                return;
            }
        }
        {
            // Save this stage's circuit to its own file: same path as -d but with `round{n}`
            // (rounds mode) or `stage{n}` (Stage D) inserted before the .txt extension. This is
            // the per-compression-stage checkpoint. In --equality_check mode this is reached only
            // after the stage passed equality, so it is the latest verified-equal circuit.
            let label = if stage_d { "stage" } else { "round" };
            let round_path = format!("{}{}{}.txt", save_base, label, i + 1);
            println!("Writing {} {} to {}", label, i + 1, round_path);
            File::create(&round_path)
                .and_then(|mut f| f.write_all(circuit.repr().as_bytes()))
                .expect("Failed to write stage circuit");
            if equality_check {
                last_good_path = Some(round_path);
                last_good_stage = i + 1;
            }
        }
        i += 1;
        if stage_d {
            // The size the next stage grows from.
            last_compressed = circuit.gates.len();
            // Evaluate the min-gen condition on the COMPRESSED circuit (the saved artifact).
            let min_gen_target = crate::replace::replace::MIN_GEN.load(Relaxed) as u32;
            let permille = crate::replace::replace::MIN_GEN_PERMILLE.load(Relaxed);
            let len = survivor_tags.len();
            let below = survivor_tags.iter().filter(|&&g| g < min_gen_target).count();
            // Per-mille of gates still below the target generation (0 = all gates reached it).
            let below_permille = if len > 0 { below * 1000 / len } else { 1000 };
            // The "fractional floor": the lowest generation once the bottom (1-frac) of gates (the
            // permanently-stuck ones) are written off — i.e. the skip-th order statistic, with
            // skip = (1-frac)*total. This is the signal the anchor actually drives, so it keeps
            // rising even when a few stuck gates pin the absolute minimum. (skip=0 -> absolute min.)
            let skip = ((1000 - permille) * len) / 1000;
            let cur_floor_gen = if len == 0 {
                0u32
            } else {
                let mut sorted = survivor_tags.clone();
                let k = skip.min(len - 1);
                *sorted.select_nth_unstable(k).1
            };
            let abs_min_gen = survivor_tags.iter().copied().min().unwrap_or(0);
            let met = len > 0 && below * 1000 <= len * (1000 - permille);
            println!(
                "[stage-D] stage {} progress: {:.1}% of {} gates at gen>={} (target {:.1}%), floor_gen {} (abs_min {})",
                i,
                100.0 - below_permille as f64 / 10.0,
                len,
                min_gen_target,
                permille as f64 / 10.0,
                cur_floor_gen,
                abs_min_gen
            );
            if met {
                println!(
                    "[stage-D] min-gen condition met after {} stage(s); stopping cadence (final {} gates)",
                    i,
                    circuit.gates.len()
                );
                break;
            }
            // No-progress guard KEYED TO min_gen: progress = the floor generation reached a new high
            // (the monotonic signal) OR the below-target fraction reached a new best. Stop only if
            // NEITHER improves for STAGE_D_STALL_LIMIT stages. (The "% at gen>=target" alone is too
            // noisy — it can dip then recover while the floor is still rising.)
            const STAGE_D_STALL_LIMIT: usize = 8;
            let mut progressed = false;
            if cur_floor_gen > best_min_gen {
                best_min_gen = cur_floor_gen;
                progressed = true;
            }
            if below_permille < best_below_permille {
                best_below_permille = below_permille;
                progressed = true;
            }
            if progressed {
                stall = 0;
            } else {
                stall += 1;
                if stall >= STAGE_D_STALL_LIMIT {
                    println!(
                        "[stage-D] no progress for {} stages (min_gen stuck at {}, best {:.1}% still below target; circuit {} gates); stopping",
                        stall,
                        best_min_gen,
                        best_below_permille as f64 / 10.0,
                        circuit.gates.len()
                    );
                    break;
                }
            }
        }
    }
    // Clear the Stage D size cap so it cannot leak into any later shooting in this process.
    crate::replace::replace::SHOOT_SIZE_CAP.store(0, std::sync::atomic::Ordering::Relaxed);

    if track && crate::replace::replace::gen_mode() {
        // Generation histogram of the final circuit: how many gates at each generation.
        let mut hist: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for &g in &survivor_tags {
            *hist.entry(g).or_insert(0) += 1;
        }
        let total = survivor_tags.len().max(1);
        let max_gen = survivor_tags.iter().copied().max().unwrap_or(0);
        let min_gen = survivor_tags.iter().copied().min().unwrap_or(0);
        let mut sorted = survivor_tags.clone();
        sorted.sort_unstable();
        let median_gen = if sorted.is_empty() { 0 } else { sorted[sorted.len() / 2] };
        let path = format!("{}.generations", save);
        match File::create(&path) {
            Ok(mut f) => {
                let _ = writeln!(
                    f,
                    "# generation histogram of final circuit. n_gates={} min_gen={} median_gen={} max_gen={}\n\
                     # generation count fraction",
                    total, min_gen, median_gen, max_gen
                );
                for (g, c) in &hist {
                    let _ = writeln!(f, "{} {} {:.4}", g, c, *c as f64 / total as f64);
                }
                println!(
                    "Generations: n_gates={} min={} median={} max={} -> {}",
                    total, min_gen, median_gen, max_gen, path
                );
            }
            Err(e) => eprintln!("Failed to write generations file {}: {}", path, e),
        }
    } else if track {
        // Survivors = original gates (by index into the pre-mixing circuit) whose tag is still
        // present in the final tag vector, i.e. never part of any replacement window.
        let mut present = vec![false; n_orig];
        for &t in &survivor_tags {
            if (t as usize) < n_orig {
                present[t as usize] = true;
            }
        }
        let n_survived = present.iter().filter(|&&p| p).count();
        let path = format!("{}.survivors", save);
        match File::create(&path) {
            Ok(mut f) => {
                let _ = writeln!(
                    f,
                    "# survivors: original gates (pre-mixing, after feistelization) never part of\n\
                     # any replacement during mixing. n_orig={} n_survived={} ({:.2}%)\n\
                     # orig_index a b c",
                    n_orig,
                    n_survived,
                    100.0 * n_survived as f64 / n_orig.max(1) as f64
                );
                for (idx, &p) in present.iter().enumerate() {
                    if p {
                        let g = orig_gates[idx];
                        let _ = writeln!(f, "{} {} {} {}", idx, g[0], g[1], g[2]);
                    }
                }
                println!(
                    "Survivors: {}/{} original gates never replaced ({:.2}%) -> {}",
                    n_survived,
                    n_orig,
                    100.0 * n_survived as f64 / n_orig.max(1) as f64,
                    path
                );
            }
            Err(e) => eprintln!("Failed to write survivors file {}: {}", path, e),
        }
    }

    println!(
        "Oversized-canon (>64-wire) lookups skipped: {}",
        crate::circuit::circuit::OVERSIZED_CANON_SKIPS.load(std::sync::atomic::Ordering::Relaxed)
    );
    println!(
        "Forced pseudo-collisions (no real collision): {}",
        crate::replace::replace::FORCED_COLLISIONS.load(std::sync::atomic::Ordering::Relaxed)
    );
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
        ) = (
            0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize, 0usize,
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
                "  hide diagnostics: eligible {} | skipped-materialized {} | attempts {} | lookup-misses {} | rejected-exposed {}",
                stats.eligible,
                stats.skipped_materialized,
                stats.attempts,
                stats.lookup_misses,
                stats.rejected_exposed
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
            "Total hide diagnostics: eligible {} | skipped-materialized {} | attempts {} | lookup-misses {} | rejected-exposed {}",
            t_eligible, t_skipped, t_attempts, t_misses, t_rejected
        );
    }
    if let Some((public_y, public_z)) = &slice_zero_random_public {
        print_slice_zero_random_public_slice(original_n, public_y, public_z);
    }
}
