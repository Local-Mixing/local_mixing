//! Sampled, provenance-preserving search for short whole-trace XOR identities.
//!
//! This is the large-n companion to `exact_trace_span`: discovery rows build a
//! GF(2) basis, validation rows reject sampled coincidences, and a locked test
//! split is evaluated only for validation survivors.  The primary feature mode
//! is a literal internal post-write wire checkpoint.  `gate-delta` is a
//! secondary mode whose features are expanded back to the two raw checkpoints
//! they abbreviate before support is counted.
//!
//! A zero-error result is still a sampled statement, never a proof on 2^n
//! inputs.  Conversely, failure to find a short witness is bounded by the
//! selected catalog, chronological basis order, sample split, and support cap.

use clap::{Parser, ValueEnum};
use local_mixing::postmix::format::{read_g57_file, read_mpmct};
use local_mixing::postmix::xgate::{XGate, max_wire};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Write as FmtWrite;
use xxhash_rust::xxh3::xxh3_64;

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum FeatureMode {
    /// Literal target value immediately after an internal physical write.
    Checkpoint,
    /// Gate firing, expanded to its before/after raw checkpoints for support.
    GateDelta,
}

impl FeatureMode {
    fn name(self) -> &'static str {
        match self {
            Self::Checkpoint => "checkpoint",
            Self::GateDelta => "gate-delta",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum WireCatalog {
    /// Every physical wire.
    All,
    /// Seven lane blocks; only logical source positions 0..n in each block.
    SourceHome,
    /// Every wire in the configured carrier-lane blocks.
    CarrierRegion,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
enum BasisFamily {
    /// One chronological basis over every eligible checkpoint.
    Global,
    /// One chronological basis per physical wire, then choose the shortest
    /// witness found for each source firing across all eligible wires.
    PerWire,
}

impl BasisFamily {
    fn name(self) -> &'static str {
        match self {
            Self::Global => "global",
            Self::PerWire => "per-wire",
        }
    }
}

impl WireCatalog {
    fn name(self) -> &'static str {
        match self {
            Self::All => "all",
            Self::SourceHome => "source-home",
            Self::CarrierRegion => "carrier-region",
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "sampled_trace_support")]
struct Args {
    /// Original/source G57 circuit.
    #[arg(long)]
    c: String,
    /// Gadgetized/mixed MPMCT1 circuit.
    #[arg(long)]
    g: String,
    /// Logical source width; random inputs occupy wires 0..n.
    #[arg(long)]
    n: usize,
    #[arg(long, value_enum, default_value_t = FeatureMode::Checkpoint)]
    feature_mode: FeatureMode,
    #[arg(long, value_enum, default_value_t = WireCatalog::All)]
    wire_catalog: WireCatalog,
    #[arg(long, value_enum, default_value_t = BasisFamily::Global)]
    basis_family: BasisFamily,
    /// Number of carrier lanes for source-home/carrier-region catalogs.
    #[arg(long, default_value_t = 7)]
    carrier_lanes: usize,
    /// Physical stride between corresponding values in adjacent carrier lanes.
    #[arg(long, default_value_t = 128)]
    carrier_stride: usize,
    /// 64-row words used for discovery/elimination only.
    #[arg(long, default_value_t = 64)]
    fit_batches: usize,
    /// Independent 64-row words used to select discovery witnesses.
    #[arg(long, default_value_t = 256)]
    validation_batches: usize,
    /// Locked 64-row words evaluated only for validation survivors.
    #[arg(long, default_value_t = 1024)]
    test_batches: usize,
    #[arg(long, default_value_t = 0x5341_4d50_4c45_0064)]
    seed: u64,
    /// Maximum raw wire-checkpoint terms; the affine constant is separate.
    #[arg(long, default_value_t = 100)]
    support_cap: usize,
    /// Output stem; writes `.meta.json` and `.witnesses.tsv`.
    #[arg(long)]
    out: String,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct Coordinate {
    gate_index: usize,
    wire: usize,
}

#[derive(Clone, Debug)]
enum LabelKind {
    Constant,
    Checkpoint(Coordinate),
    Delta {
        before: Coordinate,
        after: Coordinate,
    },
}

#[derive(Clone, Debug)]
struct Label {
    kind: LabelKind,
}

struct BasisRow {
    fit: Vec<u64>,
    combination: Vec<u64>,
}

struct Basis {
    rows: Vec<Option<BasisRow>>,
    labels: Vec<Label>,
    combination_words: usize,
    candidates: usize,
    dependent: usize,
    saturation_gate: Option<usize>,
}

impl Basis {
    fn new(fit_samples: usize) -> Self {
        Self {
            rows: std::iter::repeat_with(|| None).take(fit_samples).collect(),
            labels: Vec::new(),
            combination_words: fit_samples.div_ceil(64),
            candidates: 0,
            dependent: 0,
            saturation_gate: None,
        }
    }

    fn rank(&self) -> usize {
        self.labels.len()
    }

    fn saturated(&self) -> bool {
        self.rank() == self.rows.len()
    }

    fn offer(&mut self, signature: &[u64], label: Label, gate_index: Option<usize>) -> bool {
        self.candidates += 1;
        let mut fit = signature.to_vec();
        let mut combination = vec![0u64; self.combination_words];
        loop {
            let Some(pivot) = first_set(&fit) else {
                self.dependent += 1;
                return false;
            };
            if let Some(row) = self.rows[pivot].as_ref() {
                xor_into(&mut fit, &row.fit);
                xor_into(&mut combination, &row.combination);
                continue;
            }
            // Keep a reduced row even when pivots were discovered out of order.
            for later in pivot + 1..self.rows.len() {
                if bit(&fit, later)
                    && let Some(row) = self.rows[later].as_ref()
                {
                    xor_into(&mut fit, &row.fit);
                    xor_into(&mut combination, &row.combination);
                }
            }
            let id = self.labels.len();
            combination[id / 64] ^= 1u64 << (id % 64);
            self.rows[pivot] = Some(BasisRow { fit, combination });
            self.labels.push(label);
            if self.saturated() {
                self.saturation_gate = gate_index;
            }
            return true;
        }
    }

    fn solve(&self, target: &[u64]) -> Option<Vec<usize>> {
        let mut fit = target.to_vec();
        let mut combination = vec![0u64; self.combination_words];
        loop {
            let Some(pivot) = first_set(&fit) else {
                return Some(set_bits(&combination));
            };
            let row = self.rows[pivot].as_ref()?;
            xor_into(&mut fit, &row.fit);
            xor_into(&mut combination, &row.combination);
        }
    }
}

#[derive(Clone, Debug)]
struct Witness {
    target_gate: usize,
    coordinates: Vec<Coordinate>,
    constant: bool,
    discovery: &'static str,
    basis_terms: usize,
    validation_errors: usize,
    test_errors: Option<usize>,
}

struct BasisSummary {
    instances: usize,
    candidates: usize,
    rank: usize,
    dependent: usize,
    saturation_gate: Option<usize>,
}

#[inline]
fn bit(words: &[u64], index: usize) -> bool {
    ((words[index / 64] >> (index % 64)) & 1) != 0
}

fn first_set(words: &[u64]) -> Option<usize> {
    words.iter().enumerate().find_map(|(word_index, &word)| {
        (word != 0).then_some(word_index * 64 + word.trailing_zeros() as usize)
    })
}

#[inline]
fn xor_into(dst: &mut [u64], src: &[u64]) {
    for (left, right) in dst.iter_mut().zip(src) {
        *left ^= *right;
    }
}

fn set_bits(words: &[u64]) -> Vec<usize> {
    let mut out = Vec::new();
    for (word_index, &word) in words.iter().enumerate() {
        let mut rest = word;
        while rest != 0 {
            let bit = rest.trailing_zeros() as usize;
            out.push(word_index * 64 + bit);
            rest &= rest - 1;
        }
    }
    out
}

fn firing_into(gate: &XGate, state: &[Vec<u64>], out: &mut [u64]) {
    out.fill(!0u64);
    for &(wire, positive) in &gate.ctrls {
        for (dst, &source) in out.iter_mut().zip(&state[wire as usize]) {
            *dst &= if positive { source } else { !source };
        }
    }
    if gate.comp {
        for word in out {
            *word = !*word;
        }
    }
}

fn random_inputs(n: usize, words: usize, seed: u64, domain: u64) -> Vec<Vec<u64>> {
    let mut rng = StdRng::seed_from_u64(seed ^ domain);
    (0..n)
        .map(|_| (0..words).map(|_| rng.random::<u64>()).collect())
        .collect()
}

fn input_fingerprint(inputs: &[Vec<u64>]) -> u64 {
    let mut bytes = Vec::with_capacity(
        inputs.iter().map(|wire| wire.len()).sum::<usize>() * std::mem::size_of::<u64>(),
    );
    for wire in inputs {
        for word in wire {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
    }
    xxh3_64(&bytes)
}

fn initial_state(nw: usize, n: usize, inputs: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let words = inputs.first().map_or(0, Vec::len);
    let mut state = vec![vec![0u64; words]; nw.max(n)];
    for (dst, src) in state.iter_mut().take(n).zip(inputs) {
        dst.copy_from_slice(src);
    }
    state
}

fn source_firings(c: &[XGate], nw: usize, n: usize, inputs: &[Vec<u64>]) -> Vec<Vec<u64>> {
    let words = inputs.first().map_or(0, Vec::len);
    let mut state = initial_state(nw, n, inputs);
    let mut scratch = vec![0u64; words];
    let mut out = Vec::with_capacity(c.len());
    for gate in c {
        firing_into(gate, &state, &mut scratch);
        out.push(scratch.clone());
        xor_into(&mut state[gate.target as usize], &scratch);
    }
    out
}

fn included_wire(args: &Args, wire: usize) -> bool {
    match args.wire_catalog {
        WireCatalog::All => true,
        WireCatalog::SourceHome => {
            wire < args.carrier_lanes * args.carrier_stride && wire % args.carrier_stride < args.n
        }
        WireCatalog::CarrierRegion => wire < args.carrier_lanes * args.carrier_stride,
    }
}

fn last_writes(g: &[XGate], nw: usize) -> Vec<Option<usize>> {
    let mut last = vec![None; nw];
    for (gate_index, gate) in g.iter().enumerate() {
        last[gate.target as usize] = Some(gate_index);
    }
    last
}

fn toggle_coordinate(set: &mut BTreeSet<Coordinate>, coordinate: Coordinate) {
    if !set.remove(&coordinate) {
        set.insert(coordinate);
    }
}

fn expand_labels(labels: &[Label], selection: &[usize]) -> (bool, Vec<Coordinate>) {
    let mut constant = false;
    let mut coordinates = BTreeSet::new();
    for &id in selection {
        match labels[id].kind {
            LabelKind::Constant => constant ^= true,
            LabelKind::Checkpoint(coordinate) => toggle_coordinate(&mut coordinates, coordinate),
            LabelKind::Delta { before, after } => {
                toggle_coordinate(&mut coordinates, before);
                toggle_coordinate(&mut coordinates, after);
            }
        }
    }
    (constant, coordinates.into_iter().collect())
}

fn discover(
    args: &Args,
    c: &[XGate],
    nw_c: usize,
    g: &[XGate],
    nw_g: usize,
) -> (Vec<Witness>, BasisSummary, u64, usize) {
    const FIT_DOMAIN: u64 = 0x4649_5400_0000_0001;
    let inputs = random_inputs(args.n, args.fit_batches, args.seed, FIT_DOMAIN);
    let fingerprint = input_fingerprint(&inputs);
    let targets = source_firings(c, nw_c, args.n, &inputs);
    let mut direct: HashMap<Vec<u64>, Vec<(usize, bool)>> = HashMap::new();
    for (target_gate, target) in targets.iter().enumerate() {
        direct
            .entry(target.clone())
            .or_default()
            .push((target_gate, false));
        let complement: Vec<u64> = target.iter().map(|word| !word).collect();
        direct
            .entry(complement)
            .or_default()
            .push((target_gate, true));
    }

    let fit_samples = args.fit_batches * 64;
    let constant = vec![!0u64; args.fit_batches];
    let mut global_basis = (args.basis_family == BasisFamily::Global).then(|| {
        let mut basis = Basis::new(fit_samples);
        basis.offer(
            &constant,
            Label {
                kind: LabelKind::Constant,
            },
            None,
        );
        basis
    });
    let mut wire_bases: Vec<Option<Basis>> = (0..nw_g)
        .map(|wire| {
            (args.basis_family == BasisFamily::PerWire && included_wire(args, wire)).then(|| {
                let mut basis = Basis::new(fit_samples);
                basis.offer(
                    &constant,
                    Label {
                        kind: LabelKind::Constant,
                    },
                    None,
                );
                basis
            })
        })
        .collect();

    let last = last_writes(g, nw_g);
    let mut previous = vec![None; nw_g];
    let mut state = initial_state(nw_g, args.n, &inputs);
    let mut scratch = vec![0u64; args.fit_batches];
    let mut direct_witnesses: HashMap<usize, Witness> = HashMap::new();
    let mut eligible = 0usize;
    for (gate_index, gate) in g.iter().enumerate() {
        let target = gate.target as usize;
        firing_into(gate, &state, &mut scratch);
        xor_into(&mut state[target], &scratch);
        let after = Coordinate {
            gate_index,
            wire: target,
        };
        let kind = match args.feature_mode {
            FeatureMode::Checkpoint => Some(LabelKind::Checkpoint(after)),
            FeatureMode::GateDelta => previous[target].map(|before| LabelKind::Delta {
                before: Coordinate {
                    gate_index: before,
                    wire: target,
                },
                after,
            }),
        };
        // Filter before offering/deduplicating: every raw representative is
        // internal, never an initial checkpoint or the last checkpoint on a wire.
        let safe = included_wire(args, target) && last[target] != Some(gate_index);
        if safe && let Some(kind) = kind {
            eligible += 1;
            let signature: &[u64] = match args.feature_mode {
                FeatureMode::Checkpoint => &state[target],
                FeatureMode::GateDelta => &scratch,
            };
            if let Some(matches) = direct.get(signature) {
                let raw_selection = [Label { kind: kind.clone() }];
                let (_, coordinates) = expand_labels(&raw_selection, &[0]);
                if coordinates.len() <= args.support_cap {
                    for &(target_gate, constant) in matches {
                        direct_witnesses.entry(target_gate).or_insert(Witness {
                            target_gate,
                            coordinates: coordinates.clone(),
                            constant,
                            discovery: "direct-signature",
                            basis_terms: 1,
                            validation_errors: usize::MAX,
                            test_errors: None,
                        });
                    }
                }
            }
            match args.basis_family {
                BasisFamily::Global => {
                    let basis = global_basis.as_mut().unwrap();
                    if !basis.saturated() {
                        basis.offer(signature, Label { kind }, Some(gate_index));
                    }
                }
                BasisFamily::PerWire => {
                    let basis = wire_bases[target].as_mut().unwrap();
                    if !basis.saturated() {
                        basis.offer(signature, Label { kind }, Some(gate_index));
                    }
                }
            }
        }
        previous[target] = Some(gate_index);
    }

    let mut best: Vec<Option<Witness>> = (0..targets.len())
        .map(|target_gate| direct_witnesses.remove(&target_gate))
        .collect();
    let mut consider_basis = |basis: &Basis, discovery: &'static str| {
        for (target_gate, target) in targets.iter().enumerate() {
            let Some(selection) = basis.solve(target) else {
                continue;
            };
            let (constant, coordinates) = expand_labels(&basis.labels, &selection);
            if coordinates.len() > args.support_cap {
                continue;
            }
            let candidate = Witness {
                target_gate,
                coordinates,
                constant,
                discovery,
                basis_terms: selection.len(),
                validation_errors: usize::MAX,
                test_errors: None,
            };
            let replace = best[target_gate].as_ref().is_none_or(|current| {
                (candidate.coordinates.len(), candidate.basis_terms)
                    < (current.coordinates.len(), current.basis_terms)
            });
            if replace {
                best[target_gate] = Some(candidate);
            }
        }
    };
    match args.basis_family {
        BasisFamily::Global => consider_basis(global_basis.as_ref().unwrap(), "global-basis"),
        BasisFamily::PerWire => {
            for basis in wire_bases.iter().flatten() {
                consider_basis(basis, "per-wire-basis");
            }
        }
    }
    let witnesses = best.into_iter().flatten().collect();
    let summary = match args.basis_family {
        BasisFamily::Global => {
            let basis = global_basis.as_ref().unwrap();
            BasisSummary {
                instances: 1,
                candidates: basis.candidates,
                rank: basis.rank(),
                dependent: basis.dependent,
                saturation_gate: basis.saturation_gate,
            }
        }
        BasisFamily::PerWire => BasisSummary {
            instances: wire_bases.iter().flatten().count(),
            candidates: wire_bases
                .iter()
                .flatten()
                .map(|basis| basis.candidates)
                .sum(),
            rank: wire_bases.iter().flatten().map(Basis::rank).sum(),
            dependent: wire_bases
                .iter()
                .flatten()
                .map(|basis| basis.dependent)
                .sum(),
            saturation_gate: wire_bases
                .iter()
                .flatten()
                .filter_map(|basis| basis.saturation_gate)
                .min(),
        },
    };
    (witnesses, summary, fingerprint, eligible)
}

fn replay(
    witnesses: &[Witness],
    c: &[XGate],
    nw_c: usize,
    g: &[XGate],
    nw_g: usize,
    n: usize,
    words: usize,
    seed: u64,
    domain: u64,
) -> (Vec<usize>, u64) {
    let inputs = random_inputs(n, words, seed, domain);
    let fingerprint = input_fingerprint(&inputs);
    let targets = source_firings(c, nw_c, n, &inputs);
    let mut subscribers: HashMap<usize, Vec<usize>> = HashMap::new();
    for (witness_index, witness) in witnesses.iter().enumerate() {
        for coordinate in &witness.coordinates {
            subscribers
                .entry(coordinate.gate_index)
                .or_default()
                .push(witness_index);
        }
    }
    let mut reconstructed = vec![vec![0u64; words]; witnesses.len()];
    let mut consumed_coordinates = vec![0usize; witnesses.len()];
    for (witness_index, witness) in witnesses.iter().enumerate() {
        if witness.constant {
            reconstructed[witness_index].fill(!0u64);
        }
    }
    let mut state = initial_state(nw_g, n, &inputs);
    let mut scratch = vec![0u64; words];
    for (gate_index, gate) in g.iter().enumerate() {
        firing_into(gate, &state, &mut scratch);
        let target = gate.target as usize;
        xor_into(&mut state[target], &scratch);
        for &witness_index in subscribers.get(&gate_index).into_iter().flatten() {
            assert!(
                witnesses[witness_index]
                    .coordinates
                    .iter()
                    .any(|coordinate| coordinate.gate_index == gate_index
                        && coordinate.wire == target)
            );
            xor_into(&mut reconstructed[witness_index], &state[target]);
            consumed_coordinates[witness_index] += 1;
        }
    }
    for (witness, consumed) in witnesses.iter().zip(consumed_coordinates) {
        assert_eq!(
            consumed,
            witness.coordinates.len(),
            "not every labeled raw checkpoint was consumed during replay"
        );
    }
    let errors = witnesses
        .iter()
        .enumerate()
        .map(|(index, witness)| {
            reconstructed[index]
                .iter()
                .zip(&targets[witness.target_gate])
                .map(|(left, right)| (left ^ right).count_ones() as usize)
                .sum()
        })
        .collect();
    (errors, fingerprint)
}

fn file_xxh3(path: &str) -> u64 {
    xxh3_64(&std::fs::read(path).unwrap_or_else(|error| panic!("read {path}: {error}")))
}

fn json_string(value: &str) -> String {
    let mut out = String::from("\"");
    for character in value.chars() {
        match character {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

fn main() {
    let args = Args::parse();
    assert!(args.n > 0, "--n must be positive");
    assert!(args.fit_batches > 0, "--fit-batches must be positive");
    assert!(
        args.validation_batches > 0,
        "--validation-batches must be positive"
    );
    assert!(args.test_batches > 0, "--test-batches must be positive");
    assert!(args.support_cap > 0, "--support-cap must be positive");
    assert!(args.carrier_lanes > 0 && args.carrier_stride >= args.n);

    let c = read_g57_file(&args.c).unwrap_or_else(|error| panic!("read {}: {error}", args.c));
    let (g, declared_g) =
        read_mpmct(&args.g).unwrap_or_else(|error| panic!("read {}: {error}", args.g));
    let nw_c = (max_wire(&c) as usize + 1).max(args.n);
    let nw_g = declared_g.max(max_wire(&g) as usize + 1).max(args.n);
    eprintln!(
        "[sampled_trace_support] C={}/{} G={}/{} mode={} catalog={} basis={} fit/validation/test={}/{}/{}",
        c.len(),
        nw_c,
        g.len(),
        nw_g,
        args.feature_mode.name(),
        args.wire_catalog.name(),
        args.basis_family.name(),
        args.fit_batches * 64,
        args.validation_batches * 64,
        args.test_batches * 64,
    );

    let (mut discovered, basis, fit_fingerprint, eligible) = discover(&args, &c, nw_c, &g, nw_g);
    const VALIDATION_DOMAIN: u64 = 0x5641_4c49_4400_0002;
    const TEST_DOMAIN: u64 = 0x5445_5354_0000_0003;
    let (validation_errors, validation_fingerprint) = replay(
        &discovered,
        &c,
        nw_c,
        &g,
        nw_g,
        args.n,
        args.validation_batches,
        args.seed,
        VALIDATION_DOMAIN,
    );
    for (witness, errors) in discovered.iter_mut().zip(validation_errors) {
        witness.validation_errors = errors;
    }
    let validation_survivors: Vec<Witness> = discovered
        .iter()
        .filter(|witness| witness.validation_errors == 0)
        .cloned()
        .collect();
    // Locked test is constructed and replayed only after validation selection.
    let (test_errors, test_fingerprint) = replay(
        &validation_survivors,
        &c,
        nw_c,
        &g,
        nw_g,
        args.n,
        args.test_batches,
        args.seed,
        TEST_DOMAIN,
    );
    let mut tested_witnesses = validation_survivors;
    for (witness, errors) in tested_witnesses.iter_mut().zip(test_errors) {
        witness.test_errors = Some(errors);
    }
    let locked_survivors: Vec<&Witness> = tested_witnesses
        .iter()
        .filter(|witness| witness.test_errors == Some(0))
        .collect();

    let tsv_path = format!("{}.witnesses.tsv", args.out);
    let mut tsv = String::from(
        "target_gate\tdiscovery\traw_support\tconstant\tbasis_terms\tvalidation_errors\ttest_errors\tcoordinates_gate:wire\n",
    );
    for witness in &locked_survivors {
        let coordinates = witness
            .coordinates
            .iter()
            .map(|coordinate| format!("{}:{}", coordinate.gate_index, coordinate.wire))
            .collect::<Vec<_>>()
            .join(",");
        writeln!(
            tsv,
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            witness.target_gate,
            witness.discovery,
            witness.coordinates.len(),
            witness.constant,
            witness.basis_terms,
            witness.validation_errors,
            witness.test_errors.unwrap_or(usize::MAX),
            coordinates,
        )
        .unwrap();
    }
    std::fs::write(&tsv_path, tsv).expect("write witness TSV");

    let locked_zero = locked_survivors.len();
    let min_support = locked_survivors
        .iter()
        .map(|witness| witness.coordinates.len())
        .min();
    let meta_path = format!("{}.meta.json", args.out);
    let meta = format!(
        concat!(
            "{{\n  \"schema\": \"sampled-trace-support/v1\",\n",
            "  \"source\": {},\n  \"source_xxh3_64\": \"{:016x}\",\n",
            "  \"g\": {},\n  \"g_xxh3_64\": \"{:016x}\",\n",
            "  \"n\": {},\n  \"source_gates\": {},\n  \"g_gates\": {},\n  \"g_wires\": {},\n",
            "  \"feature_mode\": \"{}\",\n  \"wire_catalog\": \"{}\",\n  \"basis_family\": \"{}\",\n",
            "  \"carrier_lanes\": {},\n  \"carrier_stride\": {},\n  \"seed\": {},\n",
            "  \"endpoint_policy\": \"exclude initial coordinates and every wire's final coordinate before offering a feature\",\n",
            "  \"fit_samples\": {},\n  \"validation_samples\": {},\n  \"locked_test_samples\": {},\n",
            "  \"fit_rowset_xxh3_64\": \"{:016x}\",\n  \"validation_rowset_xxh3_64\": \"{:016x}\",\n  \"test_rowset_xxh3_64\": \"{:016x}\",\n",
            "  \"eligible_features\": {},\n  \"basis_instances\": {},\n  \"basis_candidates\": {},\n  \"basis_rank_sum\": {},\n  \"basis_dependent\": {},\n  \"basis_saturation_gate\": {},\n",
            "  \"support_cap_raw_checkpoints\": {},\n  \"targets\": {},\n  \"discovery_witnesses_at_most_cap\": {},\n",
            "  \"validation_zero_error_survivors\": {},\n  \"locked_test_zero_error_survivors\": {},\n  \"minimum_locked_zero_error_support\": {},\n",
            "  \"minimum_support_certified\": false,\n",
            "  \"interpretation\": \"TSV rows have zero error on validation and locked test but are exact only on the listed sampled row sets; a non-finding is bounded by catalog, order, samples, and support heuristic\",\n",
            "  \"witness_tsv\": {}\n}}\n"
        ),
        json_string(&args.c),
        file_xxh3(&args.c),
        json_string(&args.g),
        file_xxh3(&args.g),
        args.n,
        c.len(),
        g.len(),
        nw_g,
        args.feature_mode.name(),
        args.wire_catalog.name(),
        args.basis_family.name(),
        args.carrier_lanes,
        args.carrier_stride,
        args.seed,
        args.fit_batches * 64,
        args.validation_batches * 64,
        args.test_batches * 64,
        fit_fingerprint,
        validation_fingerprint,
        test_fingerprint,
        eligible,
        basis.instances,
        basis.candidates,
        basis.rank,
        basis.dependent,
        basis
            .saturation_gate
            .map_or_else(|| "null".to_string(), |gate| gate.to_string()),
        args.support_cap,
        c.len(),
        discovered.len(),
        tested_witnesses.len(),
        locked_zero,
        min_support.map_or_else(|| "null".to_string(), |support| support.to_string()),
        json_string(&tsv_path),
    );
    std::fs::write(&meta_path, meta).expect("write metadata JSON");
    eprintln!(
        "[sampled_trace_support] eligible={} basis_instances={} rank_sum={} discovered<=cap={} validation_zero={} locked_zero={} min_support={:?}; wrote {}, {}",
        eligible,
        basis.instances,
        basis.rank,
        discovered.len(),
        tested_witnesses.len(),
        locked_zero,
        min_support,
        meta_path,
        tsv_path,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basis_provenance_reconstructs_target() {
        let mut basis = Basis::new(64);
        let mut a = vec![0u64; 1];
        let mut b = vec![0u64; 1];
        a[0] = 0x5555_5555_5555_5555;
        b[0] = 0x3333_3333_3333_3333;
        basis.offer(
            &a,
            Label {
                kind: LabelKind::Checkpoint(Coordinate {
                    gate_index: 3,
                    wire: 7,
                }),
            },
            Some(3),
        );
        basis.offer(
            &b,
            Label {
                kind: LabelKind::Checkpoint(Coordinate {
                    gate_index: 9,
                    wire: 8,
                }),
            },
            Some(9),
        );
        let target = vec![a[0] ^ b[0]];
        let selected = basis.solve(&target).unwrap();
        let (constant, coordinates) = expand_labels(&basis.labels, &selected);
        assert!(!constant);
        assert_eq!(coordinates.len(), 2);
        assert!(coordinates.contains(&Coordinate {
            gate_index: 3,
            wire: 7
        }));
        assert!(coordinates.contains(&Coordinate {
            gate_index: 9,
            wire: 8
        }));
    }

    #[test]
    fn expanded_delta_cancels_shared_checkpoint() {
        let shared = Coordinate {
            gate_index: 5,
            wire: 2,
        };
        let labels = vec![
            Label {
                kind: LabelKind::Delta {
                    before: Coordinate {
                        gate_index: 1,
                        wire: 2,
                    },
                    after: shared,
                },
            },
            Label {
                kind: LabelKind::Delta {
                    before: shared,
                    after: Coordinate {
                        gate_index: 8,
                        wire: 2,
                    },
                },
            },
        ];
        let (_, coordinates) = expand_labels(&labels, &[0, 1]);
        assert_eq!(coordinates.len(), 2);
        assert!(!coordinates.contains(&shared));
    }

    #[test]
    fn sample_domains_are_distinct_and_reproducible() {
        let a = random_inputs(4, 2, 17, 1);
        let b = random_inputs(4, 2, 17, 2);
        assert_ne!(input_fingerprint(&a), input_fingerprint(&b));
        assert_eq!(a, random_inputs(4, 2, 17, 1));
    }
}
