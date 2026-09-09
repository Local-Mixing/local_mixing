//! Single-sided gate wrapping with ancilla-randomized EM parking.
//!
//! Pipeline:
//! 1. Allocate an ancilla bank (default: `n` wires).
//! 2. **Init scramble:** `2n log₂ n` random 3-wire Toffoli permutations (explicit chunks).
//!    Active wire ∈ ancilla; controls ∈ data ∪ ancilla.
//! 3. **One fused permutation chunk per plaintext gate** (typically ≤20 wires,
//!    5+6=11 with overlap-1 reuse and `k=6`): unmask → `R⁻¹` on every pending
//!    triple sharing a wire → `g` → `R` → remask. One fresh random `A_S` per gate (shared
//!    by all remasks in that chunk); unmask uses the `A_S` stored at park time.
//! 4. Plaintext circuits should be **canonicalized** first so wire reuse is usually
//!    against the most recent triple (overlap ≥1), keeping supports small.

use crate::circuit::{CircuitSeq, Permutation};
use primitive_types::U256 as u256;
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::{HashMap, HashSet};
use std::fmt::Write as _;
use std::fs;
use std::path::Path;

/// Body of a published chunk.
#[derive(Clone, Debug)]
pub enum ChunkBody {
    /// Full truth table on `wires` (`2^{wires.len()}` images).
    Perm(Permutation),
    /// Legacy irreversible reset (never emitted by the current compiler;
    /// retained so old challenge files still parse).
    Zero,
}

/// A chunk: permutation (or legacy reset) on a fixed wire set.
#[derive(Clone, Debug)]
pub struct Chunk {
    /// Physical wire indices; local bit `i` of a perm is `wires[i]`.
    pub wires: Vec<u16>,
    pub body: ChunkBody,
}

/// Obfuscated circuit as a sequence of local chunks.
#[derive(Clone, Debug)]
pub struct ChunkedCircuit {
    /// Original data wire count (ancillas are `data_wires..total_wires`).
    pub data_wires: usize,
    /// High-water mark of allocated wire indices (data + ancilla bank).
    pub total_wires: usize,
    pub chunks: Vec<Chunk>,
}

/// Parameters for the single-sided wrap compiler.
#[derive(Clone, Debug)]
pub struct SandwichParams {
    /// Ancilla bank size (default: `data_wires`).
    pub ancilla_count: usize,
    /// `|A_S|` for each EM mask (default: 6).
    pub mask_k: usize,
    /// Init scramble length (default: `2n log₂ n`).
    pub init_gates: usize,
}

impl SandwichParams {
    pub fn for_n(data_wires: usize) -> Self {
        let mask_k = 6;
        let ancilla_count = data_wires.max(mask_k);
        let init_gates =
            (2.0 * data_wires as f64 * (data_wires as f64).log2()).round() as usize;
        Self {
            ancilla_count,
            mask_k,
            init_gates,
        }
    }
}

/// One parked data wire: EM key + the ancilla subset frozen at park time.
#[derive(Clone, Debug)]
struct ParkedWire {
    wire: u16,
    ancillas: Vec<u16>,
    r: u32,
    b: bool,
}

/// `R` still waiting for `R⁻¹` after a gate that appended wrap.
#[derive(Clone, Debug)]
struct PendingWrap {
    wires: Vec<u16>,
    r: Permutation,
    /// Plaintext gate index (for future “most recent touch” pending selection).
    #[allow(dead_code)]
    source_gate: usize,
}

/// First and last plaintext gate index for each wire index.
#[derive(Clone, Debug)]
struct WireBounds {
    last: Vec<Option<usize>>,
}

impl Chunk {
    pub fn k(&self) -> usize {
        self.wires.len()
    }

    pub fn cycle_notation(&self) -> String {
        match &self.body {
            ChunkBody::Perm(p) => format!("wires {:?}  {}", self.wires, cycle_notation(p)),
            ChunkBody::Zero => format!("RESET wires {:?} -> 0", self.wires),
        }
    }

    pub fn representation_bits(&self) -> usize {
        match &self.body {
            ChunkBody::Perm(p) => self.k() * p.data.len(),
            ChunkBody::Zero => self.wires.len() * 16,
        }
    }
}

impl ChunkedCircuit {
    pub fn representation_bits(&self) -> usize {
        self.chunks.iter().map(Chunk::representation_bits).sum()
    }

    pub fn representation_bytes(&self) -> usize {
        self.representation_bits().div_ceil(8)
    }

    pub fn evaluate_bits(&self, state: &[bool]) -> Vec<bool> {
        let mut bits = vec![false; self.total_wires];
        let n = state.len().min(self.total_wires);
        bits[..n].copy_from_slice(&state[..n]);
        for chunk in &self.chunks {
            apply_chunk_bits(&mut bits, chunk);
        }
        bits
    }

    /// Evaluate with ancilla initialized to 0; return data wires only.
    pub fn evaluate_data_bits(&self, input_data: &[bool]) -> Vec<bool> {
        let out = self.evaluate_bits(input_data);
        out[..self.data_wires].to_vec()
    }

    /// Evaluate with explicit full-wire input (data + ancilla); return data only.
    pub fn evaluate_data_bits_full(&self, full_input: &[bool]) -> Vec<bool> {
        assert_eq!(full_input.len(), self.total_wires);
        let out = self.evaluate_bits(full_input);
        out[..self.data_wires].to_vec()
    }

    pub fn evaluate_data_u64(&self, input_data: u64) -> u64 {
        assert!(self.data_wires <= 64);
        let mut bits = vec![false; self.data_wires];
        for i in 0..self.data_wires {
            bits[i] = ((input_data >> i) & 1) != 0;
        }
        let out = self.evaluate_data_bits(&bits);
        bits_to_u64(&out)
    }

    /// Evaluate with the low `prefix_zeros` data wires fixed to 0 and the
    /// remaining data wires set from `free` (low bit = first free wire).
    pub fn evaluate_prefixed_zeros(&self, prefix_zeros: usize, free: u64) -> Vec<bool> {
        assert!(prefix_zeros <= self.data_wires);
        let free_bits = self.data_wires - prefix_zeros;
        assert!(free_bits <= 64);
        let mut input = vec![false; self.data_wires];
        for i in 0..free_bits {
            input[prefix_zeros + i] = ((free >> i) & 1) != 0;
        }
        self.evaluate_data_bits(&input)
    }

    pub fn save_readable(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        fs::write(path, self.to_readable_string())
    }

    pub fn load_readable(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let s = fs::read_to_string(path)?;
        Self::from_readable_string(&s).map_err(std::io::Error::other)
    }

    pub fn to_readable_string(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# sandwich chunked circuit v3 (single-sided R + EM park)");
        let _ = writeln!(out, "data_wires {}", self.data_wires);
        let _ = writeln!(out, "total_wires {}", self.total_wires);
        let _ = writeln!(out, "n_chunks {}", self.chunks.len());
        let _ = writeln!(out);
        for (i, chunk) in self.chunks.iter().enumerate() {
            let _ = writeln!(out, "# chunk {i}");
            match &chunk.body {
                ChunkBody::Zero => {
                    let _ = writeln!(out, "kind reset");
                    let _ = writeln!(out, "wires {}", join_u16(&chunk.wires));
                }
                ChunkBody::Perm(p) => {
                    let _ = writeln!(out, "kind perm");
                    let _ = writeln!(out, "wires {}", join_u16(&chunk.wires));
                    let _ = writeln!(out, "perm {}", join_usize(&p.data));
                }
            }
            let _ = writeln!(out);
        }
        out
    }

    pub fn from_readable_string(s: &str) -> Result<Self, String> {
        let mut data_wires = None;
        let mut total_wires = None;
        let mut n_chunks: Option<usize> = None;
        let mut chunks = Vec::new();

        let mut kind: Option<&str> = None;
        let mut wires: Option<Vec<u16>> = None;

        for raw in s.lines() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split_whitespace();
            let key = parts.next().ok_or("empty line")?;
            match key {
                "data_wires" => {
                    data_wires = Some(parse_one(parts.next())?);
                }
                "total_wires" => {
                    total_wires = Some(parse_one(parts.next())?);
                }
                "n_chunks" => {
                    n_chunks = Some(parse_one(parts.next())?);
                }
                "kind" => {
                    kind = Some(parts.next().ok_or("missing kind")?);
                    wires = None;
                }
                "wires" => {
                    let rest = parts.collect::<Vec<_>>().join("");
                    wires = Some(parse_u16_list(&rest)?);
                    if kind == Some("reset") {
                        chunks.push(Chunk {
                            wires: wires.take().unwrap(),
                            body: ChunkBody::Zero,
                        });
                        kind = None;
                    }
                }
                "perm" => {
                    let rest = parts.collect::<Vec<_>>().join("");
                    let data = parse_usize_list(&rest)?;
                    let w = wires.take().ok_or("perm without wires")?;
                    if kind != Some("perm") {
                        return Err("perm line without kind perm".into());
                    }
                    let expect = 1usize << w.len();
                    if data.len() != expect {
                        return Err(format!(
                            "perm len {} != 2^{} = {}",
                            data.len(),
                            w.len(),
                            expect
                        ));
                    }
                    chunks.push(Chunk {
                        wires: w,
                        body: ChunkBody::Perm(Permutation { data }),
                    });
                    kind = None;
                }
                other => return Err(format!("unknown key {other}")),
            }
        }

        let data_wires = data_wires.ok_or("missing data_wires")?;
        let total_wires = total_wires.ok_or("missing total_wires")?;
        if let Some(n) = n_chunks {
            if n != chunks.len() {
                return Err(format!("n_chunks {n} != actual {}", chunks.len()));
            }
        }
        Ok(Self {
            data_wires,
            total_wires,
            chunks,
        })
    }
}

fn join_u16(xs: &[u16]) -> String {
    xs.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn join_usize(xs: &[usize]) -> String {
    xs.iter()
        .map(|x| x.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

fn parse_one<T: std::str::FromStr>(s: Option<&str>) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    s.ok_or_else(|| "missing value".to_string())?
        .parse()
        .map_err(|e| format!("{e}"))
}

fn parse_u16_list(s: &str) -> Result<Vec<u16>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|p| p.parse::<u16>().map_err(|e| e.to_string()))
        .collect()
}

fn parse_usize_list(s: &str) -> Result<Vec<usize>, String> {
    if s.is_empty() {
        return Ok(Vec::new());
    }
    s.split(',')
        .map(|p| p.parse::<usize>().map_err(|e| e.to_string()))
        .collect()
}

/// Standard Hamming weight: number of 1-bits.
pub fn hamming_weight(bits: &[bool]) -> usize {
    bits.iter().filter(|&&b| b).count()
}

pub fn cycle_notation(perm: &Permutation) -> String {
    let n = perm.data.len();
    let mut seen = vec![false; n];
    let mut cycles: Vec<Vec<usize>> = Vec::new();
    for start in 0..n {
        if seen[start] {
            continue;
        }
        let mut cycle = Vec::new();
        let mut x = start;
        while !seen[x] {
            seen[x] = true;
            cycle.push(x);
            x = perm.data[x];
        }
        if cycle.len() > 1 {
            if let Some((i, _)) = cycle.iter().enumerate().min_by_key(|(_, v)| *v) {
                cycle.rotate_left(i);
            }
            cycles.push(cycle);
        }
    }
    if cycles.is_empty() {
        return "()".to_string();
    }
    cycles
        .iter()
        .map(|c| {
            format!(
                "({})",
                c.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        })
        .collect::<Vec<_>>()
        .join("")
}

fn apply_chunk_bits(state: &mut [bool], chunk: &Chunk) {
    match &chunk.body {
        ChunkBody::Zero => {
            for &w in &chunk.wires {
                state[w as usize] = false;
            }
        }
        ChunkBody::Perm(perm) => {
            let k = chunk.wires.len();
            debug_assert_eq!(perm.data.len(), 1 << k);
            let mut local = 0usize;
            for (i, &w) in chunk.wires.iter().enumerate() {
                if state[w as usize] {
                    local |= 1 << i;
                }
            }
            let out_local = perm.data[local];
            for (i, &w) in chunk.wires.iter().enumerate() {
                state[w as usize] = ((out_local >> i) & 1) != 0;
            }
        }
    }
}

fn get_bit_usize(state: usize, idx: usize) -> bool {
    ((state >> idx) & 1) != 0
}

fn set_bit_usize(state: usize, idx: usize, bit: bool) -> usize {
    if bit {
        state | (1 << idx)
    } else {
        state & !(1 << idx)
    }
}

fn gate_wires(gate: [u16; 3]) -> Vec<u16> {
    vec![gate[0], gate[1], gate[2]]
}

fn gate_local_perm(_gate: [u16; 3]) -> Permutation {
    // Local wire order in the chunk is [pin0, pin1, pin2] = gate order.
    CircuitSeq {
        gates: vec![[0, 1, 2]],
    }
    .perm(3)
}

fn wire_bounds(circuit: &CircuitSeq, num_wires: usize) -> WireBounds {
    let mut last = vec![None; num_wires];
    for (i, gate) in circuit.gates.iter().enumerate() {
        for &w in gate {
            let w = w as usize;
            if w < num_wires {
                last[w] = Some(i);
            }
        }
    }
    WireBounds { last }
}

fn should_append_r(gate_idx: usize, w: &[u16], bounds: &WireBounds) -> bool {
    // Append `R` unless every wire in this gate is at its last plaintext appearance.
    w.iter().any(|&wire| {
        bounds.last.get(wire as usize).copied().flatten() != Some(gate_idx)
    })
}

fn wires_intersect(a: &[u16], b: &[u16]) -> bool {
    a.iter().any(|w| b.contains(w))
}

fn wires_set_diff(from: &[u16], keep: &[u16]) -> Vec<u16> {
    from.iter().copied().filter(|w| !keep.contains(w)).collect()
}

fn union_wires(mut xs: Vec<u16>, ys: &[u16]) -> Vec<u16> {
    xs.extend_from_slice(ys);
    xs.sort_unstable();
    xs.dedup();
    xs
}

/// Take parked records for `wires` out of the compiler table (for unmasking).
fn take_parked(parks: &mut Vec<ParkedWire>, wires: &[u16]) -> Vec<ParkedWire> {
    let set: HashSet<u16> = wires.iter().copied().collect();
    let mut out = Vec::new();
    parks.retain(|p| {
        if set.contains(&p.wire) {
            out.push(p.clone());
            false
        } else {
            true
        }
    });
    out
}

/// Typical fused gate support with `k=6`; prelude unmask chunks handle rare larger unions.
#[allow(dead_code)]
const TYPICAL_CHUNK_WIRES: usize = 11;
/// Absolute truth-table limit per chunk.
const HARD_CHUNK_WIRES: usize = 20;

fn sample_ancilla_subset(bank: &[u16], k: usize, rng: &mut impl Rng) -> Vec<u16> {
    assert!(k <= bank.len(), "mask_k={k} > ancilla bank {}", bank.len());
    let mut idx: Vec<usize> = (0..bank.len()).collect();
    idx.shuffle(rng);
    let mut out: Vec<u16> = idx.into_iter().take(k).map(|i| bank[i]).collect();
    out.sort_unstable();
    out
}

fn peek_unmask_ancillas(parks: &[ParkedWire], data_support: &[u16]) -> Vec<u16> {
    let set: HashSet<u16> = data_support.iter().copied().collect();
    let mut anc: Vec<u16> = Vec::new();
    for p in parks {
        if set.contains(&p.wire) {
            anc = union_wires(anc, &p.ancillas);
        }
    }
    anc
}

/// Random fresh `A_S` for this gate's remasks; build support including unmask ancillas.
fn sample_gate_ancillas(
    bank: &[u16],
    mask_k: usize,
    data_support: &[u16],
    unmask: &[ParkedWire],
    rng: &mut impl Rng,
) -> (Vec<u16>, Vec<u16>, usize) {
    let mut unmask_anc: Vec<u16> = Vec::new();
    for p in unmask {
        unmask_anc = union_wires(unmask_anc, &p.ancillas);
    }

    let max_k = mask_k.min(HARD_CHUNK_WIRES.saturating_sub(data_support.len()));
    assert!(
        max_k >= 1,
        "fused gate data support {} leaves no room for ancilla",
        data_support.len()
    );

    // Uniform random `k`-subset of the bank, rejecting until support fits.
    for effective_k in (1..=max_k).rev() {
        for _ in 0..64 {
            let a_s = sample_ancilla_subset(bank, effective_k, rng);
            let support = union_wires(
                union_wires(data_support.to_vec(), &unmask_anc),
                &a_s,
            );
            if support.len() <= HARD_CHUNK_WIRES {
                return (support, a_s, effective_k);
            }
        }
        // Last resort: random `k`-subset of already-needed unmask wires (no support growth).
        if unmask_anc.len() >= effective_k {
            let support = union_wires(data_support.to_vec(), &unmask_anc);
            if support.len() <= HARD_CHUNK_WIRES {
                let a_s = sample_ancilla_subset(&unmask_anc, effective_k, rng);
                return (support, a_s, effective_k);
            }
        }
    }

    panic!(
        "could not sample random A_S: data={} unmask_anc={} max_k={} (support budget {})",
        data_support.len(),
        unmask_anc.len(),
        max_k,
        HARD_CHUNK_WIRES
    );
}

/// When many distinct parked `A_S` sets would exceed the wire budget, peel off
/// unmask-only chunks grouped by identical ancilla subsets.
fn drain_unmask_preludes(
    parks: &mut Vec<ParkedWire>,
    data_support: &[u16],
    pi: &[usize],
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let data_set: HashSet<u16> = data_support.iter().copied().collect();

    loop {
        let unmask_anc = peek_unmask_ancillas(parks, data_support);
        if data_support.len() + unmask_anc.len() <= HARD_CHUNK_WIRES {
            break;
        }

        let mut groups: HashMap<Vec<u16>, Vec<u16>> = HashMap::new();
        for p in parks.iter() {
            if data_set.contains(&p.wire) {
                groups
                    .entry(p.ancillas.clone())
                    .or_default()
                    .push(p.wire);
            }
        }
        let Some((anc, mut wires)) = groups
            .into_iter()
            .max_by_key(|(_, ws)| ws.len())
        else {
            break;
        };

        wires.sort_unstable();
        wires.dedup();
        while wires.len() + anc.len() > HARD_CHUNK_WIRES {
            wires.pop();
        }
        if wires.is_empty() {
            break;
        }
        let taken = take_parked(parks, &wires);
        debug_assert_eq!(taken.len(), wires.len());

        let mut support = wires.clone();
        support = union_wires(support, &anc);
        assert!(
            support.len() <= HARD_CHUNK_WIRES,
            "prelude unmask support {} exceeds budget",
            support.len()
        );

        let step = taken.clone();
        let perm = perm_from_step(&support, |mut st| {
            for p in &step {
                let pi_u = &pi[..(1 << p.ancillas.len())];
                st = xor_em_mask(st, &support, p.wire, &p.ancillas, pi_u, p.r, p.b);
            }
            st
        });
        chunks.push(Chunk {
            wires: support,
            body: ChunkBody::Perm(perm),
        });
    }
    chunks
}

/// One fused permutation for a plaintext gate:
/// unmask → `R⁻¹` (on pending triples sharing a wire) → `g` → `R` → remask.
fn build_fused_gate_chunks(
    gate_idx: usize,
    gate: [u16; 3],
    w: &[u16],
    bounds: &WireBounds,
    pending: &mut Vec<PendingWrap>,
    parks: &mut Vec<ParkedWire>,
    bank: &[u16],
    mask_k: usize,
    pi: &[usize],
    rng: &mut impl Rng,
) -> (Vec<Chunk>, Option<PendingWrap>) {
    // Pending triples that share any wire with this gate (canonical order ⇒ usually one).
    let mut reuse = Vec::new();
    pending.retain(|p| {
        if wires_intersect(&p.wires, w) {
            reuse.push(p.clone());
            false
        } else {
            true
        }
    });

    let g_local = gate_local_perm(gate);
    let append_r = should_append_r(gate_idx, w, bounds);
    let r_next = if append_r { rand_s8(rng) } else { id3() };

    let mut data_support = w.to_vec();
    for p in &reuse {
        data_support = union_wires(data_support, &p.wires);
    }

    let mut chunks = drain_unmask_preludes(parks, &data_support, pi);

    let unmask = take_parked(parks, &data_support);

    let (support, a_s, effective_k) =
        sample_gate_ancillas(bank, mask_k, &data_support, &unmask, rng);
    debug_assert!(support.len() <= HARD_CHUNK_WIRES);
    let pi_eff = &pi[..(1 << effective_k)];

    let mut remask_wires: Vec<u16> = Vec::new();
    for p in &reuse {
        remask_wires.extend(wires_set_diff(&p.wires, w));
    }
    if append_r {
        remask_wires.extend_from_slice(w);
    }
    remask_wires.sort_unstable();
    remask_wires.dedup();

    let remask_keys: Vec<(u16, u32, bool)> = remask_wires
        .iter()
        .map(|&wire| {
            (
                wire,
                rng.random_range(0u32..(1u32 << effective_k)),
                rng.random(),
            )
        })
        .collect();

    let new_parks: Vec<ParkedWire> = remask_keys
        .iter()
        .map(|&(wire, r, b)| ParkedWire {
            wire,
            ancillas: a_s.clone(),
            r,
            b,
        })
        .collect();

    let unmask_step = unmask.clone();
    let reuse_step = reuse.clone();
    let w_step = w.to_vec();
    let remask_step = remask_keys.clone();
    let a_s_step = a_s.clone();

    let perm = perm_from_step(&support, |mut st| {
        for p in &unmask_step {
            let pi_u = &pi[..(1 << p.ancillas.len())];
            st = xor_em_mask(st, &support, p.wire, &p.ancillas, pi_u, p.r, p.b);
        }
        for p in &reuse_step {
            st = apply_local_perm_on(st, &support, &p.wires, &p.r.invert());
        }
        st = apply_local_perm_on(st, &support, &w_step, &g_local);
        st = apply_local_perm_on(st, &support, &w_step, &r_next);
        for &(wire, r, b) in &remask_step {
            st = xor_em_mask(st, &support, wire, &a_s_step, pi_eff, r, b);
        }
        st
    });

    parks.extend(new_parks);

    chunks.push(Chunk {
        wires: support,
        body: ChunkBody::Perm(perm),
    });
    let new_pending = if append_r {
        Some(PendingWrap {
            wires: w.to_vec(),
            r: r_next,
            source_gate: gate_idx,
        })
    } else {
        None
    };
    (chunks, new_pending)
}

/// Epilogue: strip remaining `R` wrappers and masks on dead wires.
fn build_epilogue_chunks(
    pending: &mut Vec<PendingWrap>,
    parks: &mut Vec<ParkedWire>,
    pi: &[usize],
) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    for p in pending.drain(..) {
        let unmask = take_parked(parks, &p.wires);
        let mut support = p.wires.clone();
        for q in &unmask {
            support = union_wires(support, &q.ancillas);
        }
        let w = p.wires.clone();
        let r_inv = p.r.invert();
        let unmask_step = unmask;
        let perm = perm_from_step(&support, |mut st| {
            for q in &unmask_step {
                let pi_u = &pi[..(1 << q.ancillas.len())];
                st = xor_em_mask(st, &support, q.wire, &q.ancillas, pi_u, q.r, q.b);
            }
            st = apply_local_perm_on(st, &support, &w, &r_inv);
            st
        });
        chunks.push(Chunk {
            wires: support,
            body: ChunkBody::Perm(perm),
        });
    }
    while !parks.is_empty() {
        let wire = parks[0].wire;
        let group = take_parked(parks, std::slice::from_ref(&wire));
        let mut support: Vec<u16> = vec![wire];
        support.extend_from_slice(&group[0].ancillas);
        support.sort_unstable();
        support.dedup();
        let g = group.clone();
        let perm = perm_from_step(&support, |mut st| {
            for q in &g {
                let pi_u = &pi[..(1 << q.ancillas.len())];
                st = xor_em_mask(st, &support, q.wire, &q.ancillas, pi_u, q.r, q.b);
            }
            st
        });
        chunks.push(Chunk {
            wires: support,
            body: ChunkBody::Perm(perm),
        });
    }
    chunks
}

/// Human-readable plaintext circuit (gate list).
pub fn format_plaintext_circuit(circuit: &CircuitSeq, data_wires: usize) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# plaintext r57 circuit");
    let _ = writeln!(out, "data_wires {data_wires}");
    let _ = writeln!(out, "n_gates {}", circuit.gates.len());
    let _ = writeln!(out);
    for (i, g) in circuit.gates.iter().enumerate() {
        let _ = writeln!(
            out,
            "gate {i:4}  active={:<4} ctrl+={:<4} ctrl-={}",
            g[0], g[1], g[2]
        );
    }
    out
}

fn chunk_phase(i: usize, init_gates: usize, plain_gates: usize) -> String {
    if i < init_gates {
        "init".to_string()
    } else if i < init_gates + plain_gates {
        format!("gate:{}", i - init_gates)
    } else {
        "epilogue".to_string()
    }
}

fn cycle_notation_truncated(perm: &Permutation, max_chars: usize) -> String {
    let s = cycle_notation(perm);
    if s.len() <= max_chars {
        s
    } else {
        format!(
            "{} … [{} chars omitted, {} total]",
            &s[..max_chars],
            s.len() - max_chars,
            s.len()
        )
    }
}

/// Write the published obfuscated circuit (chunk list + permutations) as text.
pub fn write_obfuscated_circuit(
    obf: &ChunkedCircuit,
    init_gates: usize,
    plain_gates: usize,
    out: &mut impl std::io::Write,
) -> std::io::Result<()> {
    let epilogue = obf
        .chunks
        .len()
        .saturating_sub(init_gates + plain_gates);
    writeln!(out, "# obfuscated sandwich circuit (published chunks)")?;
    writeln!(out, "data_wires {}", obf.data_wires)?;
    writeln!(out, "total_wires {}", obf.total_wires)?;
    writeln!(out, "n_chunks {}", obf.chunks.len())?;
    writeln!(out, "init_gates {init_gates}")?;
    writeln!(out, "plain_gates {plain_gates}")?;
    writeln!(out, "epilogue_chunks {epilogue}")?;
    writeln!(out)?;
    writeln!(
        out,
        "# Each chunk: local bit i corresponds to wires[i]; perm is cycle notation on 0..2^k-1."
    )?;
    writeln!(out)?;

    const MAX_PERM_CHARS: usize = 8000;

    for (i, chunk) in obf.chunks.iter().enumerate() {
        let phase = chunk_phase(i, init_gates, plain_gates);
        let ws: Vec<String> = chunk.wires.iter().map(|w| w.to_string()).collect();
        match &chunk.body {
            ChunkBody::Perm(p) => {
                let perm = cycle_notation_truncated(p, MAX_PERM_CHARS);
                writeln!(
                    out,
                    "chunk {i:5}  phase={phase:<10}  k={}  wires=[{}]",
                    chunk.k(),
                    ws.join(" ")
                )?;
                writeln!(out, "  perm={perm}")?;
            }
            ChunkBody::Zero => {
                writeln!(
                    out,
                    "chunk {i:5}  phase={phase:<10}  RESET  wires=[{}]",
                    ws.join(" ")
                )?;
            }
        }
    }
    Ok(())
}

/// Human-readable obfuscated circuit (chunk list + permutations).
pub fn format_obfuscated_circuit(
    obf: &ChunkedCircuit,
    init_gates: usize,
    plain_gates: usize,
) -> String {
    let mut buf = Vec::new();
    write_obfuscated_circuit(obf, init_gates, plain_gates, &mut buf).unwrap();
    String::from_utf8(buf).unwrap()
}

/// Random 3-wire Toffoli chunk for init scramble (active ∈ ancilla).
fn emit_init_scramble(
    data_wires: usize,
    bank: &[u16],
    n_gates: usize,
    rng: &mut impl Rng,
) -> Vec<Chunk> {
    let total = data_wires + bank.len();
    let mut chunks = Vec::with_capacity(n_gates);
    for _ in 0..n_gates {
        let active = bank[rng.random_range(0..bank.len())];
        let mut pool: Vec<u16> = (0..total as u16).filter(|&w| w != active).collect();
        pool.shuffle(rng);
        let gate = [active, pool[0], pool[1]];
        chunks.push(Chunk {
            wires: gate.to_vec(),
            body: ChunkBody::Perm(gate_local_perm(gate)),
        });
    }
    chunks
}

fn id3() -> Permutation {
    Permutation::id_perm(8)
}

fn rand_s8(rng: &mut impl Rng) -> Permutation {
    let mut p = Permutation::id_perm(8);
    p.data.shuffle(rng);
    p
}

fn perm_from_step(wires: &[u16], mut step: impl FnMut(usize) -> usize) -> Permutation {
    let k = wires.len();
    assert!(
        k <= HARD_CHUNK_WIRES,
        "chunk support too large for a truth table: {k} wires"
    );
    let n = 1 << k;
    let mut data = vec![0; n];
    let mut seen = vec![false; n];
    for x in 0..n {
        let y = step(x);
        assert!(y < n, "step left local domain");
        assert!(!seen[y], "step is not injective (not a permutation)");
        seen[y] = true;
        data[x] = y;
    }
    Permutation { data }
}

fn local_index(wires: &[u16], wire: u16) -> usize {
    wires
        .iter()
        .position(|&w| w == wire)
        .unwrap_or_else(|| panic!("wire {wire} not in support {wires:?}"))
}

fn apply_local_perm_on(
    state: usize,
    support: &[u16],
    sub_wires: &[u16],
    perm: &Permutation,
) -> usize {
    let mut local = 0usize;
    for (i, &w) in sub_wires.iter().enumerate() {
        let idx = local_index(support, w);
        if get_bit_usize(state, idx) {
            local |= 1 << i;
        }
    }
    let out = perm.data[local];
    let mut new_state = state;
    for (i, &w) in sub_wires.iter().enumerate() {
        let idx = local_index(support, w);
        new_state = set_bit_usize(new_state, idx, ((out >> i) & 1) != 0);
    }
    new_state
}

/// Random permutation table for `{0,1}^k` (`π[x]` = image of `x`).
fn random_perm_table(domain: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut p: Vec<usize> = (0..domain).collect();
    p.shuffle(rng);
    p
}

fn read_anc_pattern(state: usize, support: &[u16], anc: &[u16]) -> usize {
    let mut pat = 0usize;
    for (i, &w) in anc.iter().enumerate() {
        if get_bit_usize(state, local_index(support, w)) {
            pat |= 1 << i;
        }
    }
    pat
}

/// Even–Mansour-style boolean: `LSB(π(A_S ⊕ r)) ⊕ b`.
fn eval_em_mask(pi: &[usize], anc_val: usize, r: u32, b: bool) -> bool {
    let x = anc_val ^ (r as usize);
    debug_assert!(x < pi.len());
    ((pi[x] & 1) != 0) ^ b
}

fn xor_em_mask(
    state: usize,
    support: &[u16],
    data: u16,
    anc: &[u16],
    pi: &[usize],
    r: u32,
    b: bool,
) -> usize {
    let anc_val = read_anc_pattern(state, support, anc);
    let bit = get_bit_usize(state, local_index(support, data)) ^ eval_em_mask(pi, anc_val, r, b);
    set_bit_usize(state, local_index(support, data), bit)
}

/// Single-sided-wrap obfuscate with default parameters.
pub fn obfuscate_sandwich(
    circuit: &CircuitSeq,
    data_wires: usize,
    rng: &mut impl Rng,
) -> ChunkedCircuit {
    obfuscate_sandwich_with(circuit, data_wires, &SandwichParams::for_n(data_wires), rng)
}

/// Single-sided-wrap obfuscate with explicit parameters.
pub fn obfuscate_sandwich_with(
    circuit: &CircuitSeq,
    data_wires: usize,
    params: &SandwichParams,
    rng: &mut impl Rng,
) -> ChunkedCircuit {
    let gates = &circuit.gates;
    assert!(!gates.is_empty());
    assert!(params.ancilla_count >= params.mask_k);

    let bank: Vec<u16> = (0..params.ancilla_count)
        .map(|i| (data_wires + i) as u16)
        .collect();
    let total_wires = data_wires + params.ancilla_count;
    let bounds = wire_bounds(circuit, total_wires);

    // Public EM permutation π on {0,1}^k; per-wire keys (r,b) are compiler-private.
    let pi = random_perm_table(1 << params.mask_k, rng);

    let mut parks: Vec<ParkedWire> = Vec::new();
    let mut pending: Vec<PendingWrap> = Vec::new();
    let mut chunks: Vec<Chunk> = Vec::new();

    // Phase 0: ancilla randomization (explicit 3-wire Toffoli chunks).
    chunks.extend(emit_init_scramble(
        data_wires,
        &bank,
        params.init_gates,
        rng,
    ));

    for (i, &gate) in gates.iter().enumerate() {
        let w = gate_wires(gate);
        let (gate_chunks, new_pending) = build_fused_gate_chunks(
            i,
            gate,
            &w,
            &bounds,
            &mut pending,
            &mut parks,
            &bank,
            params.mask_k,
            &pi,
            rng,
        );
        chunks.extend(gate_chunks);
        if let Some(p) = new_pending {
            pending.push(p);
        }
    }

    chunks.extend(build_epilogue_chunks(&mut pending, &mut parks, &pi));

    ChunkedCircuit {
        data_wires,
        total_wires,
        chunks,
    }
}

/// Sample a random r57 circuit, canonicalize, single-sided-wrap obfuscate.
pub fn sample_and_obfuscate(
    data_wires: usize,
    gates: usize,
    rng: &mut impl Rng,
) -> (CircuitSeq, ChunkedCircuit) {
    let mut circuit = random_circuit_rng(data_wires, gates, rng);
    circuit.canonicalize();
    let obf = obfuscate_sandwich(&circuit, data_wires, rng);
    (circuit, obf)
}

fn random_circuit_rng(n: usize, m: usize, rng: &mut impl Rng) -> CircuitSeq {
    let mut circuit = Vec::with_capacity(m);
    for _ in 0..m {
        loop {
            let mut gate = [0u16; 3];
            let mut used = vec![false; n];
            for j in 0..3 {
                loop {
                    let v = rng.random_range(0..n) as u16;
                    if !used[v as usize] {
                        used[v as usize] = true;
                        gate[j] = v;
                        break;
                    }
                }
            }
            if circuit.last() == Some(&gate) {
                continue;
            }
            circuit.push(gate);
            break;
        }
    }
    CircuitSeq { gates: circuit }
}

fn bits_to_u64(bits: &[bool]) -> u64 {
    let mut y = 0u64;
    for (i, &b) in bits.iter().enumerate() {
        if b {
            y |= 1 << i;
        }
    }
    y
}

fn eval_plain_data_bits(plain: &CircuitSeq, data_bits: &[bool]) -> Vec<bool> {
    let n = data_bits.len();
    assert!(n <= 256, "plain eval via u256 supports at most 256 data wires");
    let mut state = u256::zero();
    for (i, &b) in data_bits.iter().enumerate() {
        if b {
            state |= u256::one() << i;
        }
    }
    let out = plain.evaluate_256(state);
    (0..n)
        .map(|i| ((out >> i) & u256::one()) == u256::one())
        .collect()
}

/// Probabilistic correctness with ancilla = 0.
pub fn check_correctness_random(
    plain: &CircuitSeq,
    obf: &ChunkedCircuit,
    trials: usize,
    rng: &mut impl Rng,
) -> Result<(), String> {
    let n = obf.data_wires;
    for t in 0..trials {
        let mut input = vec![false; n];
        for b in &mut input {
            *b = rng.random();
        }
        let expect = eval_plain_data_bits(plain, &input);
        let got = obf.evaluate_data_bits(&input);
        if got != expect {
            return Err(format!(
                "mismatch on trial {t}: obfuscated data output != plaintext"
            ));
        }
    }
    Ok(())
}

/// Probabilistic correctness with *arbitrary* ancilla inputs.
pub fn check_correctness_random_dirty_ancilla(
    plain: &CircuitSeq,
    obf: &ChunkedCircuit,
    trials: usize,
    rng: &mut impl Rng,
) -> Result<(), String> {
    let n = obf.data_wires;
    let tot = obf.total_wires;
    for t in 0..trials {
        let mut full = vec![false; tot];
        for b in &mut full {
            *b = rng.random();
        }
        let expect = eval_plain_data_bits(plain, &full[..n]);
        let got = obf.evaluate_data_bits_full(&full);
        if got != expect {
            return Err(format!(
                "mismatch on trial {t} with dirty ancilla: obfuscated != plaintext"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn bits_from_u64(x: u64, n: usize) -> Vec<bool> {
        (0..n).map(|i| ((x >> i) & 1) != 0).collect()
    }

    #[test]
    fn sandwich_matches_plaintext_on_zero_ancilla() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 8;
        let m = 24;
        let (plain, obf) = sample_and_obfuscate(n, m, &mut rng);
        for x in 0..(1u64 << n).min(256) {
            let expect = plain.evaluate(x as usize) as u64;
            let got = bits_to_u64(&obf.evaluate_data_bits(&bits_from_u64(x, n)));
            assert_eq!(got, expect, "mismatch at x={x}");
        }
    }

    #[test]
    fn sandwich_matches_plaintext_on_dirty_ancilla() {
        let mut rng = StdRng::seed_from_u64(5);
        let n = 8;
        let m = 24;
        let (plain, obf) = sample_and_obfuscate(n, m, &mut rng);
        assert!(obf.total_wires > n);
        for x in 0..(1u64 << n).min(64) {
            let expect = plain.evaluate(x as usize) as u64;
            for _ in 0..8 {
                let mut full = bits_from_u64(x, n);
                full.resize(obf.total_wires, false);
                for b in full.iter_mut().skip(n) {
                    *b = rng.random();
                }
                let got = bits_to_u64(&obf.evaluate_data_bits_full(&full));
                assert_eq!(got, expect, "dirty ancilla mismatch at x={x}");
            }
        }
    }

    #[test]
    fn probabilistic_check_100_pairs_medium() {
        let mut rng = StdRng::seed_from_u64(7);
        let (plain, obf) = sample_and_obfuscate(32, 256, &mut rng);
        check_correctness_random(&plain, &obf, 100, &mut rng).unwrap();
        check_correctness_random_dirty_ancilla(&plain, &obf, 100, &mut rng).unwrap();
    }

    #[test]
    fn no_reset_chunks_emitted() {
        let mut rng = StdRng::seed_from_u64(11);
        let (_plain, obf) = sample_and_obfuscate(32, 256, &mut rng);
        let resets = obf
            .chunks
            .iter()
            .filter(|c| matches!(c.body, ChunkBody::Zero))
            .count();
        assert_eq!(resets, 0, "compiler must not emit RESET");
    }

    #[test]
    fn ancilla_bank_is_theta_n() {
        let mut rng = StdRng::seed_from_u64(11);
        let n = 32;
        let m = 256;
        let (_plain, obf) = sample_and_obfuscate(n, m, &mut rng);
        // Fixed bank of size n ⇒ total = 2n.
        assert_eq!(obf.total_wires, 2 * n);
        assert!(
            obf.total_wires < n + m,
            "ancilla grew like Θ(m): total={}",
            obf.total_wires
        );
    }

    #[test]
    fn same_support_gates_correct() {
        let mut rng = StdRng::seed_from_u64(3);
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2], [0, 1, 2], [0, 1, 2]],
        };
        let obf = obfuscate_sandwich(&circuit, 3, &mut rng);
        for x in 0..8u64 {
            let expect = circuit.evaluate(x as usize) as u64;
            assert_eq!(obf.evaluate_data_u64(x), expect);
        }
    }

    #[test]
    fn colliding_pair_correct() {
        let mut rng = StdRng::seed_from_u64(4);
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2], [3, 1, 2]],
        };
        let obf = obfuscate_sandwich(&circuit, 4, &mut rng);
        for x in 0..16u64 {
            let expect = circuit.evaluate(x as usize) as u64;
            assert_eq!(obf.evaluate_data_u64(x), expect, "x={x}");
            // Dirty ancilla too.
            let mut full = bits_from_u64(x, 4);
            full.resize(obf.total_wires, true);
            let got = bits_to_u64(&obf.evaluate_data_bits_full(&full));
            assert_eq!(got, expect, "dirty x={x}");
        }
    }

    #[test]
    fn cycle_notation_smoke() {
        let p = Permutation {
            data: vec![1, 2, 0, 3],
        };
        let s = cycle_notation(&p);
        assert!(s.contains('('), "{s}");
    }

    #[test]
    fn reuse_after_disjoint() {
        let mut rng = StdRng::seed_from_u64(100);
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2], [3, 4, 5], [0, 6, 7]],
        };
        let params = SandwichParams {
            ancilla_count: 8,
            mask_k: 6,
            init_gates: 0,
        };
        let obf = obfuscate_sandwich_with(&circuit, 8, &params, &mut rng);
        for x in 0..256u64 {
            let expect = circuit.evaluate(x as usize) as u64;
            assert_eq!(obf.evaluate_data_u64(x), expect, "x={x}");
        }
    }

    #[test]
    fn two_gate_disjoint_exhaustive() {
        let mut rng = StdRng::seed_from_u64(99);
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2], [3, 4, 5]],
        };
        let params = SandwichParams {
            ancilla_count: 8,
            mask_k: 6,
            init_gates: 0,
        };
        let obf = obfuscate_sandwich_with(&circuit, 6, &params, &mut rng);
        for x in 0..64u64 {
            let expect = circuit.evaluate(x as usize) as u64;
            assert_eq!(obf.evaluate_data_u64(x), expect, "x={x}");
        }
        check_correctness_random_dirty_ancilla(&circuit, &obf, 10, &mut rng).unwrap();
    }

    #[test]
    fn zero_init_scramble_random_correct() {
        let mut rng = StdRng::seed_from_u64(1);
        let n = 8;
        let m = 24;
        let mut circuit = random_circuit_rng(n, m, &mut rng);
        circuit.canonicalize();
        let params = SandwichParams {
            ancilla_count: 8,
            mask_k: 6,
            init_gates: 0,
        };
        let obf = obfuscate_sandwich_with(&circuit, n, &params, &mut rng);
        check_correctness_random(&circuit, &obf, 20, &mut rng).unwrap();
    }

    #[test]
    fn init_scramble_only_preserves_data_projection() {
        // Ancilla scramble alone must not change data bits.
        let mut rng = StdRng::seed_from_u64(42);
        let n = 8;
        let params = SandwichParams {
            ancilla_count: 16,
            mask_k: 6,
            init_gates: 32,
        };
        let bank: Vec<u16> = (0..params.ancilla_count)
            .map(|i| (n + i) as u16)
            .collect();
        let chunks = emit_init_scramble(n, &bank, params.init_gates, &mut rng);
        for x in 0..256u64 {
            let mut state = bits_from_u64(x, n);
            state.resize(n + params.ancilla_count, false);
            // set random-ish ancilla from x
            for i in 0..params.ancilla_count {
                state[n + i] = ((x >> (i % 8)) & 1) != 0;
            }
            let before = state[..n].to_vec();
            for c in &chunks {
                apply_chunk_bits(&mut state, c);
            }
            assert_eq!(&state[..n], &before[..], "scramble changed data at x={x}");
        }
    }
}
