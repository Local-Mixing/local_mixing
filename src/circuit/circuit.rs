// Basic implementation for circuit, gate, and permutations
use primitive_types::U256 as u256;
use primitive_types::U512 as u512;
use rand::{RngCore, seq::SliceRandom};
use rustc_hash::FxHashSet as HashSet;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering as CmpOrdering;
use std::sync::OnceLock;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Instant;
use uint::construct_uint;

construct_uint! {
    pub struct U1024(16);
}

pub static CANON4_CORE_TIME: AtomicU64 = AtomicU64::new(0);
pub static POLYCANON_CORE_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANON_BENCH_CALLS: AtomicU64 = AtomicU64::new(0);
pub static CANON4_RULE_L_TIME: AtomicU64 = AtomicU64::new(0);
pub static CANON4_RULE_L_CALLS: AtomicU64 = AtomicU64::new(0);
pub static CANON4_RULE_L_BRANCHES: AtomicU64 = AtomicU64::new(0);

/// Canonicalization-for-lookup calls skipped because a window touches more
/// than 64 distinct wires. `Monomial` is a `u64`, so such a window cannot be
/// represented without aliasing variables and potentially producing a false
/// database hit.
pub static OVERSIZED_CANON_SKIPS: AtomicU64 = AtomicU64::new(0);

/// Windows skipped while constructing polynomials because
/// `CANON_MONOMIAL_CAP` was exceeded.
pub static CANON_CAP_SKIPS: AtomicU64 = AtomicU64::new(0);

/// Windows skipped because Rule-L backtracking exceeded
/// `CANON_RULE_L_BRANCH_CAP`.
pub static CANON_RULE_L_SKIPS: AtomicU64 = AtomicU64::new(0);

/// Optional deterministic bound on the total Rule-L candidate budget across
/// one complete recursive canonicalization tree. Unset means unbounded.
pub fn canon_rule_l_branch_cap() -> Option<u64> {
    static CAP: OnceLock<Option<u64>> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("CANON_RULE_L_BRANCH_CAP")
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
            .filter(|&cap| cap > 0)
    })
}

thread_local! {
    // Reset at each canonicalize_polys_4 entry, then shared by every
    // recursive canon4_run call in that top-level canonicalization.
    static RULE_L_BRANCHES_USED: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Optional deterministic bound on the number of monomials retained per wire
/// while constructing a lookup window. Unset means unbounded.
pub fn canon_monomial_cap() -> Option<usize> {
    static CAP: OnceLock<Option<usize>> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("CANON_MONOMIAL_CAP")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&cap| cap > 0)
    })
}

fn bench_canon_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("BENCH_CANON").is_ok())
}

// ---------------------------------------------------------------------------
// Canonicalization result cache.
//
// canonicalize_polys_single is pure: the canonical polys and final order are
// fully determined by the dense-remapped, gate-canonicalized window. The
// expansion/compression games draw the same windows over and over (SAMF
// templates recur all over the circuit), so an exact process-wide cache
// skips to_polynomial + canonicalize_polys_4 entirely on repeats.
//
// CANON_CACHE_MB caps the approximate entry-byte footprint (default 256 MiB;
// 0 disables). On overflow the cache is cleared wholesale, mirroring the
// Frozen lookup cache's epoch-reset policy.
// ---------------------------------------------------------------------------
struct CanonCacheEntry {
    polys: Vec<Polynomial>,
    order: Vec<usize>,
    /// `xxh3_128(polys_repr_blob(polys)).to_le_bytes()`, precomputed at insert
    /// time so hashed lookups can return the frozen-DB key on a cache hit
    /// without deep-cloning `polys` and re-serializing/re-hashing them.
    polys_key: [u8; 16],
}

/// Outcome of the shared canonicalize-single core, before the public wrappers
/// shape it into their respective return types.
enum CanonSingleInner {
    /// Oversized window, monomial-cap skip, or Rule-L budget skip: the
    /// canonical result is the empty sentinel.
    Skip,
    /// Exact-cache hit; the entry holds polys, order, and the precomputed key.
    Cached(std::sync::Arc<CanonCacheEntry>),
    /// Freshly computed `(polys, order, polys_key)`. The key is `Some` iff the
    /// result was just inserted into the cache (where it is computed anyway).
    Fresh(Vec<Polynomial>, Vec<usize>, Option<[u8; 16]>),
}

type CanonCacheMap =
    dashmap::DashMap<Box<[u16]>, std::sync::Arc<CanonCacheEntry>, rustc_hash::FxBuildHasher>;

static CANON_CACHE_BYTES: AtomicU64 = AtomicU64::new(0);
pub static CANON_CACHE_HITS: AtomicU64 = AtomicU64::new(0);
pub static CANON_CACHE_QUERIES: AtomicU64 = AtomicU64::new(0);

fn canon_cache_cap_bytes() -> u64 {
    static CAP: OnceLock<u64> = OnceLock::new();
    *CAP.get_or_init(|| {
        std::env::var("CANON_CACHE_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(256)
            .saturating_mul(1024 * 1024)
    })
}

fn canon_cache() -> Option<&'static CanonCacheMap> {
    static CACHE: OnceLock<Option<CanonCacheMap>> = OnceLock::new();
    CACHE
        .get_or_init(|| (canon_cache_cap_bytes() > 0).then(CanonCacheMap::default))
        .as_ref()
}

fn compression_trace_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("COMPRESSION_TRACE").is_ok())
}

fn compression_trace_threshold_ms() -> u128 {
    static THRESHOLD: OnceLock<u128> = OnceLock::new();
    *THRESHOLD.get_or_init(|| {
        std::env::var("COMPRESSION_TRACE_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1_000)
    })
}

// Gate [a, pos_ctrl, neg_ctrl]: flip a UNLESS neg_ctrl=1 AND NOT pos_ctrl
// (flips when pos_ctrl=1 OR neg_ctrl=0)
// We are only concerned with gate g57
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Gate {
    pub pins: [usize; 3], //one active wire (0) and two control wires (1,2)
}

// Circuits stored as a sequence of gates [u16;3]
// Gate type is legacy
#[derive(Clone, Debug, Default, Serialize, Deserialize, Eq, Hash, PartialEq)]
pub struct CircuitSeq {
    pub gates: Vec<[u16; 3]>,
}

// Polynomial representation of circuit
pub type Monomial = u64;
pub type Polynomial = Vec<Monomial>;

// Permutations are all the possible outputs of a circuit
// On n wires permutation length is 1 << n
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permutation {
    pub data: Vec<usize>,
}

// Functions on Gate struct and [u8;3]
impl Gate {
    // Gates collide iff either active pin shares a wire with any other pin
    pub fn collides_index(gate: &[u16; 3], other: &[u16; 3]) -> bool {
        gate[0] == other[1] || gate[0] == other[2] || gate[1] == other[0] || gate[2] == other[0]
    }

    //b is "larger"
    pub fn ordered_index(gate: &[u16; 3], other: &[u16; 3]) -> bool {
        if gate[0] > other[0] {
            return false;
        } else if gate[0] == other[0] {
            if gate[1] > other[1] {
                return false;
            } else if gate[1] == other[1] {
                return gate[2] < other[2];
            }
        }
        true
    }

    // Evaluate a bit string after a single gate under gate r57
    #[inline(always)]
    pub fn evaluate_index(state: usize, gate: [u16; 3]) -> usize {
        let c1 = (state >> gate[1]) & 1;
        let c2 = (state >> gate[2]) & 1;
        state ^ (c1 | ((!c2) & 1)) << gate[0]
    }

    // Evaluate up to 128 bits
    #[inline(always)]
    pub fn evaluate_index_128(state: u128, gate: [u16; 3]) -> u128 {
        let c1 = (state >> gate[1]) & 1;
        let c2 = (state >> gate[2]) & 1;
        state ^ ((c1 | (1 ^ c2)) << gate[0])
    }

    // Evaluate up to 256 bits. Direct limb indexing: the bignum shift chains
    // cost ~6 full-width ops per gate where 3 u64 ops suffice. Wires must be
    // in range (the old full-width shifts silently evaluated out-of-range
    // wires as zero; dispatchers pick the kernel by max touched wire).
    #[inline(always)]
    pub fn evaluate_index_256(mut state: u256, gate: [u16; 3]) -> u256 {
        debug_assert!(gate[0] < 256 && gate[1] < 256 && gate[2] < 256);
        let c1 = (state.0[(gate[1] >> 6) as usize] >> (gate[1] & 63)) & 1;
        let c2 = (state.0[(gate[2] >> 6) as usize] >> (gate[2] & 63)) & 1;
        state.0[(gate[0] >> 6) as usize] ^= (c1 | (1 ^ c2)) << (gate[0] & 63);
        state
    }

    #[inline(always)]
    pub fn evaluate_index_512(mut state: u512, gate: [u16; 3]) -> u512 {
        debug_assert!(gate[0] < 512 && gate[1] < 512 && gate[2] < 512);
        let c1 = (state.0[(gate[1] >> 6) as usize] >> (gate[1] & 63)) & 1;
        let c2 = (state.0[(gate[2] >> 6) as usize] >> (gate[2] & 63)) & 1;
        state.0[(gate[0] >> 6) as usize] ^= (c1 | (1 ^ c2)) << (gate[0] & 63);
        state
    }

    #[inline(always)]
    pub fn evaluate_index_1024(mut state: U1024, gate: [u16; 3]) -> U1024 {
        debug_assert!(gate[0] < 1024 && gate[1] < 1024 && gate[2] < 1024);
        let c1 = (state.0[(gate[1] >> 6) as usize] >> (gate[1] & 63)) & 1;
        let c2 = (state.0[(gate[2] >> 6) as usize] >> (gate[2] & 63)) & 1;
        state.0[(gate[0] >> 6) as usize] ^= (c1 | (1 ^ c2)) << (gate[0] & 63);
        state
    }

    // Evaluate a list of gates
    #[inline(always)]
    pub fn evaluate_index_list(state: usize, gates: &[[u16; 3]]) -> usize {
        let mut current_wires = state;
        for g in gates {
            current_wires = Self::evaluate_index(current_wires, *g);
        }
        current_wires
    }

    /// Evaluate a g57 gate list on a single u64 state word (up to 64 wires).
    ///
    /// The narrowest kernel: the whole state lives in one register, so the
    /// walk is a pure dependency chain with no memory traffic beyond the gate
    /// stream. Dispatchers pick it when every touched wire is below 64.
    #[inline(always)]
    pub fn evaluate_index_list_64(state: u64, gates: &[[u16; 3]]) -> u64 {
        let mut s = state;
        for &g in gates {
            debug_assert!(g[0] < 64 && g[1] < 64 && g[2] < 64);
            let c1 = (s >> (g[1] & 63)) & 1;
            let c2 = (s >> (g[2] & 63)) & 1;
            s ^= (c1 | (1 ^ c2)) << (g[0] & 63);
        }
        s
    }

    #[inline(always)]
    pub fn evaluate_index_list_128(state: u128, gates: &[[u16; 3]]) -> u128 {
        let mut limbs = [state as u64, (state >> 64) as u64];
        eval_limbs::<2>(&mut limbs, gates);
        (limbs[0] as u128) | ((limbs[1] as u128) << 64)
    }

    // The per-gate work is three u64 accesses; taking/returning the whole
    // bignum by value made the loop copy 32/64/128 bytes per gate on top of
    // that. Running the limb array in place instead keeps one stack slot live
    // for the entire walk (measured 11.2 -> 3.4 ns/gate at 256 bits).
    #[inline(always)]
    pub fn evaluate_index_list_256(mut state: u256, gates: &[[u16; 3]]) -> u256 {
        eval_limbs::<4>(&mut state.0, gates);
        state
    }

    #[inline(always)]
    pub fn evaluate_index_list_512(mut state: u512, gates: &[[u16; 3]]) -> u512 {
        eval_limbs::<8>(&mut state.0, gates);
        state
    }

    #[inline(always)]
    pub fn evaluate_index_list_1024(mut state: U1024, gates: &[[u16; 3]]) -> U1024 {
        eval_limbs::<16>(&mut state.0, gates);
        state
    }

    /// Bit-sliced g57 evaluation: `state[w]` carries one bit per sample lane,
    /// so a single walk evaluates 64 independent inputs.
    ///
    /// Transposing the state this way turns "one wide word per sample" into
    /// "one word per wire", which is what makes multi-input work cheap: the
    /// per-gate cost is the same handful of u64 ops the scalar kernel pays,
    /// but it now covers 64 samples instead of one.
    ///
    /// `t ^= pos OR NOT neg` becomes `state[t] ^= state[pos] | !state[neg]`,
    /// which reproduces the scalar kernel lane by lane — including the X-gate
    /// case `pos == neg`, where `s | !s` is all ones and the target toggles
    /// unconditionally.
    ///
    /// `state.len()` must be a power of two greater than every wire index the
    /// circuit touches; size it with [`lane_state_len`].
    #[inline]
    pub fn eval_lanes_index_list(gates: &[[u16; 3]], state: &mut [u64]) {
        debug_assert!(state.len().is_power_of_two() && !state.is_empty());
        // Masking with a power-of-two length lets the bounds checks fold away
        // without unsafe; the debug_assert pins the in-range contract.
        let m = state.len() - 1;
        for &[t, x, y] in gates {
            debug_assert!((t as usize) <= m && (x as usize) <= m && (y as usize) <= m);
            let fire = state[(x as usize) & m] | !state[(y as usize) & m];
            state[(t as usize) & m] ^= fire;
        }
    }
}

/// Length of the bit-sliced lane state for a circuit touching wires `< wires`.
///
/// Rounded up to a power of two so the kernels can mask instead of
/// bounds-check; the slack is a few hundred bytes at most.
pub fn lane_state_len(wires: usize) -> usize {
    wires.max(1).next_power_of_two()
}

/// One g57 gate against a fixed-size limb array, in place.
///
/// `t ^= pos OR NOT neg`, with every wire index split into (limb, bit) exactly
/// as the fixed-width kernels above do. `L` is a const parameter so the limb
/// index masks fold to constants and the array stays in one stack slot.
#[inline(always)]
fn apply_limbs<const L: usize>(state: &mut [u64; L], gate: [u16; 3]) {
    debug_assert!(
        (gate[0] as usize) < L * 64 && (gate[1] as usize) < L * 64 && (gate[2] as usize) < L * 64
    );
    // `& (L - 1)` where L is a power of two lets LLVM drop the bounds check
    // without unsafe; the debug_assert above pins the in-range contract that
    // every dispatcher already guarantees.
    const {
        assert!(L.is_power_of_two(), "limb count must be a power of two");
    }
    let c1 = (state[((gate[1] >> 6) as usize) & (L - 1)] >> (gate[1] & 63)) & 1;
    let c2 = (state[((gate[2] >> 6) as usize) & (L - 1)] >> (gate[2] & 63)) & 1;
    state[((gate[0] >> 6) as usize) & (L - 1)] ^= (c1 | (1 ^ c2)) << (gate[0] & 63);
}

/// Walk a gate list against a limb array in place.
#[inline(always)]
fn eval_limbs<const L: usize>(state: &mut [u64; L], gates: &[[u16; 3]]) {
    for &g in gates {
        apply_limbs::<L>(state, g);
    }
}

/// Non-digit markers in the base-83 decode table.
const SEMI: u8 = 0xFD;
const TILDE: u8 = 0xFE;

/// Byte -> base-83 wire digit, with the gate separator and the `~` overflow
/// prefix folded in as sentinels above the digit range and `0xFF` for every
/// byte that is not part of the grammar. Built at compile time from the same
/// alphabet `repr()` emits, so the table cannot drift from the encoder.
const fn build_base83_decode() -> [u8; 256] {
    const ALPHABET: &[u8; 83] =
        b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?";
    let mut table = [0xFFu8; 256];
    let mut i = 0;
    while i < 83 {
        table[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table[b';' as usize] = SEMI;
    table[b'~' as usize] = TILDE;
    table
}

/// Close one `;`-delimited segment, rejecting the same malformed shapes the
/// segment-at-a-time parser did.
#[inline]
fn finish_gate(
    gates: &mut Vec<[u16; 3]>,
    wires: &[u16; 3],
    count: usize,
    overflow: u32,
    seg: &[u8],
) {
    if overflow != 0 {
        panic!("Expected wire character after ~");
    }
    if count != 3 {
        panic!(
            "Each gate must have exactly 3 wires: {:?}",
            String::from_utf8_lossy(seg)
        );
    }
    gates.push(*wires);
}

impl Permutation {
    pub fn new(data: Vec<usize>) -> Permutation {
        Permutation { data }
    }

    // Compose two permutations: (self ∘ other)[i] = self[other[i]].
    pub fn compose(&self, other: &Permutation) -> Permutation {
        if self.data.len() != other.data.len() {
            panic!("Permutation length mismatch in compose");
        }
        let data = (0..self.data.len())
            .map(|i| self.data[other.data[i]])
            .collect();
        Permutation { data }
    }
    pub fn is_perm(&self) -> bool {
        let mut temp_perm = self.clone();
        temp_perm.data.sort_unstable();
        temp_perm == Permutation::id_perm(self.data.len())
    }

    pub fn id_perm(n: usize) -> Permutation {
        let temp_data = (0..n).collect();
        Permutation { data: temp_data }
    }

    // n is the length of the permutation. For a random permutation on n bits, do 1 << n
    pub fn rand_perm(n: usize) -> Permutation {
        let mut p = Permutation::id_perm(n);
        let mut rng = rand::rng();
        p.data.shuffle(&mut rng);
        p
    }

    pub fn invert(&self) -> Permutation {
        let mut inv = vec![0; self.data.len()];
        self.data
            .iter()
            .enumerate()
            .for_each(|(i, &val)| inv[val] = i);
        Permutation { data: inv }
    }
}

impl CircuitSeq {
    // Evaluate the entire circuit with a starting input
    pub fn evaluate(&self, input: usize) -> usize {
        Gate::evaluate_index_list(input, &self.gates)
    }

    // Evaluate the circuit on a 64-bit input state (one bit per wire).
    pub fn evaluate_64(&self, input: u64) -> u64 {
        Gate::evaluate_index_list_64(input, &self.gates)
    }

    // Evaluate the circuit on a 128-bit input state (one bit per wire).
    pub fn evaluate_128(&self, input: u128) -> u128 {
        Gate::evaluate_index_list_128(input, &self.gates)
    }

    // Evaluate the circuit on a 512-bit input state (one bit per wire).
    pub fn evaluate_512(&self, input: u512) -> u512 {
        Gate::evaluate_index_list_512(input, &self.gates)
    }

    // Evaluate the circuit on a 256-bit input state (one bit per wire).
    pub fn evaluate_256(&self, input: u256) -> u256 {
        Gate::evaluate_index_list_256(input, &self.gates)
    }

    // Evaluate the circuit on a 1024-bit input state (one bit per wire).
    pub fn evaluate_1024(&self, input: U1024) -> U1024 {
        Gate::evaluate_index_list_1024(input, &self.gates)
    }

    /// True if any two adjacent gates are identical. Two identical adjacent
    /// self-inverse gates cancel, so a canonicalized circuit containing an
    /// adjacent duplicate is not minimal — `db_generation::regular` uses this
    /// to reject such candidates while rebuilding the replacement DB.
    pub fn adjacent_id(&self) -> bool {
        if self.gates.is_empty() {
            return false;
        }
        for i in 0..self.gates.len() - 1 {
            if self.gates[i] == self.gates[i + 1] {
                return true;
            }
        }
        false
    }

    /// Reconstruct CircuitSeq from a BLOB
    pub fn from_blob(blob: &[u8]) -> Self {
        assert!(blob.len() % 3 == 0, "Invalid blob length");
        let gates: Vec<[u16; 3]> = blob
            .chunks(3)
            .map(|chunk| [chunk[0] as u16, chunk[1] as u16, chunk[2] as u16])
            .collect();
        CircuitSeq { gates }
    }

    // Rewire wire i -> perm[i]
    pub fn rewire(&mut self, perm: &Permutation, n: usize) {
        if perm.data.is_empty() {
            return;
        }

        if perm.data.len() != n {
            panic!("wrong size perm! got {}, have {} wires", perm.data.len(), n);
        }

        if !perm.is_perm() {
            panic!("{:?} is not a permutation!", perm);
        }

        for gate in &mut self.gates {
            *gate = [
                perm.data[gate[0] as usize] as u16,
                perm.data[gate[1] as usize] as u16,
                perm.data[gate[2] as usize] as u16,
            ];
        }
    }

    // Representing circuit as a string
    pub fn repr(&self) -> String {
        fn wire_to_char(w: u8) -> char {
            match w {
                0..=9 => (b'0' + w) as char,          // 0-9
                10..=35 => (b'a' + (w - 10)) as char, // a-z
                36..=61 => (b'A' + (w - 36)) as char, // A-Z
                // Special characters 62..=71
                62 => '!',
                63 => '@',
                64 => '#',
                65 => '$',
                66 => '%',
                67 => '^',
                68 => '&',
                69 => '*',
                70 => '(',
                71 => ')',
                // Special characters 72..=82
                72 => '-',
                73 => '_',
                74 => '=',
                75 => '+',
                76 => '[',
                77 => ']',
                78 => '{',
                79 => '}',
                80 => '<',
                81 => '>',
                82 => '?',
                _ => panic!("Invalid wire index: {}", w),
            }
        }

        const BASE: u32 = 83; // 0..82 is base

        // Append in place: the previous version built a fresh `String` per
        // wire, i.e. three heap allocations per gate.
        fn encode_wire_into(out: &mut String, w: u32) {
            let tildes = w / BASE;
            for _ in 0..tildes {
                out.push('~');
            }
            out.push(wire_to_char((w - tildes * BASE) as u8));
        }

        // Four bytes per gate is the floor (three wire characters plus the
        // separator); wide circuits add `~` prefixes on top.
        let mut s = String::with_capacity(self.gates.len() * 4);
        for gate in &self.gates {
            for &wire in gate {
                encode_wire_into(&mut s, wire as u32);
            }
            s.push(';'); // gate separator
        }
        s
    }

    pub fn from_string(s: &str) -> Self {
        Self::from_bytes(s.as_bytes())
    }

    /// Parse the base-83 `repr()` encoding straight from bytes.
    ///
    /// The encoding is ASCII, so this is the same grammar `from_string` accepts
    /// without the UTF-8 round trip a caller would otherwise pay to read a
    /// multi-megabyte circuit file. A flat byte scan over a `;`-separated
    /// stream also avoids the per-gate `Vec` and `chars().peekable()` the
    /// previous implementation built (measured 65 -> 5 ns/gate on a 13.5M-gate
    /// circuit, i.e. ~0.9 s off a single `evaluate` invocation).
    pub fn from_bytes(raw: &[u8]) -> Self {
        const BASE: u32 = 83;

        // Byte -> wire digit, with `;` and `~` as sentinels above the digit
        // range and 0xFF for everything outside the grammar.
        static DECODE: [u8; 256] = build_base83_decode();

        // Mirror `s.trim()`: the old parser trimmed Unicode whitespace, but the
        // only characters that can appear around a base-83 body are ASCII
        // spaces and newlines, and every wire character is a non-whitespace
        // ASCII byte.
        let mut body = raw;
        while let Some((first, rest)) = body.split_first() {
            if first.is_ascii_whitespace() {
                body = rest;
            } else {
                break;
            }
        }
        while let Some((last, rest)) = body.split_last() {
            if last.is_ascii_whitespace() {
                body = rest;
            } else {
                break;
            }
        }

        // `repr()` emits one `;` per gate, so the separator count is the exact
        // gate count for anything it wrote. The `+ 1` covers a hand-written
        // body whose last segment is unterminated: without it that one extra
        // gate would double the whole buffer to hold it.
        let mut gates: Vec<[u16; 3]> =
            Vec::with_capacity(body.iter().filter(|&&b| b == b';').count() + 1);

        // One flat pass with a single table lookup per byte. The separator and
        // the `~` overflow prefix are encoded in the same table as the wire
        // digits, so the loop has one dispatch instead of the nested
        // scan-tildes / check-separator / decode structure a segment-at-a-time
        // reader needs.
        let mut wires = [0u16; 3];
        let mut count = 0usize;
        let mut overflow: u32 = 0;
        let mut seg_start = 0usize;

        for (i, &b) in body.iter().enumerate() {
            let d = DECODE[b as usize];
            if d < BASE as u8 {
                if count < 3 {
                    wires[count] = (d as u32 + overflow * BASE) as u16;
                }
                count += 1;
                overflow = 0;
            } else if d == TILDE {
                overflow += 1;
            } else if d == SEMI {
                if count != 0 || overflow != 0 {
                    finish_gate(&mut gates, &wires, count, overflow, &body[seg_start..i]);
                }
                count = 0;
                overflow = 0;
                seg_start = i + 1;
            } else {
                panic!("Invalid wire char: {}", b as char);
            }
        }
        // A body that does not end in a separator still carries a final gate,
        // matching `split(';')` on an unterminated last segment.
        if count != 0 || overflow != 0 {
            finish_gate(&mut gates, &wires, count, overflow, &body[seg_start..]);
        }
        CircuitSeq { gates }
    }

    // Gives a "pretty" circuit representation. Does not support over 83 wires
    pub fn to_string(&self, num_wires: usize) -> String {
        let mut result = String::new();

        // Local character map (0-9, a-z, A-Z)
        let wire_map_chars: Vec<char> =
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
                .chars()
                .collect();

        // --- Pretty circuit diagram ---
        for wire in 0..num_wires {
            result += &format!("{:<2} --", wire);
            for gate in &self.gates {
                if gate[0] == wire as u16 {
                    result += "( )";
                } else if gate[1] == wire as u16 {
                    result += "-●-";
                } else if gate[2] == wire as u16 {
                    result += "-○-";
                } else {
                    result += "-|-";
                }
                result.push_str("---");
            }
            result.push('\n');
        }

        // Compact circuit string (like "123;124;213;")
        let compact: String = self
            .gates
            .iter()
            .map(|g| {
                g.iter()
                    .map(|&x| wire_map_chars.get(x as usize).unwrap_or(&'?').to_string())
                    .collect::<String>()
                    + ";"
            })
            .collect();

        result.push_str("\n");
        result.push_str(&compact);

        result
    }

    // Combine two circuits
    pub fn concat(&self, other: &CircuitSeq) -> CircuitSeq {
        let mut gates = self.gates.clone();
        gates.extend_from_slice(&other.gates);
        CircuitSeq { gates }
    }

    // Returns the wires touched by a circuit
    pub fn used_wires(&self) -> Vec<u16> {
        // Stack-bitset fast path: mark wires in a [u64; 16] (covers wires
        // 0..1023, the overwhelmingly common case) in a single pass, then
        // emit the identical sorted list from the set bits. Falls back to the
        // heap-marking implementation the moment any wire is out of range.
        let mut words = [0u64; 16];
        for &[t, a, b] in &self.gates {
            if t >= 1024 || a >= 1024 || b >= 1024 {
                return self.used_wires_heap();
            }
            words[(t >> 6) as usize] |= 1u64 << (t & 63);
            words[(a >> 6) as usize] |= 1u64 << (a & 63);
            words[(b >> 6) as usize] |= 1u64 << (b & 63);
        }
        let count: usize = words.iter().map(|w| w.count_ones() as usize).sum();
        let mut out = Vec::with_capacity(count);
        for (wi, &word) in words.iter().enumerate() {
            let base = (wi as u16) << 6;
            let mut word = word;
            while word != 0 {
                out.push(base + word.trailing_zeros() as u16);
                word &= word - 1;
            }
        }
        out
    }

    // Heap fallback for circuits touching wires >= 1024 (u16 wires cap at
    // 65535). Identical to the historical implementation.
    fn used_wires_heap(&self) -> Vec<u16> {
        let Some(max_wire) = self.gates.iter().flatten().copied().max() else {
            return Vec::new();
        };
        let mut used = vec![false; max_wire as usize + 1];
        for &[target, control_a, control_b] in &self.gates {
            used[target as usize] = true;
            used[control_a as usize] = true;
            used[control_b as usize] = true;
        }
        used.into_iter()
            .enumerate()
            .filter_map(|(wire, is_used)| is_used.then_some(wire as u16))
            .collect()
    }

    /// Number of distinct wires touched by the circuit, without materializing
    /// the sorted wire list. Equals `self.used_wires().len()`.
    pub fn used_wires_len(&self) -> usize {
        let mut words = [0u64; 16];
        for &[t, a, b] in &self.gates {
            if t >= 1024 || a >= 1024 || b >= 1024 {
                return self.used_wires_heap().len();
            }
            words[(t >> 6) as usize] |= 1u64 << (t & 63);
            words[(a >> 6) as usize] |= 1u64 << (a & 63);
            words[(b >> 6) as usize] |= 1u64 << (b & 63);
        }
        words.iter().map(|w| w.count_ones() as usize).sum()
    }

    // "Bottom" function for gates
    pub fn max_wire(&self) -> usize {
        self.gates.iter().flatten().copied().max().unwrap_or(0) as usize
    }

    // Undo rewiring. Note: Recall that the number of wires in CircuitSeq is not stored
    pub fn unrewire_subcircuit(subcircuit: &CircuitSeq, used_wires: &[u16]) -> CircuitSeq {
        // Replace wires in each gate with original wires
        let new_gates: Vec<[u16; 3]> = subcircuit
            .gates
            .iter()
            .map(|&[t, c1, c2]| {
                [
                    used_wires[t as usize],
                    used_wires[c1 as usize],
                    used_wires[c2 as usize],
                ]
            })
            .collect();

        CircuitSeq { gates: new_gates }
    }

    pub fn evaluate_evolution_1024(&self, input: U1024) -> Vec<U1024> {
        let mut state = input;
        let mut evolution = Vec::with_capacity(self.gates.len() + 1);
        evolution.push(state);

        for gate in &self.gates {
            state = Gate::evaluate_index_1024(state, *gate);
            evolution.push(state);
        }

        evolution
    }

    // Probablistic check on circuit equality
    pub fn probably_equal(
        &self,
        other_circuit: &Self,
        num_wires: usize,
        num_inputs: usize,
    ) -> Result<(), String> {
        use rayon::prelude::*;

        // Arithmetic width must cover every wire either circuit TOUCHES, not
        // just the num_wires input/compare contract: primitive_types shifts
        // >= the type width silently return 0, so evaluating a circuit wider
        // than the chosen type corrupts every access to the high wires (e.g.
        // a 512-wire gadgetized circuit checked against its 256-wire source
        // was evaluated in u256 and reported non-equivalent). The num_wires
        // mask below is unchanged: inputs are drawn on num_wires bits and
        // outputs compared on num_wires bits.
        let eval_wires = num_wires
            .max(self.max_wire() + 1)
            .max(other_circuit.max_wire() + 1);

        if eval_wires > 1024 {
            // Retained from the fixed-width implementation. The bit-sliced
            // kernel below has no width ceiling of its own, so this is now an
            // artificial limit rather than a representational one.
            return Err("probabilistic equality supports up to 1024 wires".to_string());
        }
        if num_inputs == 0 {
            return Ok(());
        }

        // Bit-sliced: one walk per 64 inputs instead of one walk per input.
        // The old code drew a single wide word per sample and re-walked both
        // gate lists for each, so an m-gate check on k inputs streamed 2*m*k
        // gates; transposing to one word per wire makes that 2*m*ceil(k/64)
        // for the same per-gate cost. `num_inputs` therefore rounds up to a
        // multiple of 64 — the extra samples are free and only sharpen the
        // test.
        //
        // Input/compare contract is unchanged: samples are drawn on the low
        // `num_wires` wires (everything above starts at zero, which is what
        // masking the old wide input to `num_wires` bits did) and only those
        // wires are compared.
        let len = lane_state_len(eval_wires);
        let batches = num_inputs.div_ceil(64);

        (0..batches).into_par_iter().try_for_each(|_| {
            let mut rng = rand::rng();
            let mut mine = vec![0u64; len];
            for lane in mine[..num_wires].iter_mut() {
                *lane = rng.next_u64();
            }
            let mut theirs = mine.clone();

            Gate::eval_lanes_index_list(&self.gates, &mut mine);
            Gate::eval_lanes_index_list(&other_circuit.gates, &mut theirs);

            if mine[..num_wires] != theirs[..num_wires] {
                Err("Circuits are not equal".to_string())
            } else {
                Ok(())
            }
        })
    }

    pub fn to_polynomial(&self, n: usize, start: usize, end: usize) -> Vec<Polynomial> {
        let gates = &self.gates[start..end];
        // Wire i starts as degree 1 monomial
        let mut polys: Vec<Polynomial> = (0..n).map(|i| vec![1u64 << i]).collect();

        // Ping-pong scratch buffers reused across the gate loop: `term`
        // holds the AND-NOT product, `merged` receives the XOR merge and is
        // then swapped into polys[a], recycling the displaced allocation.
        let mut term: Vec<Monomial> = Vec::new();
        let mut merged: Vec<Monomial> = Vec::new();

        for &[a, b, c] in gates {
            // evaluate() toggles on b OR NOT(c), which is 1 + c*NOT(b) over GF(2).
            poly_and_not_into(&polys[c as usize], &polys[b as usize], &mut term);
            poly_xor_merge_into(&polys[a as usize], &term, &mut merged);
            std::mem::swap(&mut polys[a as usize], &mut merged);
            toggle_monomial(&mut polys[a as usize], 0u64);
        }

        // XOR each wire with its initial value x_i so unchanged wires become 0
        // for i in 0..n {
        //     let xi = vec![1u64 << i];
        //     polys[i] = poly_xor(polys[i].clone(), xi);
        // }

        polys
    }

    /// Like `to_polynomial`, but returns `None` when polynomial growth exceeds
    /// `cap`. The early product guard avoids allocating an intermediate AND
    /// whose raw cross product is already far beyond the useful limit.
    pub fn to_polynomial_capped(
        &self,
        n: usize,
        start: usize,
        end: usize,
        cap: usize,
    ) -> Option<Vec<Polynomial>> {
        let gates = &self.gates[start..end];
        let mut polys: Vec<Polynomial> = (0..n).map(|i| vec![1u64 << i]).collect();

        // Same reused scratch buffers as `to_polynomial`; the budget checks
        // below are unchanged and see identical lengths.
        let mut term: Vec<Monomial> = Vec::new();
        let mut merged: Vec<Monomial> = Vec::new();

        for &[a, b, c] in gates {
            if polys[b as usize]
                .len()
                .saturating_mul(polys[c as usize].len())
                > cap.saturating_mul(16)
            {
                return None;
            }

            // Keep the executor's g57 convention: a += NOT(b)*c + 1.
            poly_and_not_into(&polys[c as usize], &polys[b as usize], &mut term);
            poly_xor_merge_into(&polys[a as usize], &term, &mut merged);
            std::mem::swap(&mut polys[a as usize], &mut merged);
            toggle_monomial(&mut polys[a as usize], 0u64);
            if polys[a as usize].len() > cap {
                return None;
            }
        }

        Some(polys)
    }

    /// Compute canonical polynomials for one direction only (forward or reversed).
    /// Returns (canonical_polys, final_order, used_wires).
    /// Used by frozen compression to try forward first, then reverse on miss.
    pub fn canonicalize_polys_single(
        &self,
        reversed: bool,
    ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
        let used = self.used_wires();
        match self.canonicalize_polys_single_inner(reversed, &used) {
            CanonSingleInner::Skip => (Vec::new(), Permutation { data: Vec::new() }, used),
            CanonSingleInner::Cached(entry) => (
                entry.polys.clone(),
                Permutation {
                    data: entry.order.clone(),
                },
                used,
            ),
            CanonSingleInner::Fresh(polys, order, _) => (polys, Permutation { data: order }, used),
        }
    }

    /// Like `canonicalize_polys_single`, but returns the frozen-DB lookup key
    /// (`xxh3_128(polys_repr_blob(polys)).to_le_bytes()`) instead of the
    /// canonical polynomials, so cache hits skip the deep clone of the
    /// polynomial vector and the re-serialize/re-hash on the caller side.
    /// `None` corresponds exactly to the empty-polys skip outcome of
    /// `canonicalize_polys_single` (oversized window, monomial-cap skip, or
    /// Rule-L budget skip).
    pub fn canonicalize_polys_single_hashed(
        &self,
        reversed: bool,
    ) -> (Option<[u8; 16]>, Permutation, Vec<u16>) {
        let used = self.used_wires();
        match self.canonicalize_polys_single_inner(reversed, &used) {
            CanonSingleInner::Skip => (None, Permutation { data: Vec::new() }, used),
            CanonSingleInner::Cached(entry) => (
                Some(entry.polys_key),
                Permutation {
                    data: entry.order.clone(),
                },
                used,
            ),
            CanonSingleInner::Fresh(polys, order, key) => {
                let key = key.unwrap_or_else(|| {
                    xxhash_rust::xxh3::xxh3_128(&polys_repr_blob(&polys)).to_le_bytes()
                });
                (Some(key), Permutation { data: order }, used)
            }
        }
    }

    fn canonicalize_polys_single_inner(&self, reversed: bool, used: &[u16]) -> CanonSingleInner {
        // A u64 monomial cannot distinguish x_64 from lower variables. Treat
        // oversized lookup windows as clean misses rather than constructing an
        // overflow-aliased key.
        if used.len() > 64 {
            OVERSIZED_CANON_SKIPS.fetch_add(1, Ordering::Relaxed);
            return CanonSingleInner::Skip;
        }
        let wire_map = dense_wire_map(used);
        let mut c = CircuitSeq {
            gates: self
                .gates
                .iter()
                .map(|&[t, c1, c2]| {
                    [
                        wire_map[t as usize],
                        wire_map[c1 as usize],
                        wire_map[c2 as usize],
                    ]
                })
                .collect(),
        };
        if reversed {
            c.gates.reverse();
        }
        c.canonicalize();

        // Canonicalization is pure and windows repeat heavily in the
        // compress/expand games, so an exact process-wide cache keyed on the
        // dense-remapped, gate-canonicalized window is a straight win. The
        // reversed direction needs no key flag: it produces a different gate
        // sequence (and when it doesn't, the results coincide anyway).
        let cache = canon_cache();
        let cache_key: Option<Box<[u16]>> = cache.map(|_| {
            let mut key = Vec::with_capacity(c.gates.len() * 3);
            for g in &c.gates {
                key.extend_from_slice(g);
            }
            key.into_boxed_slice()
        });
        if let (Some(cache), Some(key)) = (cache, cache_key.as_ref()) {
            CANON_CACHE_QUERIES.fetch_add(1, Ordering::Relaxed);
            if let Some(entry) = cache.get(key) {
                CANON_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                return CanonSingleInner::Cached(std::sync::Arc::clone(entry.value()));
            }
        }

        let n = c.max_wire() as usize + 1;
        let polys = match canon_monomial_cap() {
            Some(cap) => match c.to_polynomial_capped(n, 0, c.gates.len(), cap) {
                Some(polys) => polys,
                None => {
                    CANON_CAP_SKIPS.fetch_add(1, Ordering::Relaxed);
                    return CanonSingleInner::Skip;
                }
            },
            None => c.to_polynomial(n, 0, c.gates.len()),
        };

        let bench_polys = if bench_canon_enabled() {
            Some(polys.clone())
        } else {
            None
        };

        let t4 = Instant::now();
        let canon = match canonicalize_polys_4(polys, true) {
            Ok(canon) => canon,
            Err(()) => {
                CANON_RULE_L_SKIPS.fetch_add(1, Ordering::Relaxed);
                return CanonSingleInner::Skip;
            }
        };
        let canon_elapsed = t4.elapsed();
        CANON4_CORE_TIME.fetch_add(canon_elapsed.as_nanos() as u64, Ordering::Relaxed);
        if compression_trace_enabled()
            && canon_elapsed.as_millis() >= compression_trace_threshold_ms()
        {
            eprintln!(
                "[compress-trace] slow canonicalize direction={} gates={} used_wires={} elapsed_ms={}",
                if reversed { "reverse" } else { "forward" },
                self.gates.len(),
                used.len(),
                canon_elapsed.as_millis()
            );
        }

        if let Some(polys) = bench_polys {
            let tp = Instant::now();
            let perm = crate::experimental::poly_canon_graph::canonicalize_graph(&polys, n);
            let _form = crate::experimental::poly_canon_graph::canonical_form(&polys, &perm);
            POLYCANON_CORE_TIME.fetch_add(tp.elapsed().as_nanos() as u64, Ordering::Relaxed);
            CANON_BENCH_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        // All cap exits return above, so the exact cache can contain only
        // complete, valid canonical keys—never a clean-miss sentinel.
        if let (Some(cache), Some(key)) = (cache, cache_key) {
            let polys_key = xxhash_rust::xxh3::xxh3_128(&polys_repr_blob(&canon.0)).to_le_bytes();
            let entry_bytes = (96
                + key.len() * 2
                + canon.0.iter().map(|p| 24 + p.len() * 8).sum::<usize>()
                + canon.1.data.len() * 8) as u64;
            if CANON_CACHE_BYTES.fetch_add(entry_bytes, Ordering::Relaxed) + entry_bytes
                > canon_cache_cap_bytes()
            {
                // Wholesale epoch reset, same policy as the frozen lookup cache.
                cache.clear();
                CANON_CACHE_BYTES.store(entry_bytes, Ordering::Relaxed);
            }
            cache.insert(
                key,
                std::sync::Arc::new(CanonCacheEntry {
                    polys: canon.0.clone(),
                    order: canon.1.data.clone(),
                    polys_key,
                }),
            );
            return CanonSingleInner::Fresh(canon.0, canon.1.data, Some(polys_key));
        }

        CanonSingleInner::Fresh(canon.0, canon.1.data, None)
    }

    /// Like `canonicalize_polys_single`, but absorbs pending NOTs on input wires by substituting
    /// x_w -> x_w + 1 in the polynomial form before canonicalization. This supports Stage-F-style
    /// curated lookups where the replacement consumes the pending NOT instead of materializing a
    /// standalone NOT gadget.
    pub fn canonicalize_polys_single_neg(
        &self,
        negated_inputs: &[u16],
    ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
        let used = self.used_wires();
        if used.len() > 64 {
            OVERSIZED_CANON_SKIPS.fetch_add(1, Ordering::Relaxed);
            return (Vec::new(), Permutation { data: Vec::new() }, used);
        }
        let wire_map = dense_wire_map(&used);
        let mut c = CircuitSeq {
            gates: self
                .gates
                .iter()
                .map(|&[t, c1, c2]| {
                    [
                        wire_map[t as usize],
                        wire_map[c1 as usize],
                        wire_map[c2 as usize],
                    ]
                })
                .collect(),
        };
        c.canonicalize();

        // Sorted mapped negation multiset. Substitutions of distinct input
        // variables commute and a repeated variable is an involution applied
        // twice, so the sorted list fully determines the outcome and doubles
        // as the cache-key component.
        let mut mapped_negs: Vec<u16> = negated_inputs
            .iter()
            .filter_map(|&w| match wire_map.get(w as usize) {
                Some(&m) if m != u16::MAX => Some(m),
                _ => None,
            })
            .collect();
        mapped_negs.sort_unstable();

        // Same exact result-cache treatment as the plain path. The leading
        // u16::MAX tag namespaces neg entries away from plain window keys
        // (whose first element is a dense wire index < 64, given the
        // used.len() <= 64 guard above), and the negation-count element keeps
        // the encoding prefix-unambiguous.
        let cache = canon_cache();
        let cache_key: Option<Box<[u16]>> = cache.map(|_| {
            let mut key = Vec::with_capacity(2 + mapped_negs.len() + c.gates.len() * 3);
            key.push(u16::MAX);
            key.push(mapped_negs.len() as u16);
            key.extend_from_slice(&mapped_negs);
            for g in &c.gates {
                key.extend_from_slice(g);
            }
            key.into_boxed_slice()
        });
        if let (Some(cache), Some(key)) = (cache, cache_key.as_ref()) {
            CANON_CACHE_QUERIES.fetch_add(1, Ordering::Relaxed);
            if let Some(entry) = cache.get(key) {
                CANON_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                return (
                    entry.polys.clone(),
                    Permutation {
                        data: entry.order.clone(),
                    },
                    used,
                );
            }
        }

        let n = c.max_wire() as usize + 1;
        let mut polys = match canon_monomial_cap() {
            Some(cap) => match c.to_polynomial_capped(n, 0, c.gates.len(), cap) {
                Some(polys) => polys,
                None => {
                    CANON_CAP_SKIPS.fetch_add(1, Ordering::Relaxed);
                    return (Vec::new(), Permutation { data: Vec::new() }, used);
                }
            },
            None => c.to_polynomial(n, 0, c.gates.len()),
        };
        for &mapped in &mapped_negs {
            let mapped = mapped as usize;
            if mapped >= 64 {
                return (Vec::new(), Permutation { data: Vec::new() }, used);
            }
            for p in polys.iter_mut() {
                substitute_input_negation(p, mapped);
            }
        }

        let canon = match canonicalize_polys_4(polys, true) {
            Ok(canon) => canon,
            Err(()) => {
                CANON_RULE_L_SKIPS.fetch_add(1, Ordering::Relaxed);
                return (Vec::new(), Permutation { data: Vec::new() }, used);
            }
        };

        // Only complete successes are inserted, mirroring the plain path: a
        // hit therefore always replays a deterministic success.
        if let (Some(cache), Some(key)) = (cache, cache_key) {
            let polys_key = xxhash_rust::xxh3::xxh3_128(&polys_repr_blob(&canon.0)).to_le_bytes();
            let entry_bytes = (96
                + key.len() * 2
                + canon.0.iter().map(|p| 24 + p.len() * 8).sum::<usize>()
                + canon.1.data.len() * 8) as u64;
            if CANON_CACHE_BYTES.fetch_add(entry_bytes, Ordering::Relaxed) + entry_bytes
                > canon_cache_cap_bytes()
            {
                cache.clear();
                CANON_CACHE_BYTES.store(entry_bytes, Ordering::Relaxed);
            }
            cache.insert(
                key,
                std::sync::Arc::new(CanonCacheEntry {
                    polys: canon.0.clone(),
                    order: canon.1.data.clone(),
                    polys_key,
                }),
            );
        }
        (canon.0, canon.1, used)
    }
}

/// Dense old-wire -> compact-index map for a sorted `used_wires` list.
/// Entries for unused wires are `u16::MAX`; used wires map to their position.
/// Replaces per-call `HashMap<u16, u16>` construction on canonicalization hot paths.
fn dense_wire_map(used: &[u16]) -> Vec<u16> {
    let len = used.last().map_or(0, |&w| w as usize + 1);
    let mut map = vec![u16::MAX; len];
    for (i, &w) in used.iter().enumerate() {
        map[w as usize] = i as u16;
    }
    map
}

pub fn polynomial_from_terms<I>(terms: I) -> Polynomial
where
    I: IntoIterator<Item = Monomial>,
{
    let mut terms: Vec<Monomial> = terms.into_iter().collect();
    normalize_polynomial(&mut terms);
    terms
}

pub fn normalize_polynomial(poly: &mut Polynomial) {
    poly.sort_unstable();

    let mut write = 0usize;
    let mut read = 0usize;
    while read < poly.len() {
        let m = poly[read];
        let mut count = 1usize;
        read += 1;
        while read < poly.len() && poly[read] == m {
            count += 1;
            read += 1;
        }
        if count % 2 == 1 {
            poly[write] = m;
            write += 1;
        }
    }
    poly.truncate(write);
}

pub fn substitute_input_negation(poly: &mut Polynomial, w: usize) {
    // Substituting x_w -> x_w + 1 toggles, for every monomial containing x_w,
    // the same monomial with x_w cleared. The rests are pairwise distinct
    // (rest | bit reconstructs its unique source), so the sequential
    // binary-search toggles the old implementation performed are exactly one
    // sorted symmetric-difference merge.
    let bit = 1u64 << w;
    let mut rests: Vec<Monomial> = poly
        .iter()
        .filter(|&&m| m & bit != 0)
        .map(|&m| m & !bit)
        .collect();
    if rests.is_empty() {
        return;
    }
    // Rests inherit the poly's sort order (the cleared bit is common to every
    // source monomial); sort defensively anyway — it is a no-op then.
    rests.sort_unstable();
    poly_xor_assign(poly, rests);
}

fn toggle_monomial(poly: &mut Polynomial, m: Monomial) {
    match poly.binary_search(&m) {
        Ok(pos) => {
            poly.remove(pos);
        }
        Err(pos) => {
            poly.insert(pos, m);
        }
    }
}

fn poly_xor_assign(poly: &mut Polynomial, terms: Polynomial) {
    let old = std::mem::take(poly);
    let mut merged = Vec::with_capacity(old.len().max(terms.len()));
    let mut i = 0usize;
    let mut j = 0usize;

    while i < old.len() && j < terms.len() {
        match old[i].cmp(&terms[j]) {
            CmpOrdering::Less => {
                merged.push(old[i]);
                i += 1;
            }
            CmpOrdering::Greater => {
                merged.push(terms[j]);
                j += 1;
            }
            CmpOrdering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    merged.extend_from_slice(&old[i..]);
    merged.extend_from_slice(&terms[j..]);
    *poly = merged;
}

#[cfg_attr(not(test), allow(dead_code))]
fn poly_and_not(poly_1: &Polynomial, poly_2: &Polynomial) -> Polynomial {
    let mut terms = Vec::with_capacity(poly_1.len() * (poly_2.len() + 1));
    for &m1 in poly_1 {
        terms.push(m1);
        for &m2 in poly_2 {
            terms.push(m1 | m2);
        }
    }
    polynomial_from_terms(terms)
}

/// Allocation-reusing form of `poly_and_not`: same raw term stream and the
/// same sort+cancel normalization, written into a caller-owned scratch vec.
fn poly_and_not_into(poly_1: &[Monomial], poly_2: &[Monomial], out: &mut Vec<Monomial>) {
    out.clear();
    out.reserve(poly_1.len() * (poly_2.len() + 1));
    for &m1 in poly_1 {
        out.push(m1);
        for &m2 in poly_2 {
            out.push(m1 | m2);
        }
    }
    normalize_polynomial(out);
}

/// Allocation-reusing form of `poly_xor_assign`: merges the sorted symmetric
/// difference of `a` and `b` into a caller-owned scratch vec.
fn poly_xor_merge_into(a: &[Monomial], b: &[Monomial], out: &mut Vec<Monomial>) {
    out.clear();
    let mut i = 0usize;
    let mut j = 0usize;
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            CmpOrdering::Less => {
                out.push(a[i]);
                i += 1;
            }
            CmpOrdering::Greater => {
                out.push(b[j]);
                j += 1;
            }
            CmpOrdering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
}

/// Cancel adjacent duplicate gates to a fixed point, keeping `tags` (when
/// present) in lockstep with the gate vector. Stack-style single pass:
/// produces the identical final sequence to the historical
/// drain-with-backtrack loop (`aa -> empty` rewriting is confluent, and the
/// backtrack re-examined exactly the stack top) without the O(n) tail
/// memmove per removal.
pub fn cancel_adjacent_duplicates<T: Copy>(
    gates: &mut Vec<[u16; 3]>,
    mut tags: Option<&mut Vec<T>>,
) {
    let mut write = 0usize;
    for read in 0..gates.len() {
        if write > 0 && gates[write - 1] == gates[read] {
            write -= 1;
        } else {
            gates[write] = gates[read];
            if let Some(tags) = tags.as_deref_mut() {
                tags[write] = tags[read];
            }
            write += 1;
        }
    }
    gates.truncate(write);
    if let Some(tags) = tags {
        tags.truncate(write);
    }
}

// Display polynomials

pub fn polys_repr_blob(polys: &Vec<Polynomial>) -> Vec<u8> {
    let total: usize = polys.iter().map(|p| p.len() + 1).sum();
    let mut bytes = Vec::with_capacity(total * 8);
    let mut scratch: Vec<u64> = Vec::new();
    for poly in polys {
        // Canonical polys are already monomial-sorted; only re-sort when a
        // caller hands us an unsorted polynomial.
        if poly.is_sorted() {
            for m in poly {
                bytes.extend_from_slice(&m.to_le_bytes());
            }
        } else {
            scratch.clear();
            scratch.extend_from_slice(poly);
            scratch.sort_unstable();
            for m in &scratch {
                bytes.extend_from_slice(&m.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // separator
    }
    bytes
}

/// Initial ranking method
/// Degree counts of a polynomial: [count_of_max_possible_deg, ..., count_of_deg_0]
/// Padded to max_possible_degree+1 entries so Vec comparison is always over equal-length
/// vectors and correctly ranks e.g. one degree-2 monomial above two degree-1 monomials.
fn degree_counts(poly: &Polynomial, max_possible_degree: usize) -> Vec<usize> {
    let mut counts = vec![0usize; max_possible_degree + 1];
    for m in poly {
        let deg = m.count_ones() as usize;
        counts[deg] += 1;
    }
    counts.reverse();
    counts
}

// Static accumulators for time spent in each rule (in nanoseconds)
static TIME_RULE_2_1: AtomicU64 = AtomicU64::new(0);
static TIME_RULE_2_2: AtomicU64 = AtomicU64::new(0);
static TIME_RULE_2_4: AtomicU64 = AtomicU64::new(0);
static TIME_RULE_2_5: AtomicU64 = AtomicU64::new(0);
static TIME_RULE_L: AtomicU64 = AtomicU64::new(0);

// Given two orderings that produce the same canonical form,
// build the automorphism: sigma[order_a[pos]] = order_b[pos]
fn automorphism_from_orders(order_a: &[usize], order_b: &[usize], n: usize) -> Vec<usize> {
    let mut sigma = vec![0usize; n];
    for pos in 0..order_a.len() {
        sigma[order_a[pos]] = order_b[pos];
    }
    sigma
}

/// After canonicalization, trim trailing polynomials that are uninformative.
/// Starting from the last polynomial, remove P_i if both conditions hold:
///   1. P_i is trivial — its only monomial is the single variable x_i (bitmask 1 << i)
///   2. x_i does not appear in any other polynomial in the full list
/// Stop as soon as we reach a P_i that is non-trivial OR whose variable x_i
/// appears in some other polynomial. Keep everything from that point forward.
///
/// Returns the trimmed polynomial list. The permutation is left unchanged.
pub fn trim_canonicalized(polynomials: Vec<Polynomial>) -> Vec<Polynomial> {
    let n = polynomials.len();
    let mut keep_up_to = n; // exclusive upper bound — trim everything at or after this

    for i in (0..n).rev() {
        let bit = 1u64 << i;

        // Check if P_i is trivial: exactly one monomial which is just x_i
        let is_trivial =
            polynomials[i].len() == 1 && polynomials[i].iter().next().copied().unwrap() == bit;

        if !is_trivial {
            // Non-trivial polynomial — stop trimming here
            break;
        }

        // Check if x_i appears in any other polynomial (including higher degree monomials)
        let used_elsewhere = polynomials
            .iter()
            .enumerate()
            .any(|(j, poly)| j != i && poly.iter().any(|&m| m & bit != 0));

        if used_elsewhere {
            // x_i is referenced by another polynomial — stop trimming here
            break;
        }

        // P_i is trivial and x_i is unused elsewhere — trim it
        keep_up_to = i;
    }

    polynomials[..keep_up_to].to_vec()
}

const MONOMIAL_RANK_KEY_LEN_4: usize = 65;

const MONOMIAL_RANK_PREFIX_LEN_4: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MonomialRankKey4 {
    degree: u8,
    /// Big-endian packing of `encoded_ranks[..16]`. Comparing prefixes as
    /// integers equals comparing the first 16 bytes lexicographically, so the
    /// full 65-byte compare only runs on prefix ties (rare: it requires two
    /// monomials agreeing on their 16 highest-priority rank slots).
    prefix: u128,
    encoded_ranks: [u8; MONOMIAL_RANK_KEY_LEN_4],
}

impl Ord for MonomialRankKey4 {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        // Identical ordering to `encoded_ranks.cmp(&other.encoded_ranks)`:
        // lexicographic byte compare == big-endian integer compare on the
        // packed prefix, and the tail settles prefix ties.
        self.prefix.cmp(&other.prefix).then_with(|| {
            self.encoded_ranks[MONOMIAL_RANK_PREFIX_LEN_4..]
                .cmp(&other.encoded_ranks[MONOMIAL_RANK_PREFIX_LEN_4..])
        })
    }
}

impl PartialOrd for MonomialRankKey4 {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MonomialLevelKey4 {
    rank_key: MonomialRankKey4,
    coeff: usize,
}

type LevelEntry4 = (Monomial, usize, MonomialLevelKey4);

fn monomial_rank_key_4(m: Monomial, vr: &[usize], _n: usize) -> MonomialRankKey4 {
    let mut encoded_ranks = [0u8; MONOMIAL_RANK_KEY_LEN_4];
    let mut degree = 0usize;
    let mut mm = m;
    while mm != 0 {
        let v = mm.trailing_zeros() as usize;
        debug_assert!(vr[v] < u8::MAX as usize);
        encoded_ranks[degree] = (vr[v] + 1) as u8;
        degree += 1;
        mm &= mm - 1;
    }
    encoded_ranks[..degree].sort_unstable();
    let prefix = u128::from_be_bytes(
        encoded_ranks[..MONOMIAL_RANK_PREFIX_LEN_4]
            .try_into()
            .unwrap(),
    );
    MonomialRankKey4 {
        degree: degree as u8,
        prefix,
        encoded_ranks,
    }
}

fn monomial_level_key_4(m: Monomial, coeff: usize, vr: &[usize], n: usize) -> MonomialLevelKey4 {
    MonomialLevelKey4 {
        rank_key: monomial_rank_key_4(m, vr, n),
        coeff,
    }
}

fn cmp_level_key_4(a: &MonomialLevelKey4, b: &MonomialLevelKey4) -> CmpOrdering {
    b.rank_key
        .degree
        .cmp(&a.rank_key.degree)
        .then_with(|| a.rank_key.cmp(&b.rank_key))
        .then_with(|| b.coeff.cmp(&a.coeff))
}

// ── Compact level entries ────────────────────────────────────────────────────
// Valid iff every monomial degree <= MONOMIAL_RANK_PREFIX_LEN_4 (16): the fat
// key's 65-byte tail is then all zeros for every entry, so the full comparison
// collapses to (degree desc, prefix asc, coeff desc) — the same total order
// and the same equalities as the fat key. 32 bytes instead of ~128: 4x less
// build traffic, 4x smaller sort moves, and a 21-byte equality compare in the
// level walk. `canonicalize_polys_4` checks the degree bound once per top
// call (`compact_ok`); class polys and D-class polys reuse the same
// monomials, so that one check covers every scan in the recursion. Measured
// on the DB-armed fmix recipe the bound always holds; the fat path stays as
// the deg>16 fallback.

#[derive(Clone, Copy)]
struct LevelEntryC {
    prefix: u128,
    m: Monomial,
    coeff: u32,
    degree: u8,
}

// Shared rank packing for the compact paths: (degree, big-endian prefix of
// the +1-encoded, ascending-sorted rank bytes). Identical to
// monomial_rank_key_4 restricted to the first 16 slots; callers must uphold
// degree <= 16 (`compact_ok`).
#[inline]
fn rank_prefix_c(m: Monomial, vr: &[usize]) -> (u8, u128) {
    let mut ranks = [0u8; MONOMIAL_RANK_PREFIX_LEN_4];
    let mut degree = 0usize;
    let mut mm = m;
    while mm != 0 {
        let v = mm.trailing_zeros() as usize;
        debug_assert!(vr[v] < u8::MAX as usize);
        ranks[degree] = (vr[v] + 1) as u8;
        degree += 1;
        mm &= mm - 1;
    }
    ranks[..degree].sort_unstable();
    (degree as u8, u128::from_be_bytes(ranks))
}

#[inline]
fn level_entry_c(m: Monomial, coeff: usize, vr: &[usize]) -> LevelEntryC {
    let (degree, prefix) = rank_prefix_c(m, vr);
    LevelEntryC {
        prefix,
        m,
        // Class-poly coefficients count contributing wires, so they never
        // exceed n <= 64.
        coeff: coeff as u32,
        degree,
    }
}

#[inline]
fn cmp_level_c(a: &LevelEntryC, b: &LevelEntryC) -> CmpOrdering {
    b.degree
        .cmp(&a.degree)
        .then_with(|| a.prefix.cmp(&b.prefix))
        .then_with(|| b.coeff.cmp(&a.coeff))
}

#[inline]
fn eq_level_c(a: &LevelEntryC, b: &LevelEntryC) -> bool {
    a.degree == b.degree && a.prefix == b.prefix && a.coeff == b.coeff
}

fn sorted_level_entries_c(cp: &[(Monomial, usize)], vr: &[usize], entries: &mut Vec<LevelEntryC>) {
    entries.clear();
    entries.extend(cp.iter().map(|&(m, c)| level_entry_c(m, c, vr)));
    entries.sort_by(cmp_level_c);
}

fn sorted_level_entries_4(
    cp: &[(Monomial, usize)],
    vr: &[usize],
    n: usize,
    entries: &mut Vec<LevelEntry4>,
) {
    entries.clear();
    entries.extend(
        cp.iter()
            .map(|&(m, c)| (m, c, monomial_level_key_4(m, c, vr, n))),
    );
    entries.sort_by(|a, b| cmp_level_key_4(&a.2, &b.2));
}

// Count how many monomials in a level each wire appears in.
fn wire_freq_4(level: &[LevelEntry4], n: usize, freq: &mut Vec<usize>) {
    freq.resize(n, 0);
    freq.fill(0);
    for &(m, _, _) in level {
        let mut mm = m;
        while mm != 0 {
            freq[mm.trailing_zeros() as usize] += 1;
            mm &= mm - 1;
        }
    }
}

// Freq counts restricted to tied wires — the only entries split decisions
// read (untied wires never belong to a multi-member rank group). Masking
// before the bit walk skips most of the per-monomial popcount work.
fn wire_freq_tied_4(level: &[LevelEntry4], tied_mask: u64, n: usize, freq: &mut Vec<usize>) {
    freq.resize(n, 0);
    freq.fill(0);
    for &(m, _, _) in level {
        let mut mm = m & tied_mask;
        while mm != 0 {
            freq[mm.trailing_zeros() as usize] += 1;
            mm &= mm - 1;
        }
    }
}

fn wire_freq_c(level: &[LevelEntryC], n: usize, freq: &mut Vec<usize>) {
    freq.resize(n, 0);
    freq.fill(0);
    for e in level {
        let mut mm = e.m;
        while mm != 0 {
            freq[mm.trailing_zeros() as usize] += 1;
            mm &= mm - 1;
        }
    }
}

fn wire_freq_tied_c(level: &[LevelEntryC], tied_mask: u64, n: usize, freq: &mut Vec<usize>) {
    freq.resize(n, 0);
    freq.fill(0);
    for e in level {
        let mut mm = e.m & tied_mask;
        while mm != 0 {
            freq[mm.trailing_zeros() as usize] += 1;
            mm &= mm - 1;
        }
    }
}

// Split the FIRST (highest-priority) tied wire group whose members have different
// frequencies. Higher frequency → higher priority (lower rank number).
// Returns the split group's wire bitmask (None = no split) so the caller can
// track which class polys the split can possibly affect (cleanskip).
fn split_by_freq_4(
    vr: &mut Vec<usize>,
    n: usize,
    freq: &[usize],
    tied: &mut Vec<usize>,
    sorted: &mut Vec<usize>,
    sub_ranks: &mut Vec<usize>,
) -> Option<u64> {
    let max_rank = *vr.iter().max().unwrap_or(&0);
    for cur_rank in 0..=max_rank {
        tied.clear();
        tied.extend((0..n).filter(|&v| vr[v] == cur_rank));
        if tied.len() <= 1 {
            continue;
        }
        let first_freq = freq[tied[0]];
        if tied.iter().all(|&v| freq[v] == first_freq) {
            continue;
        }

        sorted.clear();
        sorted.extend_from_slice(tied);
        sorted.sort_by(|&a, &b| freq[b].cmp(&freq[a]));

        let mut sub_rank = 0usize;
        sub_ranks.clear();
        sub_ranks.resize(sorted.len(), 0);
        for i in 1..sorted.len() {
            if freq[sorted[i]] != freq[sorted[i - 1]] {
                sub_rank += 1;
            }
            sub_ranks[i] = sub_rank;
        }
        for v in 0..n {
            if vr[v] > cur_rank {
                vr[v] += sub_rank;
            }
        }
        for (i, &v) in sorted.iter().enumerate() {
            vr[v] = cur_rank + sub_ranks[i];
        }
        let group_mask = if n <= 64 {
            sorted.iter().fold(0u64, |acc, &v| acc | (1u64 << v))
        } else {
            // Degenerate width: report "could affect anything" so cleanskip
            // conservatively clears every flag.
            u64::MAX
        };
        return Some(group_mask);
    }
    None
}

// Tied groups precomputed once per master iteration (vr is constant between
// splits), stored flat to avoid allocs. groups_meta holds
// (rank, member bitmask, members start, members end) in ascending rank order,
// multi-member groups only. n <= 64 callers only.
fn tied_groups_4(
    vr: &[usize],
    n: usize,
    groups_meta: &mut Vec<(usize, u64, usize, usize)>,
    groups_members: &mut Vec<usize>,
) {
    groups_meta.clear();
    groups_members.clear();
    debug_assert!(n <= 64);
    let mut count = [0u8; 64];
    for &r in vr {
        count[r] += 1;
    }
    for r in 0..n {
        if count[r] > 1 {
            let start = groups_members.len();
            let mut mask = 0u64;
            for v in 0..n {
                if vr[v] == r {
                    groups_members.push(v);
                    mask |= 1u64 << v;
                }
            }
            groups_meta.push((r, mask, start, groups_members.len()));
        }
    }
}

// split_by_freq_4 with the rank iteration replaced by the precomputed group
// list. Identical selection: groups visited in ascending rank order; a group
// whose mask misses the level union has all-zero freqs (the old code also
// skipped it as all-equal); the stable freq-desc sort sees members in
// ascending wire id exactly like the old (0..n).filter build. Returns the
// split group's mask.
#[allow(clippy::too_many_arguments)]
fn split_by_freq_groups_4(
    vr: &mut Vec<usize>,
    n: usize,
    freq: &[usize],
    groups_meta: &[(usize, u64, usize, usize)],
    groups_members: &[usize],
    level_union: u64,
    sorted: &mut Vec<usize>,
    sub_ranks: &mut Vec<usize>,
) -> Option<u64> {
    for &(cur_rank, gmask, s, e) in groups_meta {
        if gmask & level_union == 0 {
            continue;
        }
        let members = &groups_members[s..e];
        let f0 = freq[members[0]];
        if members.iter().all(|&v| freq[v] == f0) {
            continue;
        }
        sorted.clear();
        sorted.extend_from_slice(members);
        sorted.sort_by(|&a, &b| freq[b].cmp(&freq[a]));
        let mut sub_rank = 0usize;
        sub_ranks.clear();
        sub_ranks.resize(sorted.len(), 0);
        for i in 1..sorted.len() {
            if freq[sorted[i]] != freq[sorted[i - 1]] {
                sub_rank += 1;
            }
            sub_ranks[i] = sub_rank;
        }
        for v in 0..n {
            if vr[v] > cur_rank {
                vr[v] += sub_rank;
            }
        }
        for (i, &v) in sorted.iter().enumerate() {
            vr[v] = cur_rank + sub_ranks[i];
        }
        return Some(gmask);
    }
    None
}

// Remapped polynomial key for tiebreak #1: replace each variable with its var_rank,
// sort ranks within each monomial, then sort monomials (highest priority first).
fn poly_key_4(poly: &Polynomial, vr: &[usize], n: usize) -> Vec<MonomialRankKey4> {
    let mut terms: Vec<MonomialRankKey4> = poly
        .iter()
        .map(|&m| monomial_rank_key_4(m, vr, n))
        .collect();
    terms.sort_by(|a, b| b.degree.cmp(&a.degree).then(a.cmp(b)));
    terms
}

// Compact tiebreak-#1 key, exact under the same deg <= 16 precondition as
// LevelEntryC: with the 65-byte tail all zeros, the fat rank-key ordering is
// its prefix ordering, so (degree, prefix) pairs sorted by
// (degree desc, prefix asc) reproduce poly_key_4's term order.
fn poly_key_c(poly: &Polynomial, vr: &[usize]) -> Vec<(u8, u128)> {
    let mut terms: Vec<(u8, u128)> = poly.iter().map(|&m| rank_prefix_c(m, vr)).collect();
    terms.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    terms
}

// Keyed-wire comparison matching Vec<MonomialRankKey4>::cmp under the
// compact precondition: element order is prefix order (the fat Ord ignores
// the degree byte and the tail is zero), and equal prefixes imply equal
// degrees (a degree-d key has exactly d nonzero prefix bytes), so
// lexicographic-by-prefix plus the length tiebreak is the identical total
// order — and its Equal class is the identical equality.
fn cmp_poly_key_c(a: &[(u8, u128)], b: &[(u8, u128)]) -> CmpOrdering {
    let common = a.len().min(b.len());
    for i in 0..common {
        match a[i].1.cmp(&b[i].1) {
            CmpOrdering::Equal => {}
            other => return other,
        }
    }
    a.len().cmp(&b.len())
}

fn push_flat_canonical_form_4(
    polynomials: &[Polynomial],
    final_order: &[usize],
    wire_to_pos: &mut Vec<usize>,
    monomials: &mut Vec<Monomial>,
    out: &mut Vec<Option<Monomial>>,
) {
    let n = polynomials.len();
    wire_to_pos.resize(n, 0);
    for (pos, &wire) in final_order.iter().enumerate() {
        wire_to_pos[wire] = pos;
    }

    out.clear();
    for &wire in final_order {
        monomials.clear();
        monomials.extend(polynomials[wire].iter().map(|&m| {
            let mut r = 0u64;
            let mut mm = m;
            while mm != 0 {
                r |= 1u64 << wire_to_pos[mm.trailing_zeros() as usize];
                mm &= mm - 1;
            }
            r
        }));
        monomials.sort_unstable();
        out.extend(monomials.iter().copied().map(Some));
        out.push(None);
    }
}

// `groups`: the (groups_meta, groups_members) tied-group precompute for the
// n <= 64 fast path; None keeps the legacy per-level rank rescan (n > 64).
// Returns the split group's wire mask, None when no split fired.
#[allow(clippy::too_many_arguments)]
fn scan_class_poly_levels_4(
    cp: &[(Monomial, usize)],
    vr: &mut Vec<usize>,
    n: usize,
    tied_mask: u64,
    level_entries: &mut Vec<LevelEntry4>,
    freq: &mut Vec<usize>,
    tied: &mut Vec<usize>,
    sorted: &mut Vec<usize>,
    sub_ranks: &mut Vec<usize>,
    groups: Option<(&[(usize, u64, usize, usize)], &[usize])>,
) -> Option<u64> {
    sorted_level_entries_4(cp, vr, n, level_entries);
    let mut start = 0usize;
    while start < level_entries.len() {
        let mut end = start + 1;
        let mut level_union = level_entries[start].0;
        while end < level_entries.len() && level_entries[end].2 == level_entries[start].2 {
            level_union |= level_entries[end].0;
            end += 1;
        }
        // A level containing no tied wire gives every member of every tied
        // group frequency 0, so no split can fire — skip the freq count.
        if level_union & tied_mask != 0 {
            let split = if let Some((gm, gmem)) = groups {
                wire_freq_tied_4(&level_entries[start..end], tied_mask, n, freq);
                split_by_freq_groups_4(vr, n, freq, gm, gmem, level_union, sorted, sub_ranks)
            } else {
                wire_freq_4(&level_entries[start..end], n, freq);
                split_by_freq_4(vr, n, freq, tied, sorted, sub_ranks)
            };
            if split.is_some() {
                return split;
            }
        }
        start = end;
    }
    None
}

// Compact-entry scan: byte-for-byte the same level grouping and split
// decisions as scan_class_poly_levels_4 whenever all degrees <= 16
// (`compact_ok`).
#[allow(clippy::too_many_arguments)]
fn scan_class_poly_levels_c(
    cp: &[(Monomial, usize)],
    vr: &mut Vec<usize>,
    n: usize,
    tied_mask: u64,
    entries: &mut Vec<LevelEntryC>,
    freq: &mut Vec<usize>,
    tied: &mut Vec<usize>,
    sorted: &mut Vec<usize>,
    sub_ranks: &mut Vec<usize>,
    groups: Option<(&[(usize, u64, usize, usize)], &[usize])>,
) -> Option<u64> {
    sorted_level_entries_c(cp, vr, entries);
    let mut start = 0usize;
    while start < entries.len() {
        let mut end = start + 1;
        let mut level_union = entries[start].m;
        while end < entries.len() && eq_level_c(&entries[end], &entries[start]) {
            level_union |= entries[end].m;
            end += 1;
        }
        if level_union & tied_mask != 0 {
            let split = if let Some((gm, gmem)) = groups {
                wire_freq_tied_c(&entries[start..end], tied_mask, n, freq);
                split_by_freq_groups_4(vr, n, freq, gm, gmem, level_union, sorted, sub_ranks)
            } else {
                wire_freq_c(&entries[start..end], n, freq);
                split_by_freq_4(vr, n, freq, tied, sorted, sub_ranks)
            };
            if split.is_some() {
                return split;
            }
        }
        start = end;
    }
    None
}

// Bitmask of wires whose current rank is shared with another wire. Callers
// only use this to skip scans that provably cannot split anything, so for the
// (unreachable in practice) n > 64 case it degrades to "skip nothing".
fn tied_mask_4(vr: &[usize]) -> u64 {
    let n = vr.len();
    if n > 64 {
        return u64::MAX;
    }
    let mut count = [0u8; 64];
    for &r in vr {
        count[r] += 1;
    }
    let mut mask = 0u64;
    for (v, &r) in vr.iter().enumerate() {
        if count[r] > 1 {
            mask |= 1u64 << v;
        }
    }
    mask
}

fn has_ties_4(vr: &[usize]) -> bool {
    // Rank values stay contiguous in 0..n, so a bitmask detects duplicates in
    // one pass for the common n <= 64 case (monomials are u64 bitmasks, so n
    // never exceeds 64 on the polynomial-canonicalization paths).
    if vr.len() <= 64 {
        let mut seen = 0u64;
        for &r in vr {
            let bit = 1u64 << r;
            if seen & bit != 0 {
                return true;
            }
            seen |= bit;
        }
        return false;
    }
    let n = vr.len();
    (0..n).any(|v| (0..n).any(|u| u != v && vr[u] == vr[v]))
}

// Sort concatenated (monomial, 1) pairs and merge duplicates into counts.
// Produces the same monomial-ascending (monomial, count) sequence a BTreeMap
// build would, without per-insert tree traversal.
fn coalesce_class_poly(sum: &mut Vec<(Monomial, usize)>) {
    sum.sort_unstable_by_key(|&(m, _)| m);
    let mut write = 0usize;
    for read in 0..sum.len() {
        if write > 0 && sum[write - 1].0 == sum[read].0 {
            sum[write - 1].1 += sum[read].1;
        } else {
            sum[write] = sum[read];
            write += 1;
        }
    }
    sum.truncate(write);
}

// Is `b` reachable from `a` within `candidates` under known automorphisms that
// preserve the current rank coloring? Mirrors `is_same_orbit` (Rule L of the
// legacy canonicalizer); the coloring filter is what makes automorphisms
// discovered elsewhere in the recursion safe to reuse at this node, and each
// usable automorphism is applied in both directions (the group is closed
// under inversion).
// Does the orbit of `w` within `candidates` under `usable` (a set of
// coloring-preserving automorphisms, closed under inversion) contain any
// already-tried candidate? Because `usable` contains every inverse,
// reachability is symmetric, so this is exactly "exists t in tried reachable
// from t to w" — the historical per-(tried, candidate) check — computed with
// one BFS per candidate instead of one per pair. The caller maintains
// `usable` incrementally (vr is constant across one candidate loop).
fn canon4_orbit_hits_tried(
    w: usize,
    tried: &[usize],
    candidates: &[usize],
    usable: &[Vec<usize>],
) -> bool {
    if usable.is_empty() || tried.is_empty() {
        return false;
    }
    let cset: HashSet<usize> = candidates.iter().copied().collect();
    let tset: HashSet<usize> = tried.iter().copied().collect();
    let mut visited: HashSet<usize> = HashSet::default();
    let mut frontier = vec![w];
    visited.insert(w);
    while let Some(x) = frontier.pop() {
        for aut in usable {
            let img = aut[x];
            if tset.contains(&img) {
                return true;
            }
            if cset.contains(&img) && visited.insert(img) {
                frontier.push(img);
            }
        }
    }
    false
}

// Core loop: refine var_rank until fully resolved, then return final_order.
// `known_auts` accumulates automorphisms of the polynomial system discovered
// whenever two Rule L trials produce identical canonical forms; they are
// shared across the whole recursion to prune orbit-equivalent candidates.
// Per-invocation scratch buffers for canon4_run, pooled per thread and per
// recursion depth (each recursive call pops its own frame). Reuse kills the
// ~50+ heap allocations a fresh call would make; contents are always cleared
// before use, so pooling is invisible to the algorithm.
#[derive(Default)]
struct Canon4Frame {
    level_entries: Vec<LevelEntry4>,
    entries_c: Vec<LevelEntryC>,
    groups_meta: Vec<(usize, u64, usize, usize)>,
    groups_members: Vec<usize>,
    clean: Vec<bool>,
    freq_scratch: Vec<usize>,
    tied_scratch: Vec<usize>,
    sorted_scratch: Vec<usize>,
    sub_ranks_scratch: Vec<usize>,
    d_class_poly: Vec<(Monomial, usize)>,
    tied_buf: Vec<usize>,
    keyed_buf: Vec<(usize, Vec<MonomialRankKey4>)>,
    keyed_buf_c: Vec<(usize, Vec<(u8, u128)>)>,
    sub_ranks_buf: Vec<usize>,
    best_canonical: Vec<Option<Monomial>>,
    trial_canonical: Vec<Option<Monomial>>,
    canonical_monomials: Vec<Monomial>,
    wire_to_pos: Vec<usize>,
    tried: Vec<usize>,
    usable_auts: Vec<Vec<usize>>,
}

thread_local! {
    static CANON4_FRAMES: std::cell::RefCell<Vec<Canon4Frame>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[allow(clippy::too_many_arguments)]
fn canon4_run(
    polynomials: &[Polynomial],
    class_polys: &[Vec<(Monomial, usize)>],
    class_unions: &[u64],
    vr: Vec<usize>,
    allow_rule_l: bool,
    known_auts: &mut Vec<Vec<usize>>,
    compact_ok: bool,
) -> Result<Vec<usize>, ()> {
    let mut frame = CANON4_FRAMES
        .with(|pool| pool.borrow_mut().pop())
        .unwrap_or_default();
    let result = canon4_run_inner(
        polynomials,
        class_polys,
        class_unions,
        vr,
        allow_rule_l,
        known_auts,
        compact_ok,
        &mut frame,
    );
    CANON4_FRAMES.with(|pool| pool.borrow_mut().push(frame));
    result
}

#[allow(clippy::too_many_arguments)]
fn canon4_run_inner(
    polynomials: &[Polynomial],
    class_polys: &[Vec<(Monomial, usize)>],
    class_unions: &[u64],
    mut vr: Vec<usize>,
    allow_rule_l: bool,
    known_auts: &mut Vec<Vec<usize>>,
    compact_ok: bool,
    frame: &mut Canon4Frame,
) -> Result<Vec<usize>, ()> {
    let n = polynomials.len();
    let Canon4Frame {
        level_entries,
        entries_c,
        groups_meta,
        groups_members,
        clean,
        freq_scratch,
        tied_scratch,
        sorted_scratch,
        sub_ranks_scratch,
        d_class_poly,
        tied_buf,
        keyed_buf,
        keyed_buf_c,
        sub_ranks_buf,
        best_canonical,
        trial_canonical,
        canonical_monomials,
        wire_to_pos,
        tried,
        usable_auts,
    } = frame;
    level_entries.clear();
    entries_c.clear();
    // One reservation per canon4_run: every scan reuses this buffer, and no
    // static class poly can exceed the largest one's length.
    let max_cp_len = class_polys.iter().map(|cp| cp.len()).max().unwrap_or(0);
    if compact_ok {
        entries_c.reserve(max_cp_len);
    } else {
        level_entries.reserve(max_cp_len);
    }
    freq_scratch.clear();
    tied_scratch.clear();
    sorted_scratch.clear();
    sub_ranks_scratch.clear();
    d_class_poly.clear();
    // The tied-group precompute and cleanskip apply on the n <= 64 fast path
    // only (monomials are u64 bitmasks, so canonicalization callers never
    // exceed it; the degenerate n > 64 path keeps the legacy per-level rank
    // rescan and never skips).
    let fast = n <= 64;
    // Cleanskip state is per canon4_run invocation: clean[i] = the last scan
    // of class poly i in THIS run returned no-split and every split since had
    // a group disjoint from its union. A split renumbers ranks
    // order-isomorphically outside its own group, so such a rescan must
    // return no-split again; skipping it cannot change vr and the result
    // stays byte-identical.
    clean.clear();
    clean.resize(class_polys.len(), false);

    'master: loop {
        if !has_ties_4(&vr) {
            break;
        }

        // Wires still sharing a rank; scans over polys that touch none of
        // them can never split anything and are skipped wholesale.
        let tied_mask = tied_mask_4(&vr);
        if fast {
            tied_groups_4(&vr, n, groups_meta, groups_members);
        }
        let groups: Option<(&[(usize, u64, usize, usize)], &[usize])> = if fast {
            Some((groups_meta.as_slice(), groups_members.as_slice()))
        } else {
            None
        };

        // Phase 1: scan P_{C_i} monomial levels; split by wire frequency.
        // Any split of the first splittable group → restart.
        for (idx, (cp, &cp_union)) in class_polys.iter().zip(class_unions).enumerate() {
            if cp_union & tied_mask == 0 {
                continue;
            }
            // A clean poly's rescan is provably a no-split; skipping it
            // leaves vr untouched, so the outcome is byte-identical.
            if fast && clean[idx] {
                continue;
            }
            let scan_res = if compact_ok {
                scan_class_poly_levels_c(
                    cp,
                    &mut vr,
                    n,
                    tied_mask,
                    entries_c,
                    freq_scratch,
                    tied_scratch,
                    sorted_scratch,
                    sub_ranks_scratch,
                    groups,
                )
            } else {
                scan_class_poly_levels_4(
                    cp,
                    &mut vr,
                    n,
                    tied_mask,
                    level_entries,
                    freq_scratch,
                    tied_scratch,
                    sorted_scratch,
                    sub_ranks_scratch,
                    groups,
                )
            };
            match scan_res {
                Some(gmask) => {
                    for (j, &u) in class_unions.iter().enumerate() {
                        if u & gmask != 0 {
                            clean[j] = false;
                        }
                    }
                    continue 'master;
                }
                None => {
                    clean[idx] = true;
                }
            }
        }

        if !has_ties_4(&vr) {
            break;
        }

        // Tiebreak #1: for each tied group, compare remapped polynomial keys.
        // First group where keys differ → split and restart. The compact and
        // fat branches compute the identical stable sort and sub-rank
        // partition (cmp_poly_key_c == Vec<MonomialRankKey4>::cmp under
        // compact_ok, including its Equal classes).
        let max_rank = *vr.iter().max().unwrap_or(&0);
        for cur_rank in 0..=max_rank {
            tied_buf.clear();
            tied_buf.extend((0..n).filter(|&v| vr[v] == cur_rank));
            if tied_buf.len() <= 1 {
                continue;
            }

            let mut sub_rank = 0usize;
            sub_ranks_buf.clear();
            sub_ranks_buf.resize(tied_buf.len(), 0);
            if compact_ok {
                keyed_buf_c.clear();
                keyed_buf_c.extend(
                    tied_buf
                        .iter()
                        .map(|&v| (v, poly_key_c(&polynomials[v], &vr))),
                );
                keyed_buf_c.sort_by(|a, b| cmp_poly_key_c(&a.1, &b.1));
                for i in 1..keyed_buf_c.len() {
                    if keyed_buf_c[i - 1].1 != keyed_buf_c[i].1 {
                        sub_rank += 1;
                    }
                    sub_ranks_buf[i] = sub_rank;
                }
            } else {
                keyed_buf.clear();
                keyed_buf.extend(
                    tied_buf
                        .iter()
                        .map(|&v| (v, poly_key_4(&polynomials[v], &vr, n))),
                );
                keyed_buf.sort_by(|a, b| a.1.cmp(&b.1));
                for i in 1..keyed_buf.len() {
                    if keyed_buf[i - 1].1 != keyed_buf[i].1 {
                        sub_rank += 1;
                    }
                    sub_ranks_buf[i] = sub_rank;
                }
            }
            if sub_rank > 0 {
                // This split's group is exactly the tied group at cur_rank;
                // clear clean flags for every class poly it touches.
                let gmask = if n <= 64 {
                    tied_buf.iter().fold(0u64, |acc, &v| acc | (1u64 << v))
                } else {
                    u64::MAX
                };
                for (j, &u) in class_unions.iter().enumerate() {
                    if u & gmask != 0 {
                        clean[j] = false;
                    }
                }
                for v in 0..n {
                    if vr[v] > cur_rank {
                        vr[v] += sub_rank;
                    }
                }
                if compact_ok {
                    for (i, &(v, _)) in keyed_buf_c.iter().enumerate() {
                        vr[v] = cur_rank + sub_ranks_buf[i];
                    }
                } else {
                    for (i, &(v, _)) in keyed_buf.iter().enumerate() {
                        vr[v] = cur_rank + sub_ranks_buf[i];
                    }
                }
                continue 'master;
            }
        }

        if !has_ties_4(&vr) {
            break;
        }

        // Tiebreak #2: dynamic class polys P_{D_i} from current rank groups.
        // Apply same monomial-level scanning as Phase 1.
        let max_rank_val = *vr.iter().max().unwrap_or(&0);
        for rk in 0..=max_rank_val {
            d_class_poly.clear();
            let mut d_union = 0u64;
            for w in 0..n {
                if vr[w] == rk {
                    for &m in &polynomials[w] {
                        d_class_poly.push((m, 1usize));
                        d_union |= m;
                    }
                }
            }
            if d_class_poly.is_empty() || d_union & tied_mask == 0 {
                continue;
            }
            coalesce_class_poly(d_class_poly);
            let dscan_res = if compact_ok {
                scan_class_poly_levels_c(
                    d_class_poly,
                    &mut vr,
                    n,
                    tied_mask,
                    entries_c,
                    freq_scratch,
                    tied_scratch,
                    sorted_scratch,
                    sub_ranks_scratch,
                    groups,
                )
            } else {
                scan_class_poly_levels_4(
                    d_class_poly,
                    &mut vr,
                    n,
                    tied_mask,
                    level_entries,
                    freq_scratch,
                    tied_scratch,
                    sorted_scratch,
                    sub_ranks_scratch,
                    groups,
                )
            };
            if let Some(gmask) = dscan_res {
                for (j, &u) in class_unions.iter().enumerate() {
                    if u & gmask != 0 {
                        clean[j] = false;
                    }
                }
                continue 'master;
            }
        }

        // Rule L: try each wire in the first tied group as the sole winner.
        // Take the candidate that produces the lexicographically smallest canonical form.
        let tied_rank = if n <= 64 {
            (0..n)
                .filter(|&v| tied_mask & (1u64 << v) != 0)
                .map(|v| vr[v])
                .min()
        } else {
            (0..n)
                .filter(|&v| (0..n).filter(|&u| vr[u] == vr[v]).count() > 1)
                .map(|v| vr[v])
                .min()
        };

        if let Some(tr) = tied_rank {
            if !allow_rule_l {
                return Err(());
            }
            let candidates: Vec<usize> = (0..n).filter(|&v| vr[v] == tr).collect();
            // Charge the full tied-group candidate budget at every recursive
            // node. The thread-local accumulator is reset once per top-level
            // canonicalization, so the cap covers the entire Rule-L tree.
            if let Some(cap) = canon_rule_l_branch_cap() {
                let used = RULE_L_BRANCHES_USED.with(|branches| {
                    let used = branches.get().saturating_add(candidates.len() as u64);
                    branches.set(used);
                    used
                });
                if used > cap {
                    return Err(());
                }
            }
            let rule_l_start = Instant::now();
            CANON4_RULE_L_CALLS.fetch_add(1, Ordering::Relaxed);
            CANON4_RULE_L_BRANCHES.fetch_add(candidates.len() as u64, Ordering::Relaxed);
            best_canonical.clear();
            trial_canonical.clear();
            canonical_monomials.clear();
            wire_to_pos.clear();
            let mut have_best = false;
            let mut best_order: Vec<usize> = Vec::new();
            tried.clear();
            // Coloring-compatible automorphisms (plus inverses) for this
            // node's vr, synced lazily as the recursion appends to
            // known_auts. vr is constant across the candidate loop, so the
            // usability filter never changes and each aut is examined once.
            usable_auts.clear();
            let mut auts_synced = 0usize;

            for &w in &candidates {
                // Orbit pruning: a candidate reachable from an already-tried
                // one via a coloring-preserving automorphism completes to the
                // same canonical form, and equal forms never replace the best
                // (strict `<` below), so skipping it leaves the result
                // byte-identical while collapsing factorial symmetric blowup.
                while auts_synced < known_auts.len() {
                    let aut = &known_auts[auts_synced];
                    if aut.iter().enumerate().all(|(v, &img)| vr[img] == vr[v]) {
                        let mut inv = vec![0usize; aut.len()];
                        for (v, &img) in aut.iter().enumerate() {
                            inv[img] = v;
                        }
                        usable_auts.push(aut.clone());
                        usable_auts.push(inv);
                    }
                    auts_synced += 1;
                }
                if canon4_orbit_hits_tried(w, &tried, &candidates, &usable_auts) {
                    continue;
                }

                let mut trial_vr = vr.clone();
                for v in 0..n {
                    if trial_vr[v] > tr {
                        trial_vr[v] += 1;
                    }
                }
                for &other in &candidates {
                    if other != w {
                        trial_vr[other] = tr + 1;
                    }
                }

                let trial_order = canon4_run(
                    polynomials,
                    class_polys,
                    class_unions,
                    trial_vr,
                    true,
                    known_auts,
                    compact_ok,
                )?;
                push_flat_canonical_form_4(
                    polynomials,
                    &trial_order,
                    wire_to_pos,
                    canonical_monomials,
                    trial_canonical,
                );

                if !have_best {
                    best_canonical.clear();
                    best_canonical.extend_from_slice(&trial_canonical);
                    best_order = trial_order;
                    have_best = true;
                } else if trial_canonical == best_canonical {
                    // Two completions with the same canonical form yield an
                    // automorphism; record it for pruning everywhere in the
                    // recursion (usability is re-checked per node).
                    known_auts.push(automorphism_from_orders(&best_order, &trial_order, n));
                } else if trial_canonical < best_canonical {
                    best_canonical.clear();
                    best_canonical.extend_from_slice(&trial_canonical);
                    best_order = trial_order;
                }

                tried.push(w);
            }
            let rule_l_elapsed = rule_l_start.elapsed();
            CANON4_RULE_L_TIME.fetch_add(rule_l_elapsed.as_nanos() as u64, Ordering::Relaxed);
            if compression_trace_enabled()
                && rule_l_elapsed.as_millis() >= compression_trace_threshold_ms()
            {
                eprintln!(
                    "[compress-trace] slow rule_l n={} tied_rank={} branches={} elapsed_ms={}",
                    n,
                    tr,
                    candidates.len(),
                    rule_l_elapsed.as_millis()
                );
            }
            return Ok(best_order);
        }

        break;
    }

    let mut final_order: Vec<usize> = (0..n).collect();
    final_order.sort_by_key(|&w| (vr[w], w));
    Ok(final_order)
}

pub fn canonicalize_polys_4(
    mut polynomials: Vec<Polynomial>,
    allow_rule_l: bool,
) -> Result<(Vec<Polynomial>, Permutation), ()> {
    let n = polynomials.len();
    if n == 0 {
        return Ok((vec![], Permutation { data: vec![] }));
    }
    // Compact level keys are exact iff every rank key's 65-byte tail stays
    // zero, i.e. every monomial degree <= MONOMIAL_RANK_PREFIX_LEN_4 (16).
    // Class polys and D-class polys reuse these same monomials, so one
    // top-level check covers every scan in the recursion.
    let compact_ok = !polynomials.iter().any(|p| {
        p.iter()
            .any(|m| m.count_ones() > MONOMIAL_RANK_PREFIX_LEN_4 as u32)
    });
    // Each direction gets an independent budget, while every recursive
    // canon4_run invocation in this call shares the same accumulator.
    if canon_rule_l_branch_cap().is_some() {
        RULE_L_BRANCHES_USED.with(|branches| branches.set(0));
    }
    for poly in &mut polynomials {
        normalize_polynomial(poly);
    }
    let max_degree = n;

    // Group wires by degree profile; highest-profile group = P_{C_1}.
    let mut profiles: Vec<(usize, Vec<usize>)> = (0..n)
        .map(|i| (i, degree_counts(&polynomials[i], max_degree)))
        .collect();
    profiles.sort_by(|a, b| b.1.cmp(&a.1));

    let mut class_groups: Vec<Vec<usize>> = Vec::new();
    {
        let mut current = vec![profiles[0].0];
        for i in 1..profiles.len() {
            if profiles[i].1 == profiles[i - 1].1 {
                current.push(profiles[i].0);
            } else {
                class_groups.push(current.clone());
                current = vec![profiles[i].0];
            }
        }
        class_groups.push(current);
    }

    // Build P_{C_i}: sum of polynomials in each class group (natural-number
    // coefficients), as monomial-sorted (monomial, count) vectors.
    let class_polys: Vec<Vec<(Monomial, usize)>> = class_groups
        .iter()
        .map(|group| {
            let mut sum: Vec<(Monomial, usize)> = Vec::new();
            for &wire in group {
                sum.extend(polynomials[wire].iter().map(|&m| (m, 1usize)));
            }
            coalesce_class_poly(&mut sum);
            sum
        })
        .collect();

    // Static per-class-poly wire unions let canon4_run skip scans over polys
    // that touch no still-tied wire.
    let class_unions: Vec<u64> = class_polys
        .iter()
        .map(|cp| cp.iter().fold(0u64, |acc, &(m, _)| acc | m))
        .collect();

    // All wires start tied; canon4_run refines iteratively.
    let mut known_auts: Vec<Vec<usize>> = Vec::new();
    let final_order = canon4_run(
        &polynomials,
        &class_polys,
        &class_unions,
        vec![0usize; n],
        allow_rule_l,
        &mut known_auts,
        compact_ok,
    )?;

    let mut wire_to_pos = vec![0usize; n];
    for (pos, &wire) in final_order.iter().enumerate() {
        wire_to_pos[wire] = pos;
    }
    let remap_monomial = |m: Monomial| -> Monomial {
        let mut result = 0u64;
        let mut mm = m;
        while mm != 0 {
            result |= 1u64 << wire_to_pos[mm.trailing_zeros() as usize];
            mm &= mm - 1;
        }
        result
    };
    let canonical: Vec<Polynomial> = final_order
        .iter()
        .map(|&wire| polynomial_from_terms(polynomials[wire].iter().map(|&m| remap_monomial(m))))
        .collect();
    let canonical = trim_canonicalized(canonical);
    Ok((canonical, Permutation { data: final_order }))
}

pub fn monomial_degree(m: u64) -> u32 {
    m.count_ones()
}

fn mono_compressed_str(m: u64, n: usize) -> String {
    if m == 0 {
        return "I".into();
    }
    (0..n)
        .filter(|&i| (m >> i) & 1 == 1)
        .map(|i| format!("{}", i))
        .collect::<Vec<_>>()
        .join("•")
}

pub fn poly_to_compressed_str(poly: &Polynomial, n: usize) -> String {
    if poly.is_empty() {
        return "i".into();
    }
    let mut terms: Vec<u64> = poly.iter().copied().collect();
    terms.sort_by_key(|&m| (monomial_degree(m), m));
    terms
        .iter()
        .map(|&m| mono_compressed_str(m, n))
        .collect::<Vec<_>>()
        .join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::random_circuit;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::BTreeSet;

    #[test]
    fn polynomial_from_terms_sorts_and_cancels_pairs() {
        assert_eq!(polynomial_from_terms([5, 3, 5, 1, 3, 5, 7, 7]), vec![1, 5]);
    }

    // The byte parser replaced a char-by-char one. This is the old
    // implementation verbatim, so any grammar drift shows up as a diff.
    fn ref_from_string(s: &str) -> CircuitSeq {
        fn char_to_wire(c: char) -> u8 {
            match c {
                '0'..='9' => c as u8 - b'0',
                'a'..='z' => c as u8 - b'a' + 10,
                'A'..='Z' => c as u8 - b'A' + 36,
                '!' => 62,
                '@' => 63,
                '#' => 64,
                '$' => 65,
                '%' => 66,
                '^' => 67,
                '&' => 68,
                '*' => 69,
                '(' => 70,
                ')' => 71,
                '-' => 72,
                '_' => 73,
                '=' => 74,
                '+' => 75,
                '[' => 76,
                ']' => 77,
                '{' => 78,
                '}' => 79,
                '<' => 80,
                '>' => 81,
                '?' => 82,
                _ => panic!("Invalid wire char: {}", c),
            }
        }
        const BASE: u32 = 83;
        let gates: Vec<[u16; 3]> = s
            .trim()
            .split(';')
            .filter(|part| !part.is_empty())
            .map(|gate_str| {
                let mut chars = gate_str.chars().peekable();
                let mut wires = Vec::new();
                while chars.peek().is_some() {
                    let mut overflow = 0;
                    while chars.peek() == Some(&'~') {
                        overflow += 1;
                        chars.next();
                    }
                    let c = chars.next().expect("Expected wire character after ~");
                    wires.push((char_to_wire(c) as u32 + overflow * BASE) as u16);
                }
                assert_eq!(wires.len(), 3, "Each gate must have exactly 3 wires");
                [wires[0], wires[1], wires[2]]
            })
            .collect();
        CircuitSeq { gates }
    }

    #[test]
    fn opt_equiv_from_bytes_matches_char_parser_reference() {
        let mut rng = StdRng::seed_from_u64(0x5eed_2026);
        // Widths that exercise plain characters, one `~` and several `~`.
        for &wires in &[8usize, 83, 84, 166, 167, 300, 1024] {
            for _ in 0..40 {
                let m = rng.random_range(0..60usize);
                let gates: Vec<[u16; 3]> = (0..m)
                    .map(|_| {
                        [
                            rng.random_range(0..wires) as u16,
                            rng.random_range(0..wires) as u16,
                            rng.random_range(0..wires) as u16,
                        ]
                    })
                    .collect();
                let text = CircuitSeq { gates }.repr();
                assert_eq!(
                    CircuitSeq::from_string(&text).gates,
                    ref_from_string(&text).gates,
                    "parser drift on {text}"
                );
            }
        }
    }

    #[test]
    fn opt_equiv_from_bytes_keeps_whitespace_and_empty_segment_handling() {
        // Every wire alphabet class, plus one- and two-`~` overflow prefixes
        // (wires 83 and 175).
        let body = "0az;A!?;~0~~9~1;))(;";
        let expect = ref_from_string(body).gates;
        for wrapped in [
            body.to_string(),
            format!("  {body}\n"),
            format!("\n\t{body}  \r\n"),
            // Empty segments between separators are skipped, not gates.
            body.replace(';', ";;"),
        ] {
            assert_eq!(
                CircuitSeq::from_string(&wrapped).gates,
                expect,
                "mismatch on {wrapped:?}"
            );
        }
        // Empty and separator-only input yield no gates, as before.
        assert!(CircuitSeq::from_string("").gates.is_empty());
        assert!(CircuitSeq::from_string("   \n").gates.is_empty());
        assert!(CircuitSeq::from_string(";;;").gates.is_empty());
    }

    #[test]
    #[should_panic(expected = "Invalid wire char")]
    fn from_bytes_rejects_a_non_alphabet_character() {
        CircuitSeq::from_string("01 2;");
    }

    #[test]
    #[should_panic(expected = "exactly 3 wires")]
    fn from_bytes_rejects_a_short_gate() {
        CircuitSeq::from_string("01;");
    }

    // The bit-sliced kernel must reproduce the scalar one lane for lane,
    // including the X-gate spelling (pos == neg), which fires unconditionally.
    #[test]
    fn opt_equiv_lane_kernel_matches_scalar_evaluation() {
        let mut rng = StdRng::seed_from_u64(0xfeed_babe);
        for &wires in &[1usize, 2, 8, 64, 65, 128, 200, 256] {
            for trial in 0..25 {
                let m = rng.random_range(0..80usize);
                let gates: Vec<[u16; 3]> = (0..m)
                    .map(|_| {
                        let t = rng.random_range(0..wires) as u16;
                        let x = rng.random_range(0..wires) as u16;
                        // Every third gate is an X gate: controls equal.
                        let y = if trial % 3 == 0 && rng.random_bool(0.5) {
                            x
                        } else {
                            rng.random_range(0..wires) as u16
                        };
                        [t, x, y]
                    })
                    .collect();

                let len = lane_state_len(wires);
                let mut lanes = vec![0u64; len];
                for lane in lanes[..wires].iter_mut() {
                    *lane = rng.random();
                }
                let seed = lanes.clone();
                Gate::eval_lanes_index_list(&gates, &mut lanes);

                // Each of the 64 lanes must equal the scalar walk on the
                // sample that lane carries.
                for bit in 0..64 {
                    let mut scalar = U1024::zero();
                    for w in 0..wires {
                        if (seed[w] >> bit) & 1 == 1 {
                            scalar = scalar | (U1024::one() << w);
                        }
                    }
                    let out = Gate::evaluate_index_list_1024(scalar, &gates);
                    for w in 0..wires {
                        let want = ((out >> w) & U1024::one()) != U1024::zero();
                        let got = (lanes[w] >> bit) & 1 == 1;
                        assert_eq!(want, got, "wire {w} lane {bit} wires={wires}");
                    }
                }
            }
        }
    }

    #[test]
    fn opt_equiv_evaluate_index_list_64_matches_wider_kernels() {
        let mut rng = StdRng::seed_from_u64(0xc0ffee);
        for _ in 0..200 {
            let m = rng.random_range(0..64usize);
            let gates: Vec<[u16; 3]> = (0..m)
                .map(|_| {
                    let x = rng.random_range(0..64u16);
                    [
                        rng.random_range(0..64u16),
                        x,
                        if rng.random_bool(0.15) {
                            x
                        } else {
                            rng.random_range(0..64u16)
                        },
                    ]
                })
                .collect();
            let input: u64 = rng.random();
            let got = Gate::evaluate_index_list_64(input, &gates);
            let want = Gate::evaluate_index_list_128(input as u128, &gates);
            assert_eq!(got as u128, want);
            let mut v = u256::zero();
            v.0[0] = input;
            assert_eq!(Gate::evaluate_index_list_256(v, &gates).0[0], got);
        }
    }

    // Limb-indexed wide kernels must be bit-exact against the original
    // full-width bignum shift formulation for every in-range wire, including
    // limb boundaries.
    #[test]
    fn opt_equiv_limb_kernels_match_bignum_shifts() {
        fn ref_256(state: u256, gate: [u16; 3]) -> u256 {
            let one = u256::one();
            let c1 = (state >> gate[1]) & one;
            let c2 = (state >> gate[2]) & one;
            state ^ ((c1 | (one ^ c2)) << gate[0])
        }
        fn ref_512(state: u512, gate: [u16; 3]) -> u512 {
            let one = u512::one();
            let c1 = (state >> gate[1]) & one;
            let c2 = (state >> gate[2]) & one;
            state ^ ((c1 | (one ^ c2)) << gate[0])
        }
        fn ref_1024(state: U1024, gate: [u16; 3]) -> U1024 {
            let one = U1024::one();
            let c1 = (state >> gate[1]) & one;
            let c2 = (state >> gate[2]) & one;
            state ^ ((c1 | (one ^ c2)) << gate[0])
        }
        let mut state = 0x0dd_ba11_5eed_2026u64;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        let boundary = [0u64, 1, 63, 64, 65, 127, 128, 191, 255];
        for trial in 0..4000 {
            let pick = |next: &mut dyn FnMut(u64) -> u64, width: u64| -> u16 {
                if trial % 3 == 0 {
                    boundary[next(boundary.len() as u64) as usize].min(width - 1) as u16
                } else {
                    next(width) as u16
                }
            };
            let mut bytes = [0u8; 128];
            for b in bytes.iter_mut() {
                *b = next(256) as u8;
            }
            let g256 = [
                pick(&mut next, 256),
                pick(&mut next, 256),
                pick(&mut next, 256),
            ];
            let s256 = u256::from_little_endian(&bytes[..32]);
            assert_eq!(Gate::evaluate_index_256(s256, g256), ref_256(s256, g256));
            let g512 = [
                pick(&mut next, 512),
                pick(&mut next, 512),
                pick(&mut next, 512),
            ];
            let s512 = u512::from_little_endian(&bytes[..64]);
            assert_eq!(Gate::evaluate_index_512(s512, g512), ref_512(s512, g512));
            let g1024 = [
                pick(&mut next, 1024),
                pick(&mut next, 1024),
                pick(&mut next, 1024),
            ];
            let s1024 = U1024::from_little_endian(&bytes);
            assert_eq!(
                Gate::evaluate_index_1024(s1024, g1024),
                ref_1024(s1024, g1024)
            );
        }
    }

    // The u512 probably_equal arm (300 eval wires) must keep both verdicts.
    #[test]
    fn opt_equiv_probably_equal_512_arm_agrees() {
        let c = random_circuit(300, 400);
        // Equivalent pair: append a cancelling gate pair (target not among its
        // controls, so the two applications undo each other).
        let mut c2 = c.clone();
        c2.gates.push([5, 20, 30]);
        c2.gates.push([5, 20, 30]);
        assert!(c.probably_equal(&c2, 300, 64).is_ok());
        // Non-equivalent pair: [0,1,1] fires on every input (c1 | !c1), so
        // wire 0 differs on every drawn input.
        let mut c3 = c.clone();
        c3.gates.push([0, 1, 1]);
        assert!(c.probably_equal(&c3, 300, 64).is_err());
    }

    // The stack-style pass must reproduce the historical drain-with-backtrack
    // cancellation exactly, including tag lockstep.
    #[test]
    fn opt_equiv_cancel_adjacent_duplicates_matches_drain_reference() {
        fn reference(gates: &mut Vec<[u16; 3]>, tags: Option<&mut Vec<u32>>) {
            let mut tags = tags;
            let mut j = 0;
            while j < gates.len().saturating_sub(1) {
                if gates[j] == gates[j + 1] {
                    gates.drain(j..=j + 1);
                    if let Some(tags) = tags.as_deref_mut() {
                        tags.drain(j..=j + 1);
                    }
                    j = j.saturating_sub(2);
                } else {
                    j += 1;
                }
            }
        }
        let mut state = 0xdead_beef_cafe_1234u64;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        for len in [0usize, 1, 2, 3, 10, 200, 2000] {
            // A tiny alphabet makes adjacent duplicates and cascades common.
            let mut gates: Vec<[u16; 3]> = (0..len)
                .map(|_| [next(3) as u16, next(3) as u16, next(3) as u16])
                .collect();
            let mut tags: Vec<u32> = (0..len as u32).collect();
            let mut ref_gates = gates.clone();
            let mut ref_tags = tags.clone();
            cancel_adjacent_duplicates(&mut gates, Some(&mut tags));
            reference(&mut ref_gates, Some(&mut ref_tags));
            assert_eq!(gates, ref_gates, "len={len}");
            assert_eq!(tags, ref_tags, "len={len}");
        }
    }

    #[test]
    fn polynomial_xor_assign_cancels_shared_terms() {
        let mut left = vec![1, 3, 8];
        poly_xor_assign(&mut left, vec![3, 5, 8]);
        assert_eq!(left, vec![1, 5]);
    }

    #[test]
    fn to_polynomial_keeps_terms_sorted_and_cancelled() {
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let polys = circuit.to_polynomial(3, 0, 1);
        assert_eq!(polys[0], vec![0, 1, 4, 6]);
        assert_eq!(polys[1], vec![2]);
        assert_eq!(polys[2], vec![4]);
    }

    #[test]
    fn evaluate_1024_handles_wires_above_512() {
        let circuit = CircuitSeq {
            gates: vec![[900, 901, 902]],
        };
        let one = U1024::one();

        let flipped = circuit.evaluate_1024(U1024::zero());
        assert_eq!((flipped >> 900) & one, one);

        let blocked = circuit.evaluate_1024(one << 902);
        assert_eq!((blocked >> 900) & one, U1024::zero());
    }

    #[test]
    fn evaluate_128_matches_256_for_supported_wires() {
        let mut rng = fastrand::Rng::with_seed(0x6576_616c_3132_38);
        for _ in 0..500 {
            let n = rng.usize(3..=128);
            let m = rng.usize(0..=(3 * n));
            fastrand::seed(rng.u64(..));
            let circuit = random_circuit(n, m);
            let input = rng.u128(..);
            let mask = if n < 128 { (1u128 << n) - 1 } else { u128::MAX };

            let output_128 = circuit.evaluate_128(input) & mask;
            let output_256 = circuit.evaluate_256(u256::from(input)) & u256::from(mask);
            assert_eq!(u256::from(output_128), output_256, "n={n} m={m}");
        }
    }

    fn old_toggle(poly: &mut BTreeSet<Monomial>, m: Monomial) {
        if !poly.remove(&m) {
            poly.insert(m);
        }
    }

    fn old_xor(mut left: BTreeSet<Monomial>, right: BTreeSet<Monomial>) -> BTreeSet<Monomial> {
        for m in right {
            old_toggle(&mut left, m);
        }
        left
    }

    fn old_and(left: &BTreeSet<Monomial>, right: &BTreeSet<Monomial>) -> BTreeSet<Monomial> {
        let mut result = BTreeSet::new();
        for &m1 in left {
            for &m2 in right {
                old_toggle(&mut result, m1 | m2);
            }
        }
        result
    }

    fn old_not(poly: BTreeSet<Monomial>) -> BTreeSet<Monomial> {
        old_xor(BTreeSet::from([0u64]), poly)
    }

    fn old_hashset_style_to_polynomial(circuit: &CircuitSeq, n: usize) -> Vec<Polynomial> {
        let mut polys: Vec<BTreeSet<Monomial>> =
            (0..n).map(|i| BTreeSet::from([1u64 << i])).collect();

        for &[a, b, c] in &circuit.gates {
            let not_b = old_not(polys[b as usize].clone());
            let term = old_and(&polys[c as usize], &not_b);
            let mut new_a = old_xor(polys[a as usize].clone(), term);
            old_toggle(&mut new_a, 0u64);
            polys[a as usize] = new_a;
        }

        polys
            .into_iter()
            .map(|poly| poly.into_iter().collect())
            .collect()
    }

    #[test]
    fn to_polynomial_matches_old_hashset_style_implementation() {
        let mut rng = fastrand::Rng::with_seed(0x706f_6c79_7665_6375);
        for _ in 0..200 {
            let n = rng.usize(3..=12);
            let m = rng.usize(0..=(3 * n));
            fastrand::seed(rng.u64(..));
            let circuit = random_circuit(n, m);

            assert_eq!(
                circuit.to_polynomial(n, 0, circuit.gates.len()),
                old_hashset_style_to_polynomial(&circuit, n)
            );
        }
    }

    #[test]
    fn to_polynomial_capped_matches_unbounded_and_bails_cleanly() {
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let expected = circuit.to_polynomial(3, 0, circuit.gates.len());

        assert_eq!(
            circuit.to_polynomial_capped(3, 0, circuit.gates.len(), 4),
            Some(expected)
        );
        assert_eq!(
            circuit.to_polynomial_capped(3, 0, circuit.gates.len(), 3),
            None
        );
    }

    #[test]
    fn canonicalize_skips_window_over_64_distinct_wires() {
        // 22 disjoint triples touch 66 distinct wires. Building u64
        // monomials for this window would alias variables above bit 63.
        let gates: Vec<[u16; 3]> = (0..22u16)
            .map(|gate| [3 * gate, 3 * gate + 1, 3 * gate + 2])
            .collect();
        let circuit = CircuitSeq { gates };
        assert_eq!(circuit.used_wires().len(), 66);

        let skips_before = OVERSIZED_CANON_SKIPS.load(Ordering::Relaxed);
        assert!(circuit.canonicalize_polys_single(false).0.is_empty());
        assert!(circuit.canonicalize_polys_single(true).0.is_empty());
        assert!(circuit.canonicalize_polys_single_neg(&[]).0.is_empty());
        assert!(OVERSIZED_CANON_SKIPS.load(Ordering::Relaxed) >= skips_before.saturating_add(3));

        let small = CircuitSeq {
            gates: vec![[0, 1, 2], [1, 2, 0]],
        };
        assert!(!small.canonicalize_polys_single(false).0.is_empty());
    }

    fn eval_poly(poly: &Polynomial, input: usize) -> usize {
        poly.iter().fold(0usize, |acc, &monomial| {
            let term = if monomial == 0 || ((input as u64) & monomial) == monomial {
                1
            } else {
                0
            };
            acc ^ term
        })
    }

    fn eval_polys(polys: &[Polynomial], input: usize) -> usize {
        polys.iter().enumerate().fold(0usize, |acc, (wire, poly)| {
            acc | (eval_poly(poly, input) << wire)
        })
    }

    #[test]
    fn probably_equal_widens_to_actual_circuit_wires() {
        // Both circuits compute w2 ^= (w1 AND NOT w0) on the 200-wire compare
        // contract, via a zero-initialized aux wire. A's aux lives at wire
        // 300/301 — beyond u256 — so the old num_wires-based width dispatch
        // evaluated it in u256 where shifts >= 256 silently return 0,
        // corrupting the aux accesses and reporting a false "not equal".
        // Width must follow the circuits' actual max wire.
        let a = CircuitSeq {
            gates: vec![[300, 0, 1], [2, 301, 300]],
        };
        let b = CircuitSeq {
            gates: vec![[250, 0, 1], [2, 251, 250]],
        };
        assert!(
            a.probably_equal(&b, 200, 512).is_ok(),
            "equivalent circuits reported non-equal — width dispatch regressed"
        );
        assert!(b.probably_equal(&a, 200, 512).is_ok());

        // Sanity: genuinely different circuits are still detected at the
        // widened evaluation width.
        let c = CircuitSeq {
            gates: vec![[2, 1, 0]],
        };
        assert!(a.probably_equal(&c, 200, 512).is_err());
    }

    // The packed-prefix rank-key comparator must order and equate exactly like
    // the full 65-byte lexicographic compare it replaces.
    #[test]
    fn opt_equiv_rank_key_prefix_cmp_matches_lexicographic() {
        let mut state = 0x7261_6e6b_6b65_7934u64;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        let mut keys: Vec<MonomialRankKey4> = Vec::new();
        // Raw random byte arrays with a tiny alphabet and frequent short
        // fills, so prefix ties (and ties resolved only past byte 16) are
        // common.
        for _ in 0..200 {
            let mut encoded_ranks = [0u8; MONOMIAL_RANK_KEY_LEN_4];
            let filled = if next(2) == 0 {
                next(6) as usize
            } else {
                next(1 + MONOMIAL_RANK_KEY_LEN_4 as u64) as usize
            };
            for slot in 0..filled {
                encoded_ranks[slot] = next(4) as u8;
            }
            let degree = encoded_ranks.iter().filter(|&&b| b != 0).count() as u8;
            let prefix = u128::from_be_bytes(
                encoded_ranks[..MONOMIAL_RANK_PREFIX_LEN_4]
                    .try_into()
                    .unwrap(),
            );
            keys.push(MonomialRankKey4 {
                degree,
                prefix,
                encoded_ranks,
            });
        }
        // Realistic keys from random monomials under random rank vectors.
        for _ in 0..200 {
            let n = 1 + next(16) as usize;
            let vr: Vec<usize> = (0..n).map(|_| next(n as u64) as usize).collect();
            let m: Monomial = next(1u64 << n);
            keys.push(monomial_rank_key_4(m, &vr, n));
        }
        for a in &keys {
            for b in &keys {
                assert_eq!(a.cmp(b), a.encoded_ranks.cmp(&b.encoded_ranks));
                assert_eq!(*a == *b, a.encoded_ranks == b.encoded_ranks);
            }
        }
    }

    // The stack-bitset used_wires fast path (and the len-only variant) must
    // match the historical marking implementation, on either side of the
    // 1024-wire fallback boundary.
    #[test]
    fn opt_equiv_used_wires_matches_marking_reference() {
        fn reference(gates: &[[u16; 3]]) -> Vec<u16> {
            let Some(max_wire) = gates.iter().flatten().copied().max() else {
                return Vec::new();
            };
            let mut used = vec![false; max_wire as usize + 1];
            for &[target, control_a, control_b] in gates {
                used[target as usize] = true;
                used[control_a as usize] = true;
                used[control_b as usize] = true;
            }
            used.into_iter()
                .enumerate()
                .filter_map(|(wire, is_used)| is_used.then_some(wire as u16))
                .collect()
        }
        let mut state = 0x7573_6564_7769_7265u64;
        let mut next = |m: u64| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) % m
        };
        for len in [0usize, 1, 5, 40, 300] {
            for &wire_span in &[3u64, 60, 1023, 1024, 1500, 65535] {
                let gates: Vec<[u16; 3]> = (0..len)
                    .map(|_| {
                        [
                            next(wire_span + 1) as u16,
                            next(wire_span + 1) as u16,
                            next(wire_span + 1) as u16,
                        ]
                    })
                    .collect();
                let circuit = CircuitSeq { gates };
                let expected = reference(&circuit.gates);
                assert_eq!(circuit.used_wires(), expected, "len={len} span={wire_span}");
                assert_eq!(circuit.used_wires_len(), expected.len());
            }
        }
    }

    // Scratch-buffer to_polynomial/_capped must reproduce the historical
    // allocate-per-gate pipeline exactly, including the capped budget checks.
    fn reference_to_polynomial(
        circuit: &CircuitSeq,
        n: usize,
        start: usize,
        end: usize,
    ) -> Vec<Polynomial> {
        let gates = &circuit.gates[start..end];
        let mut polys: Vec<Polynomial> = (0..n).map(|i| vec![1u64 << i]).collect();
        for &[a, b, c] in gates {
            let term = poly_and_not(&polys[c as usize], &polys[b as usize]);
            poly_xor_assign(&mut polys[a as usize], term);
            toggle_monomial(&mut polys[a as usize], 0u64);
        }
        polys
    }

    fn reference_to_polynomial_capped(
        circuit: &CircuitSeq,
        n: usize,
        start: usize,
        end: usize,
        cap: usize,
    ) -> Option<Vec<Polynomial>> {
        let gates = &circuit.gates[start..end];
        let mut polys: Vec<Polynomial> = (0..n).map(|i| vec![1u64 << i]).collect();
        for &[a, b, c] in gates {
            if polys[b as usize]
                .len()
                .saturating_mul(polys[c as usize].len())
                > cap.saturating_mul(16)
            {
                return None;
            }
            let term = poly_and_not(&polys[c as usize], &polys[b as usize]);
            poly_xor_assign(&mut polys[a as usize], term);
            toggle_monomial(&mut polys[a as usize], 0u64);
            if polys[a as usize].len() > cap {
                return None;
            }
        }
        Some(polys)
    }

    #[test]
    fn opt_equiv_to_polynomial_scratch_reuse_matches_reference() {
        let mut rng = fastrand::Rng::with_seed(0x7363_7261_7463_6821);
        for _ in 0..200 {
            let n = rng.usize(3..=12);
            let m = rng.usize(0..=(3 * n));
            fastrand::seed(rng.u64(..));
            let circuit = random_circuit(n, m);
            let start = rng.usize(0..=circuit.gates.len());
            let end = rng.usize(start..=circuit.gates.len());
            assert_eq!(
                circuit.to_polynomial(n, start, end),
                reference_to_polynomial(&circuit, n, start, end),
                "n={n} m={m} start={start} end={end}"
            );
            for cap in [1usize, 2, 4, 8, 64, usize::MAX] {
                assert_eq!(
                    circuit.to_polynomial_capped(n, start, end, cap),
                    reference_to_polynomial_capped(&circuit, n, start, end, cap),
                    "n={n} m={m} start={start} end={end} cap={cap}"
                );
            }
        }
    }

    // Merge-based input negation must match the historical per-rest
    // binary-search toggles on normalized polynomials.
    #[test]
    fn opt_equiv_substitute_input_negation_matches_toggle_reference() {
        fn reference(poly: &mut Polynomial, w: usize) {
            let bit = 1u64 << w;
            let rests: Vec<Monomial> = poly
                .iter()
                .filter(|&&m| m & bit != 0)
                .map(|&m| m & !bit)
                .collect();
            for rest in rests {
                toggle_monomial(poly, rest);
            }
        }
        let mut rng = fastrand::Rng::with_seed(0x6e65_6761_7465_7331);
        for _ in 0..500 {
            let k = rng.usize(1..=12);
            let terms: Vec<Monomial> = (0..rng.usize(0..=40))
                .map(|_| rng.u64(..) & ((1u64 << k) - 1))
                .collect();
            let poly = polynomial_from_terms(terms);
            let w = rng.usize(0..k);
            let mut new_poly = poly.clone();
            let mut old_poly = poly.clone();
            substitute_input_negation(&mut new_poly, w);
            reference(&mut old_poly, w);
            assert_eq!(new_poly, old_poly, "k={k} w={w} poly={poly:?}");
            // Involution sanity: substituting the same wire twice restores
            // the input.
            substitute_input_negation(&mut new_poly, w);
            assert_eq!(new_poly, poly, "k={k} w={w}");
        }
    }

    // The hashed single-direction variant must agree with the plain variant
    // on order and used wires, and its key must be exactly
    // xxh3_128(polys_repr_blob(plain polys)) in both fresh and cached paths.
    #[test]
    fn opt_equiv_canonicalize_polys_single_hashed_matches_plain() {
        use xxhash_rust::xxh3::xxh3_128;
        let mut rng = fastrand::Rng::with_seed(0x6861_7368_6564_2101);
        for trial in 0..60 {
            let n = rng.usize(3..=10);
            let m = rng.usize(1..=(3 * n));
            fastrand::seed(rng.u64(..));
            let circuit = random_circuit(n, m);
            for reversed in [false, true] {
                let (hash, h_order, h_used) = circuit.canonicalize_polys_single_hashed(reversed);
                let (polys, p_order, p_used) = circuit.canonicalize_polys_single(reversed);
                assert_eq!(h_order.data, p_order.data, "trial={trial} rev={reversed}");
                assert_eq!(h_used, p_used);
                assert_eq!(
                    hash,
                    Some(xxh3_128(&polys_repr_blob(&polys)).to_le_bytes()),
                    "trial={trial} rev={reversed}"
                );
                // Second call exercises the cached path when the cache is on.
                let (hash2, order2, used2) = circuit.canonicalize_polys_single_hashed(reversed);
                assert_eq!(hash2, hash);
                assert_eq!(order2.data, p_order.data);
                assert_eq!(used2, p_used);
            }
        }
        // Oversized windows keep the skip contract: no key, empty order.
        let gates: Vec<[u16; 3]> = (0..22u16)
            .map(|gate| [3 * gate, 3 * gate + 1, 3 * gate + 2])
            .collect();
        let big = CircuitSeq { gates };
        let (hash, order, used) = big.canonicalize_polys_single_hashed(false);
        assert_eq!(hash, None);
        assert!(order.data.is_empty());
        assert_eq!(used.len(), 66);
    }

    // The cached neg variant must match the historical uncached pipeline
    // (per-input substitution in caller order) on fresh and hit paths, and an
    // empty negation set must coincide with the plain canonicalization.
    #[test]
    fn opt_equiv_canon_single_neg_cache_matches_uncached_reference() {
        fn reference_neg(
            circuit: &CircuitSeq,
            negated_inputs: &[u16],
        ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
            let used = circuit.used_wires();
            if used.len() > 64 {
                return (Vec::new(), Permutation { data: Vec::new() }, used);
            }
            let wire_map = dense_wire_map(&used);
            let mut c = CircuitSeq {
                gates: circuit
                    .gates
                    .iter()
                    .map(|&[t, c1, c2]| {
                        [
                            wire_map[t as usize],
                            wire_map[c1 as usize],
                            wire_map[c2 as usize],
                        ]
                    })
                    .collect(),
            };
            c.canonicalize();
            let n = c.max_wire() as usize + 1;
            let mut polys = c.to_polynomial(n, 0, c.gates.len());
            for &w in negated_inputs {
                let mapped = match wire_map.get(w as usize) {
                    Some(&mw) if mw != u16::MAX => mw as usize,
                    _ => continue,
                };
                for p in polys.iter_mut() {
                    // Sequential toggle substitution, as the historical code
                    // did.
                    let bit = 1u64 << mapped;
                    let rests: Vec<Monomial> = p
                        .iter()
                        .filter(|&&mm| mm & bit != 0)
                        .map(|&mm| mm & !bit)
                        .collect();
                    for rest in rests {
                        toggle_monomial(p, rest);
                    }
                }
            }
            let canon = match canonicalize_polys_4(polys, true) {
                Ok(canon) => canon,
                Err(()) => return (Vec::new(), Permutation { data: Vec::new() }, used),
            };
            (canon.0, canon.1, used)
        }

        let mut rng = fastrand::Rng::with_seed(0x6e65_675f_6361_6368);
        for trial in 0..60 {
            let n = rng.usize(3..=10);
            let m = rng.usize(1..=(3 * n));
            fastrand::seed(rng.u64(..));
            let circuit = random_circuit(n, m);
            // Random negation list: in-range wires (used or not), a possible
            // duplicate (involution), and a possible out-of-range wire to
            // exercise the skip filter.
            let mut negs: Vec<u16> = (0..rng.usize(0..=4))
                .map(|_| rng.u16(0..n as u16))
                .collect();
            if rng.bool() && !negs.is_empty() {
                let dup = negs[0];
                negs.push(dup);
            }
            if rng.bool() {
                negs.push(500);
            }
            let expected = reference_neg(&circuit, &negs);
            let first = circuit.canonicalize_polys_single_neg(&negs);
            let second = circuit.canonicalize_polys_single_neg(&negs);
            assert_eq!(first.0, expected.0, "trial={trial} negs={negs:?}");
            assert_eq!(first.1.data, expected.1.data, "trial={trial} negs={negs:?}");
            assert_eq!(first.2, expected.2);
            assert_eq!(second.0, expected.0, "hit path, trial={trial}");
            assert_eq!(second.1.data, expected.1.data);
            assert_eq!(second.2, expected.2);
            // Empty negation set must coincide with plain canonicalization.
            let plain = circuit.canonicalize_polys_single(false);
            let neg_empty = circuit.canonicalize_polys_single_neg(&[]);
            assert_eq!(neg_empty.0, plain.0, "trial={trial}");
            assert_eq!(neg_empty.1.data, plain.1.data);
            assert_eq!(neg_empty.2, plain.2);
        }
    }

    #[test]
    fn to_polynomial_matches_evaluate_control_semantics() {
        let cases = [
            CircuitSeq {
                gates: vec![[0, 1, 2]],
            },
            CircuitSeq {
                gates: vec![[3, 0, 1], [1, 2, 3], [2, 3, 0], [0, 1, 4]],
            },
        ];

        for circuit in cases {
            let n = 5;
            let mask = (1usize << n) - 1;
            let polys = circuit.to_polynomial(n, 0, circuit.gates.len());
            for input in 0..(1usize << n) {
                assert_eq!(
                    eval_polys(&polys, input) & mask,
                    circuit.evaluate(input) & mask,
                    "input={input:#b} circuit={:?}",
                    circuit.gates
                );
            }
        }
    }

    fn canon_hash(seed: u64, n_wires: u16, gates: usize) -> u128 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut c = CircuitSeq { gates: Vec::new() };
        while c.gates.len() < gates {
            let a = rng.random_range(0..n_wires);
            let b = rng.random_range(0..n_wires);
            let d = rng.random_range(0..n_wires);
            if a != b && a != d && b != d {
                c.gates.push([a, b, d]);
            }
        }
        let (polys, _, _) = c.canonicalize_polys_single(seed % 2 == 0);
        xxhash_rust::xxh3::xxh3_128(&polys_repr_blob(&polys))
    }

    // Golden canonical-form hashes. The canonical form defines every curated-DB key, so any
    // change to these values means the DB has been silently invalidated. Do not regenerate
    // casually. Regenerated 2026-07-18 after fixing a swapped-argument bug in to_polynomial
    // (the g57 monomial was b*NOT(c); the executor and from_g57 use NOT(b)*c) — this realigns
    // our DB keys with the source/upstream convention. Regenerate via the #[ignore]d
    // regenerate_canon_golden test only when the canonical form legitimately changes.
    #[test]
    fn canonical_form_golden() {
        const GOLDEN: &[(&str, &str)] = &[
            ("G0", "6fa0209f74c6aca5a629ea4de7b882dd"),
            ("H0", "f8decf2b941d92e0016521e783bea5a8"),
            ("G1", "541946580565039aca15fdbd19755175"),
            ("H1", "25fb31deb42fd599dc90d9dea6fba89d"),
            ("G2", "5d551f5eba5474d33a871ecb0a8e46b7"),
            ("H2", "36a43931107c7d80f1fffbc5527c313a"),
            ("G3", "fd20b272d990c65207f9ffa0b3582b31"),
            ("H3", "1502ecfb3f88e59418564f2185548dce"),
            ("G4", "b6d47169d8e88efc8d965ff190d6c21f"),
            ("H4", "56986adc1dfa776e9de5480a80e9ab46"),
            ("G5", "fdf065eb905e344adb801bf88319a929"),
            ("H5", "e924c80a183b499d7b576bde07129131"),
            ("G6", "fbda8128fafdf7924880c95813641b7a"),
            ("H6", "245dd8d448f23884fbbac2b87aacaa86"),
            ("G7", "f7594dc87bdccb66a5e77d816b05b06a"),
            ("H7", "e015d3a5ec42bab54220110ad22f2612"),
            ("G8", "c29cd6d9be717edd6850b647f5370c0c"),
            ("H8", "602c1fad413f6816a32e4d299d096879"),
            ("G9", "e36265eba0672f79bbd54ed94c388564"),
            ("H9", "3fd134840c7913907227cf9205bdac9e"),
            ("G10", "d19251954f40b19ccb15c700f246276b"),
            ("H10", "59e94f3dedd110d92c1dae1e0a74cade"),
            ("G11", "b3cb06c6bb4c1f8f3f8e612e2622f2bf"),
            ("H11", "5168e7504437bd05203e558be87d7a5a"),
            ("G12", "3a2eebdc712c2cb9f8896f11169e4026"),
            ("H12", "c6ad8d964f866af06894dae44cad7911"),
            ("G13", "586eb625594255c9621e813a91669339"),
            ("H13", "2bfc4e6ff9208214514c899ffc57bb62"),
            ("G14", "1192d8249e2773dd389c2e8611bf256c"),
            ("H14", "d28943f055eb4a63aa4e4f4c4670ba13"),
            ("G15", "bca4629cf33574b8bd9eec6527e0c95f"),
            ("H15", "3523ee73c966859cabd1a5e626cdc51f"),
            ("G16", "2f0196c07918909d574b35e423bc60c7"),
            ("H16", "b76ea7f0d46c738a83d5ccdfacec678c"),
            ("G17", "1f0f68761b56f660f88c9ebdc2177bb7"),
            ("H17", "bd6e51d525fbeab79ec34b154ee398b4"),
            ("G18", "1d12417189ed645c0d68e74c85c3e4bd"),
            ("H18", "360c53cc42d3591e7c3558384e5d7a7f"),
            ("G19", "ac596d2604e6963258e9de7746f1efd7"),
            ("H19", "c24ea23211b28ac22faac1480c51289b"),
            ("G20", "e020d8c2d1dd740c9ef7e71d0ae52ac9"),
            ("H20", "a5b37a68dddb773429da7672d6c4870d"),
            ("G21", "39b636d94d180700abfe4e0b299e62c8"),
            ("H21", "c7b6d35f0b75ee1a3a23910657d13935"),
            ("G22", "b0a48bb2e87a13245269b1e309380d07"),
            ("H22", "53c37dd76b64c030b4c50b3ffd68a2c9"),
            ("G23", "631171335b8ff30b1a4406c1913cfed8"),
            ("H23", "acf0954bed92446336cd7dd618c46a22"),
            ("G24", "6d1677f347a24adaddd438be1758575f"),
            ("H24", "227c2822c97ec6a2add6ef947e57557b"),
            ("G25", "00c835f56a23564e21bc44105f1169b2"),
            ("H25", "5fd407309e7f74231d8cc0890f7d63f6"),
            ("G26", "783cd181e4f96d3bf9656d5aa0737b98"),
            ("H26", "d8dc7d0de786e49c90e97f77cebc19bf"),
            ("G27", "5d739004d2e1c3232660d923c341950f"),
            ("H27", "e854979b9d3fe892e12223aa67f261a1"),
            ("G28", "bcceaba13911caa90caddbd5c9090af3"),
            ("H28", "7a70480cb8f8a667167669f7a1128d14"),
            ("G29", "39c9d5702ff15e41a0504d0e6cbefe39"),
            ("H29", "ece8b5e185ca0c4cb818fc3fea4cf033"),
        ];
        for (tag, want) in GOLDEN {
            let (series, seed_s) = tag.split_at(1);
            let seed: u64 = seed_s.parse().unwrap();
            let got = match series {
                "G" => canon_hash(seed, 10, 8),
                _ => canon_hash(seed.wrapping_mul(0x9e37), 14, 12),
            };
            assert_eq!(
                format!("{:032x}", got),
                *want,
                "canonical form changed for {}",
                tag
            );
        }
    }

    #[test]
    #[ignore]
    fn regenerate_canon_golden() {
        for i in 0..30u64 {
            let g = canon_hash(i, 10, 8);
            let h = canon_hash(i.wrapping_mul(0x9e37), 14, 12);
            println!("(\"G{i}\", \"{:032x}\"),", g);
            println!("(\"H{i}\", \"{:032x}\"),", h);
        }
    }

    #[test]
    fn substitute_input_negation_flips_that_variable() {
        // eval a polynomial (XOR of monomials; monomial bits = AND of those input vars; 0 = const).
        fn eval(poly: &Polynomial, x: u64) -> u8 {
            let mut v = 0u8;
            for &m in poly {
                if x & m == m {
                    v ^= 1;
                }
            }
            v
        }
        let n = 4;
        for _ in 0..40 {
            let c = random_circuit(n, 8);
            let polys = c.to_polynomial(n, 0, c.gates.len());
            for w in 0..n {
                let mut flipped = polys.clone();
                for p in flipped.iter_mut() {
                    substitute_input_negation(p, w);
                }
                // substituting x_w -> x_w+1 means flipped(x) == original(x ^ (1<<w)) for every wire.
                for x in 0..(1u64 << n) {
                    for i in 0..n {
                        assert_eq!(
                            eval(&flipped[i], x),
                            eval(&polys[i], x ^ (1u64 << w)),
                            "w={w} x={x} wire={i}"
                        );
                    }
                }
            }
        }
    }

    // The three canon4 scan configurations must be interchangeable: the
    // legacy per-level rank rescan (fat entries, groups=None), the tied-group
    // precompute (fat entries, groups=Some), and the compact-entry scan
    // (deg<=16). Same split verdict, same split-group mask, same vr after.
    #[test]
    fn opt_equiv_canon4_scan_paths_agree() {
        let mut rng = StdRng::seed_from_u64(0x5ca9_2026);
        for trial in 0..400 {
            // Wide trials force degrees > 16 to exercise the fat fallback
            // beside the group precompute; narrow trials cover compact too.
            let wide = trial % 4 == 3;
            let n = if wide {
                rng.random_range(17..=22usize)
            } else {
                rng.random_range(4..=12usize)
            };
            let ranks = rng.random_range(1..=n);
            let vr0: Vec<usize> = (0..n).map(|_| rng.random_range(0..ranks)).collect();
            let terms = rng.random_range(1..=12usize);
            let mut cp: Vec<(Monomial, usize)> = (0..terms)
                .map(|_| {
                    let m = if n >= 64 {
                        rng.random_range(0..u64::MAX)
                    } else {
                        rng.random_range(0..(1u64 << n))
                    };
                    (m, 1usize)
                })
                .collect();
            coalesce_class_poly(&mut cp);
            let tied_mask = tied_mask_4(&vr0);
            let mut groups_meta = Vec::new();
            let mut groups_members = Vec::new();
            tied_groups_4(&vr0, n, &mut groups_meta, &mut groups_members);
            let groups = Some((groups_meta.as_slice(), groups_members.as_slice()));

            let scratch = || (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let (mut le, mut f, mut t, mut s, mut sr) = scratch();
            let mut vr_legacy = vr0.clone();
            let res_legacy = scan_class_poly_levels_4(
                &cp,
                &mut vr_legacy,
                n,
                tied_mask,
                &mut le,
                &mut f,
                &mut t,
                &mut s,
                &mut sr,
                None,
            );
            let (mut le, mut f, mut t, mut s, mut sr) = scratch();
            let mut vr_groups = vr0.clone();
            let res_groups = scan_class_poly_levels_4(
                &cp,
                &mut vr_groups,
                n,
                tied_mask,
                &mut le,
                &mut f,
                &mut t,
                &mut s,
                &mut sr,
                groups,
            );
            assert_eq!(res_legacy, res_groups, "trial={trial} n={n}");
            assert_eq!(vr_legacy, vr_groups, "trial={trial} n={n}");

            let compact_ok = cp.iter().all(|&(m, _)| m.count_ones() <= 16);
            if compact_ok {
                let mut ec = Vec::new();
                let (_, mut f, mut t, mut s, mut sr) = scratch();
                let mut vr_c = vr0.clone();
                let res_c = scan_class_poly_levels_c(
                    &cp, &mut vr_c, n, tied_mask, &mut ec, &mut f, &mut t, &mut s, &mut sr, groups,
                );
                assert_eq!(res_legacy, res_c, "compact trial={trial} n={n}");
                assert_eq!(vr_legacy, vr_c, "compact trial={trial} n={n}");
            }
        }
    }

    // Compact tiebreak-#1 keys must reproduce the fat keyed comparison
    // exactly (ordering AND equality classes) whenever all degrees <= 16.
    #[test]
    fn opt_equiv_canon4_poly_key_compact_matches_fat() {
        let mut rng = StdRng::seed_from_u64(0x9e37_2026);
        for trial in 0..2000 {
            let n = rng.random_range(2..=14usize);
            let ranks = rng.random_range(1..=n);
            let vr: Vec<usize> = (0..n).map(|_| rng.random_range(0..ranks)).collect();
            let mut poly = |force_share: Option<&Polynomial>| -> Polynomial {
                // Bias toward shared monomials so equality cases actually occur.
                let terms = rng.random_range(0..=6usize);
                (0..terms)
                    .map(|_| match force_share {
                        Some(other) if !other.is_empty() && rng.random_range(0..2u8) == 0 => {
                            other[rng.random_range(0..other.len())]
                        }
                        _ => rng.random_range(0..(1u64 << n)),
                    })
                    .collect()
            };
            let a = poly(None);
            let b = poly(Some(&a));
            let fat_a = poly_key_4(&a, &vr, n);
            let fat_b = poly_key_4(&b, &vr, n);
            let c_a = poly_key_c(&a, &vr);
            let c_b = poly_key_c(&b, &vr);
            assert_eq!(
                cmp_poly_key_c(&c_a, &c_b),
                fat_a.cmp(&fat_b),
                "trial={trial} n={n}"
            );
            assert_eq!(c_a == c_b, fat_a == fat_b, "trial={trial} n={n}");
            // Term order must match one-to-one: (degree, prefix) of fat terms.
            assert_eq!(
                c_a,
                fat_a
                    .iter()
                    .map(|k| (k.degree, k.prefix))
                    .collect::<Vec<_>>(),
                "trial={trial} n={n}"
            );
        }
    }

    // End-to-end: the shipped compact/split2/cleanskip canonicalizer must
    // stay deterministic and produce a canonical form invariant under wire
    // relabeling — the property every curated-DB key relies on. Narrow trials
    // run the compact path, wide trials (a forced degree-17+ monomial) run
    // the fat fallback through the same driver.
    #[test]
    fn opt_equiv_canon4_form_invariant_under_relabeling() {
        let mut rng = StdRng::seed_from_u64(0xfab1e_2026);
        for trial in 0..72 {
            let wide = trial % 6 == 5;
            let n = if wide {
                rng.random_range(17..=20usize)
            } else {
                rng.random_range(3..=9usize)
            };
            let mut polys: Vec<Polynomial> = (0..n)
                .map(|_| {
                    let terms = rng.random_range(1..=5usize);
                    (0..terms)
                        .map(|_| rng.random_range(0..(1u64 << n)))
                        .collect()
                })
                .collect();
            if wide {
                // Guarantee a degree > 16 monomial so compact_ok is false.
                let w = rng.random_range(0..n);
                polys[w].push((1u64 << n) - 1);
            }
            let (canon_a, _) = canonicalize_polys_4(polys.clone(), true).unwrap();
            let (canon_again, _) = canonicalize_polys_4(polys.clone(), true).unwrap();
            assert_eq!(canon_a, canon_again, "determinism trial={trial} n={n}");

            let mut sigma: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.random_range(0..=i);
                sigma.swap(i, j);
            }
            let mut relabeled: Vec<Polynomial> = vec![Vec::new(); n];
            for w in 0..n {
                relabeled[sigma[w]] = polys[w]
                    .iter()
                    .map(|&m| {
                        let mut r = 0u64;
                        let mut mm = m;
                        while mm != 0 {
                            let v = mm.trailing_zeros() as usize;
                            r |= 1u64 << sigma[v];
                            mm &= mm - 1;
                        }
                        r
                    })
                    .collect();
            }
            let (canon_b, _) = canonicalize_polys_4(relabeled, true).unwrap();
            assert_eq!(
                canon_a, canon_b,
                "relabeling trial={trial} n={n} sigma={sigma:?}"
            );
        }
    }
}

// Choose the smallest lexigraphical ordering
impl CircuitSeq {
    pub fn canonicalize(&mut self) {
        for i in 1..self.gates.len() {
            //index in base_gates of current gate
            let gi_index = self.gates[i];
            let mut to_swap: Option<usize> = None;

            let mut j = i;
            while j > 0 {
                j -= 1;
                let gj_index = self.gates[j];

                if Gate::collides_index(&gi_index, &gj_index) {
                    break;
                } else if !Gate::ordered_index(&gj_index, &gi_index) {
                    to_swap = Some(j);
                }
            }
            if let Some(pos) = to_swap {
                let g = self.gates[i];
                self.gates.remove(i);
                self.gates.insert(pos, g);
            }
        }
    }
}
