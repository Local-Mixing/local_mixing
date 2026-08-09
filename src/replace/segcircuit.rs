//! Segmented circuit for Stage C (many short, adaptively-located shooting passes).
//!
//! A flat `Vec<gate>` makes every local edit O(N) (the splice shifts the whole tail) and every
//! min-generation anchor lookup O(N) (a full scan). For the Stage-C strategy — thousands of short
//! passes at adaptive locations — that overhead dominates. This structure stores the circuit as a
//! list of bounded-size chunks so that:
//!   * a local edit (replace a window inside one region) touches only its chunk — O(chunk);
//!   * a minimum-generation anchor is found via cached per-chunk minima — O(#chunks + chunk).
//!
//! Each chunk will later also carry a *pending inbound SAMF* (the lazy global-propagation tail);
//! that field is added in the next step. This step is just the storage + edits + gen lookup.

use crate::replace::replace::Tag;
use rand::Rng;

const TARGET_CHUNK: usize = 1024; // soft target size; chunks split above 2x, merge below 1/2x.

/// Compact accumulated SAMF state on `n` wires: a wire PERMUTATION plus a pending per-wire
/// NEGATION mask. This is the lazy "tail" each chunk will carry — equivalent to the existing
/// `Transpositions` + negation_mask representation, but stored as fixed n-size arrays so that
/// composing two tails is O(n) (rather than concatenating ever-growing transposition lists) and
/// relabeling a gate is O(1).
///
/// Convention matches the existing code: `perm[w]` is where wire `w` currently lives (i.e.
/// `Transpositions::evaluate(w)`); composing tail A (applied first) then B gives
/// `perm[w] = B.perm[A.perm[w]]`, and B's negation plus A's negation transported through B.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamfTail {
    pub perm: Vec<u16>,
    pub neg: Vec<u8>,
}

impl SamfTail {
    pub fn identity(n: usize) -> Self {
        SamfTail {
            perm: (0..n as u16).collect(),
            neg: vec![0u8; n],
        }
    }

    pub fn is_identity(&self) -> bool {
        self.neg.iter().all(|&b| b == 0) && self.perm.iter().enumerate().all(|(i, &p)| p as usize == i)
    }

    /// Build a compact tail from the existing (Transpositions, negation_mask) representation.
    pub fn from_transpositions(
        t: &crate::replace::transpositions::Transpositions,
        neg: &[u8],
        n: usize,
    ) -> Self {
        SamfTail {
            perm: (0..n as u16).map(|w| t.evaluate(w)).collect(),
            neg: neg.to_vec(),
        }
    }

    /// Compose: `self` applied first, then `other`. Matches `self_t.concat(&other_t)` for perms
    /// and the code's negation-combination (other's neg + self's neg transported through other).
    pub fn then(&self, other: &SamfTail) -> SamfTail {
        let n = self.perm.len();
        let perm: Vec<u16> = (0..n)
            .map(|w| other.perm[self.perm[w] as usize])
            .collect();
        let mut neg = other.neg.clone();
        for w in 0..n {
            if self.neg[w] == 1 {
                neg[other.perm[w] as usize] ^= 1;
            }
        }
        SamfTail { perm, neg }
    }

    /// The inverse tail: `self.then(&self.invert())` and `self.invert().then(self)` are identity.
    pub fn invert(&self) -> SamfTail {
        let n = self.perm.len();
        let mut perm = vec![0u16; n];
        for w in 0..n {
            perm[self.perm[w] as usize] = w as u16;
        }
        // inv.neg[u] = self.neg[self.perm[u]] (negation transported so the composition cancels).
        let neg: Vec<u8> = (0..n).map(|u| self.neg[self.perm[u] as usize]).collect();
        SamfTail { perm, neg }
    }

    /// Decompose this tail's permutation into a list of transpositions (neg_type 0) whose ordered
    /// application (`Transpositions::evaluate`) reproduces the permutation. Lets a SamfTail be
    /// folded into the shoot's `Transpositions` t_list (the negation rides separately in the mask).
    pub fn perm_to_swaps(&self) -> Vec<(u16, u16, u16)> {
        let n = self.perm.len();
        let mut visited = vec![false; n];
        let mut swaps = Vec::new();
        for start in 0..n {
            if visited[start] {
                continue;
            }
            if self.perm[start] as usize == start {
                visited[start] = true;
                continue;
            }
            // Trace the cycle start -> perm[start] -> ...; emit (c0,c1),(c0,c2),...,(c0,ck).
            let mut cycle = vec![start];
            visited[start] = true;
            let mut cur = self.perm[start] as usize;
            while cur != start {
                cycle.push(cur);
                visited[cur] = true;
                cur = self.perm[cur] as usize;
            }
            for j in 1..cycle.len() {
                swaps.push((cycle[0] as u16, cycle[j] as u16, 0));
            }
        }
        swaps
    }

    /// Relabel a gate's three wires by the permutation (does NOT apply negations — those are
    /// flushed as NOT gates when a wire is read, exactly as the existing code does).
    #[inline]
    pub fn relabel(&self, gate: [u16; 3]) -> [u16; 3] {
        [
            self.perm[gate[0] as usize],
            self.perm[gate[1] as usize],
            self.perm[gate[2] as usize],
        ]
    }
}

struct Chunk {
    gates: Vec<[u16; 3]>,
    tags: Vec<Tag>,
    min_gen: u32, // cached minimum generation of `tags` (u32::MAX when empty)
}

impl Chunk {
    fn new(gates: Vec<[u16; 3]>, tags: Vec<Tag>) -> Self {
        let min_gen = tags.iter().map(|t| t.generation()).min().unwrap_or(u32::MAX);
        Chunk {
            gates,
            tags,
            min_gen,
        }
    }
    fn recompute_min(&mut self) {
        self.min_gen = self.tags.iter().map(|t| t.generation()).min().unwrap_or(u32::MAX);
    }
    fn len(&self) -> usize {
        self.gates.len()
    }
}

pub struct SegCircuit {
    chunks: Vec<Chunk>,
    total: usize,
    // Cumulative chunk-start offsets: offsets[i] = global index of chunks[i]'s first gate.
    // Rebuilt (O(#chunks)) after any chunk-length mutation — i.e. at every splice exit and on
    // construction — so `locate` is O(log #chunks) via partition_point instead of a linear scan.
    offsets: Vec<usize>,
}

impl SegCircuit {
    /// Build from a flat circuit + parallel tags (tags must match length).
    pub fn from_flat(gates: &[[u16; 3]], tags: &[Tag]) -> Self {
        assert_eq!(gates.len(), tags.len(), "gates/tags length mismatch");
        let mut chunks = Vec::new();
        let mut i = 0;
        while i < gates.len() {
            let end = (i + TARGET_CHUNK).min(gates.len());
            chunks.push(Chunk::new(gates[i..end].to_vec(), tags[i..end].to_vec()));
            i = end;
        }
        let mut out = SegCircuit {
            chunks,
            total: gates.len(),
            offsets: Vec::new(),
        };
        out.rebuild_offsets();
        out
    }

    /// Recompute the cumulative chunk-start offsets after any chunk-length mutation.
    fn rebuild_offsets(&mut self) {
        self.offsets.clear();
        self.offsets.reserve(self.chunks.len());
        let mut acc = 0usize;
        for c in &self.chunks {
            self.offsets.push(acc);
            acc += c.len();
        }
    }

    /// Flatten back to a single gate list + tag list.
    pub fn to_flat(&self) -> (Vec<[u16; 3]>, Vec<Tag>) {
        let mut gates = Vec::with_capacity(self.total);
        let mut tags = Vec::with_capacity(self.total);
        for c in &self.chunks {
            gates.extend_from_slice(&c.gates);
            tags.extend_from_slice(&c.tags);
        }
        (gates, tags)
    }

    pub fn len(&self) -> usize {
        self.total
    }

    pub fn is_empty(&self) -> bool {
        self.total == 0
    }

    /// Locate a global index: returns (chunk index, offset within chunk). Panics if out of range.
    /// O(log #chunks) binary search over the cumulative chunk-start offsets.
    fn locate(&self, index: usize) -> (usize, usize) {
        if index < self.total {
            // First chunk whose start is > index, minus one; safe because offsets[0] == 0.
            let ci = self.offsets.partition_point(|&start| start <= index) - 1;
            debug_assert!(index - self.offsets[ci] < self.chunks[ci].len());
            return (ci, index - self.offsets[ci]);
        }
        panic!("index {index} out of range (len {})", self.total);
    }

    /// Original linear-scan locate, kept as the test oracle for the offsets-based version.
    #[cfg(test)]
    fn locate_reference(&self, index: usize) -> (usize, usize) {
        let mut acc = 0;
        for (ci, c) in self.chunks.iter().enumerate() {
            if index < acc + c.len() {
                return (ci, index - acc);
            }
            acc += c.len();
        }
        panic!("index {index} out of range (len {})", self.total);
    }

    /// The minimum generation across all gates (u32::MAX if empty).
    pub fn min_gen(&self) -> u32 {
        self.chunks
            .iter()
            .map(|c| c.min_gen)
            .min()
            .unwrap_or(u32::MAX)
    }

    /// Pick a uniformly random gate whose generation equals the current global minimum, returning
    /// its global index. Cost: O(#chunks) to find min-gen chunks + O(chunk) to sample within.
    pub fn random_min_gen_index(&self, rng: &mut impl Rng) -> Option<usize> {
        if self.total == 0 {
            return None;
        }
        let m = self.min_gen();
        // Reservoir-sample one position with tag == m across all chunks, tracking the global base.
        let mut chosen: Option<usize> = None;
        let mut seen = 0usize;
        let mut base = 0usize;
        for c in &self.chunks {
            if c.min_gen == m {
                for (off, &t) in c.tags.iter().enumerate() {
                    if t.generation() == m {
                        seen += 1;
                        if rng.random_range(0..seen) == 0 {
                            chosen = Some(base + off);
                        }
                    }
                }
            }
            base += c.len();
        }
        chosen
    }

    /// "Fractional floor" generation: the smallest generation `g` such that the number of gates
    /// with generation `< g` is `<= skip`. With `skip = floor((1-frac)*total)` this is the lowest
    /// generation once the bottom `(1-frac)` of gates (e.g. permanently-stuck, never-colliding ones)
    /// are written off. `skip = 0` reduces to the absolute minimum. O(N) + O(#distinct gens).
    pub fn frac_min_gen(&self, skip: usize) -> u32 {
        if self.total == 0 {
            return 0;
        }
        let mut hist: std::collections::BTreeMap<u32, usize> = std::collections::BTreeMap::new();
        for c in &self.chunks {
            for &t in &c.tags {
                *hist.entry(t.generation()).or_insert(0) += 1;
            }
        }
        let mut acc = 0usize;
        let mut last = 0u32;
        for (g, count) in hist {
            last = g;
            acc += count;
            if acc > skip {
                return g;
            }
        }
        last
    }

    /// Uniformly random gate whose generation tag == `g` (global index), or None if there is none.
    pub fn random_index_at_gen(&self, g: u32, rng: &mut impl Rng) -> Option<usize> {
        let mut chosen: Option<usize> = None;
        let mut seen = 0usize;
        let mut base = 0usize;
        for c in &self.chunks {
            if c.min_gen <= g {
                for (off, &t) in c.tags.iter().enumerate() {
                    if t.generation() == g {
                        seen += 1;
                        if rng.random_range(0..seen) == 0 {
                            chosen = Some(base + off);
                        }
                    }
                }
            }
            base += c.len();
        }
        chosen
    }

    /// Anchor at the fractional floor: pick a random gate at generation `frac_min_gen(skip)`. With
    /// `skip = 0` this is exactly `random_min_gen_index`; with `skip > 0` it skips the stuck bottom.
    pub fn random_frac_min_gen_index(&self, skip: usize, rng: &mut impl Rng) -> Option<usize> {
        if self.total == 0 {
            return None;
        }
        if skip == 0 {
            return self.random_min_gen_index(rng);
        }
        let g = self.frac_min_gen(skip);
        self.random_index_at_gen(g, rng)
    }

    /// The frac-min-gen level `g` together with ALL global indices whose generation == g, in a
    /// single O(n) scan. Lets a caller pick several spaced anchors per batch (SHOOT_PARALLEL)
    /// without paying an O(n) `random_frac_min_gen_index` per anchor.
    pub fn frac_min_gen_indices(&self, skip: usize) -> (u32, Vec<usize>) {
        if self.total == 0 {
            return (0, Vec::new());
        }
        let g = if skip == 0 {
            self.min_gen()
        } else {
            self.frac_min_gen(skip)
        };
        let mut idxs = Vec::new();
        let mut base = 0usize;
        for c in &self.chunks {
            if c.min_gen <= g {
                for (off, &t) in c.tags.iter().enumerate() {
                    if t.generation() == g {
                        idxs.push(base + off);
                    }
                }
            }
            base += c.len();
        }
        (g, idxs)
    }

    /// Read a contiguous range [start, start+len) as flat gate + tag vectors.
    pub fn read_range(&self, start: usize, len: usize) -> (Vec<[u16; 3]>, Vec<Tag>) {
        let mut g = Vec::with_capacity(len);
        let mut t = Vec::with_capacity(len);
        if len == 0 {
            return (g, t);
        }
        let (mut ci, mut off) = self.locate(start);
        let mut need = len;
        while need > 0 && ci < self.chunks.len() {
            let c = &self.chunks[ci];
            let take = (c.len() - off).min(need);
            g.extend_from_slice(&c.gates[off..off + take]);
            t.extend_from_slice(&c.tags[off..off + take]);
            need -= take;
            ci += 1;
            off = 0;
        }
        (g, t)
    }

    /// Count gates whose generation tag is < `target` (skips chunks already fully at/above it, so
    /// the cost shrinks toward O(#chunks) as the round converges).
    pub fn count_below(&self, target: u32) -> usize {
        let mut c = 0;
        for ch in &self.chunks {
            if ch.min_gen < target {
                c += ch.tags.iter().filter(|&&t| t.generation() < target).count();
            }
        }
        c
    }

    /// Count gates at generation 0 (for the [gen] progress log).
    pub fn count_zero(&self) -> usize {
        self.chunks
            .iter()
            .map(|ch| {
                if ch.min_gen == 0 {
                    ch.tags.iter().filter(|&&t| t.generation() == 0).count()
                } else {
                    0
                }
            })
            .sum()
    }

    /// Pick a uniformly random gate whose generation == `gen` and whose global index lies in
    /// [lo, hi), or None if there is none. Used for CENTRALIZE: prioritize central-band gates at
    /// generation (min_gen + 1). Cost O(circuit) but only invoked when CENTRALIZE is enabled.
    pub fn random_index_with_gen_in_range(
        &self,
        generation: u32,
        lo: usize,
        hi: usize,
        rng: &mut impl Rng,
    ) -> Option<usize> {
        let mut chosen: Option<usize> = None;
        let mut seen = 0usize;
        let mut base = 0usize;
        for c in &self.chunks {
            let cend = base + c.len();
            // skip chunks entirely outside [lo, hi) or with no gate at `generation`
            if cend > lo && base < hi && c.min_gen <= generation {
                for (off, &t) in c.tags.iter().enumerate() {
                    let idx = base + off;
                    if idx >= lo && idx < hi && t.generation() == generation {
                        seen += 1;
                        if rng.random_range(0..seen) == 0 {
                            chosen = Some(idx);
                        }
                    }
                }
            }
            base = cend;
        }
        chosen
    }

    /// Read the gate + tag at a global index.
    pub fn get(&self, index: usize) -> ([u16; 3], Tag) {
        let (ci, off) = self.locate(index);
        (self.chunks[ci].gates[off], self.chunks[ci].tags[off])
    }

    /// Replace the `remove_len` gates starting at `index` with `new_gates`/`new_tags`.
    /// Local edits inside one chunk are O(chunk); edits spanning chunks merge the span first.
    /// Oversized chunks are re-split afterwards so sizes stay bounded.
    pub fn splice(
        &mut self,
        index: usize,
        remove_len: usize,
        new_gates: &[[u16; 3]],
        new_tags: &[Tag],
    ) {
        assert_eq!(new_gates.len(), new_tags.len(), "gates/tags length mismatch");
        assert!(index + remove_len <= self.total, "splice out of range");
        if self.chunks.is_empty() {
            self.chunks
                .push(Chunk::new(new_gates.to_vec(), new_tags.to_vec()));
            self.total = new_gates.len();
            self.rebuild_offsets();
            return;
        }
        let (start_ci, start_off) = self.locate(index);
        // Find the end chunk/offset (index + remove_len). If remove_len == 0 it's an insertion.
        let end = index + remove_len;
        // Determine the last chunk the removed range touches.
        let mut end_ci = start_ci;
        let mut acc = index - start_off; // global base of start chunk
        while end_ci < self.chunks.len() && acc + self.chunks[end_ci].len() < end {
            acc += self.chunks[end_ci].len();
            end_ci += 1;
        }
        // end_off within end_ci
        let end_off = end - acc;

        if start_ci == end_ci {
            // Local edit within a single chunk.
            let c = &mut self.chunks[start_ci];
            c.gates.splice(start_off..end_off, new_gates.iter().copied());
            c.tags.splice(start_off..end_off, new_tags.iter().copied());
            c.recompute_min();
        } else {
            // Merge the touched chunks [start_ci, end_ci] into one, edit, then re-split.
            let mut mg: Vec<[u16; 3]> = Vec::new();
            let mut mt: Vec<Tag> = Vec::new();
            for ci in start_ci..=end_ci {
                mg.extend_from_slice(&self.chunks[ci].gates);
                mt.extend_from_slice(&self.chunks[ci].tags);
            }
            // local offsets within the merged buffer
            let merged_start = start_off;
            let merged_end = mg.len() - (self.chunks[end_ci].len() - end_off);
            mg.splice(merged_start..merged_end, new_gates.iter().copied());
            mt.splice(merged_start..merged_end, new_tags.iter().copied());
            let merged = Chunk::new(mg, mt);
            self.chunks.splice(start_ci..=end_ci, std::iter::once(merged));
        }
        // Update total and rebalance the (possibly) oversized/empty edited chunk.
        self.total = self.total - remove_len + new_gates.len();
        self.rebalance_around(start_ci);
        self.rebuild_offsets();
    }

    /// Split an oversized chunk into target-sized pieces; drop an empty chunk.
    fn rebalance_around(&mut self, ci: usize) {
        if ci >= self.chunks.len() {
            return;
        }
        if self.chunks[ci].len() == 0 {
            self.chunks.remove(ci);
            return;
        }
        if self.chunks[ci].len() > 2 * TARGET_CHUNK {
            let c = self.chunks.remove(ci);
            let mut pieces = Vec::new();
            let mut i = 0;
            while i < c.gates.len() {
                let end = (i + TARGET_CHUNK).min(c.gates.len());
                pieces.push(Chunk::new(c.gates[i..end].to_vec(), c.tags[i..end].to_vec()));
                i = end;
            }
            self.chunks.splice(ci..ci, pieces);
        }
    }
}

/// Ledger of UNRESOLVED SAMFs, each a wire-permutation+negation tail attached at the position it
/// starts applying from (it relabels every gate at that position and to its right). Kept sorted by
/// position.
///
/// Why this is enough (no balanced BST needed): a shooting pass over a span only has to resolve the
/// SAMFs whose positions fall *inside* its span. SAMFs entirely to the pass's left (or entirely to
/// its right) relabel everything the pass touches *uniformly*, and collision / curated-equivalence
/// are invariant under a uniform relabel — so the pass can work in the relative space that ignores
/// them. The crossed SAMFs are a contiguous range of this sorted ledger; the pass composes them
/// into its working tail as it reaches each position, and at the end replaces that whole range with
/// a single new entry (the crossed tails composed with the pass's own tail) at the pass's far edge.
#[derive(Clone, Debug, Default)]
pub struct SamfLedger {
    entries: Vec<(usize, SamfTail)>, // sorted ascending by position
}

impl SamfLedger {
    pub fn new() -> Self {
        SamfLedger {
            entries: Vec::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Insert a SAMF that starts applying at `pos` (keeps the ledger sorted; merges into an
    /// existing entry at the same position by composing — existing first, then the new one).
    pub fn insert(&mut self, pos: usize, tail: SamfTail) {
        match self.entries.binary_search_by_key(&pos, |e| e.0) {
            Ok(i) => {
                let combined = self.entries[i].1.then(&tail);
                self.entries[i].1 = combined;
            }
            Err(i) => self.entries.insert(i, (pos, tail)),
        }
    }

    /// The inbound tail for a FORWARD pass that started at `start`, evaluated at position `x`:
    /// compose (in position order) every ledger entry whose position is in `[start, x]`. Entries
    /// left of `start` are the ignorable uniform factor; entries right of `x` are not yet crossed.
    pub fn forward_inbound(&self, start: usize, x: usize, n: usize) -> SamfTail {
        let mut acc = SamfTail::identity(n);
        for (p, t) in &self.entries {
            if *p < start {
                continue;
            }
            if *p > x {
                break;
            }
            acc = acc.then(t);
        }
        acc
    }

    /// Remove and return all entries strictly inside (lo, hi), in position order (the entries a
    /// pass crossing the window (lo, hi) must resolve). Entries at exactly `lo` relabel the whole
    /// window uniformly and are left in place (the pass ignores them).
    pub fn drain_internal(&mut self, lo: usize, hi: usize) -> Vec<(usize, SamfTail)> {
        let mut out = Vec::new();
        self.entries.retain(|(p, t)| {
            if *p > lo && *p < hi {
                out.push((*p, t.clone()));
                false
            } else {
                true
            }
        });
        out
    }

    /// Position of the first entry strictly greater than `pos`, or `usize::MAX` if none (used to
    /// bound a window so it does not cross a ledger entry).
    pub fn next_after(&self, pos: usize) -> usize {
        self.entries
            .iter()
            .find(|(p, _)| *p > pos)
            .map(|(p, _)| *p)
            .unwrap_or(usize::MAX)
    }

    /// Position of the last entry strictly less than `pos`, or 0 if none.
    pub fn prev_before(&self, pos: usize) -> usize {
        self.entries
            .iter()
            .rev()
            .find(|(p, _)| *p < pos)
            .map(|(p, _)| *p)
            .unwrap_or(0)
    }

    /// Shift every entry position at/after `from` by `delta` (after a splice changed the length).
    pub fn shift_from(&mut self, from: usize, delta: isize) {
        for (p, _) in self.entries.iter_mut() {
            if *p >= from {
                *p = (*p as isize + delta) as usize;
            }
        }
    }

    /// Snapshot of all entries (for the final flush), in position order.
    pub fn entries(&self) -> &[(usize, SamfTail)] {
        &self.entries
    }

    /// Apply EVERY ledger entry to the suffix of `gates` at/after its position, in position order,
    /// producing the fully-resolved (true) permutation of the circuit. Permutation only (negations
    /// are handled by the end-of-round unsamf that emits NOT gates); used as the reference oracle
    /// in tests and, later, by the pre-compression sweep. Clears the ledger.
    pub fn materialize_perm(&mut self, gates: &mut [[u16; 3]]) {
        for (p, t) in &self.entries {
            for g in gates.iter_mut().skip(*p) {
                *g = t.relabel(*g);
            }
        }
        self.entries.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    fn flat(n: usize) -> (Vec<[u16; 3]>, Vec<Tag>) {
        let gates: Vec<[u16; 3]> = (0..n).map(|i| [i as u16, (i + 1) as u16, (i + 2) as u16]).collect();
        let tags: Vec<Tag> = (0..n).map(Tag::survivor).collect();
        (gates, tags)
    }

    #[test]
    fn roundtrip_flat() {
        for n in [0usize, 1, 5, 1023, 1024, 1025, 5000] {
            let (g, t) = flat(n);
            let sc = SegCircuit::from_flat(&g, &t);
            assert_eq!(sc.len(), n);
            let (g2, t2) = sc.to_flat();
            assert_eq!(g, g2);
            assert_eq!(t, t2);
        }
    }

    #[test]
    fn local_splice_matches_vec() {
        let (mut g, mut t) = flat(5000);
        let mut sc = SegCircuit::from_flat(&g, &t);
        // a local replacement well inside one chunk
        let idx = 1500;
        let new_g = vec![[9, 9, 9], [8, 8, 8], [7, 7, 7]];
        let new_t = vec![Tag(100), Tag(101), Tag(102)];
        sc.splice(idx, 2, &new_g, &new_t);
        g.splice(idx..idx + 2, new_g.iter().copied());
        t.splice(idx..idx + 2, new_t.iter().copied());
        let (g2, t2) = sc.to_flat();
        assert_eq!(g, g2);
        assert_eq!(t, t2);
        assert_eq!(sc.len(), g.len());
    }

    #[test]
    fn cross_chunk_splice_matches_vec() {
        let (mut g, mut t) = flat(5000);
        let mut sc = SegCircuit::from_flat(&g, &t);
        // a removal spanning a chunk boundary (~1024)
        let idx = 1000;
        let rem = 100; // crosses into the next chunk
        let new_g = vec![[1, 2, 3]];
        let new_t = vec![Tag(42)];
        sc.splice(idx, rem, &new_g, &new_t);
        g.splice(idx..idx + rem, new_g.iter().copied());
        t.splice(idx..idx + rem, new_t.iter().copied());
        let (g2, t2) = sc.to_flat();
        assert_eq!(g, g2);
        assert_eq!(t, t2);
    }

    #[test]
    fn big_insert_rebalances_and_matches() {
        let (mut g, mut t) = flat(100);
        let mut sc = SegCircuit::from_flat(&g, &t);
        let new_g: Vec<[u16; 3]> = (0..5000).map(|i| [i as u16, 0, 1]).collect();
        let new_t: Vec<Tag> = vec![Tag(7); 5000];
        sc.splice(50, 10, &new_g, &new_t);
        g.splice(50..60, new_g.iter().copied());
        t.splice(50..60, new_t.iter().copied());
        let (g2, t2) = sc.to_flat();
        assert_eq!(g, g2);
        assert_eq!(t, t2);
    }

    #[test]
    fn min_gen_and_anchor() {
        // tags with a unique minimum
        let g: Vec<[u16; 3]> = (0..3000).map(|i| [i as u16, 0, 1]).collect();
        let mut t: Vec<Tag> = vec![Tag(5); 3000];
        t[1234] = Tag(2); // unique min
        let sc = SegCircuit::from_flat(&g, &t);
        assert_eq!(sc.min_gen(), 2);
        let mut rng = StdRng::seed_from_u64(1);
        for _ in 0..20 {
            assert_eq!(sc.random_min_gen_index(&mut rng), Some(1234));
        }
    }

    use crate::replace::transpositions::Transpositions;

    fn rand_transpositions(n: usize, k: usize, rng: &mut impl Rng) -> (Transpositions, Vec<u8>) {
        let mut ts = Vec::new();
        for _ in 0..k {
            let a = rng.random_range(0..n) as u16;
            let mut b = rng.random_range(0..n) as u16;
            while b == a {
                b = rng.random_range(0..n) as u16;
            }
            ts.push((a, b, rng.random_range(0..4u16)));
        }
        let mut neg = vec![0u8; n];
        for w in 0..n {
            neg[w] = rng.random_range(0..2u8);
        }
        (Transpositions { transpositions: ts }, neg)
    }

    #[test]
    fn samftail_perm_matches_evaluate() {
        let n = 40;
        let mut rng = StdRng::seed_from_u64(0xa1);
        for _ in 0..50 {
            let (t, neg) = rand_transpositions(n, 30, &mut rng);
            let tail = SamfTail::from_transpositions(&t, &neg, n);
            for w in 0..n as u16 {
                assert_eq!(tail.perm[w as usize], t.evaluate(w));
            }
        }
    }

    // Combine two SAMF rounds the way the shooting driver does, returning the compact result.
    fn combine_like_code(
        ta: &Transpositions,
        na: &[u8],
        tb: &Transpositions,
        nb: &[u8],
        n: usize,
    ) -> (Transpositions, Vec<u8>) {
        let t = ta.concat(tb);
        let mut neg = nb.to_vec();
        for w in 0..n {
            if na[w] == 1 {
                neg[tb.evaluate(w as u16) as usize] ^= 1;
            }
        }
        (t, neg)
    }

    #[test]
    fn samftail_then_matches_concat() {
        let n = 40;
        let mut rng = StdRng::seed_from_u64(0xb2);
        for _ in 0..50 {
            let (ta, na) = rand_transpositions(n, 20, &mut rng);
            let (tb, nb) = rand_transpositions(n, 20, &mut rng);
            let a = SamfTail::from_transpositions(&ta, &na, n);
            let b = SamfTail::from_transpositions(&tb, &nb, n);
            let combined = a.then(&b);
            let (tc, nc) = combine_like_code(&ta, &na, &tb, &nb, n);
            let expect = SamfTail::from_transpositions(&tc, &nc, n);
            assert_eq!(combined, expect);
        }
    }

    fn rand_perm_tail(n: usize, rng: &mut impl Rng) -> SamfTail {
        let mut perm: Vec<u16> = (0..n as u16).collect();
        for i in (1..n).rev() {
            let j = rng.random_range(0..=i);
            perm.swap(i, j);
        }
        let neg: Vec<u8> = (0..n).map(|_| rng.random_range(0..2u8)).collect();
        SamfTail { perm, neg }
    }

    #[test]
    fn ledger_materialize_matches_per_gate_inbound() {
        let n = 20;
        let mut rng = StdRng::seed_from_u64(0xd4);
        for _ in 0..30 {
            let m = 200;
            let gates: Vec<[u16; 3]> = (0..m)
                .map(|_| {
                    [
                        rng.random_range(0..n) as u16,
                        rng.random_range(0..n) as u16,
                        rng.random_range(0..n) as u16,
                    ]
                })
                .collect();
            // ledger: a handful of perm-only entries at distinct sorted positions
            let mut ledger = SamfLedger::new();
            let mut positions: Vec<usize> = (0..8).map(|_| rng.random_range(0..m)).collect();
            positions.sort_unstable();
            positions.dedup();
            for &p in &positions {
                let mut t = rand_perm_tail(n, &mut rng);
                t.neg = vec![0u8; n]; // perm-only for materialize_perm
                ledger.insert(p, t);
            }
            // eager materialization
            let mut eager = gates.clone();
            ledger.clone().materialize_perm(&mut eager);
            // per-gate: relabel by compose of all entries with pos <= x  (= forward_inbound(0,x))
            for x in 0..m {
                let inbound = ledger.forward_inbound(0, x, n);
                assert_eq!(eager[x], inbound.relabel(gates[x]), "x={x}");
            }
        }
    }

    #[test]
    fn ledger_ignore_left_compose_crossed() {
        // T_start . forward_inbound(start, x) == forward_inbound(0, x): the uniform left factor
        // composes with the crossed factor to give the full inbound (the relabeling-invariance
        // that lets a pass ignore SAMFs left of its anchor).
        let n = 16;
        let mut rng = StdRng::seed_from_u64(0xe5);
        for _ in 0..30 {
            let m = 100;
            let mut ledger = SamfLedger::new();
            let mut positions: Vec<usize> = (0..10).map(|_| rng.random_range(0..m)).collect();
            positions.sort_unstable();
            positions.dedup();
            for &p in &positions {
                ledger.insert(p, rand_perm_tail(n, &mut rng));
            }
            for &start in &[0usize, 20, 50, 80] {
                let t_left = if start == 0 {
                    SamfTail::identity(n)
                } else {
                    ledger.forward_inbound(0, start - 1, n)
                };
                for x in start..m {
                    let combined = t_left.then(&ledger.forward_inbound(start, x, n));
                    assert_eq!(combined, ledger.forward_inbound(0, x, n), "start={start} x={x}");
                }
            }
        }
    }

    #[test]
    fn ledger_insert_sorted_and_merges() {
        let n = 8;
        let mut rng = StdRng::seed_from_u64(0xf6);
        let mut ledger = SamfLedger::new();
        let a = rand_perm_tail(n, &mut rng);
        let b = rand_perm_tail(n, &mut rng);
        ledger.insert(5, a.clone());
        ledger.insert(2, b.clone());
        ledger.insert(5, b.clone()); // merge into existing pos-5 entry: a then b
        assert_eq!(ledger.len(), 2);
        // pos 2 = b, pos 5 = a.then(b)
        assert_eq!(ledger.forward_inbound(2, 2, n), b);
        assert_eq!(ledger.forward_inbound(5, 5, n), a.then(&b));
    }

    #[test]
    fn perm_to_swaps_reproduces_perm() {
        let n = 40;
        let mut rng = StdRng::seed_from_u64(0x9a7);
        for _ in 0..100 {
            let tail = rand_perm_tail(n, &mut rng);
            let swaps = tail.perm_to_swaps();
            let t = Transpositions {
                transpositions: swaps,
            };
            for w in 0..n as u16 {
                assert_eq!(t.evaluate(w), tail.perm[w as usize]);
            }
        }
    }

    #[test]
    fn samftail_invert_cancels() {
        let n = 40;
        let mut rng = StdRng::seed_from_u64(0xc3);
        for _ in 0..50 {
            let (t, neg) = rand_transpositions(n, 25, &mut rng);
            let tail = SamfTail::from_transpositions(&t, &neg, n);
            let id = SamfTail::identity(n);
            assert_eq!(tail.then(&tail.invert()), id);
            assert_eq!(tail.invert().then(&tail), id);
        }
    }

    #[test]
    fn anchor_uniform_over_ties() {
        let g: Vec<[u16; 3]> = (0..1000).map(|i| [i as u16, 0, 1]).collect();
        let mut t: Vec<Tag> = vec![Tag(9); 1000];
        t[10] = Tag(1);
        t[500] = Tag(1);
        t[900] = Tag(1);
        let sc = SegCircuit::from_flat(&g, &t);
        let mut rng = StdRng::seed_from_u64(7);
        let mut counts = std::collections::HashMap::new();
        for _ in 0..3000 {
            let i = sc.random_min_gen_index(&mut rng).unwrap();
            assert!(i == 10 || i == 500 || i == 900);
            *counts.entry(i).or_insert(0) += 1;
        }
        // each tie should be hit a healthy fraction of the time
        for k in [10, 500, 900] {
            assert!(counts[&k] > 500, "tie {k} under-sampled: {:?}", counts);
        }
    }
}
