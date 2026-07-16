//! Compression-only provenance for loose post-mix fragments.
//!
//! The circuit representation intentionally stays free of accounting data.  A
//! [`ProvenanceArena`] instead assigns an implicit leaf to every uncomplemented
//! gate in the input tape and gives the compressor one compact [`ProvId`] per
//! live gate.  Pairwise rewrites add exact-union nodes.  ANF and frozen-database
//! rewrites add group-boundary nodes because their replacement gates do not
//! have a canonical one-to-one ancestry.
//!
//! Resolving output roots therefore gives two useful, deduplicated quantities:
//! an exact lower bound which stops at group boundaries, and an inclusive
//! group-attributed count which traverses them.  Leaves and internal nodes both
//! use `u32` identifiers; no per-gate fragment set is materialized.

use super::xgate::XGate;

/// Compact handle to one provenance leaf or internal arena node.
///
/// Plain-input leaves occupy `0..input_plain_fragments`.  Internal nodes follow
/// them, and [`ProvId::EMPTY`] is reserved for gates with no loose-fragment
/// ancestry (notably structural g57s already present in the input).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct ProvId(u32);

impl ProvId {
    pub const EMPTY: Self = Self(u32::MAX);

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == Self::EMPTY.0
    }

    /// Stable numeric representation, useful in compact trace sidecars.
    #[inline]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Exact pairwise operation which produced a lineage node.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathKind {
    DirectPass1,
    Catalogue,
    DirectLater,
    Parity,
}

/// Rewrite boundary at which gate-by-gate ancestry ceases to be canonical.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupKind {
    Anf,
    Database,
}

/// One method recorded anywhere below a provenance root.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MethodKind {
    DirectPass1,
    Catalogue,
    DirectLater,
    Anf,
    Database,
    Parity,
}

/// Cumulative compression methods in a provenance subtree.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct MethodMask(u8);

impl MethodMask {
    const DIRECT_PASS1: u8 = 1 << 0;
    const CATALOGUE: u8 = 1 << 1;
    const DIRECT_LATER: u8 = 1 << 2;
    const ANF: u8 = 1 << 3;
    const DATABASE: u8 = 1 << 4;
    const PARITY: u8 = 1 << 5;

    #[inline]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline]
    pub const fn contains(self, method: MethodKind) -> bool {
        self.0 & method_bit(method) != 0
    }

    #[inline]
    const fn with(self, method: MethodKind) -> Self {
        Self(self.0 | method_bit(method))
    }

    #[inline]
    const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

const fn method_bit(method: MethodKind) -> u8 {
    match method {
        MethodKind::DirectPass1 => MethodMask::DIRECT_PASS1,
        MethodKind::Catalogue => MethodMask::CATALOGUE,
        MethodKind::DirectLater => MethodMask::DIRECT_LATER,
        MethodKind::Anf => MethodMask::ANF,
        MethodKind::Database => MethodMask::DATABASE,
        MethodKind::Parity => MethodMask::PARITY,
    }
}

const fn path_method(path: PathKind) -> MethodKind {
    match path {
        PathKind::DirectPass1 => MethodKind::DirectPass1,
        PathKind::Catalogue => MethodKind::Catalogue,
        PathKind::DirectLater => MethodKind::DirectLater,
        PathKind::Parity => MethodKind::Parity,
    }
}

const fn group_method(group: GroupKind) -> MethodKind {
    match group {
        GroupKind::Anf => MethodKind::Anf,
        GroupKind::Database => MethodKind::Database,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NodeKind {
    Exact,
    Group(GroupKind),
}

/// Fixed-size internal node.  Large fragment sets are represented by edges,
/// never copied into individual gates.
#[derive(Clone, Copy, Debug)]
struct Node {
    left: ProvId,
    right: ProvId,
    methods: MethodMask,
    kind: NodeKind,
}

/// Deduplicated input-fragment coverage, including the fragments' widths at
/// the start of compression.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FragmentCoverage {
    pub total: u64,
    /// `by_initial_width[w]` is the number of covered input fragments that had
    /// exactly `w` controls.  Trailing zero buckets are retained so results from
    /// the same arena have a stable shape.
    pub by_initial_width: Vec<u64>,
}

/// Exact and group-attributed coverage for a set of current/event roots.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResolvedCoverage {
    /// Provenance reachable without crossing an ANF or database boundary.
    pub exact: FragmentCoverage,
    /// Provenance reachable when group/window participation is attributed to
    /// every replacement gate produced by that rewrite.
    pub inclusive: FragmentCoverage,
    /// Union of all method flags below the supplied roots.
    pub methods: MethodMask,
}

/// Compact provenance DAG for one compression run.
#[derive(Clone, Debug)]
pub struct ProvenanceArena {
    leaf_widths: Vec<u16>,
    input_width_histogram: Vec<u64>,
    nodes: Vec<Node>,
}

impl ProvenanceArena {
    /// Initialize provenance at the `fcompress` input boundary.
    ///
    /// The returned sidecar is position-aligned with `gates`.  Every plain gate
    /// gets a distinct implicit leaf, while complemented gates get
    /// [`ProvId::EMPTY`] and therefore do not enter the loose-fragment
    /// denominator.
    pub fn from_gates(gates: &[XGate]) -> (Self, Vec<ProvId>) {
        let plain = gates.iter().filter(|gate| !gate.comp).count();
        assert!(
            plain < u32::MAX as usize,
            "too many loose input fragments for u32 provenance ids"
        );

        let mut leaf_widths = Vec::with_capacity(plain);
        let mut roots = Vec::with_capacity(gates.len());
        let mut input_width_histogram = Vec::<u64>::new();
        for gate in gates {
            if gate.comp {
                roots.push(ProvId::EMPTY);
                continue;
            }
            let id =
                ProvId(u32::try_from(leaf_widths.len()).expect("loose-fragment id exceeds u32"));
            let width = u16::try_from(gate.width())
                .expect("an XGate cannot have more than u16::MAX distinct controls");
            leaf_widths.push(width);
            let bucket = width as usize;
            if input_width_histogram.len() <= bucket {
                input_width_histogram.resize(bucket + 1, 0);
            }
            input_width_histogram[bucket] += 1;
            roots.push(id);
        }

        (
            Self {
                leaf_widths,
                input_width_histogram,
                nodes: Vec::new(),
            },
            roots,
        )
    }

    #[inline]
    pub fn input_plain_fragments(&self) -> usize {
        self.leaf_widths.len()
    }

    #[inline]
    pub fn input_width_histogram(&self) -> &[u64] {
        &self.input_width_histogram
    }

    #[inline]
    pub fn internal_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Record an exact pairwise rewrite.  Unary nodes are deliberately retained
    /// when one side is empty so method metadata (for example parity combined
    /// with a plain-derived cube) is not lost.
    pub fn exact_union(&mut self, left: ProvId, right: ProvId, path: PathKind) -> ProvId {
        if left.is_empty() && right.is_empty() {
            return ProvId::EMPTY;
        }
        self.assert_valid(left);
        self.assert_valid(right);
        let methods = self
            .methods(left)
            .union(self.methods(right))
            .with(path_method(path));
        self.push_node(Node {
            left,
            right,
            methods,
            kind: NodeKind::Exact,
        })
    }

    /// Attribute a non-canonical replacement group/window to all of its input
    /// roots.  Empty roots are ignored.  The returned boundary is opaque to
    /// exact resolution and transparent to inclusive resolution.
    pub fn group_union<I>(&mut self, roots: I, group: GroupKind) -> ProvId
    where
        I: IntoIterator<Item = ProvId>,
    {
        let mut level: Vec<ProvId> = roots.into_iter().filter(|root| !root.is_empty()).collect();
        if level.is_empty() {
            return ProvId::EMPTY;
        }
        for &root in &level {
            self.assert_valid(root);
        }

        // A balanced internal union keeps traversal depth logarithmic even if a
        // future compressor uses much larger groups than today's caps.
        while level.len() > 1 {
            let mut next = Vec::with_capacity(level.len().div_ceil(2));
            for pair in level.chunks(2) {
                if pair.len() == 1 {
                    next.push(pair[0]);
                } else {
                    next.push(self.aggregate(pair[0], pair[1]));
                }
            }
            level = next;
        }
        let child = level[0];
        let methods = self.methods(child).with(group_method(group));
        self.push_node(Node {
            left: child,
            right: ProvId::EMPTY,
            methods,
            kind: NodeKind::Group(group),
        })
    }

    /// Cumulative method metadata below `root` in O(1).
    #[inline]
    pub fn methods(&self, root: ProvId) -> MethodMask {
        if root.is_empty() || self.is_leaf(root) {
            return MethodMask::default();
        }
        self.node(root).methods
    }

    /// Resolve one or more current gates or recorded recovery-event roots.
    /// Input fragment IDs are deduplicated across all roots.
    pub fn resolve<I>(&self, roots: I) -> ResolvedCoverage
    where
        I: IntoIterator<Item = ProvId>,
    {
        let roots: Vec<ProvId> = roots.into_iter().collect();
        for &root in &roots {
            self.assert_valid(root);
        }
        let methods = roots.iter().fold(MethodMask::default(), |mask, &root| {
            mask.union(self.methods(root))
        });
        let exact_bits = self.resolve_bits(&roots, false);
        let inclusive_bits = self.resolve_bits(&roots, true);
        ResolvedCoverage {
            exact: self.summarize(&exact_bits),
            inclusive: self.summarize(&inclusive_bits),
            methods,
        }
    }

    #[inline]
    fn is_leaf(&self, id: ProvId) -> bool {
        !id.is_empty() && (id.0 as usize) < self.leaf_widths.len()
    }

    fn aggregate(&mut self, left: ProvId, right: ProvId) -> ProvId {
        debug_assert!(!left.is_empty() && !right.is_empty());
        let methods = self.methods(left).union(self.methods(right));
        self.push_node(Node {
            left,
            right,
            methods,
            kind: NodeKind::Exact,
        })
    }

    fn push_node(&mut self, node: Node) -> ProvId {
        let raw = self
            .leaf_widths
            .len()
            .checked_add(self.nodes.len())
            .and_then(|id| u32::try_from(id).ok())
            .filter(|&id| id != ProvId::EMPTY.0)
            .expect("provenance arena exhausted u32 ids");
        self.nodes.push(node);
        ProvId(raw)
    }

    fn node(&self, id: ProvId) -> &Node {
        self.assert_valid(id);
        assert!(!self.is_leaf(id) && !id.is_empty());
        &self.nodes[id.0 as usize - self.leaf_widths.len()]
    }

    fn assert_valid(&self, id: ProvId) {
        if id.is_empty() {
            return;
        }
        let limit = self.leaf_widths.len() + self.nodes.len();
        assert!((id.0 as usize) < limit, "invalid provenance id {}", id.0);
    }

    fn resolve_bits(&self, roots: &[ProvId], cross_groups: bool) -> Vec<u64> {
        let mut leaves = vec![0u64; self.leaf_widths.len().div_ceil(64)];
        let mut seen_nodes = vec![0u64; self.nodes.len().div_ceil(64)];
        let mut stack = Vec::<ProvId>::new();

        // Drain each root immediately, avoiding a second roots-sized worklist
        // when millions of output gates are supplied.
        for &root in roots {
            stack.push(root);
            while let Some(id) = stack.pop() {
                if id.is_empty() {
                    continue;
                }
                if self.is_leaf(id) {
                    set_bit(&mut leaves, id.0 as usize);
                    continue;
                }
                let node_index = id.0 as usize - self.leaf_widths.len();
                if test_and_set(&mut seen_nodes, node_index) {
                    continue;
                }
                let node = &self.nodes[node_index];
                if matches!(node.kind, NodeKind::Group(_)) && !cross_groups {
                    continue;
                }
                stack.push(node.left);
                stack.push(node.right);
            }
        }
        leaves
    }

    fn summarize(&self, bits: &[u64]) -> FragmentCoverage {
        let mut by_initial_width = vec![0u64; self.input_width_histogram.len()];
        let mut total = 0u64;
        for (word_index, &word) in bits.iter().enumerate() {
            let mut live = word;
            while live != 0 {
                let bit = live.trailing_zeros() as usize;
                live &= live - 1;
                let leaf = word_index * 64 + bit;
                if leaf >= self.leaf_widths.len() {
                    break;
                }
                total += 1;
                by_initial_width[self.leaf_widths[leaf] as usize] += 1;
            }
        }
        FragmentCoverage {
            total,
            by_initial_width,
        }
    }
}

#[inline]
fn set_bit(words: &mut [u64], bit: usize) {
    words[bit / 64] |= 1u64 << (bit % 64);
}

/// Return true when the bit was already set.
#[inline]
fn test_and_set(words: &mut [u64], bit: usize) -> bool {
    let mask = 1u64 << (bit % 64);
    let word = &mut words[bit / 64];
    let old = *word & mask != 0;
    *word |= mask;
    old
}

#[cfg(test)]
mod tests {
    use super::*;

    fn conj(target: u16, lits: &[(u16, bool)]) -> XGate {
        XGate::conj(target, lits.iter().copied()).unwrap()
    }

    fn histogram(entries: &[(usize, u64)]) -> Vec<u64> {
        let mut out = vec![0; entries.iter().map(|&(width, _)| width).max().unwrap_or(0) + 1];
        for &(width, count) in entries {
            out[width] = count;
        }
        out
    }

    #[test]
    fn initialization_assigns_only_plain_input_leaves() {
        let gates = vec![
            conj(0, &[]),
            XGate::from_g57([0, 1, 2]),
            conj(0, &[(1, true)]),
            conj(0, &[(1, false), (2, false), (3, true)]),
        ];
        let (arena, roots) = ProvenanceArena::from_gates(&gates);
        assert_eq!(arena.input_plain_fragments(), 3);
        assert_eq!(roots[0].as_u32(), 0);
        assert_eq!(roots[1], ProvId::EMPTY);
        assert_eq!(roots[2].as_u32(), 1);
        assert_eq!(roots[3].as_u32(), 2);
        assert_eq!(
            arena.input_width_histogram(),
            histogram(&[(0, 1), (1, 1), (3, 1)])
        );
    }

    #[test]
    fn exact_cascade_deduplicates_leaves_and_preserves_initial_widths() {
        let gates = vec![
            conj(0, &[(1, true)]),
            conj(0, &[(1, false), (2, false), (3, true)]),
            conj(0, &[(1, false), (2, false), (3, false)]),
        ];
        let (mut arena, roots) = ProvenanceArena::from_gates(&gates);
        let two_cc = arena.exact_union(roots[1], roots[2], PathKind::Catalogue);
        let g57 = arena.exact_union(roots[0], two_cc, PathKind::DirectLater);
        let coverage = arena.resolve([g57, g57]);

        assert_eq!(coverage.exact.total, 3);
        assert_eq!(coverage.exact, coverage.inclusive);
        assert_eq!(
            coverage.exact.by_initial_width,
            histogram(&[(1, 1), (3, 2)])
        );
        assert!(coverage.methods.contains(MethodKind::Catalogue));
        assert!(coverage.methods.contains(MethodKind::DirectLater));
        assert!(!coverage.methods.contains(MethodKind::DirectPass1));
    }

    #[test]
    fn group_boundary_separates_exact_from_inclusive_attribution() {
        let gates = vec![
            conj(0, &[(1, true)]),
            conj(0, &[(2, true), (3, false)]),
            conj(0, &[(4, true), (5, false), (6, true)]),
        ];
        let (mut arena, roots) = ProvenanceArena::from_gates(&gates);
        let anf = arena.group_union([roots[1], roots[2]], GroupKind::Anf);
        let final_g57 = arena.exact_union(roots[0], anf, PathKind::DirectLater);
        let coverage = arena.resolve([final_g57]);

        assert_eq!(coverage.exact.total, 1);
        assert_eq!(
            coverage.exact.by_initial_width,
            histogram(&[(1, 1), (2, 0), (3, 0)])
        );
        assert_eq!(coverage.inclusive.total, 3);
        assert_eq!(
            coverage.inclusive.by_initial_width,
            histogram(&[(1, 1), (2, 1), (3, 1)])
        );
        assert!(coverage.methods.contains(MethodKind::Anf));
        assert!(coverage.methods.contains(MethodKind::DirectLater));
    }

    #[test]
    fn shared_database_group_is_counted_once_across_many_outputs() {
        let gates = vec![
            conj(0, &[(1, true)]),
            conj(0, &[(1, false), (2, false)]),
            XGate::from_g57([3, 4, 5]),
        ];
        let (mut arena, roots) = ProvenanceArena::from_gates(&gates);
        let db = arena.group_union(roots, GroupKind::Database);
        let coverage = arena.resolve([db, db, db]);

        assert_eq!(coverage.exact.total, 0);
        assert_eq!(coverage.inclusive.total, 2);
        assert_eq!(
            coverage.inclusive.by_initial_width,
            histogram(&[(1, 1), (2, 1)])
        );
        assert!(coverage.methods.contains(MethodKind::Database));
    }

    #[test]
    fn metadata_records_every_requested_compression_path() {
        let gates = vec![conj(0, &[(1, true)]), conj(0, &[(2, false)])];
        let (mut arena, roots) = ProvenanceArena::from_gates(&gates);
        let pass1 = arena.exact_union(roots[0], ProvId::EMPTY, PathKind::DirectPass1);
        let catalogue = arena.exact_union(pass1, roots[1], PathKind::Catalogue);
        let parity = arena.exact_union(catalogue, ProvId::EMPTY, PathKind::Parity);
        let anf = arena.group_union([parity], GroupKind::Anf);
        let later = arena.exact_union(anf, ProvId::EMPTY, PathKind::DirectLater);
        let db = arena.group_union([later], GroupKind::Database);
        let methods = arena.methods(db);

        for method in [
            MethodKind::DirectPass1,
            MethodKind::Catalogue,
            MethodKind::DirectLater,
            MethodKind::Anf,
            MethodKind::Database,
            MethodKind::Parity,
        ] {
            assert!(methods.contains(method), "missing {method:?}");
        }
    }

    #[test]
    fn empty_ancestry_stays_empty() {
        let gates = vec![XGate::from_g57([0, 1, 2])];
        let (mut arena, roots) = ProvenanceArena::from_gates(&gates);
        assert_eq!(roots, vec![ProvId::EMPTY]);
        assert_eq!(
            arena.exact_union(ProvId::EMPTY, ProvId::EMPTY, PathKind::Parity),
            ProvId::EMPTY
        );
        assert_eq!(
            arena.group_union([ProvId::EMPTY], GroupKind::Anf),
            ProvId::EMPTY
        );
        assert_eq!(arena.resolve([ProvId::EMPTY]), ResolvedCoverage::default());
    }
}
