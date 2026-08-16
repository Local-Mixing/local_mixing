// Vec-backed doubly-linked list of gates. Node ids are stable across splices,
// so the episode worklist can hold (id, stamp) pairs; `stamp` increments when a
// slot is freed/reallocated, letting stale worklist entries be detected.
use super::xgate::XGate;
use rand::Rng;

pub const NIL: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    L,
    R,
}

impl Dir {
    pub fn opposite(self) -> Dir {
        match self {
            Dir::L => Dir::R,
            Dir::R => Dir::L,
        }
    }
}

/// Words in a collision mask: 2 covers wires 0..=127 (production runs at 128
/// wires). A gate touching any wire >= 64 * MASK_WORDS has no mask and poisons
/// the arena's fast path (`masks_ok` = false), falling back to
/// `XGate::collides` — same predicate, just slower.
pub const MASK_WORDS: usize = 2;

/// Per-node collision mask mirroring the node's gate, kept in sync by every
/// arena mutator that writes a gate (`from_gates`, `alloc`, `replace_gate`).
/// `mask_collides` computes exactly the `XGate::collides` predicate with three
/// mask operations instead of literal-list walks.
///
/// read: bit w set iff the gate has a control literal on wire w.
/// pol:  bit w set iff that literal is POSITIVE (meaningful only where read is set).
/// tgt:  the target wire; comp: complemented conjunction.
#[derive(Clone, Copy)]
pub struct GateMask {
    pub read: [u64; MASK_WORDS],
    pub pol: [u64; MASK_WORDS],
    pub tgt: u16,
    pub comp: bool,
}

impl GateMask {
    #[inline]
    pub fn of(g: &XGate) -> Option<GateMask> {
        const LIM: u16 = (64 * MASK_WORDS) as u16;
        if g.target >= LIM {
            return None;
        }
        let mut read = [0u64; MASK_WORDS];
        let mut pol = [0u64; MASK_WORDS];
        for &(w, p) in &g.ctrls {
            if w >= LIM {
                return None;
            }
            read[(w >> 6) as usize] |= 1u64 << (w & 63);
            if p {
                pol[(w >> 6) as usize] |= 1u64 << (w & 63);
            }
        }
        Some(GateMask {
            read,
            pol,
            tgt: g.target,
            comp: g.comp,
        })
    }

    #[inline]
    fn zero() -> GateMask {
        GateMask {
            read: [0; MASK_WORDS],
            pol: [0; MASK_WORDS],
            tgt: 0,
            comp: false,
        }
    }

    #[inline]
    fn reads(&self, w: u16) -> bool {
        (self.read[(w >> 6) as usize] >> (w & 63)) & 1 != 0
    }
}

pub struct Arena {
    gates: Vec<XGate>,
    prev: Vec<u32>,
    next: Vec<u32>,
    stamp: Vec<u32>,
    linked: Vec<bool>,
    head: u32,
    tail: u32,
    len: usize,
    free: Vec<u32>,
    // Parallel collision-mask array (see GateMask). masks_ok = false disables
    // the fast path (some wire >= 64 * MASK_WORDS appeared); collides_ids then
    // falls back to XGate::collides. With the Mixer's num_wires <= 64 *
    // MASK_WORDS — checked at Mixer construction — the fallback never engages.
    masks: Vec<GateMask>,
    masks_ok: bool,
}

impl Arena {
    pub fn from_gates(gs: Vec<XGate>) -> Arena {
        let n = gs.len();
        assert!(n > 0, "empty circuit");
        let mut masks: Vec<GateMask> = Vec::with_capacity(n);
        let mut masks_ok = true;
        for g in &gs {
            match GateMask::of(g) {
                Some(m) => masks.push(m),
                None => {
                    masks_ok = false;
                    masks.push(GateMask::zero());
                }
            }
        }
        let mut a = Arena {
            gates: gs,
            prev: (0..n)
                .map(|i| if i == 0 { NIL } else { (i - 1) as u32 })
                .collect(),
            next: (0..n)
                .map(|i| if i + 1 == n { NIL } else { (i + 1) as u32 })
                .collect(),
            stamp: vec![0; n],
            linked: vec![true; n],
            head: 0,
            tail: (n - 1) as u32,
            len: n,
            free: Vec::new(),
            masks,
            masks_ok,
        };
        a.stamp.shrink_to_fit();
        a
    }

    /// Fast collides on node ids: mask algebra when every wire fits the mask
    /// words — the exact same predicate as XGate::collides (see the proof
    /// there) — else fallback.
    #[inline]
    pub fn collides_ids(&self, a: u32, b: u32) -> bool {
        if !self.masks_ok {
            return XGate::collides(self.gate(a), self.gate(b));
        }
        Self::mask_collides(&self.masks[a as usize], &self.masks[b as usize])
    }

    #[inline]
    pub fn mask_collides(ma: &GateMask, mb: &GateMask) -> bool {
        // Neither reads the other's target => commute.
        if !ma.reads(mb.tgt) && !mb.reads(ma.tgt) {
            return false;
        }
        if ma.comp || mb.comp {
            return true;
        }
        // Separation exemption: a shared control wire with opposite polarity.
        let mut sep = 0u64;
        for w in 0..MASK_WORDS {
            sep |= (ma.read[w] & mb.read[w]) & (ma.pol[w] ^ mb.pol[w]);
        }
        sep == 0
    }

    #[inline]
    pub fn masks_ok(&self) -> bool {
        self.masks_ok
    }

    pub fn len(&self) -> usize {
        self.len
    }

    // Allocated slot count (linked or not): the id-indexed vectors' size.
    pub fn capacity(&self) -> usize {
        self.gates.len()
    }

    pub fn head(&self) -> u32 {
        self.head
    }

    pub fn gate(&self, id: u32) -> &XGate {
        &self.gates[id as usize]
    }

    pub fn stamp(&self, id: u32) -> u32 {
        self.stamp[id as usize]
    }

    pub fn is_linked(&self, id: u32) -> bool {
        self.linked[id as usize]
    }

    pub fn neighbor(&self, id: u32, dir: Dir) -> u32 {
        match dir {
            Dir::L => self.prev[id as usize],
            Dir::R => self.next[id as usize],
        }
    }

    // Remove from the list; the slot stays allocated (gate, stamp intact) so the
    // node can be relinked elsewhere.
    pub fn unlink(&mut self, id: u32) {
        debug_assert!(self.linked[id as usize]);
        let (p, n) = (self.prev[id as usize], self.next[id as usize]);
        if p == NIL {
            self.head = n;
        } else {
            self.next[p as usize] = n;
        }
        if n == NIL {
            self.tail = p;
        } else {
            self.prev[n as usize] = p;
        }
        self.linked[id as usize] = false;
        self.len -= 1;
    }

    // Relink an allocated-but-unlinked node after `after` (NIL = at head).
    pub fn link_after(&mut self, id: u32, after: u32) {
        debug_assert!(!self.linked[id as usize]);
        let next = if after == NIL {
            self.head
        } else {
            self.next[after as usize]
        };
        self.prev[id as usize] = after;
        self.next[id as usize] = next;
        if after == NIL {
            self.head = id;
        } else {
            self.next[after as usize] = id;
        }
        if next == NIL {
            self.tail = id;
        } else {
            self.prev[next as usize] = id;
        }
        self.linked[id as usize] = true;
        self.len += 1;
    }

    pub fn link_before(&mut self, id: u32, before: u32) {
        let after = if before == NIL {
            self.tail
        } else {
            self.prev[before as usize]
        };
        self.link_after(id, after);
    }

    // Allocate a fresh node (unlinked) for `gate`.
    fn alloc(&mut self, gate: XGate) -> u32 {
        let m = GateMask::of(&gate).unwrap_or_else(|| {
            self.masks_ok = false;
            GateMask::zero()
        });
        if let Some(id) = self.free.pop() {
            self.gates[id as usize] = gate;
            self.masks[id as usize] = m;
            self.stamp[id as usize] = self.stamp[id as usize].wrapping_add(1);
            id
        } else {
            self.gates.push(gate);
            self.masks.push(m);
            self.prev.push(NIL);
            self.next.push(NIL);
            self.stamp.push(0);
            self.linked.push(false);
            (self.gates.len() - 1) as u32
        }
    }

    pub fn insert_after(&mut self, after: u32, gate: XGate) -> u32 {
        let id = self.alloc(gate);
        self.link_after(id, after);
        id
    }

    // Rewrite a linked node's gate in place (position and links preserved).
    // Bumps the stamp so anything holding (id, stamp) — undo-journal entries —
    // sees the node as touched.
    pub fn replace_gate(&mut self, id: u32, gate: XGate) {
        debug_assert!(self.linked[id as usize]);
        self.masks[id as usize] = GateMask::of(&gate).unwrap_or_else(|| {
            self.masks_ok = false;
            GateMask::zero()
        });
        self.gates[id as usize] = gate;
        self.stamp[id as usize] = self.stamp[id as usize].wrapping_add(1);
    }

    // Mark a linked node as touched without changing it: bumps the stamp so
    // anything holding (id, stamp) — undo-journal entries — sees the node as
    // modified. Used when state an undo would restore (the ancestry litter of
    // a reused cross pivot) changes even though the gate itself does not.
    pub fn touch(&mut self, id: u32) {
        debug_assert!(self.linked[id as usize]);
        self.stamp[id as usize] = self.stamp[id as usize].wrapping_add(1);
    }

    // Free an unlinked node's slot for reuse.
    pub fn free_node(&mut self, id: u32) {
        debug_assert!(!self.linked[id as usize]);
        self.stamp[id as usize] = self.stamp[id as usize].wrapping_add(1);
        self.free.push(id);
    }

    pub fn random_linked(&self, rng: &mut impl Rng) -> u32 {
        let cap = self.gates.len();
        for _ in 0..64 {
            let id = rng.random_range(0..cap) as u32;
            if self.linked[id as usize] {
                return id;
            }
        }
        // Occupancy is normally near 100%; fall back to a scan from a random point.
        let start = rng.random_range(0..cap);
        for off in 0..cap {
            let id = ((start + off) % cap) as u32;
            if self.linked[id as usize] {
                return id;
            }
        }
        unreachable!("no linked nodes");
    }

    pub fn to_vec(&self) -> Vec<XGate> {
        let mut out = Vec::with_capacity(self.len);
        let mut cur = self.head;
        while cur != NIL {
            out.push(self.gates[cur as usize].clone());
            cur = self.next[cur as usize];
        }
        out
    }

    // Position of a linked node (O(n) walk; trace/debug use only).
    pub fn index_of(&self, id: u32) -> usize {
        debug_assert!(self.linked[id as usize]);
        let mut cur = self.head;
        let mut i = 0usize;
        while cur != NIL {
            if cur == id {
                return i;
            }
            i += 1;
            cur = self.next[cur as usize];
        }
        panic!("index_of on unlinked node");
    }

    pub fn ids_in_order(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.len);
        let mut cur = self.head;
        while cur != NIL {
            out.push(cur);
            cur = self.next[cur as usize];
        }
        out
    }
}
