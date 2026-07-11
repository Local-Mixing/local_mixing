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
}

impl Arena {
    pub fn from_gates(gs: Vec<XGate>) -> Arena {
        let n = gs.len();
        assert!(n > 0, "empty circuit");
        let mut a = Arena {
            gates: gs,
            prev: (0..n).map(|i| if i == 0 { NIL } else { (i - 1) as u32 }).collect(),
            next: (0..n).map(|i| if i + 1 == n { NIL } else { (i + 1) as u32 }).collect(),
            stamp: vec![0; n],
            linked: vec![true; n],
            head: 0,
            tail: (n - 1) as u32,
            len: n,
            free: Vec::new(),
        };
        a.stamp.shrink_to_fit();
        a
    }

    pub fn len(&self) -> usize {
        self.len
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
        let next = if after == NIL { self.head } else { self.next[after as usize] };
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
        let after = if before == NIL { self.tail } else { self.prev[before as usize] };
        self.link_after(id, after);
    }

    // Allocate a fresh node (unlinked) for `gate`.
    fn alloc(&mut self, gate: XGate) -> u32 {
        if let Some(id) = self.free.pop() {
            self.gates[id as usize] = gate;
            self.stamp[id as usize] = self.stamp[id as usize].wrapping_add(1);
            id
        } else {
            self.gates.push(gate);
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
        self.gates[id as usize] = gate;
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
