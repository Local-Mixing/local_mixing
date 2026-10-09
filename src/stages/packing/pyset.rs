//! CPython 3.9 `setobject.c` emulator for int keys (docs/STAGE4_RUST_DESIGN.md §4.4).
//!
//! The oracle's ANF caps abort on a PREFIX of the smaller operand's iteration
//! order, and that order is CPython's slot-table order, so the stage-4 sets
//! must reproduce the table exactly: probe sequence (LINEAR_PROBES = 9,
//! PERTURB_SHIFT = 5), growth rule, `insert_clean` resizes (including the
//! small-table purge), `set_merge` (verbatim fast path) and 3.9's add, which
//! does NOT reuse dummy slots.
//!
//! Layout: the slot table holds (hash, arena index) pairs (16 bytes per
//! slot; a linear group of 10 slots is scanned branch-free); keys live in an
//! arena. Iteration order is a function of the slot table only. Invariant: `arena.len() == fill` (every add appends
//! one key; a resize compacts the arena in new table order and sets
//! `fill = used`).
use super::mono::PyIntKey;

const MINSIZE: usize = 8;
const LINEAR_PROBES: usize = 9;
const PERTURB_SHIFT: u32 = 5;
const NULL: u32 = u32::MAX;
const DUMMY: u32 = u32::MAX - 1;

#[derive(Clone, Copy)]
#[repr(C)]
struct Slot {
    hash: u64,
    idx: u32,
}

const EMPTY: Slot = Slot { hash: 0, idx: NULL };

#[derive(Clone)]
pub struct PySet<K: PyIntKey> {
    slots: Vec<Slot>,
    arena: Vec<K>,
    fill: usize,
    used: usize,
}

impl<K: PyIntKey> Default for PySet<K> {
    fn default() -> Self {
        Self::new()
    }
}

enum Probe {
    Found(usize),
    Free(usize),
}

impl<K: PyIntKey> PySet<K> {
    /// `set()`: an empty 8-slot table.
    pub fn new() -> Self {
        PySet {
            slots: vec![EMPTY; MINSIZE],
            arena: Vec::new(),
            fill: 0,
            used: 0,
        }
    }

    /// The literal `{k}` (BUILD_SET: an empty set, then add).
    pub fn singleton(k: K) -> Self {
        let mut s = Self::new();
        s.add(k);
        s
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.used
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.used == 0
    }

    #[inline]
    fn mask(&self) -> usize {
        self.slots.len() - 1
    }

    /// Slot count (for tests and memory accounting).
    pub fn table_size(&self) -> usize {
        self.slots.len()
    }

    /// The probe sequence shared by `set_add_entry` and `set_lookkey`: stop at
    /// the first NULL slot or at the active slot holding `key`.
    #[inline]
    fn probe(&self, key: &K, h: u64) -> Probe {
        let mask = self.mask();
        let mut i = (h as usize) & mask;
        let mut perturb = h;
        loop {
            if i + LINEAR_PROBES <= mask {
                // The 10-slot linear group, scanned branch-free: candidates are
                // NULL slots and active slots with an equal hash; the first
                // NULL or the first candidate holding `key` decides, exactly
                // as the slot-by-slot loop of setobject.c.
                let grp: &[Slot; LINEAR_PROBES + 1] =
                    self.slots[i..=i + LINEAR_PROBES].try_into().unwrap();
                let mut cand = 0u32;
                for (j, sl) in grp.iter().enumerate() {
                    let null = (sl.idx == NULL) as u32;
                    let hit = ((sl.hash == h) & (sl.idx < DUMMY)) as u32;
                    cand |= (null | hit) << j;
                }
                while cand != 0 {
                    let j = cand.trailing_zeros() as usize;
                    let sl = grp[j];
                    if sl.idx == NULL {
                        return Probe::Free(i + j);
                    }
                    if self.arena[sl.idx as usize] == *key {
                        return Probe::Found(i + j);
                    }
                    cand &= cand - 1;
                }
            } else {
                let sl = self.slots[i];
                if sl.idx == NULL {
                    return Probe::Free(i);
                }
                if sl.hash == h && sl.idx != DUMMY && self.arena[sl.idx as usize] == *key {
                    return Probe::Found(i);
                }
            }
            perturb >>= PERTURB_SHIFT;
            i = (i
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb as usize))
                & mask;
        }
    }

    pub fn contains(&self, k: &K) -> bool {
        matches!(self.probe(k, k.py_hash()), Probe::Found(_))
    }

    #[inline]
    fn found_unused(&mut self, slot: usize, key: K, h: u64) {
        self.fill += 1;
        self.used += 1;
        self.slots[slot] = Slot {
            hash: h,
            idx: self.arena.len() as u32,
        };
        self.arena.push(key);
        if self.fill * 5 >= self.mask() * 3 {
            let minused = if self.used > 50_000 {
                self.used * 2
            } else {
                self.used * 4
            };
            self.resize(minused);
        }
    }

    /// `set.add`.
    pub fn add(&mut self, k: K) {
        let h = k.py_hash();
        if let Probe::Free(slot) = self.probe(&k, h) {
            self.found_unused(slot, k, h);
        }
    }

    /// `set.discard`: the slot becomes a dummy; the table never shrinks here.
    pub fn discard(&mut self, k: &K) -> bool {
        match self.probe(k, k.py_hash()) {
            Probe::Found(slot) => {
                self.slots[slot].idx = DUMMY;
                self.used -= 1;
                true
            }
            Probe::Free(_) => false,
        }
    }

    /// `if k in s: s.discard(k) else: s.add(k)`. The discard lookup and the
    /// add probe visit the same slots and stop at the same NULL, so one probe
    /// serves both.
    #[inline]
    pub fn toggle(&mut self, k: K) {
        let h = k.py_hash();
        match self.probe(&k, h) {
            Probe::Found(slot) => {
                self.slots[slot].idx = DUMMY;
                self.used -= 1;
            }
            Probe::Free(slot) => self.found_unused(slot, k, h),
        }
    }

    fn insert_clean(slots: &mut [Slot], h: u64, ix: u32) {
        let mask = slots.len() - 1;
        let mut i = (h as usize) & mask;
        let mut perturb = h;
        loop {
            if slots[i].idx == NULL {
                break;
            }
            if i + LINEAR_PROBES <= mask
                && let Some(j) = (i + 1..=i + LINEAR_PROBES).find(|&j| slots[j].idx == NULL)
            {
                i = j;
                break;
            }
            perturb >>= PERTURB_SHIFT;
            i = (i
                .wrapping_mul(5)
                .wrapping_add(1)
                .wrapping_add(perturb as usize))
                & mask;
        }
        slots[i] = Slot { hash: h, idx: ix };
    }

    /// `set_table_resize`.
    fn resize(&mut self, minused: usize) {
        let mut newsize = MINSIZE;
        while newsize <= minused {
            newsize <<= 1;
        }
        if newsize == MINSIZE && self.mask() == MINSIZE - 1 && self.fill == self.used {
            return; // small table without dummies: nothing to do
        }
        let old_slots = std::mem::replace(&mut self.slots, vec![EMPTY; newsize]);
        let old_arena = std::mem::take(&mut self.arena);
        self.arena.reserve(self.used);
        for sl in old_slots {
            if sl.idx < DUMMY {
                let new_ix = self.arena.len() as u32;
                self.arena.push(old_arena[sl.idx as usize]);
                Self::insert_clean(&mut self.slots, sl.hash, new_ix);
            }
        }
        self.fill = self.used;
    }

    /// `set_merge(self, other)`.
    fn merge(&mut self, other: &Self) {
        if other.used == 0 {
            return;
        }
        if (self.fill + other.used) * 5 >= self.mask() * 3 {
            self.resize((self.used + other.used) * 2);
        }
        if self.fill == 0 && self.mask() == other.mask() && other.fill == other.used {
            // No dummies in `other`: its arena holds exactly its live keys.
            self.slots.copy_from_slice(&other.slots);
            self.arena.clone_from(&other.arena);
            self.fill = other.fill;
            self.used = other.used;
            return;
        }
        if self.fill == 0 {
            self.fill = other.used;
            self.used = other.used;
            self.arena.reserve(other.used);
            for sl in &other.slots {
                if sl.idx < DUMMY {
                    let new_ix = self.arena.len() as u32;
                    self.arena.push(other.arena[sl.idx as usize]);
                    Self::insert_clean(&mut self.slots, sl.hash, new_ix);
                }
            }
            return;
        }
        for k in other.iter() {
            self.add(*k);
        }
    }

    /// `set(other)`.
    pub fn copy_of(other: &Self) -> Self {
        let mut s = Self::new();
        s.merge(other);
        s
    }

    /// `a ^ b`: a copy of `b`, then `a`'s keys toggled in `a`'s table order.
    pub fn xor(a: &Self, b: &Self) -> Self {
        let mut s = Self::copy_of(b);
        s.xor_assign(a);
        s
    }

    /// `self ^= b`: `b`'s keys toggled in `b`'s table order.
    pub fn xor_assign(&mut self, b: &Self) {
        for sl in &b.slots {
            if sl.idx < DUMMY {
                let k = b.arena[sl.idx as usize];
                let h = sl.hash;
                match self.probe(&k, h) {
                    Probe::Found(slot) => {
                        self.slots[slot].idx = DUMMY;
                        self.used -= 1;
                    }
                    Probe::Free(slot) => self.found_unused(slot, k, h),
                }
            }
        }
    }

    /// Keys in table order (skipping NULL and dummy slots).
    pub fn iter(&self) -> impl Iterator<Item = &K> + '_ {
        self.slots
            .iter()
            .filter(|sl| sl.idx < DUMMY)
            .map(move |sl| &self.arena[sl.idx as usize])
    }

    /// Keys in table order, copied (the operand of a product loop).
    pub fn to_vec(&self) -> Vec<K> {
        self.iter().copied().collect()
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/pyset.rs"]
mod tests;
