//! Exact live-node population indexes, canaries and common splice bookkeeping.
use super::*;

impl Mixer {
    // ---- merge-partner index maintenance ----

    pub(crate) fn index_add(&mut self, id: u32) {
        let k = key_of(self.arena.gate(id));
        let bucket = self.index.entry(k).or_default();
        let idu = id as usize;
        if self.index_pos.len() <= idu {
            self.index_pos.resize(idu + 1, NIL);
        }
        debug_assert_eq!(
            self.index_pos[idu], NIL,
            "id already present in merge index"
        );
        self.index_pos[idu] = bucket.len() as u32;
        bucket.push(id);
        self.indexed_count += 1;
        self.side_add(id);
    }

    pub(crate) fn index_remove(&mut self, id: u32) {
        let k = key_of(self.arena.gate(id));
        let idu = id as usize;
        let pos = *self
            .index_pos
            .get(idu)
            .filter(|&&p| p != NIL)
            .expect("id missing from merge-index positions") as usize;
        let (moved, bucket_empty) = {
            let bucket = self.index.get_mut(&k).expect("index bucket missing");
            assert_eq!(bucket.get(pos), Some(&id), "merge-index position drift");
            bucket.swap_remove(pos);
            (bucket.get(pos).copied(), bucket.is_empty())
        };
        if let Some(moved) = moved {
            self.index_pos[moved as usize] = pos as u32;
        }
        self.index_pos[idu] = NIL;
        if bucket_empty {
            self.index.remove(&k);
        }
        self.indexed_count -= 1;
        self.side_remove(id);
    }

    // ---- split-stage population-index maintenance ----
    //
    // These hooks stay with the shared merge index because every splice and
    // in-place rewrite must update both structures atomically. The Stage-4
    // selection/reporting algorithms that consume the indexes live in
    // postprocessing::splitting.

    pub(super) fn side_add(&mut self, id: u32) {
        let idu = id as usize;
        if self.comp_pos.len() <= idu {
            self.comp_pos.resize(idu + 1, NIL);
            self.wt_pos.resize(idu + 1, NIL);
        }
        let g = self.arena.gate(id);
        let (comp, elig, t) = (g.comp, g.comp || g.ctrls.len() == 1, g.target as usize);
        if comp {
            self.comp_pos[idu] = self.comp_ids.len() as u32;
            self.comp_ids.push(id);
        }
        if elig {
            let b = &mut self.wt_buckets[t];
            self.wt_pos[idu] = b.len() as u32;
            b.push(id);
        }
    }

    pub(super) fn side_remove(&mut self, id: u32) {
        let idu = id as usize;
        let p = self.comp_pos[idu];
        if p != NIL {
            self.comp_ids.swap_remove(p as usize);
            if let Some(&moved) = self.comp_ids.get(p as usize) {
                self.comp_pos[moved as usize] = p;
            }
            self.comp_pos[idu] = NIL;
        }
        let q = self.wt_pos[idu];
        if q != NIL {
            let t = self.arena.gate(id).target as usize;
            let b = &mut self.wt_buckets[t];
            b.swap_remove(q as usize);
            if let Some(&moved) = b.get(q as usize) {
                self.wt_pos[moved as usize] = q;
            }
            self.wt_pos[idu] = NIL;
        }
    }

    // Bulk (re)build: construction and resume, where the merge index is built
    // without going through index_add.
    pub(super) fn rebuild_side_index(&mut self) {
        self.comp_ids.clear();
        self.comp_pos = vec![NIL; self.arena.capacity()];
        self.wt_buckets = vec![Vec::new(); self.num_wires];
        self.wt_pos = vec![NIL; self.arena.capacity()];
        for id in self.arena.ids_in_order() {
            self.side_add(id);
        }
    }

    /// Re-anchor split canaries off a node about to die. This is a shared
    /// mutation hook: DB replacement, crossing, merging, and splitting can all
    /// remove nodes after canaries have been planted.
    pub(crate) fn evict_taps(&mut self, id: u32) {
        if self.taps.is_empty() {
            return;
        }
        let Some(list) = self.tap_at.remove(&id) else {
            return;
        };
        let mut to = self.arena.neighbor(id, Dir::L);
        if to == NIL {
            to = self.arena.neighbor(id, Dir::R);
        }
        if to == NIL {
            return;
        }
        for &t in &list {
            self.taps[t as usize].anchor = to;
        }
        self.tap_at.entry(to).or_default().extend(list);
    }

    // ---- splicing (fsplit engine semantics) ----

    pub(crate) fn splice_replace_one(&mut self, id: u32, gates: Vec<XGate>) -> Vec<u32> {
        let mut cursor = self.arena.neighbor(id, Dir::L);
        self.evict_taps(id);
        self.index_remove(id);
        self.arena.unlink(id);
        self.arena.free_node(id);
        let mut ids = Vec::with_capacity(gates.len());
        for g in gates {
            cursor = self.arena.insert_after(cursor, g);
            self.index_add(cursor);
            ids.push(cursor);
        }
        ids
    }
}
