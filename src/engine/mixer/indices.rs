//! The exact merge-partner index and common splice bookkeeping.
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
    }

    // ---- splicing (fsplit engine semantics) ----

    pub(crate) fn splice_replace_one(&mut self, id: u32, gates: Vec<XGate>) -> Vec<u32> {
        let mut cursor = self.arena.neighbor(id, Dir::L);
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
