//! The nonlinear part of two masks is Q_a XOR Q_b. Treat that as an edge
//! between quadratic-pair labels. These rows are independent iff the graph
//! is a forest. Linear mask terms and constants do not help against affine
//! predictors because all band wires and the constant are available.
//!
//! Ported verbatim from the `stage2_compute_pilot` generator (`src/mask_rank.rs`).

use super::fire::Mask;
use std::collections::BTreeMap;

#[derive(Default)]
pub struct Forest {
    ids: BTreeMap<(u16, u16), usize>,
    parent: Vec<usize>,
}

impl Forest {
    fn id(&mut self, m: Mask) -> usize {
        let key = (m[0].min(m[1]), m[0].max(m[1]));
        if let Some(&i) = self.ids.get(&key) {
            return i;
        }
        let i = self.parent.len();
        self.parent.push(i);
        self.ids.insert(key, i);
        i
    }

    fn root(&mut self, mut i: usize) -> usize {
        while self.parent[i] != i {
            self.parent[i] = self.parent[self.parent[i]];
            i = self.parent[i];
        }
        i
    }

    pub fn connected(&mut self, a: Mask, b: Mask) -> bool {
        let i = self.id(a);
        let j = self.id(b);
        self.root(i) == self.root(j)
    }

    pub fn add(&mut self, a: Mask, b: Mask) -> bool {
        let i = self.id(a);
        let j = self.id(b);
        let x = self.root(i);
        let y = self.root(j);
        if x == y {
            return false;
        }
        self.parent[x] = y;
        true
    }

    pub fn without_role(masks: &[Vec<Mask>], omit: usize) -> Self {
        let mut f = Self::default();
        for (w, ms) in masks.iter().enumerate() {
            if w != omit && ms.len() == 2 {
                assert!(f.add(ms[0], ms[1]), "existing mask forest has a cycle");
            }
        }
        f
    }
}

pub fn deficit(masks: &[Vec<Mask>]) -> usize {
    let mut f = Forest::default();
    masks
        .iter()
        .filter(|ms| ms.len() == 2)
        .filter(|ms| !f.add(ms[0], ms[1]))
        .count()
}
