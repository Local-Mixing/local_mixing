use rustc_hash::{FxHashMap, FxHashSet};
use xxhash_rust::xxh3::Xxh3Default;

use crate::circuit::circuit::{Monomial, Permutation, Polynomial, trim_canonicalized};

type Hash = u128;
type NodeId = u128;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NodeType {
    Variable,
    Monomial,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Node {
    pub id: NodeId,
    pub hash: Option<Hash>,
    pub node_type: NodeType,
}

#[derive(Debug, Default)]
pub struct Graph {
    variables: Vec<Node>,
    monomials: FxHashMap<Monomial, Node>,
    vars_to_mono: FxHashMap<NodeId, FxHashSet<Monomial>>,
    mono_to_vars: FxHashMap<Monomial, FxHashSet<NodeId>>,
    mono_to_poly: FxHashMap<Monomial, FxHashSet<NodeId>>,
    poly_to_mono: FxHashMap<NodeId, FxHashSet<Monomial>>,
    wires: u64,
}

impl Graph {
    pub fn add_poly(&mut self, out_idx: u64, p: &Polynomial) {
        for &m in p {
            self.monomials.insert(
                m,
                Node {
                    id: m as NodeId,
                    hash: None,
                    node_type: NodeType::Monomial,
                },
            );

            for i in 0..(self.wires as u128) {
                if (m >> i) & 1 == 1 {
                    self.vars_to_mono.entry(i).or_default().insert(m);
                    self.mono_to_vars.entry(m).or_default().insert(i);
                }
            }

            self.mono_to_poly
                .entry(m)
                .or_default()
                .insert(out_idx as NodeId);
            self.poly_to_mono
                .entry(out_idx as u128)
                .or_default()
                .insert(m);
        }
    }

    pub fn new(wires: u64) -> Self {
        let mut variables = Vec::with_capacity(wires as usize);

        for i in 0..wires {
            variables.push(Node {
                id: i as u128,
                hash: None,
                node_type: NodeType::Variable,
            });
        }

        Self {
            variables,
            monomials: FxHashMap::default(),
            vars_to_mono: FxHashMap::default(),
            mono_to_vars: FxHashMap::default(),
            mono_to_poly: FxHashMap::default(),
            poly_to_mono: FxHashMap::default(),
            wires,
        }
    }

    pub fn push_hashes(&mut self) {
        for (m, vars) in &self.mono_to_vars {
            let mut hashes: Vec<Hash> = vars
                .iter()
                .map(|v| self.variables[*v as usize].hash.unwrap_or_default())
                .collect();
            hashes.sort_unstable();

            let mut hasher = Xxh3Default::new();
            hasher.update(b"M>");
            hasher.update(&self.monomials[m].hash.unwrap_or_default().to_le_bytes());
            for hash in hashes {
                hasher.update(&hash.to_le_bytes());
            }

            let result = hasher.digest128();
            if let Some(node) = self.monomials.get_mut(m) {
                node.hash = Some(result);
            }
        }

        for (idx, monomials) in &self.poly_to_mono {
            let mut hashes: Vec<Hash> = monomials
                .iter()
                .map(|monomial| {
                    self.monomials
                        .get(monomial)
                        .and_then(|node| node.hash)
                        .unwrap_or_default()
                })
                .collect();
            hashes.sort_unstable();

            let mut hasher = Xxh3Default::new();
            hasher.update(b"P>");
            hasher.update(
                &self.variables[*idx as usize]
                    .hash
                    .unwrap_or_default()
                    .to_le_bytes(),
            );
            for hash in hashes {
                hasher.update(&hash.to_le_bytes());
            }

            let result = hasher.digest128();
            if let Some(node) = self.variables.get_mut(*idx as usize) {
                node.hash = Some(result);
            }
        }
    }

    pub fn pull_hashes(&mut self) {
        for (m, outputs) in &self.mono_to_poly {
            let mut hashes: Vec<Hash> = outputs
                .iter()
                .map(|output| self.variables[*output as usize].hash.unwrap_or_default())
                .collect();
            hashes.sort_unstable();

            let mut hasher = Xxh3Default::new();
            hasher.update(b"M<");
            hasher.update(&self.monomials[m].hash.unwrap_or_default().to_le_bytes());
            for hash in hashes {
                hasher.update(&hash.to_le_bytes());
            }

            let result = hasher.digest128();
            if let Some(node) = self.monomials.get_mut(m) {
                node.hash = Some(result);
            }
        }

        for (idx, monomials) in &self.vars_to_mono {
            let mut hashes: Vec<Hash> = monomials
                .iter()
                .map(|monomial| {
                    self.monomials
                        .get(monomial)
                        .and_then(|node| node.hash)
                        .unwrap_or_default()
                })
                .collect();
            hashes.sort_unstable();

            let mut hasher = Xxh3Default::new();
            hasher.update(b"P<");
            hasher.update(
                &self.variables[*idx as usize]
                    .hash
                    .unwrap_or_default()
                    .to_le_bytes(),
            );
            for hash in hashes {
                hasher.update(&hash.to_le_bytes());
            }

            let result = hasher.digest128();
            if let Some(node) = self.variables.get_mut(*idx as usize) {
                node.hash = Some(result);
            }
        }
    }

    pub fn extract_perm(&self) -> Permutation {
        let mut pairs: Vec<(usize, Hash)> = self
            .variables
            .iter()
            .enumerate()
            .map(|(i, node)| (i, node.hash.unwrap_or_default()))
            .collect();
        pairs.sort_by(|a, b| a.1.cmp(&b.1));

        let mut perm = vec![0; self.variables.len()];
        for (new_idx, (old_idx, _)) in pairs.iter().enumerate() {
            perm[*old_idx] = new_idx;
        }

        Permutation::new(perm)
    }
}

pub fn canonicalize_graph(polys: &[Polynomial], n: usize) -> Permutation {
    let mut graph = Graph::new(n as u64);

    for (i, p) in polys.iter().enumerate() {
        graph.add_poly(i as u64, p);
    }

    for _ in 0..n {
        graph.push_hashes();
        graph.pull_hashes();
    }

    graph.extract_perm()
}

pub fn canonical_form(polys: &[Polynomial], perm: &Permutation) -> Vec<Polynomial> {
    let n = polys.len();
    let mut canonical = vec![Polynomial::new(); n];
    for (old_wire, poly) in polys.iter().enumerate() {
        let new_wire = perm.data[old_wire];
        canonical[new_wire] = poly
            .iter()
            .map(|&monomial| {
                let mut remapped = 0u64;
                for wire in 0..n {
                    if monomial & (1u64 << wire) != 0 {
                        remapped |= 1u64 << perm.data[wire];
                    }
                }
                remapped
            })
            .collect();
    }
    trim_canonicalized(canonical)
}
