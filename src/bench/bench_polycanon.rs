use cryptography::hash::sha2;
use local_mixing::bench_support::{SEEDS, default_m, gen_polys, selected_n_grid, trimmed_polys};
use local_mixing::circuit::circuit::{
    Monomial, Permutation, Polynomial, polynomial_from_terms, trim_canonicalized,
};
use local_mixing::random::random_data::random_circuit;
use std::collections::{HashMap, HashSet};
use std::hint::black_box;
use std::time::Instant;

const K: usize = 5;
const VARIANT: &str = "wl";

type Hash = [u8; 32];
type NodeId = u128;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NodeType {
    Variable,
    Monomial,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Node {
    id: NodeId,
    hash: Option<Hash>,
    node_type: NodeType,
}

#[derive(Debug, Default)]
struct Graph {
    variables: Vec<Node>,
    monomials: HashMap<Monomial, Node>,
    vars_to_mono: HashMap<NodeId, HashSet<Monomial>>,
    mono_to_vars: HashMap<Monomial, HashSet<NodeId>>,
    mono_to_poly: HashMap<Monomial, HashSet<NodeId>>,
    poly_to_mono: HashMap<NodeId, HashSet<Monomial>>,
    wires: u64,
}

impl Graph {
    fn new(wires: u64) -> Self {
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
            monomials: HashMap::new(),
            vars_to_mono: HashMap::new(),
            mono_to_vars: HashMap::new(),
            mono_to_poly: HashMap::new(),
            poly_to_mono: HashMap::new(),
            wires,
        }
    }

    fn add_poly(&mut self, out_idx: u64, p: &Polynomial) {
        for &m in p {
            self.monomials.entry(m).or_insert(Node {
                id: (1u128 << 64) | (m as u128),
                hash: None,
                node_type: NodeType::Monomial,
            });

            for i in 0..(self.wires as u128) {
                if (m >> i) & 1 == 1 {
                    self.vars_to_mono.entry(i).or_default().insert(m);
                    self.mono_to_vars.entry(m).or_default().insert(i);
                }
            }

            self.mono_to_poly
                .entry(m)
                .or_default()
                .insert(out_idx as u128);
            self.poly_to_mono
                .entry(out_idx as u128)
                .or_default()
                .insert(m);
        }
    }

    fn push_hashes(&mut self) {
        for (m, vars) in self.mono_to_vars.iter() {
            let mut hashes: Vec<Hash> = vars
                .iter()
                .map(|v| self.variables[*v as usize].hash.unwrap_or_default())
                .collect();
            hashes.sort();

            let mut hasher = sha2::Sha256::new();
            hasher.update(b"M>");
            for h in hashes {
                hasher.update(&h);
            }

            if let Some(node) = self.monomials.get_mut(m) {
                node.hash = Some(hasher.finalize().into());
            }
        }

        for (idx, monomials) in self.poly_to_mono.iter() {
            let mut hashes: Vec<Hash> = monomials
                .iter()
                .map(|m| self.monomials[m].hash.unwrap_or_default())
                .collect();
            hashes.sort();

            let mut hasher = sha2::Sha256::new();
            hasher.update(b"P>");
            for h in hashes {
                hasher.update(&h);
            }

            if let Some(node) = self.variables.get_mut(*idx as usize) {
                node.hash = Some(hasher.finalize().into());
            }
        }
    }

    fn pull_hashes(&mut self) {
        for (m, polys) in self.mono_to_poly.iter() {
            let mut hashes: Vec<Hash> = polys
                .iter()
                .map(|v| self.variables[*v as usize].hash.unwrap_or_default())
                .collect();
            hashes.sort();

            let mut hasher = sha2::Sha256::new();
            hasher.update(b"P<");
            for h in hashes {
                hasher.update(&h);
            }

            if let Some(node) = self.monomials.get_mut(m) {
                node.hash = Some(hasher.finalize().into());
            }
        }

        for (idx, monomials) in self.vars_to_mono.iter() {
            let mut hashes: Vec<Hash> = monomials
                .iter()
                .map(|m| self.monomials[m].hash.unwrap_or_default())
                .collect();
            hashes.sort();

            let mut hasher = sha2::Sha256::new();
            hasher.update(b"M<");
            for h in hashes {
                hasher.update(&h);
            }

            if let Some(node) = self.variables.get_mut(*idx as usize) {
                node.hash = Some(hasher.finalize().into());
            }
        }
    }

    fn extract_perm(&self) -> Permutation {
        let mut pairs: Vec<(usize, Hash)> = self
            .variables
            .iter()
            .enumerate()
            .map(|(i, n)| (i, n.hash.unwrap_or_default()))
            .collect();
        pairs.sort_by(|a, b| a.1.cmp(&b.1));

        let mut perm = vec![0; self.variables.len()];
        for (new_idx, (old_idx, _)) in pairs.iter().enumerate() {
            perm[*old_idx] = new_idx;
        }

        Permutation::new(perm)
    }
}

fn canonicalize_graph(polys: &[Polynomial], n: usize) -> Permutation {
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

fn canonical_form(polys: &[Polynomial], perm: &Permutation) -> Vec<Polynomial> {
    let n = polys.len();
    let mut canonical = vec![Polynomial::new(); n];
    for (old_wire, poly) in polys.iter().enumerate() {
        let new_wire = perm.data[old_wire];
        canonical[new_wire] = polynomial_from_terms(poly.iter().map(|&m| {
            let mut remapped = 0u64;
            for wire in 0..n {
                if m & (1u64 << wire) != 0 {
                    remapped |= 1u64 << perm.data[wire];
                }
            }
            remapped
        }));
    }
    trim_canonicalized(canonical)
}

fn deterministic_wire_perm(seed: u64, n: usize) -> Permutation {
    let mut data: Vec<usize> = (0..n).collect();
    let mut rng = fastrand::Rng::with_seed(seed ^ 0x9e37_79b9_7f4a_7c15);
    rng.shuffle(&mut data);

    Permutation::new(data)
}

fn valid_round_trip(seed: u64, n: usize, m: usize) -> bool {
    fastrand::seed(seed);
    let mut ckt = random_circuit(n, m);

    let polys = trimmed_polys(&ckt);
    let alpha = canonicalize_graph(&polys, polys.len());
    let alpha_form = canonical_form(&polys, &alpha);

    let p = deterministic_wire_perm(seed, n);
    ckt.rewire(&p, n);

    let rewired_polys = trimmed_polys(&ckt);
    let beta = canonicalize_graph(&rewired_polys, rewired_polys.len());
    let beta_form = canonical_form(&rewired_polys, &beta);
    alpha_form == beta_form
}

fn median_nanos(samples: &mut [u128]) -> u128 {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn main() {
    println!("algo,n,m,seed,variant,nanos,valid");

    for n in selected_n_grid() {
        let m = default_m(n);

        for &seed in SEEDS {
            let valid = valid_round_trip(seed, n, m);
            let polys = gen_polys(seed, n, m);

            let trimmed_n = polys.len();

            let warmup = polys.clone();
            black_box(canonicalize_graph(black_box(&warmup), trimmed_n));

            let inputs: Vec<Vec<Polynomial>> = (0..K).map(|_| polys.clone()).collect();
            let mut samples = Vec::with_capacity(K);

            for input in inputs {
                let start = Instant::now();
                let perm = canonicalize_graph(black_box(&input), trimmed_n);
                let nanos = start.elapsed().as_nanos();
                black_box(perm);
                samples.push(nanos);
            }

            let nanos = median_nanos(&mut samples);
            println!(
                "polycanon,{n},{m},{seed},{VARIANT},{nanos},{}",
                u8::from(valid)
            );
        }
    }
}
