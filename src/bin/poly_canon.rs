use cryptography::hash::sha2;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use clap::Parser;
use local_mixing::{
    circuit::circuit::{Monomial, Polynomial, poly_to_compressed_str, Permutation},
    random::random_data::random_circuit,
};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', default_value_t = 32)]
    wires: usize,

    #[arg(short = 'm', long)]
    gates: Option<usize>,
}

type Hash = [u8; 32];
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
    monomials: HashMap<Monomial, Node>,

    // I -> T edges are multiplication
    // T -> O edges are addition

    // Give a variable node, what monomials is it a member of?
    vars_to_mono: HashMap<NodeId, HashSet<Monomial>>,

    // Given a monomial, what are its constituent variables?
    mono_to_vars: HashMap<Monomial, HashSet<NodeId>>,

    // Given a monomial, which polynomial outputs is it a member of?
    mono_to_poly: HashMap<Monomial, HashSet<NodeId>>,

    // Give a polynomial, what are its constituent monomials?
    poly_to_mono: HashMap<NodeId, HashSet<Monomial>>,

    wires: u64,
}

impl Graph {
    pub fn add_poly(&mut self, out_idx: u64, p: Polynomial) {
        let start = Instant::now();

        // Construct each monomial.
        for m in p {
            let t = self.monomials.entry(m).or_insert(Node {
                id: ((1 as u128) << 64) | (m as u128),
                hash: None,
                node_type: NodeType::Monomial,
            });

            // Insert variables relationships
            for i in 0..(self.wires as u128) {
                if (m >> i) & 1 == 1 {
                    // x_i is in this monomial.
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
        eprintln!("add_poly: out_idx={} time={:.6}s", out_idx, start.elapsed().as_secs_f64());
    }

    pub fn new(wires: u64) -> Self {
        let mut variables = Vec::with_capacity(wires as usize);

        for i in 0..wires {
            let node = Node {
                id: i as u128,
                hash: None,
                node_type: NodeType::Variable,
            };
            variables.push(node);
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

    pub fn print_wire_hashes(&self) {
        let start = Instant::now();

        println!("--");
        for (i, n) in self.variables.iter().enumerate() {
            if let Some(h) = n.hash {
                println!(
                    "{:2} {}",
                    i,
                    h.iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<Vec<_>>()
                        .join("")
                );
            } else {
                println!("{:2} -", i);
            }
        }

        // Print sets of wires with matching hashes
        let mut groups: HashMap<Hash, Vec<usize>> = HashMap::new();
        for (i, n) in self.variables.iter().enumerate() {
            if let Some(h) = n.hash {
                groups.entry(h).or_default().push(i);
            }
        }

        let mut unique = true;

        for (_hash, wires) in groups.into_iter().filter(|(_, wires)| wires.len() > 1) {
            println!("identical: {:?}", wires);
            unique = false;
        }
        if unique {
            println!("[ all unique ]");
        }
        eprintln!("print_wire_hashes: time={:.6}s", start.elapsed().as_secs_f64());
    }

    pub fn push_hashes(&mut self) {
        let start = Instant::now();

        // Compute monomial hashes from their constituent variable hashes.
        // For canonical order, sort variable ids numerically.
        for (m, vars) in self.mono_to_vars.iter() {
            let mut hashes: Vec<Hash> = vars
                .into_iter()
                .map(|v| self.variables[*v as usize].hash.unwrap_or_default())
                .collect();
            hashes.sort();

            let mut hasher = sha2::Sha256::new();
            // Domain separation
            hasher.update(b"M>");
            for h in hashes {
                hasher.update(&h);
            }

            let res: Hash = hasher.finalize().into();
            if let Some(node) = self.monomials.get_mut(m) {
                node.hash = Some(res);
            }
        }

        for (idx, m) in self.poly_to_mono.iter() {
            let mut hashes: Vec<Hash> = m
                .into_iter()
                .map(|n| self.monomials[n].hash.unwrap_or_default())
                .collect();
            hashes.sort();

            let mut hasher = sha2::Sha256::new();
            hasher.update(b"P>");
            for h in hashes {
                hasher.update(&h);
            }

            let res = hasher.finalize().into();
            if let Some(node) = self.variables.get_mut(*idx as usize) {
                node.hash = Some(res);
            }
        }
        eprintln!("push_hashes: time={:.6}s", start.elapsed().as_secs_f64());
    }

    pub fn pull_hashes(&mut self) {
        let start = Instant::now();

        // Compute monomial hashes from their constituent variable hashes.
        // For canonical order, sort variable ids numerically.
        for (m, vars) in self.mono_to_poly.iter() {
            let mut hashes: Vec<Hash> = vars
                .into_iter()
                .map(|v| self.variables[*v as usize].hash.unwrap_or_default())
                .collect();
            hashes.sort();

            let mut hasher = sha2::Sha256::new();
            // Domain separation
            hasher.update(b"P<");
            for h in hashes {
                hasher.update(&h);
            }

            let res: Hash = hasher.finalize().into();
            if let Some(node) = self.monomials.get_mut(m) {
                node.hash = Some(res);
            }
        }

        for (idx, m) in self.vars_to_mono.iter() {
            let mut hashes: Vec<Hash> = m
                .into_iter()
                .map(|n| self.monomials[n].hash.unwrap_or_default())
                .collect();
            hashes.sort();

            let mut hasher = sha2::Sha256::new();
            hasher.update(b"M<");
            for h in hashes {
                hasher.update(&h);
            }

            let res = hasher.finalize().into();
            if let Some(node) = self.variables.get_mut(*idx as usize) {
                node.hash = Some(res);
            }
        }
        eprintln!("pull_hashes: time={:.6}s", start.elapsed().as_secs_f64());
    }

    pub fn extract_perm(&self) -> Permutation {
        let start = Instant::now();

        // Sort the node hashes, and output the permutation that would have to be applied to self.variables for them to be in that order
        // Build vector of (old_index, hash_bytes)
        let mut pairs: Vec<(usize, [u8; 32])> = self
            .variables
            .iter()
            .enumerate()
            .map(|(i, n)| (i, n.hash.unwrap_or([0u8; 32])))
            .collect();

        // Sort by hash bytes lexicographically
        pairs.sort_by(|a, b| a.1.cmp(&b.1));

        // Build permutation p where p[old_index] = new_index
        let mut perm: Vec<usize> = vec![0; self.variables.len()];
        for (new_idx, (old_idx, _)) in pairs.iter().enumerate() {
            perm[*old_idx] = new_idx;
        }
        let p = Permutation::new(perm);
        eprintln!("extract_perm: n={} time={:.6}s", self.variables.len(), start.elapsed().as_secs_f64());
        p
    }
}

fn main() {
    let args = Args::parse();

    let n = args.wires;
    let m = args
        .gates
        .unwrap_or(2 * ((n as f64) * (n as f64).ln()) as usize);

    let mut ckt = random_circuit(n, m);

    let alpha: Permutation;
    let beta: Permutation;

    {
        let mut g = Graph::new(n as u64);

        let start = Instant::now();
        let poly = ckt.to_polynomial(n, 0, m);
        eprintln!("to_poly: time={:.6}s", start.elapsed().as_secs_f64());

        for (i, p) in poly.iter().enumerate() {
            // println!("y{} = {}", i, poly_to_compressed_str(&p, n));

            g.add_poly(i as u64, p.clone());
        }

        for _ in 0..n {
            g.push_hashes();
            g.pull_hashes();
        }
        g.print_wire_hashes();
        alpha = g.extract_perm();
    }

    let p =Permutation::rand_perm(n);
    ckt.rewire(&p, n);

    let full_start: Instant;
    {
        let mut g = Graph::new(n as u64);

        let poly = ckt.to_polynomial(n, 0, m);
        full_start = Instant::now();

        for (i, p) in poly.iter().enumerate() {
            // println!("y{} = {}", i, poly_to_compressed_str(&p, n));
            g.add_poly(i as u64, p.clone());
        }

        // I think this needs to run as many times as there are wires, in the worst case
        for _ in 0..n {
            g.push_hashes();
            g.pull_hashes();
        }
        g.print_wire_hashes();
        beta = g.extract_perm();
    }
    eprintln!("full_canon: time={:.6}s", full_start.elapsed().as_secs_f64());

    println!("Rewired Perm:   {:?}", p.data);

    let r = beta.invert().compose(&alpha);
    println!("Inferred Comp.: {:?}", r.data);

    assert_eq!(p, r);
}
