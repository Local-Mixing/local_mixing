//! Quadratic-sharing obfuscation as a **gate-57-only** reversible circuit
//! (distinct pins on every gate).
//!
//! Pipeline (polynomials live only in the compiler except at share / reconstruct):
//! 1. Init scramble on the ancilla bank (scratch untouched).
//! 2. Share each data wire with a **sparse** random quadratic on ~½ the bank.
//! 3. Per plaintext gate-57 — **homomorphic**, never cleartext:
//!    substitute sharings into `a ^= 1 ⊕ c2 ⊕ c1 c2`, expand ANF (deg ≤ 4),
//!    XOR the update onto the active carrier (low deg then high deg), and leave
//!    a fresh degree-2 mask `R`. High-degree monomials use dirty-scratch native
//!    products (scratch may start dirty; restored by the gadget).
//! 4. Final reconstruct.
//!
//! Does not replace [`crate::sandwich`].

mod gadgets;
mod poly;

use crate::circuit::CircuitSeq;
use gadgets::{emit_and, emit_cnot, emit_monomial_xor, emit_not};
use poly::{random_quadratic, Poly};
use rand::seq::SliceRandom;
use rand::Rng;
use std::collections::HashSet;

pub use gadgets::{
    emit_and as emit_and_pub, emit_and3 as emit_and3_pub, emit_and4 as emit_and4_pub,
    emit_cnot as emit_cnot_pub, emit_not as emit_not_pub,
};

/// Parameters for the quadratic-share compiler.
#[derive(Clone, Debug)]
pub struct MqShareParams {
    pub ancilla_count: usize,
    /// Borrowed workspace for deg-3/4 products (may be dirty; restored by gadgets).
    pub scratch_count: usize,
    pub init_gates: usize,
    /// Fraction of the ancilla bank used in each quadratic mask (~1/2).
    pub share_frac: f64,
}

impl MqShareParams {
    pub fn for_n(data_wires: usize) -> Self {
        let ancilla_count = data_wires.max(8);
        let init_gates =
            (2.0 * data_wires as f64 * (data_wires as f64).log2()).round() as usize;
        Self {
            ancilla_count,
            scratch_count: 4,
            init_gates,
            share_frac: 0.5,
        }
    }
}

/// Layout: `data | ancilla bank | scratch`. All may be dirty on input.
#[derive(Clone, Debug)]
pub struct MqShareCircuit {
    pub data_wires: usize,
    pub ancilla_count: usize,
    pub scratch_count: usize,
    pub total_wires: usize,
    pub gates: Vec<[u16; 3]>,
}

impl MqShareCircuit {
    pub fn as_circuit_seq(&self) -> CircuitSeq {
        CircuitSeq {
            gates: self.gates.clone(),
        }
    }

    pub fn bank_end(&self) -> usize {
        self.data_wires + self.ancilla_count
    }

    pub fn evaluate_data_bits(&self, input_data: &[bool]) -> Vec<bool> {
        assert_eq!(input_data.len(), self.data_wires);
        let mut bits = vec![false; self.total_wires];
        bits[..self.data_wires].copy_from_slice(input_data);
        eval_gates(&mut bits, &self.gates);
        bits[..self.data_wires].to_vec()
    }

    /// Fully dirty input (data + bank + scratch) allowed.
    pub fn evaluate_data_bits_full(&self, full: &[bool]) -> Vec<bool> {
        assert_eq!(full.len(), self.total_wires);
        let mut bits = full.to_vec();
        eval_gates(&mut bits, &self.gates);
        bits[..self.data_wires].to_vec()
    }
}

fn eval_gates(bits: &mut [bool], gates: &[[u16; 3]]) {
    for g in gates {
        debug_assert!(g[0] != g[1] && g[0] != g[2] && g[1] != g[2]);
        let c1 = bits[g[1] as usize];
        let c2 = bits[g[2] as usize];
        bits[g[0] as usize] ^= c1 || !c2;
    }
}

#[derive(Clone, Debug)]
struct Share {
    carrier: u16,
    ancillas: Vec<u16>,
    mask: Poly,
}

struct Compiler<'a, R: Rng> {
    data_wires: usize,
    bank: Vec<u16>,
    scratch: Vec<u16>,
    params: &'a MqShareParams,
    shares: Vec<Share>,
    gates: Vec<[u16; 3]>,
    rng: &'a mut R,
}

impl<'a, R: Rng> Compiler<'a, R> {
    fn total_wires(&self) -> usize {
        self.data_wires + self.bank.len() + self.scratch.len()
    }

    fn bank_end(&self) -> usize {
        self.data_wires + self.bank.len()
    }

    /// Helpers from data∪bank (scratch reserved for products).
    fn pick_helpers(&self, forbid: &HashSet<u16>, n: usize) -> Vec<u16> {
        let mut out = Vec::new();
        for w in 0..self.bank_end() as u16 {
            if !forbid.contains(&w) {
                out.push(w);
                if out.len() == n {
                    return out;
                }
            }
        }
        panic!("need {n} helper wires outside {forbid:?}");
    }

    fn pick_scratches(&self, n: usize, forbid: &HashSet<u16>) -> Vec<u16> {
        let mut out = Vec::new();
        for &w in &self.scratch {
            if !forbid.contains(&w) {
                out.push(w);
                if out.len() == n {
                    return out;
                }
            }
        }
        panic!("need {n} scratch wires");
    }

    /// XOR any deg≤4 poly onto `target`. Deg≥3 uses dirty-scratch native products.
    fn emit_xor_poly(&mut self, target: u16, poly: &Poly) {
        assert!(
            poly.degree() <= 4,
            "emit_xor_poly: degree {} > 4",
            poly.degree()
        );
        for m in poly.terms() {
            let mut forbid: HashSet<u16> = m.iter().copied().collect();
            forbid.insert(target);
            match m.len() {
                0 => {
                    let hs = self.pick_helpers(&forbid, 2);
                    emit_not(target, hs[0], hs[1], &mut self.gates);
                }
                1 => {
                    let hs = self.pick_helpers(&forbid, 1);
                    emit_cnot(target, m[0], hs[0], &mut self.gates);
                }
                2 => {
                    emit_and(target, m[0], m[1], &mut self.gates);
                }
                3 | 4 => {
                    let need = if m.len() == 3 { 1 } else { 2 };
                    let scratches = self.pick_scratches(need, &forbid);
                    for &s in &scratches {
                        forbid.insert(s);
                    }
                    let hs = self.pick_helpers(&forbid, 1);
                    emit_monomial_xor(target, m, &scratches, hs[0], &mut self.gates);
                }
                _ => unreachable!(),
            }
        }
    }

    fn sample_ancilla_subset(&mut self) -> Vec<u16> {
        let k = ((self.bank.len() as f64) * self.params.share_frac)
            .round()
            .max(2.0) as usize;
        let k = k.min(self.bank.len());
        let mut idx = self.bank.clone();
        idx.shuffle(self.rng);
        let mut out = idx[..k].to_vec();
        out.sort_unstable();
        out
    }

    fn share_wire(&mut self, wire: usize) {
        let ancillas = self.sample_ancilla_subset();
        let mask = random_quadratic(&ancillas, self.rng);
        debug_assert!(mask.term_count() > 0);
        self.emit_xor_poly(wire as u16, &mask);
        self.shares[wire] = Share {
            carrier: wire as u16,
            ancillas,
            mask,
        };
    }

    fn unshare_wire(&mut self, wire: usize) {
        let share = self.shares[wire].clone();
        self.emit_xor_poly(share.carrier, &share.mask);
        self.shares[wire].mask = Poly::zero();
        self.shares[wire].ancillas.clear();
    }

    fn emit_init_scramble(&mut self) {
        // Active ∈ bank; controls ∈ data ∪ bank (never scratch).
        let ctrl_hi = self.bank_end() as u16;
        for _ in 0..self.params.init_gates {
            let active = self.bank[self.rng.random_range(0..self.bank.len())];
            let mut c1 = self.rng.random_range(0..ctrl_hi);
            while c1 == active {
                c1 = self.rng.random_range(0..ctrl_hi);
            }
            let mut c2 = self.rng.random_range(0..ctrl_hi);
            while c2 == active || c2 == c1 {
                c2 = self.rng.random_range(0..ctrl_hi);
            }
            self.gates.push([active, c1, c2]);
        }
    }

    /// Homomorphic gate-57: never unmasks to cleartext.
    ///
    /// Logical update `a ^= 1 ⊕ c2 ⊕ c1·c2`. With sharings
    /// `a=wa⊕Qa`, `c1=wb⊕Qb`, `c2=wc⊕Qc`, expand into a physical polynomial
    /// of degree ≤4, then XOR `M = Qa ⊕ delta ⊕ R` onto `wa` so the new
    /// encoding is `wa' ⊕ R` with fresh sparse quadratic `R`.
    /// Deg≤2 terms first, then deg 3/4 cleanup terms (more gates).
    fn emit_gate57(&mut self, gate: [u16; 3]) {
        let a = gate[0] as usize;
        let b = gate[1] as usize;
        let c = gate[2] as usize;

        let sa = self.shares[a].clone();
        let sb = self.shares[b].clone();
        let sc = self.shares[c].clone();

        let pb = Poly::var(sb.carrier).add(&sb.mask);
        let pc = Poly::var(sc.carrier).add(&sc.mask);

        // delta = 1 ⊕ c2 ⊕ c1 c2  (gate-57 ANF)
        let delta = Poly::one().add(&pc).add(&pb.mul(&pc));

        let new_anc = self.sample_ancilla_subset();
        let r = random_quadratic(&new_anc, self.rng);
        debug_assert!(r.term_count() > 0);

        // M = Qa ⊕ delta ⊕ R  ⇒  wa ⊕ M ⊕ R = a ⊕ delta = a'
        let m = sa.mask.add(&delta).add(&r);
        assert!(m.degree() <= 4, "update deg {}", m.degree());

        let (low, high) = m.split_degree(2);
        // Homomorphic low-degree part, then cancel deg-3/4 by XORing them on.
        self.emit_xor_poly(sa.carrier, &low);
        self.emit_xor_poly(sa.carrier, &high);

        self.shares[a] = Share {
            carrier: sa.carrier,
            ancillas: new_anc,
            mask: r,
        };
        // Controls keep their existing sharings unchanged.
    }
}

pub fn obfuscate_mqshare(
    circuit: &CircuitSeq,
    data_wires: usize,
    rng: &mut impl Rng,
) -> MqShareCircuit {
    obfuscate_mqshare_with(circuit, data_wires, &MqShareParams::for_n(data_wires), rng)
}

pub fn obfuscate_mqshare_with(
    circuit: &CircuitSeq,
    data_wires: usize,
    params: &MqShareParams,
    rng: &mut impl Rng,
) -> MqShareCircuit {
    assert!(params.ancilla_count >= 4);
    assert!(params.scratch_count >= 2);

    let bank: Vec<u16> = (0..params.ancilla_count)
        .map(|i| (data_wires + i) as u16)
        .collect();
    let scratch: Vec<u16> = (0..params.scratch_count)
        .map(|i| (data_wires + params.ancilla_count + i) as u16)
        .collect();

    let mut comp = Compiler {
        data_wires,
        bank,
        scratch,
        params,
        shares: (0..data_wires)
            .map(|i| Share {
                carrier: i as u16,
                ancillas: Vec::new(),
                mask: Poly::zero(),
            })
            .collect(),
        gates: Vec::new(),
        rng,
    };

    comp.emit_init_scramble();
    for w in 0..data_wires {
        comp.share_wire(w);
    }
    for &g in &circuit.gates {
        for &pin in &g {
            assert!((pin as usize) < data_wires);
        }
        comp.emit_gate57(g);
    }
    for w in 0..data_wires {
        comp.unshare_wire(w);
    }

    let total_wires = comp.total_wires();
    MqShareCircuit {
        data_wires,
        ancilla_count: params.ancilla_count,
        scratch_count: params.scratch_count,
        total_wires,
        gates: comp.gates,
    }
}

pub fn sample_and_obfuscate_mqshare(
    data_wires: usize,
    n_gates: usize,
    rng: &mut impl Rng,
) -> (CircuitSeq, MqShareCircuit) {
    let mut circuit = random_circuit_rng(data_wires, n_gates, rng);
    circuit.canonicalize();
    let obf = obfuscate_mqshare(&circuit, data_wires, rng);
    (circuit, obf)
}

fn random_circuit_rng(n: usize, m: usize, rng: &mut impl Rng) -> CircuitSeq {
    let mut gates = Vec::with_capacity(m);
    for _ in 0..m {
        loop {
            let mut gate = [0u16; 3];
            let mut used = vec![false; n];
            for j in 0..3 {
                loop {
                    let v = rng.random_range(0..n) as u16;
                    if !used[v as usize] {
                        used[v as usize] = true;
                        gate[j] = v;
                        break;
                    }
                }
            }
            if gates.last() == Some(&gate) {
                continue;
            }
            gates.push(gate);
            break;
        }
    }
    CircuitSeq { gates }
}

fn eval_plain_data_bits(plain: &CircuitSeq, data_bits: &[bool]) -> Vec<bool> {
    let n = data_bits.len();
    if n <= usize::BITS as usize {
        let mut state = 0usize;
        for (i, &b) in data_bits.iter().enumerate() {
            if b {
                state |= 1 << i;
            }
        }
        let out = plain.evaluate(state);
        return (0..n).map(|i| ((out >> i) & 1) != 0).collect();
    }
    let mut bits = data_bits.to_vec();
    for g in &plain.gates {
        let c1 = bits[g[1] as usize];
        let c2 = bits[g[2] as usize];
        bits[g[0] as usize] ^= c1 || !c2;
    }
    bits
}

pub fn check_correctness_random(
    plain: &CircuitSeq,
    obf: &MqShareCircuit,
    trials: usize,
    rng: &mut impl Rng,
) -> Result<(), String> {
    let n = obf.data_wires;
    for t in 0..trials {
        let mut input = vec![false; n];
        for b in &mut input {
            *b = rng.random();
        }
        let expect = eval_plain_data_bits(plain, &input);
        let got = obf.evaluate_data_bits(&input);
        if got != expect {
            return Err(format!("mismatch on trial {t}"));
        }
    }
    Ok(())
}

pub fn check_correctness_random_dirty_ancilla(
    plain: &CircuitSeq,
    obf: &MqShareCircuit,
    trials: usize,
    rng: &mut impl Rng,
) -> Result<(), String> {
    let n = obf.data_wires;
    for t in 0..trials {
        let mut full = vec![false; obf.total_wires];
        for b in &mut full {
            *b = rng.random();
        }
        let expect = eval_plain_data_bits(plain, &full[..n]);
        let got = obf.evaluate_data_bits_full(&full);
        if got != expect {
            return Err(format!("dirty-ancilla mismatch on trial {t}"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn bits_from_u64(x: u64, n: usize) -> Vec<bool> {
        (0..n).map(|i| ((x >> i) & 1) != 0).collect()
    }

    fn bits_to_u64(bits: &[bool]) -> u64 {
        let mut y = 0u64;
        for (i, &b) in bits.iter().enumerate() {
            if b {
                y |= 1 << i;
            }
        }
        y
    }

    #[test]
    fn no_shared_pins() {
        let mut rng = StdRng::seed_from_u64(1);
        let (_p, obf) = sample_and_obfuscate_mqshare(5, 10, &mut rng);
        for g in &obf.gates {
            assert!(g[0] != g[1] && g[0] != g[2] && g[1] != g[2], "{g:?}");
        }
    }

    #[test]
    fn single_gate_exhaustive() {
        let mut rng = StdRng::seed_from_u64(1);
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let params = MqShareParams {
            ancilla_count: 8,
            scratch_count: 4,
            init_gates: 0,
            share_frac: 0.5,
        };
        let obf = obfuscate_mqshare_with(&circuit, 3, &params, &mut rng);
        for x in 0..8u64 {
            let expect = circuit.evaluate(x as usize) as u64;
            let got = bits_to_u64(&obf.evaluate_data_bits(&bits_from_u64(x, 3)));
            assert_eq!(got, expect, "x={x}");
        }
    }

    #[test]
    fn single_gate_full_dirty() {
        let mut rng = StdRng::seed_from_u64(2);
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2]],
        };
        let params = MqShareParams {
            ancilla_count: 8,
            scratch_count: 4,
            init_gates: 0,
            share_frac: 0.5,
        };
        let obf = obfuscate_mqshare_with(&circuit, 3, &params, &mut rng);
        for x in 0..8u64 {
            let expect = circuit.evaluate(x as usize) as u64;
            for _ in 0..8 {
                let mut full = bits_from_u64(x, 3);
                full.resize(obf.total_wires, false);
                for b in full.iter_mut().skip(3) {
                    *b = rng.random();
                }
                let got = bits_to_u64(&obf.evaluate_data_bits_full(&full));
                assert_eq!(got, expect, "dirty x={x}");
            }
        }
    }

    #[test]
    fn two_gates_overlap() {
        let mut rng = StdRng::seed_from_u64(3);
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2], [0, 1, 2]],
        };
        let params = MqShareParams {
            ancilla_count: 8,
            scratch_count: 4,
            init_gates: 4,
            share_frac: 0.5,
        };
        let obf = obfuscate_mqshare_with(&circuit, 3, &params, &mut rng);
        for x in 0..8u64 {
            let expect = circuit.evaluate(x as usize) as u64;
            assert_eq!(
                bits_to_u64(&obf.evaluate_data_bits(&bits_from_u64(x, 3))),
                expect
            );
        }
    }

    #[test]
    fn random_small_correct() {
        let mut rng = StdRng::seed_from_u64(7);
        let (plain, obf) = sample_and_obfuscate_mqshare(6, 12, &mut rng);
        check_correctness_random(&plain, &obf, 40, &mut rng).unwrap();
        check_correctness_random_dirty_ancilla(&plain, &obf, 20, &mut rng).unwrap();
    }

    fn decode(bits: &[bool], share: &Share) -> bool {
        bits[share.carrier as usize] ^ share.mask.eval(bits)
    }

    /// Compile with checkpoints after share and after each homomorphic gate.
    fn traced_obfuscate(
        circuit: &CircuitSeq,
        data_wires: usize,
        params: &MqShareParams,
        rng: &mut impl Rng,
    ) -> (MqShareCircuit, Vec<(usize, Vec<Share>)>) {
        let bank: Vec<u16> = (0..params.ancilla_count)
            .map(|i| (data_wires + i) as u16)
            .collect();
        let scratch: Vec<u16> = (0..params.scratch_count)
            .map(|i| (data_wires + params.ancilla_count + i) as u16)
            .collect();
        let mut comp = Compiler {
            data_wires,
            bank,
            scratch,
            params,
            shares: (0..data_wires)
                .map(|i| Share {
                    carrier: i as u16,
                    ancillas: Vec::new(),
                    mask: Poly::zero(),
                })
                .collect(),
            gates: Vec::new(),
            rng,
        };
        let mut checkpoints = Vec::new();
        comp.emit_init_scramble();
        for w in 0..data_wires {
            comp.share_wire(w);
        }
        checkpoints.push((comp.gates.len(), comp.shares.clone()));
        for &g in &circuit.gates {
            comp.emit_gate57(g);
            for s in &comp.shares {
                assert!(
                    s.mask.term_count() > 0,
                    "empty mask mid-circuit would be cleartext encoding"
                );
            }
            checkpoints.push((comp.gates.len(), comp.shares.clone()));
        }
        for w in 0..data_wires {
            comp.unshare_wire(w);
        }
        let total_wires = comp.total_wires();
        (
            MqShareCircuit {
                data_wires,
                ancilla_count: params.ancilla_count,
                scratch_count: params.scratch_count,
                total_wires,
                gates: comp.gates,
            },
            checkpoints,
        )
    }

    #[test]
    fn no_plaintext_midcircuit_sharing_invariant() {
        // After share and after every gate (before unshare):
        //   decode(carrier, mask) == logical value, and mask is non-empty.
        // Never clears masks to run a bare plaintext gate.
        let mut rng = StdRng::seed_from_u64(42);
        let circuit = CircuitSeq {
            gates: vec![[0, 1, 2], [1, 0, 2], [2, 0, 1]],
        };
        let params = MqShareParams {
            ancilla_count: 8,
            scratch_count: 4,
            init_gates: 8,
            share_frac: 0.5,
        };
        let n = 3;
        let (obf, checkpoints) = traced_obfuscate(&circuit, n, &params, &mut rng);

        for trial in 0..30 {
            let mut full = vec![false; obf.total_wires];
            for b in &mut full {
                *b = rng.random();
            }
            let logical_in: Vec<bool> = full[..n].to_vec();

            for (cp_idx, (prefix_len, shares)) in checkpoints.iter().enumerate() {
                let mut bits = full.clone();
                eval_gates(&mut bits, &obf.gates[..*prefix_len]);

                let mut logical = logical_in.clone();
                let n_logical_gates = cp_idx; // 0 = post-share
                for g in circuit.gates.iter().take(n_logical_gates) {
                    let c1 = logical[g[1] as usize];
                    let c2 = logical[g[2] as usize];
                    logical[g[0] as usize] ^= c1 || !c2;
                }

                for (w, share) in shares.iter().enumerate() {
                    assert!(
                        share.mask.term_count() > 0,
                        "trial {trial} cp {cp_idx}: wire {w} empty mask"
                    );
                    assert_eq!(
                        decode(&bits, share),
                        logical[w],
                        "trial {trial} cp {cp_idx}: decode wire {w}"
                    );
                }
            }
        }

        for x in 0..8u64 {
            let expect = circuit.evaluate(x as usize) as u64;
            assert_eq!(
                bits_to_u64(&obf.evaluate_data_bits(&bits_from_u64(x, 3))),
                expect
            );
        }
    }
}
