//! Masking-safe native SAMF packets for heterogeneous circuits.
//!
//! A normal three-CNOT XOR swap temporarily writes `a XOR b` to one wire. If
//! `a` and `b` are the two carriers of one shared value, that intermediate is
//! the unmasked logical value. The packets here use a dedicated independent
//! random mask wire `r` and restore it exactly:
//!
//! ```text
//! a ^= r; a ^= b; b ^= a; a ^= b; b ^= r
//! ```
//!
//! The net action is `swap(a,b)`, while every prefix retains either the share
//! mask or `r`. Optional output negations give the four legacy signed-SAMF
//! types. Reversing a packet is its exact circuit inverse and is the native
//! unsamf operation.

use crate::circuit::xgate::XGate;
use rand::Rng;

pub fn masked_swap_packet(a: u16, b: u16, random_mask: u16) -> Vec<XGate> {
    assert!(a != b && a != random_mask && b != random_mask);
    vec![
        XGate::cnot(a, random_mask),
        XGate::cnot(a, b),
        XGate::cnot(b, a),
        XGate::cnot(a, b),
        XGate::cnot(b, random_mask),
    ]
}

/// Signed swap using the same convention as legacy SAMFs after the swap:
/// 0=plain, 1=negate `a`, 2=negate `b`, 3=negate both.
pub fn signed_masked_swap_packet(
    a: u16,
    b: u16,
    random_mask: u16,
    negation_type: u16,
) -> Vec<XGate> {
    let mut packet = masked_swap_packet(a, b, random_mask);
    match negation_type {
        0 => {}
        1 => packet.push(XGate::x_gate(a)),
        2 => packet.push(XGate::x_gate(b)),
        3 => {
            packet.push(XGate::x_gate(a));
            packet.push(XGate::x_gate(b));
        }
        _ => panic!("invalid SAMF negation type {negation_type}"),
    }
    packet
}

pub fn inverse_packet(packet: &[XGate]) -> Vec<XGate> {
    packet.iter().rev().cloned().collect()
}

pub fn conjugate_gate_by_swap(gate: &XGate, a: u16, b: u16) -> XGate {
    let swap = |wire: u16| {
        if wire == a {
            b
        } else if wire == b {
            a
        } else {
            wire
        }
    };
    let mut controls = gate
        .ctrls
        .iter()
        .map(|&(wire, polarity)| (swap(wire), polarity))
        .collect::<crate::circuit::xgate::Lits>();
    controls.sort_unstable();
    XGate {
        target: swap(gate.target),
        comp: gate.comp,
        ctrls: controls,
    }
}

/// Insert disjoint native SAMF/unsamf brackets after all heterogeneous
/// rewrites have finished. Interiors never touch `random_mask`, and processing
/// stops after insertion, so each opening packet starts with the independent
/// helper restored and each closing packet is its exact reverse.
pub fn insert_masked_swap_samfs(
    gates: &mut Vec<XGate>,
    data_wires: usize,
    random_mask: u16,
    requested: usize,
    rng: &mut impl Rng,
) -> usize {
    assert_eq!(random_mask as usize, data_wires);
    assert!(
        gates
            .iter()
            .all(|gate| gate.target != random_mask && !gate.reads(random_mask)),
        "dedicated SAMF mask wire is already in use"
    );
    if data_wires < 2 || gates.is_empty() || requested == 0 {
        return 0;
    }

    let original_len = gates.len();
    let count = requested.min(original_len);
    let chunk = original_len.div_ceil(count);
    let mut edits = Vec::with_capacity(count);
    for index in 0..count {
        let start = index * chunk;
        if start >= original_len {
            break;
        }
        let end = ((index + 1) * chunk).min(original_len);
        let touched: Vec<u16> = gates[start..end]
            .iter()
            .flat_map(|gate| {
                std::iter::once(gate.target).chain(gate.ctrls.iter().map(|&(wire, _)| wire))
            })
            .filter(|&wire| (wire as usize) < data_wires)
            .collect();
        if touched.is_empty() {
            continue;
        }
        let a = touched[rng.random_range(0..touched.len())];
        let b = loop {
            let candidate = rng.random_range(0..data_wires) as u16;
            if candidate != a {
                break candidate;
            }
        };
        edits.push((start, end, a, b));
    }

    for &(start, end, a, b) in edits.iter().rev() {
        for gate in &mut gates[start..end] {
            *gate = conjugate_gate_by_swap(gate, a, b);
        }
        let opening = masked_swap_packet(a, b, random_mask);
        let closing = inverse_packet(&opening);
        gates.splice(end..end, closing);
        gates.splice(start..start, opening);
    }
    edits.len()
}

#[cfg(test)]
#[path = "../../../tests/stages/preprocessing/samf_tests.rs"]
mod tests;
