use crate::circuit::{CircuitSeq, Gate};

use rand::Rng;

// Computes a completely random circuit on n wires and m gates
pub fn random_circuit(n: usize, m: usize) -> CircuitSeq {
    random_circuit_with_draw(n, m, |upper| fastrand::usize(..upper))
}

fn random_circuit_with_draw(
    n: usize,
    m: usize,
    mut draw: impl FnMut(usize) -> usize,
) -> CircuitSeq {
    assert!(n >= 3, "random circuits need at least 3 wires");
    assert!(
        n <= u16::MAX as usize + 1,
        "random circuit wire count exceeds u16 wire indices"
    );
    let mut circuit = Vec::with_capacity(m);

    for _ in 0..m {
        loop {
            // pick 3 distinct pins
            let mut gate = [0u16; 3];
            for j in 0..3 {
                loop {
                    let v = draw(n) as u16;
                    if !gate[..j].contains(&v) {
                        gate[j] = v;
                        break;
                    }
                }
            }

            // check against last gate to avoid duplicates
            if circuit.last() == Some(&gate) {
                continue;
            } else {
                circuit.push(gate);
                break;
            }
        }
    }

    CircuitSeq { gates: circuit }
}

// Shoots a random gate left or right without collisions
pub fn shoot_random_gate(circuit: &mut CircuitSeq, rounds: usize) {
    let mut rng = rand::rng();
    let len = circuit.gates.len();

    if len == 0 {
        return;
    }

    for _ in 0..rounds {
        let gate_idx = rng.random_range(0..len);
        let go_left: bool = rng.random_bool(0.5);

        if go_left {
            // Shoot left
            let mut target = gate_idx;
            while target > 0 {
                if Gate::collides_index(&circuit.gates[target - 1], &circuit.gates[gate_idx]) {
                    break;
                }
                target -= 1;
            }
            target = rng.random_range(target..=gate_idx);
            if target != gate_idx {
                let gate = circuit.gates.remove(gate_idx);
                circuit.gates.insert(target, gate);
            }
        } else {
            // Shoot right
            let mut target = gate_idx;
            while target + 1 < len {
                if Gate::collides_index(&circuit.gates[target + 1], &circuit.gates[gate_idx]) {
                    break;
                }
                target += 1;
            }
            target = rng.random_range(gate_idx..=target);
            if target != gate_idx {
                let gate = circuit.gates.remove(gate_idx);
                circuit.gates.insert(target, gate);
            }
        }
    }
}

#[cfg(test)]
#[path = "../../tests/unit/circuit/randomize/tests.rs"]
mod tests;

/// Draw a wire outside the excluded coordinates, preserving rejection sampling.
pub(crate) fn random_wire_except(
    total: usize,
    excluded: &[usize],
    rng: &mut impl rand::Rng,
) -> usize {
    loop {
        let w = rng.random_range(0..total);
        if !excluded.contains(&w) {
            return w;
        }
    }
}
