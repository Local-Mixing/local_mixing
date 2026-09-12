use super::*;
use crate::circuit::random_circuit;
use crate::db_mixing::replace::Tag;

// Verbatim pre-rotate eviction (Vec::remove + Vec::insert on gates, member
// AND tags) kept as the reference implementation so the rotate-based
// contiguous_convex can be checked against it byte-for-byte.
fn contiguous_convex_reference(
    circuit: &mut CircuitSeq,
    ordered_convex_gates: &mut Vec<usize>,
    num_wires: usize,
    tags: &mut Vec<Tag>,
) -> Option<(usize, usize)> {
    let track = !tags.is_empty();
    if ordered_convex_gates.len() < 2 {
        return None;
    }
    if !is_convex(num_wires, circuit, &ordered_convex_gates) {
        panic!("not convex");
    }
    let mut member = vec![false; circuit.gates.len()];
    for &idx in ordered_convex_gates.iter() {
        member[idx] = true;
    }
    let mut start = *ordered_convex_gates.first().unwrap();
    let mut end = *ordered_convex_gates.last().unwrap();
    loop {
        let mut moved = false;
        let mut p = start + 1;
        while p < end {
            if member[p] {
                p += 1;
                continue;
            }
            let can_left =
                (start..p).all(|i| !Gate::collides_index(&circuit.gates[i], &circuit.gates[p]));
            if can_left {
                let gate = circuit.gates.remove(p);
                circuit.gates.insert(start, gate);
                member.remove(p);
                member.insert(start, false);
                if track {
                    let t = tags.remove(p);
                    tags.insert(start, t);
                }
                start += 1;
                moved = true;
                break;
            }
            let can_right = ((p + 1)..=end)
                .all(|i| !Gate::collides_index(&circuit.gates[i], &circuit.gates[p]));
            if can_right {
                let gate = circuit.gates.remove(p);
                circuit.gates.insert(end, gate);
                member.remove(p);
                member.insert(end, false);
                if track {
                    let t = tags.remove(p);
                    tags.insert(end, t);
                }
                end -= 1;
                moved = true;
                break;
            }
            p += 1;
        }
        if !moved {
            break;
        }
    }
    if (start..=end).any(|i| !member[i]) {
        return None;
    }
    *ordered_convex_gates = (start..=end).collect();
    Some((start, end))
}

#[test]
fn opt_equiv_contiguous_convex_rotate_matches_remove_insert() {
    use rand::SeedableRng;
    let mut exercised = 0usize;
    for seed in 0..200u64 {
        fastrand::seed(seed);
        let num_wires = 6 + (seed % 5) as usize; // 6..=10 wires
        let num_gates = 60 + (seed % 40) as usize;
        let circuit = random_circuit(num_wires, num_gates);
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed ^ 0x5EED);
        let (mut sel, _) = match seed % 3 {
            0 => find_convex_subcircuit_max_wires(30, num_wires, &circuit, &mut rng),
            1 => find_convex_subcircuit_max_gates(21, num_wires, &circuit, &mut rng),
            _ => simple_find_convex_subcircuit(num_wires, &circuit, &mut rng),
        };
        if sel.is_empty() {
            continue;
        }
        sel.sort();
        exercised += 1;

        // Alternate the --track-survivors tag path on and off.
        let track = seed % 2 == 0;
        let mk_tags = |len: usize| -> Vec<Tag> {
            if track {
                (0..len).map(|i| Tag(i as u64)).collect()
            } else {
                Vec::new()
            }
        };

        let mut a = circuit.clone();
        let mut a_sel = sel.clone();
        let mut a_tags = mk_tags(circuit.gates.len());
        let got = contiguous_convex(&mut a, &mut a_sel, num_wires, &mut a_tags);

        let mut b = circuit.clone();
        let mut b_sel = sel.clone();
        let mut b_tags = mk_tags(circuit.gates.len());
        let want = contiguous_convex_reference(&mut b, &mut b_sel, num_wires, &mut b_tags);

        assert_eq!(got, want, "range mismatch at seed {seed}");
        assert_eq!(a.gates, b.gates, "gate order mismatch at seed {seed}");
        assert_eq!(a_sel, b_sel, "selection mismatch at seed {seed}");
        assert_eq!(a_tags, b_tags, "tag order mismatch at seed {seed}");
    }
    assert!(
        exercised >= 10,
        "too few convex sets found ({exercised}) — eviction path not exercised"
    );
}
