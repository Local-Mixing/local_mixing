use super::*;
use crate::db_mixing::db_replace::polys_equivalent;
use crate::engine::xpoly::XPolyBudget;

fn limits() -> BlockLimits {
    BlockLimits {
        max_span: 100,
        max_gates: 100,
        max_support: 64,
    }
}

fn assert_legal_gather(gates: &[XGate], plan: &ConvexBlock) {
    let mut indices: Vec<usize> = plan.span.clone().collect();
    for (new, &old) in plan.permutation.iter().enumerate() {
        let mut at = indices.iter().position(|&i| i == old).unwrap();
        while at > new {
            assert!(!XGate::collides(
                &gates[indices[at - 1]],
                &gates[indices[at]]
            ));
            indices.swap(at - 1, at);
            at -= 1;
        }
    }
    assert_eq!(indices, plan.permutation);
    assert_eq!(
        polys_equivalent(
            &gates[plan.span.clone()],
            &plan.reordered_span(gates),
            XPolyBudget::default()
        ),
        Some(true)
    );
    let gathered = plan.reordered_span(gates);
    assert_eq!(
        &gathered[plan.block.start - plan.span.start..plan.block.end - plan.span.start],
        &plan.window(gates)
    );
}

#[test]
fn includes_every_alternate_dependency_path() {
    // 0 -> 1 -> 3 and 0 -> 2 -> 3, despite 1 and 2 commuting.
    let gates = vec![
        XGate::x_gate(0),
        XGate::cnot(1, 0),
        XGate::cnot(2, 0),
        XGate::conj(3, [(1, true), (2, true)]).unwrap(),
    ];
    let plan = plan_convex_block(&gates, 0, 3, limits()).unwrap();
    assert_eq!(plan.selected, vec![0, 1, 2, 3]);
    assert_legal_gather(&gates, &plan);
}

#[test]
fn gathers_commuting_endpoints_around_ancestors_and_descendants() {
    let gates = vec![
        XGate::cnot(0, 1),
        XGate::cnot(3, 0),
        XGate::x_gate(2),
        XGate::x_gate(5),
        XGate::cnot(0, 2),
    ];
    let plan = plan_convex_block(&gates, 0, 4, limits()).unwrap();
    assert!(!XGate::collides(&gates[0], &gates[4]));
    // 1 lies on a real path 0 -> 1 -> 4, though the endpoints commute.
    assert_eq!(plan.selected, vec![0, 1, 4]);
    assert_eq!(plan.permutation, vec![2, 0, 1, 4, 3]);
    assert_legal_gather(&gates, &plan);
}

#[test]
fn excludes_independent_gap_and_keeps_global_indices() {
    let gates = vec![
        XGate::x_gate(7),
        XGate::x_gate(0),
        XGate::x_gate(6),
        XGate::x_gate(0),
        XGate::x_gate(8),
    ];
    let plan = plan_convex_block(&gates, 1, 3, limits()).unwrap();
    assert_eq!(plan.selected, vec![1, 3]);
    assert_eq!(plan.permutation, vec![1, 3, 2]);
    assert_eq!(plan.block, 1..3);
    assert_legal_gather(&gates, &plan);
}

#[test]
fn distinguishes_span_gate_support_and_endpoint_failures() {
    let gates = vec![XGate::x_gate(0), XGate::cnot(1, 0), XGate::cnot(2, 1)];
    assert_eq!(
        plan_convex_block(
            &gates,
            0,
            2,
            BlockLimits {
                max_span: 2,
                ..limits()
            }
        )
        .unwrap_err(),
        BlockError::SpanCap
    );
    assert_eq!(
        plan_convex_block(
            &gates,
            0,
            2,
            BlockLimits {
                max_gates: 2,
                ..limits()
            }
        )
        .unwrap_err(),
        BlockError::GateCap
    );
    assert_eq!(
        plan_convex_block(
            &gates,
            0,
            2,
            BlockLimits {
                max_support: 2,
                ..limits()
            }
        )
        .unwrap_err(),
        BlockError::SupportCap
    );
    assert_eq!(
        plan_convex_block(&gates, 1, 1, limits()).unwrap_err(),
        BlockError::InvalidEndpoints
    );
    assert_eq!(
        plan_convex_block(&gates, 0, 3, limits()).unwrap_err(),
        BlockError::InvalidEndpoints
    );
}

#[test]
fn gathering_preserves_all_collision_edges_for_small_gate_alphabet() {
    let alphabet = [
        XGate::x_gate(0),
        XGate::x_gate(1),
        XGate::cnot(0, 1),
        XGate::cnot(1, 0),
        XGate::cnot(2, 1),
    ];
    for code in 0..alphabet.len().pow(4) {
        let mut code = code;
        let mut gates = Vec::new();
        for _ in 0..4 {
            gates.push(alphabet[code % alphabet.len()].clone());
            code /= alphabet.len();
        }
        let plan = plan_convex_block(&gates, 0, 3, limits()).unwrap();
        assert_legal_gather(&gates, &plan);
    }
}
