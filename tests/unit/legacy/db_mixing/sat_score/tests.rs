use super::{output_cone_stats, output_cone_stats_for_wires, score_subcircuit};

#[test]
fn empty_circuit_scores_zero() {
    let score = score_subcircuit(&[], 4, 1);
    assert_eq!(score.score, 0.0);
    assert_eq!(score.gates, 0);
    assert_eq!(score.wires, 0);
}

#[test]
fn mixed_circuit_scores_above_repeated_template() {
    let repeated = [[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]];
    let mixed = [[0, 1, 2], [3, 0, 4], [5, 3, 6], [7, 5, 0]];

    let repeated_score = score_subcircuit(&repeated, 8, 7);
    let mixed_score = score_subcircuit(&mixed, 8, 7);

    assert!(mixed_score.score > repeated_score.score);
    assert!(mixed_score.nonlinear_depth > repeated_score.nonlinear_depth);
    assert!(repeated_score.repeated_template_penalty > mixed_score.repeated_template_penalty);
}

#[test]
fn output_cone_counts_only_relevant_active_writes() {
    let gates = [[0, 1, 2], [3, 0, 4], [5, 3, 6], [7, 5, 0]];
    let stats = output_cone_stats(&gates, 7, 1).unwrap();

    assert_eq!(stats.cone_gates, 4);
    assert_eq!(stats.cone_wires, 8);
    assert_eq!(stats.cone_input_wires, 8);

    let unrelated = output_cone_stats(&gates, 8, 1).unwrap();
    assert_eq!(unrelated.cone_gates, 0);
    assert_eq!(unrelated.cone_wires, 1);
}

#[test]
fn output_cone_counts_noncontiguous_output_wires() {
    let gates = [[0, 1, 2], [3, 4, 5], [6, 0, 3]];
    let stats = output_cone_stats_for_wires(&gates, &[0, 3]).unwrap();

    assert_eq!(stats.output_bits, 2);
    assert_eq!(stats.cone_gates, 2);
    assert_eq!(stats.cone_wires, 6);
}

#[test]
fn bcp_resistance_stays_in_unit_interval() {
    let gates = [[0, 1, 2], [3, 0, 4], [5, 3, 6], [7, 5, 0]];
    let score = super::bcp_resistance(&gates, 8, &[7], 123);

    assert!((0.0..=1.0).contains(&score));
}

#[test]
fn unit_propagation_detects_simple_contradiction() {
    let clauses = vec![vec![1], vec![-1]];
    let result = super::unit_propagate(&clauses, 1, &[]);

    assert!(result.is_none());
}
