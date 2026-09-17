use super::*;
use clap::error::ErrorKind;

#[test]
fn automatic_pieces_accept_default_fixed_options() {
    for minimum in [2, 1024, usize::MAX] {
        let args = Args::try_parse_from([
            "circuit_mixer",
            "--input",
            "input.txt",
            "--parallel-target-piece-gates",
            &minimum.to_string(),
        ])
        .unwrap();
        assert_eq!(args.min_block_size, Some(minimum));
        assert_eq!(args.pieces, 1);
        assert_eq!(args.piece_min_len, 1024);
        assert_eq!(args.piece_threads, 0);
        // Automatic mode must run rounds even when the initial count is one.
        assert!(args.piecewise_enabled());
    }
}

#[test]
fn automatic_pieces_reject_invalid_minimum() {
    for minimum in ["0", "1", "-2", "2.5", "abc", "18446744073709551616"] {
        assert!(
            Args::try_parse_from([
                "circuit_mixer",
                "--input",
                "input.txt",
                "--parallel-target-piece-gates",
                minimum,
            ])
            .is_err(),
            "accepted {minimum}"
        );
    }
}

#[test]
fn automatic_pieces_conflict_with_explicit_fixed_options() {
    for (option, value) in [
        ("--parallel-pieces", "1"),
        ("--parallel-pieces", "4"),
        ("--piece-min-len", "1024"),
    ] {
        for automatic_first in [false, true] {
            let automatic = ["--parallel-target-piece-gates", "2"];
            let fixed = [option, value];
            let mut argv = vec!["circuit_mixer", "--input", "input.txt"];
            if automatic_first {
                argv.extend(automatic);
                argv.extend(fixed);
            } else {
                argv.extend(fixed);
                argv.extend(automatic);
            }
            let error = Args::try_parse_from(argv).unwrap_err();
            assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
        }
    }
}

#[test]
fn fixed_piece_defaults_and_floor_remain_supported() {
    let args = Args::try_parse_from(["circuit_mixer", "--input", "input.txt"]).unwrap();
    assert_eq!(args.pieces, 1);
    assert_eq!(args.min_block_size, None);
    assert!(!args.piecewise_enabled());
    let args = Args::try_parse_from([
        "circuit_mixer",
        "--input",
        "input.txt",
        "--parallel-pieces",
        "4",
        "--piece-min-len",
        "64",
    ])
    .unwrap();
    assert_eq!(args.pieces, 4);
    assert_eq!(args.piece_min_len, 64);
    assert_eq!(args.min_block_size, None);
    assert!(args.piecewise_enabled());
}

#[test]
fn descriptive_db_parallel_and_leakage_flags_parse_and_appear_in_help() {
    let args = Args::try_parse_from([
        "circuit_mixer",
        "--input",
        "input.txt",
        "--db-mixing",
        "--parallel-pieces",
        "4",
        "--parallel-threads",
        "5",
        "--leakage-repair",
        "--leakage-repair-seed",
        "91",
    ])
    .unwrap();
    assert!(args.db_mixing);
    assert_eq!((args.pieces, args.piece_threads), (4, 5));
    assert!(args.quality.qc);
    assert_eq!(args.quality.qc_seed, 91);
    let error = Args::try_parse_from([
        "circuit_mixer",
        "--input",
        "input.txt",
        "--parallel-pieces",
        "4",
        "--parallel-target-piece-gates",
        "10000",
    ])
    .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::ArgumentConflict);
    let args = Args::try_parse_from([
        "circuit_mixer",
        "--input",
        "input.txt",
        "--parallel-target-piece-gates",
        "10000",
    ])
    .unwrap();
    assert_eq!(args.min_block_size, Some(10000));
    let help = Args::try_parse_from(["circuit_mixer", "--help"])
        .unwrap_err()
        .to_string();
    assert!(help.contains("--db-mixing"));
    assert!(help.contains("--leakage-repair"));
}
