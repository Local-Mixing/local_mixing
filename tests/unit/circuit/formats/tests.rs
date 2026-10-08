use super::*;

#[test]
fn read_mpmct_rejects_invalid_control_sets() {
    let cases: [(&str, &str); 3] = [
        ("mpmct1 4 1\n1 0 2 1 1 3 0\n", "control on its own target"),
        ("mpmct1 4 1\n0 0 2 2 1 2 1\n", "duplicate"),
        ("mpmct1 4 1\n0 0 2 2 1 2 0\n", "contradictory"),
    ];
    for (content, what) in cases {
        let dir = std::env::temp_dir();
        let path = dir.join(format!(
            "mpmct_invalid_{}_{}.mpmct1",
            std::process::id(),
            what.len()
        ));
        std::fs::write(&path, content).unwrap();
        let res = read_mpmct(path.to_str().unwrap());
        let _ = std::fs::remove_file(&path);
        assert!(res.is_err(), "case `{what}` was not rejected");
    }
}

#[test]
fn mpmct_writer_matches_golden_bytes_and_roundtrips() {
    let gates = vec![
        XGate::x_gate(12),
        XGate {
            target: 7,
            comp: true,
            ctrls: [(2, false), (9, true)].into_iter().collect(),
        },
        XGate::conj(3, [(1, false), (15, true)]).unwrap(),
    ];
    let path = std::env::temp_dir().join(format!(
        "local_mixing_mpmct_golden_{}.txt",
        std::process::id()
    ));
    let path_str = path.to_str().expect("temporary path is UTF-8");

    write_mpmct(path_str, &gates, 16).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        bytes,
        b"mpmct1 16 3\n12 0 0\n7 1 2 2 0 9 1\n3 0 2 1 0 15 1\n"
    );

    let (roundtrip, num_wires) = read_mpmct(path_str).unwrap();
    assert_eq!(num_wires, 16);
    assert_eq!(roundtrip, gates);
    std::fs::remove_file(path).ok();
}

fn read_str(body: &str, tag: &str) -> io::Result<(Vec<XGate>, usize)> {
    let path = std::env::temp_dir().join(format!(
        "local_mixing_mpmct_{tag}_{}.txt",
        std::process::id()
    ));
    std::fs::write(&path, body).unwrap();
    let out = read_mpmct(path.to_str().unwrap());
    std::fs::remove_file(&path).ok();
    out
}

// The byte reader must accept the same whitespace and blank lines as a
// `lines()` + `split_whitespace()` + `parse()` reader.
#[test]
fn opt_equiv_read_mpmct_keeps_whitespace_and_blank_line_tolerance() {
    let want = vec![
        XGate::x_gate(12),
        XGate {
            target: 7,
            comp: true,
            ctrls: [(2, false), (9, true)].into_iter().collect(),
        },
    ];
    // Blank lines, CRLF, leading/inner/trailing spaces and tabs, a
    // trailing newline, and out-of-order controls that must be sorted.
    for (tag, body) in [
        ("plain", "mpmct1 16 2\n12 0 0\n7 1 2 2 0 9 1\n"),
        ("noeol", "mpmct1 16 2\n12 0 0\n7 1 2 2 0 9 1"),
        ("blank", "mpmct1 16 2\n12 0 0\n\n   \n7 1 2 2 0 9 1\n\n"),
        ("crlf", "mpmct1 16 2\r\n12 0 0\r\n7 1 2 2 0 9 1\r\n"),
        (
            "spacey",
            "mpmct1  16   2\n  12 0 0  \n\t7  1 2   2 0 9 1\t\n",
        ),
        ("unsorted", "mpmct1 16 2\n12 0 0\n7 1 2 9 1 2 0\n"),
    ] {
        let (got, wires) = read_str(body, tag).unwrap_or_else(|e| panic!("{tag}: {e}"));
        assert_eq!(wires, 16, "{tag}");
        assert_eq!(got, want, "{tag}");
    }
}

#[test]
fn read_mpmct_rejects_malformed_input() {
    assert!(read_str("", "empty").is_err());
    assert!(read_str("g57 16 2\n", "wrongtag").is_err());
    assert!(read_str("mpmct1 16\n", "shorthdr").is_err());
    assert!(read_str("mpmct1 x 2\n", "nonnumhdr").is_err());
    // k=2 announced but only one control pair present.
    assert!(read_str("mpmct1 16 1\n7 1 2 2 0\n", "shortgate").is_err());
    // Header count disagrees with the body.
    assert!(read_str("mpmct1 16 5\n12 0 0\n", "countdrift").is_err());
}

// The packed writers (`esop1`/`anf1`) lost their production callers with the
// historic `fcompress` stage; this pins their bytes and the round trip through
// both readers (`read_packed` and the expanding `read_mpmct` dispatch).
#[test]
fn packed_writers_match_golden_bytes_and_roundtrip() {
    let mut esop = vec![
        PackedGate {
            target: 4,
            terms: vec![vec![(1, true), (2, false)], vec![], vec![(0, true)]],
        },
        PackedGate {
            target: 0,
            terms: vec![vec![(3, false)]],
        },
    ];
    for g in &mut esop {
        g.sort_terms();
    }
    assert_eq!(esop[0].term_count(), 3);
    assert_eq!(esop[0].terms[0], Vec::<(u16, bool)>::new());
    assert!(!esop[0].is_anf());

    let anf = vec![PackedGate {
        target: 2,
        terms: vec![vec![(0, true)], vec![(1, true), (3, true)]],
    }];
    assert!(anf[0].is_anf());

    let dir = std::env::temp_dir();
    let pid = std::process::id();
    for (tag, packed, is_anf, golden) in [
        (
            "esop1",
            &esop,
            false,
            b"esop1 5 2\n4 3 0 1 0 1 2 1 1 2 0\n0 1 1 3 0\n".as_slice(),
        ),
        ("anf1", &anf, true, b"anf1 5 1\n2 2 1 0 2 1 3\n".as_slice()),
    ] {
        let path = dir.join(format!("local_mixing_packed_{tag}_{pid}.txt"));
        let path_str = path.to_str().expect("temporary path is UTF-8");
        if is_anf {
            write_anf1(path_str, packed, 5).unwrap();
        } else {
            write_esop1(path_str, packed, 5).unwrap();
        }
        assert_eq!(std::fs::read(&path).unwrap(), golden, "{tag} bytes");

        let (back, wires) = read_packed(path_str).unwrap();
        assert_eq!((wires, &back), (5, packed), "{tag} read_packed");
        if is_anf {
            assert_eq!(read_anf1(path_str).unwrap().0, *packed);
        }
        let (expanded, wires) = read_mpmct(path_str).unwrap();
        assert_eq!(wires, 5);
        assert_eq!(expanded, expand_packed(packed), "{tag} read_mpmct");
        std::fs::remove_file(&path).ok();
    }
}
