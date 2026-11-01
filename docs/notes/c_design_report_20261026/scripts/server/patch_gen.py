from pathlib import Path
p=Path("generator/src/main.rs");s=p.read_text()
s=s.replace('        "--mask-count",\n    ];','        "--mask-count",\n        "--source-file",\n    ];',1)
old="    let source = prepare_source(n, m, seed, None).unwrap();"
new='''    let supplied = args
        .iter()
        .position(|a| a == "--source-file")
        .map(|i| {
            // g57 triples `a x y` per line (a ^= x OR NOT y), optional `#` comments.
            let text = fs::read_to_string(&args[i + 1]).expect("read --source-file");
            let gates: Vec<[u16; 3]> = text
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(|l| {
                    let v: Vec<u16> = l.split_whitespace().map(|t| t.parse().expect("wire")).collect();
                    assert!(v.len() == 3 && v[0] != v[1] && v[0] != v[2] && v[1] != v[2], "bad gate line: {l}");
                    [v[0], v[1], v[2]]
                })
                .collect();
            local_mixing::circuit::CircuitSeq { gates }
        });
    let source = prepare_source(n, m, seed, supplied).unwrap();'''
assert old in s and '"--source-file"' in s; s=s.replace(old,new,1); p.write_text(s); print("patched")
