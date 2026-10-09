//! `stage2_typed_params_byte_identical` (docs/STAGE4_RUST_DESIGN.md §11.4):
//! the stage-2 construction knobs that used to be environment reads inside
//! the library are typed `EmbeddedMaskingParams` fields now, mapped from the
//! same environment names only by gen's adapter. For every case of
//! `tests/fixtures/stage2_typed/cases.txt` (presets A and L8, two retirement
//! configurations, chains/rotation/ancilla reserve/drift window/legacy slice
//! guard, and drift off) and three calibration seeds, gen's three outputs
//! must hash exactly as the PRE-refactor binary's did (`expected.txt`).
use local_mixing::stages::packing::blake2b::blake2b;
use std::fs;
use std::path::Path;
use std::process::Command;

#[test]
fn stage2_typed_params_byte_identical() {
    let fx = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage2_typed");
    let expected: Vec<(String, String, String)> = fs::read_to_string(fx.join("expected.txt"))
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            (f[0].to_string(), f[1].to_string(), f[2].to_string())
        })
        .collect();
    let dir = std::env::temp_dir().join(format!("stage2_typed_{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let mut checked = 0;
    for line in fs::read_to_string(fx.join("cases.txt")).unwrap().lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let (name, env) = line.split_once('|').unwrap();
        let name = name.trim();
        for seed in ["11", "1013", "424242"] {
            let out = dir.join(format!("{name}_{seed}.mpmct1"));
            let mut cmd = Command::new(env!("CARGO_BIN_EXE_gen_sandwich_gadget"));
            cmd.env_clear().env("PATH", "/usr/bin:/bin");
            for kv in env.split_whitespace() {
                let (k, v) = kv.split_once('=').unwrap();
                cmd.env(k, v);
            }
            let o = cmd
                .arg(&out)
                .args(["16", "32", "32", "64", "1", "320", seed, seed, seed])
                .output()
                .unwrap();
            assert!(
                o.status.success(),
                "{name} {seed}: {}",
                String::from_utf8_lossy(&o.stderr)
            );
            let mut bytes = Vec::new();
            for suffix in ["", ".source_c.g57", ".sandwich.mpmct1"] {
                let mut p = out.as_os_str().to_owned();
                p.push(suffix);
                bytes.extend(fs::read(&p).unwrap());
            }
            let got: String = blake2b(&bytes, 32)
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            let want = &expected
                .iter()
                .find(|(n, s, _)| n == name && s == seed)
                .unwrap_or_else(|| panic!("no expected hash for {name} {seed}"))
                .2;
            assert_eq!(&got, want, "{name} seed {seed}: stage-2 bytes changed");
            checked += 1;
        }
    }
    assert_eq!(checked, expected.len());
    let _ = fs::remove_dir_all(dir);
}
