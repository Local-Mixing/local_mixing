//! `hmap_affine --exposure` (now on the library C-view engine,
//! `stages::packing::cview`) against outputs of the PINNED baseline binary
//! built from 492564f0 (`tests/fixtures/stage4/cview_n8/regen.sh` refuses any
//! other binary). Every case must reproduce stdout and every TSV byte for
//! byte, and must not write a TSV the baseline did not write.
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage4/cview_n8")
}

fn scratch() -> PathBuf {
    let path = std::env::temp_dir().join(format!(
        "tdp_hmap_affine_exposure_{}_{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let _ = fs::remove_dir_all(&path);
    fs::create_dir_all(&path).unwrap();
    path
}

struct Case {
    name: String,
    allcuts: bool,
    witness: bool,
    args: Vec<String>,
}

fn cases() -> Vec<Case> {
    let dir = fixture_dir();
    let text = fs::read_to_string(dir.join("cases.tsv")).unwrap();
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.splitn(4, '\t').collect();
            assert_eq!(f.len(), 4, "cases.tsv line: {l}");
            Case {
                name: f[0].to_string(),
                allcuts: f[1] == "1",
                witness: f[2] == "1",
                args: f[3]
                    .split_whitespace()
                    .map(|a| match a.strip_prefix("@/") {
                        Some(rest) => dir.join(rest).to_string_lossy().into_owned(),
                        None => a.to_string(),
                    })
                    .collect(),
            }
        })
        .collect()
}

#[test]
fn exposure_outputs_match_the_pinned_baseline() {
    let expected = fixture_dir().join("expected");
    let cases = cases();
    assert!(cases.len() >= 9, "fixture cases present");
    for case in cases {
        let work = scratch();
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_hmap_affine"));
        cmd.env_clear()
            .env("PATH", "/usr/bin:/bin")
            .current_dir(&work)
            .args(&case.args)
            .args(["--out", &case.name]);
        if case.allcuts {
            cmd.env("EXPOSURE_ALLCUTS", format!("{}.allcuts.tsv", case.name));
        }
        if case.witness {
            cmd.env("EXPOSURE_WITNESS", format!("{}.wit.tsv", case.name));
        }
        let out = cmd.output().unwrap();
        assert!(
            out.status.success(),
            "{}: {}",
            case.name,
            String::from_utf8_lossy(&out.stderr)
        );
        let want_stdout = fs::read(expected.join(format!("{}.stdout", case.name))).unwrap();
        assert!(
            out.stdout == want_stdout,
            "{}: stdout differs\n--- got\n{}\n--- want\n{}",
            case.name,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&want_stdout)
        );
        for (suffix, on) in [
            ("exposure.tsv", true),
            ("allcuts.tsv", case.allcuts),
            ("wit.tsv", case.witness),
        ] {
            let file = format!("{}.{suffix}", case.name);
            let got = work.join(&file);
            let want = expected.join(&file);
            assert_eq!(got.exists(), on, "{file}: produced iff requested");
            assert_eq!(want.exists(), on, "{file}: baseline produced iff requested");
            if on {
                assert!(
                    fs::read(&got).unwrap() == fs::read(&want).unwrap(),
                    "{file} differs from the baseline"
                );
            }
        }
        let _ = fs::remove_dir_all(&work);
    }
}
