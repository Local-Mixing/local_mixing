use super::*;
use crate::stages::packing::mono::Mono;
use crate::stages::packing::pyrand::PyRandom;
use std::path::Path;

type K = Mono<8>;

fn fixture(name: &str) -> String {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/stage4")
        .join(name);
    std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn order(s: &PySet<u64>) -> Vec<u64> {
    s.iter().copied().collect()
}

#[test]
fn dummy_not_reused_on_add() {
    // CPython 3.9.6: {1,2,3}; discard 2; add 10 -> [10, 1, 3] (Appendix A)
    let mut s = PySet::new();
    for k in [1u64, 2, 3] {
        s.add(k);
    }
    s.discard(&2);
    s.add(10);
    assert_eq!(order(&s), vec![10, 1, 3]);
}

#[test]
fn resize_used_le_50000_quadruples_and_gt_50000_doubles() {
    let mut s = PySet::new();
    for k in 0u64..5 {
        s.add(k);
    }
    // fill 5: 5*5 >= 7*3 -> resize(used*4 = 20) -> 32 slots
    assert_eq!(s.table_size(), 32);
    let mut big = PySet::new();
    let mut sizes = Vec::new();
    for k in 0u64..200_000 {
        big.add(k);
        if sizes.last() != Some(&big.table_size()) {
            sizes.push(big.table_size());
        }
    }
    // growth by used*4 up to used <= 50000, then used*2
    assert_eq!(
        sizes,
        vec![
            8, 32, 128, 512, 2048, 8192, 32768, 131_072, 262_144, 524_288
        ]
    );
}

#[test]
fn small_table_purge_newsize_8() {
    // keys 8, 16 (all hash to slot 0 of an 8-slot table) with dummies:
    // purge rebuilds the small table in old table order.
    let mut s = PySet::new();
    for k in [8u64, 16, 24, 32] {
        s.add(k);
    }
    for k in [8u64, 16, 24] {
        s.discard(&k);
    }
    assert_eq!(s.table_size(), 8);
    s.add(40); // fill 5 -> resize(used*4 = 8) -> newsize 16
    assert_eq!(s.table_size(), 16);
    assert_eq!(order(&s), vec![32, 40]); // CPython 3.9.6
    let mut t = PySet::new();
    t.add(1u64);
    t.add(2);
    t.discard(&1);
    t.discard(&2);
    t.add(3);
    t.add(4);
    t.add(5); // fill 5, used 3 -> resize(12) -> 16
    assert_eq!(order(&t), vec![3, 4, 5]);
}

#[test]
fn merge_fast_path_and_slow_path_agree_with_cpython_rules() {
    let mut a = PySet::new();
    for k in [5u64, 13, 21] {
        a.add(k);
    }
    let c = PySet::copy_of(&a); // same mask, no dummies: verbatim
    assert_eq!(order(&c), order(&a));
    a.discard(&13);
    let d = PySet::copy_of(&a); // dummies: insert_clean in table order
    assert_eq!(order(&d), vec![5, 21]);
}

#[test]
fn xor_is_copy_b_then_toggle_a() {
    let mut a = PySet::new();
    let mut b = PySet::new();
    for k in [1u64, 9, 17] {
        a.add(k);
    }
    for k in [9u64, 2] {
        b.add(k);
    }
    let x = PySet::xor(&a, &b);
    let mut manual = PySet::copy_of(&b);
    for k in a.iter() {
        manual.toggle(*k);
    }
    assert_eq!(order(&x), order(&manual));
    let mut y = PySet::copy_of(&a);
    y.xor_assign(&b);
    assert_eq!(y.len(), 3);
    assert!(!y.contains(&9) && y.contains(&2));
}

fn replay_small(text: &str) -> (usize, usize) {
    let mut sets: Vec<PySet<K>> = (0..4).map(|_| PySet::new()).collect();
    let key = |s: &str| K::from_hex(s).expect("trace key fits 512 bits");
    let (mut checked, mut bad) = (0, 0);
    for line in text.lines() {
        let q: Vec<&str> = line.split_whitespace().collect();
        if q.is_empty() {
            continue;
        }
        let ix = |i: usize| q[i].parse::<usize>().unwrap();
        match q[0] {
            "T" => sets = (0..4).map(|_| PySet::new()).collect(),
            "N" => sets[ix(1)] = PySet::new(),
            "L" => sets[ix(1)] = PySet::singleton(key(q[2])),
            "G" => sets[ix(1)].toggle(key(q[2])),
            "A" => sets[ix(1)].add(key(q[2])),
            "D" => {
                sets[ix(1)].discard(&key(q[2]));
            }
            "C" => sets[ix(1)] = PySet::copy_of(&sets[ix(2)]),
            "X" => sets[ix(1)] = PySet::xor(&sets[ix(2)], &sets[ix(3)]),
            "I" => {
                let b = sets[ix(2)].clone();
                sets[ix(1)].xor_assign(&b);
            }
            "O" => {
                checked += 1;
                let got: Vec<String> = sets[ix(1)].iter().map(|k| format!("{k:x}")).collect();
                if got != q[2..] {
                    bad += 1;
                }
            }
            op => panic!("bad trace op {op}"),
        }
    }
    (checked, bad)
}

#[test]
fn replays_cpython39_traces() {
    let (checked, bad) = replay_small(&fixture("pyset39_traces.txt"));
    assert!(checked > 1000, "only {checked} orders in the trace file");
    assert_eq!(
        bad, 0,
        "{bad} of {checked} iteration orders differ from CPython 3.9"
    );
}

const FNV_OFF: u64 = 0xCBF2_9CE4_8422_2325;
const FNV_PRIME: u64 = 0x100_0000_01B3;

fn digest(s: &PySet<K>) -> u64 {
    let mut h = FNV_OFF;
    for k in s.iter() {
        for &l in &k.0 {
            h = (h ^ l).wrapping_mul(FNV_PRIME);
        }
    }
    h
}

fn large_key(idx: u64) -> K {
    let mut k = K::bit(64 + (idx % 400) as usize);
    k.0[0] |= idx;
    k
}

#[test]
fn replays_large_traces_checkpoints() {
    // pyset_selftest.py large_trace: toggles over 2*target keys, 3*target steps
    for line in fixture("pyset39_large.txt").lines() {
        let q: Vec<&str> = line.split_whitespace().collect();
        if q.is_empty() {
            continue;
        }
        let seed: u64 = q[1].parse().unwrap();
        let target: u64 = q[2].parse().unwrap();
        let mut r = PyRandom::new(seed);
        let (universe, steps) = (2 * target, 3 * target);
        let every = steps / 4;
        let mut s: PySet<K> = PySet::new();
        let mut got = Vec::new();
        for step in 1..=steps {
            s.toggle(large_key(r.genrand_u32() as u64 % universe));
            if step % every == 0 {
                let c = PySet::copy_of(&s);
                let x = PySet::xor(&c, &PySet::singleton(large_key(step)));
                got.push(format!(
                    "{}:{:016x}:{:016x}:{:016x}",
                    s.len(),
                    digest(&s),
                    digest(&c),
                    digest(&x)
                ));
            }
        }
        assert_eq!(got, q[3..], "large trace seed {seed} target {target}");
    }
}

#[test]
fn arena_compaction_keeps_order() {
    let mut s: PySet<u64> = PySet::new();
    let mut r = PyRandom::new(3);
    let mut mirror = std::collections::BTreeSet::new();
    for _ in 0..20_000 {
        let k = (r.genrand_u32() % 3000) as u64;
        s.toggle(k);
        if !mirror.remove(&k) {
            mirror.insert(k);
        }
    }
    let mut keys = order(&s);
    keys.sort_unstable();
    assert_eq!(keys, mirror.into_iter().collect::<Vec<_>>());
    assert_eq!(s.len(), keys.len());
}
