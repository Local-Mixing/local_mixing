use crate::circuit::circuit;
use crate::{
    circuit::{CircuitSeq, Gate, Permutation},
    rainbow::canonical::PermStore,
};
use crate::rainbow::canonical;
use crate::rainbow::database::{self, Persist};

use rayon::prelude::*;
use dashmap::DashMap;
use std::collections::HashMap;

use std::sync::{
    Arc,
    atomic::{AtomicI64, Ordering},
};
use std::thread;
use std::time::{Duration, Instant};

// PR struct
#[derive(Clone)]
pub struct PR {
    p: Permutation,
    r: Vec<u8>,
    canonical: bool,
}

// Atomic counters
static N_PERMS: AtomicI64 = AtomicI64::new(0);
static CKT_CHECK: AtomicI64 = AtomicI64::new(0);
static SKIP_INV: AtomicI64 = AtomicI64::new(0);
static SKIP_ID: AtomicI64 = AtomicI64::new(0);
static OWN_INV_COUNT: AtomicI64 = AtomicI64::new(0);
static CKT_I: AtomicI64 = AtomicI64::new(0);

/// Parallel circuit builder returning a ParallelIterator of PR
//TODO: (J: Canonicalises the circuit ( i.e. runs fast canon, brute force canon etc. ). Fix the parallel part.)
pub fn build_circuit_rayon(
    n: usize,
    m: usize,
    circuits: impl ParallelIterator<Item = Vec<usize>> + Send,
    base_gates: Arc<Vec<[usize;3]>>,
) -> impl ParallelIterator<Item = PR> + Send {
    circuits.map(|circuit| CircuitSeq { gates: circuit })
        .flat_map(move |mut c| {
            CKT_CHECK.fetch_add(1, Ordering::Relaxed);

            c.canonicalize(&base_gates);

            if c.adjacent_id() {
                SKIP_ID.fetch_add(1, Ordering::Relaxed);
                return vec![].into_par_iter();
            }

            let per = c.permutation(n, &base_gates);
            let can_per = per.canonical();
            let is_canonical = per == can_per.perm;

            vec![PR {
                p: can_per.perm,
                r: c.repr_bytes(),
                canonical: is_canonical,
            }].into_par_iter()
        })
}


/// Process a single PR and update DashMap store
fn process_pr(pr: PR, circuit_store: &DashMap<String, PermStore>) {
    CKT_I.fetch_add(1, Ordering::Relaxed);

    let p = &pr.p;
    let ph = p.repr();
    let ip = p.invert();
    let own_inv = *p == ip;

    if !own_inv && circuit_store.contains_key(&ip.repr()) {
        SKIP_INV.fetch_add(1, Ordering::Relaxed);
        return;
    }

    if own_inv { OWN_INV_COUNT.fetch_add(1, Ordering::Relaxed); }

    let mut store = circuit_store.entry(ph.clone())
        .or_insert_with(|| PermStore::NewPermStore(p.clone()));

    if pr.canonical {
        if store.contains_canonical { store.add_circuit(&pr.r); }
        else { store.replace(&pr.r); }
        store.contains_canonical = true;
    } else if !store.contains_any_circuit { store.add_circuit(&pr.r); }
    else { store.increment(); }
    store.contains_any_circuit = true;

    N_PERMS.fetch_add(1, Ordering::Relaxed);
}

/// Convert DashMap into HashMap and save
fn save_circuit_store(n: usize, m: usize, circuit_store: &DashMap<String, PermStore>) {
    let mut save_map = HashMap::new();
    for r in circuit_store.iter() {
        let v = r.value();
        save_map.insert(
            r.key().clone(),
            database::make_persist(v.perm.clone(), v.circuits.clone(), v.count)
        );
    }
    Persist::save(n, m, &save_map);

    println!("Canonical perms stored: {}", circuit_store.len());
    println!("Total circuits checked: {}", CKT_CHECK.load(Ordering::Relaxed));
    println!("Skipped trivial ID circuits: {}", SKIP_ID.load(Ordering::Relaxed));
    println!("Skipped inverse circuits: {}", SKIP_INV.load(Ordering::Relaxed));
    println!("Circuits that are own inverse: {}", OWN_INV_COUNT.load(Ordering::Relaxed));
}

/// Spawn a thread to track progress
fn spawn_progress_tracker(total_circuits: i64, done: Arc<AtomicI64>) {
    thread::spawn(move || {
        let start = Instant::now();
        let mut last_count = 0;
        loop {
            thread::sleep(Duration::from_secs(1));
            if done.load(Ordering::Relaxed) != 0 { break; }

            let elapsed = start.elapsed().as_secs_f64();
            let ci = CKT_I.load(Ordering::Relaxed) as f64;
            if elapsed < 1.0 || ci < 10.0 || last_count == 0 { last_count = ci as i64; continue; }

            let kper_second = ci / elapsed / 1000.0;
            let now_kper = (ci - last_count as f64) / 1000.0;
            let eta = ((total_circuits as f64 - ci) / kper_second / 1000.0) as i32;

            println!(
                "@ {:.2}M circuits, now {:.2}k/s, avg {:.2}k/s ETA: {} sec",
                ci / 1_000_000.0,
                now_kper,
                kper_second,
                eta
            );

            last_count = ci as i64;
        }
    });
}

/// Main entry for generating new circuits
pub fn main_rainbow_generate(n: usize, m: usize) {
    assert!(n >= 3 && m >= 1, "Invalid circuit size");

    let base_gates = Arc::new(circuit::base_gates(n));

    canonical::init(n);

    let total_circuits = (base_gates.len() as i64) * ((base_gates.len() - 1) as i64).pow((m-1) as u32);
    let circuits = circuit::par_all_circuits(n, m);

    let circuit_store: Arc<DashMap<String, PermStore>> = Arc::new(DashMap::new());
    let done = Arc::new(AtomicI64::new(0));

    spawn_progress_tracker(total_circuits, Arc::clone(&done));

    // Parallel processing
    build_circuit_rayon(n, m, circuits, base_gates)
        .for_each(|pr| process_pr(pr, &circuit_store));

    done.store(1, Ordering::Relaxed);

    save_circuit_store(n, m, &circuit_store);
}

/// Main entry for loading existing circuits
pub fn main_rainbow_load(n: usize, m: usize, load: &str) {
    assert!(n >= 3 && m >= 1, "Invalid circuit size");

    let base_gates = Arc::new(circuit::base_gates(n));

    canonical::init(n);

    let store = Persist::load(n, m);
    let store_arc = Arc::new(store);
    let prev_count: i64 = store_arc.values().map(|p| p.circuits.len() as i64).sum();
    let total_circuits = 2 * prev_count * (base_gates.len() as i64);

    let circuits = circuit::build_from(n, m, &store_arc);

    let circuit_store: Arc<DashMap<String, PermStore>> = Arc::new(DashMap::new());
    let done = Arc::new(AtomicI64::new(0));

    spawn_progress_tracker(total_circuits, Arc::clone(&done));

    build_circuit_rayon(n, m, circuits, base_gates)
        .for_each(|pr| process_pr(pr, &circuit_store));

    done.store(1, Ordering::Relaxed);

    save_circuit_store(n, m, &circuit_store);
}