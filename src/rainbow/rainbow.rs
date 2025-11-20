use crate::circuit::circuit;
use crate::{
    circuit::{CircuitSeq, Permutation},
    rainbow::canonical::PermStore,
};
use crate::rainbow::{PersistPermStore, canonical};
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
use crate::random::random_data::base_gates;
use smallvec::SmallVec;
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

pub fn build_from(
    num_wires: usize,
    num_gates: usize,
    store: &Arc<HashMap<Vec<u8>, PersistPermStore>>,
) -> impl ParallelIterator<Item = Vec<usize>>{
    let n_base = base_gates(num_wires).len();
    store
        .par_iter() // parallelize over each stored permutation since threads are independent
        .flat_map_iter(move |(_key, perm)| {
            // iterator over the circuits for a given perm
            perm.circuits.iter().flat_map(move |circuit| { // use iter to avoid using more threads than the max
                // allocate prefix on stack (max 64 gates, adjust if needed)
                // use smallvec for stack efficiency
                let mut prefix: SmallVec<[usize; 64]> = SmallVec::with_capacity(num_gates);
                for &b in circuit.iter() { 
                    prefix.push(b as usize);
                }

                // iterator over base_gates, yielding q1 and q2
                (0..n_base).flat_map(move |idx| {
                        // skip consecutive duplicate
                        // before, we had s-1, but s was already num_gates - 1
                        if num_gates >= 2 && idx == prefix[num_gates - 2] {
                            return None;
                        }

                        // q1 = prefix + g
                        let mut q1 = SmallVec::<[usize; 64]>::with_capacity(num_gates);
                        q1.extend_from_slice(&prefix[..num_gates - 1]);
                        q1.push(idx);

                        // q2 = g + prefix
                        let mut q2 = SmallVec::<[usize; 64]>::with_capacity(num_gates);
                        q2.push(idx);
                        q2.extend_from_slice(&prefix[..num_gates - 1]);

                        // yield both q1 and q2 as Vec<usize>
                        Some([q1.into_vec(), q2.into_vec()])
                    })
                    .flatten() // flatten the array of Vecs into iterator
            })
        })
}

pub fn build_circuit_rayon(
    n: usize,
    m: usize,
    circuits: impl ParallelIterator<Item = Vec<usize>> + Send,
    base_gates: Arc<Vec<[u8;3]>>,
) -> impl ParallelIterator<Item = PR> + Send {
    circuits.map(move |circuit| CircuitSeq { gates: circuit.iter().map(|&i| { base_gates[i]}).collect(), })
        .flat_map(move |mut c| {
            CKT_CHECK.fetch_add(1, Ordering::Relaxed);

            c.canonicalize();

            if c.adjacent_id() {
                SKIP_ID.fetch_add(1, Ordering::Relaxed);
                return vec![].into_par_iter();
            }

            let per = c.permutation(n);
            let can_per = per.canonical();
            let is_canonical = per == can_per.perm;

            vec![PR {
                p: can_per.perm,
                r: c.repr_blob(),
                canonical: is_canonical,
            }].into_par_iter()
        })
}

/// Process a single PR and update DashMap store
fn process_pr(pr: PR, circuit_store: &DashMap<Vec<u8>, PermStore>) {
    CKT_I.fetch_add(1, Ordering::Relaxed);

    let p = &pr.p;
    let ph = p.repr_blob();
    let ip = p.invert();
    let own_inv = *p == ip;

    if !own_inv && circuit_store.contains_key(&ip.repr_blob()) {
        SKIP_INV.fetch_add(1, Ordering::Relaxed);
        return;
    }

    if own_inv { OWN_INV_COUNT.fetch_add(1, Ordering::Relaxed); }

    let mut store = circuit_store.entry(ph.clone())
        .or_insert_with(|| PermStore::new_perm_store(p.clone()));

    if pr.canonical {
        if store.contains_canonical { store.add_circuit(&pr.r); }
        else { store.replace(&pr.r); }
        store.contains_canonical = true;
    } else if !store.contains_any_circuit { store.add_circuit(&pr.r); }
    else { store.increment(); }
    store.contains_any_circuit = true;

    N_PERMS.fetch_add(1, Ordering::Relaxed);
}

//scratch
//load old circuits
//append + prepend
//do removal checks
//add to dashmap (perm_blob, permstore)
pub fn expand_m1(
    n: usize,
    circuit_store: &DashMap<Vec<u8>, Vec<Vec<u8>>>,
) {
    let gates = base_gates(n);

    for &gate in gates.iter() {
        // build circuit length 1
        let mut c = CircuitSeq { gates: vec![gate] };
        c.canonicalize();

        // compute permutation
        let p = c.permutation(n);
        let perm_blob = p.repr_blob();
        let c_blob = c.repr_blob();
        let mut entry = circuit_store.entry(perm_blob.clone()).or_insert(Vec::new());
        if !entry.iter().any(|e| e == &c_blob) {
            entry.push(c_blob);
        }
    }
}

pub fn build_and_process_all(
    persist_map: &Arc<std::collections::HashMap<Vec<u8>, Vec<Vec<u8>>>>,
    circuit_store: &DashMap<Vec<u8>, Vec<Vec<u8>>>,
    base_gates: Arc<Vec<[u8; 3]>>,
) {
    // Parallel over persist_map, but inner insertions must NOT hold mutable refs across threads.
    persist_map.par_iter().for_each(|(perm_blob, circuits_list)| {
        circuits_list.par_iter().for_each(|circuit| {
            base_gates.par_iter().for_each(|base_gate| {
                // Prepend
                let mut prepend_version = Vec::with_capacity(3 + circuit.len());
                prepend_version.extend_from_slice(&CircuitSeq::repr_blob_gate(base_gate));
                prepend_version.extend_from_slice(circuit);

                // Append
                let mut append_version = Vec::with_capacity(circuit.len() + 3);
                append_version.extend_from_slice(circuit);
                append_version.extend_from_slice(&CircuitSeq::repr_blob_gate(base_gate));

                let mut entry = circuit_store.entry(perm_blob.clone()).or_insert(Vec::new());
                if !entry.iter().any(|e| e == &prepend_version) {
                    entry.push(prepend_version);
                }
                if !entry.iter().any(|e| e == &append_version) {
                    entry.push(append_version);
                }
            });
        });
    });
}

/// Convert DashMap into HashMap and save
fn save_circuit_store(n: usize, m: usize, circuit_store: &DashMap<Vec<u8>, Vec<Vec<u8>>>) {
    let mut save_map = HashMap::new();
    for r in circuit_store.iter() {
        let v = r.value();
        save_map.insert(
            r.key().clone(),
            v.clone(),
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

/// Main entry for loading existing circuits
pub fn main_rainbow_load(n: usize, m: usize, _load: &str) {
    assert!(n >= 3 && m >= 1, "Invalid circuit size");

    let base_gates = Arc::new(circuit::base_gates(n));

    canonical::init(n);

    if m == 1 {
        let circuit_store: Arc<DashMap<Vec<u8>, Vec<Vec<u8>>>> = Arc::new(DashMap::new());
        let done = Arc::new(AtomicI64::new(0));

        // expand_m1(n, &circuit_store);

        done.store(1, Ordering::Relaxed);

        save_circuit_store(n, m, &circuit_store);
    } else {
        let store = Persist::load(n, m);
        let store_arc = Arc::new(store);
        let prev_count: i64 = store_arc.values().map(|p| p.len() as i64).sum();
        let total_circuits = 2 * prev_count * (base_gates.len() as i64);

        let circuit_store: Arc<DashMap<Vec<u8>, Vec<Vec<u8>>>> = Arc::new(DashMap::new());
        let done = Arc::new(AtomicI64::new(0));

        spawn_progress_tracker(total_circuits, Arc::clone(&done));

        build_and_process_all(&store_arc, &circuit_store, base_gates);

        done.store(1, Ordering::Relaxed);

        save_circuit_store(n, m, &circuit_store);
    }
}