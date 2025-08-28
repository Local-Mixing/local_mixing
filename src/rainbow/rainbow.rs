use crate::circuit::control_functions::Gate_Control_Func;
use crate::circuit::circuit;
use crate::{
            rainbow::constants::{self, CONTROL_FUNC_TABLE},
            circuit::{Circuit, Gate, Permutation},
            rainbow::{Canonicalization, CandSet},
            rainbow::canonical::PermStore,
            rainbow::database::make_persist,
            };
use crate::rainbow::canonical;
use std::sync::Mutex;
use std::sync::atomic::{AtomicI64, Ordering, AtomicBool};
use std::sync::{Arc};
use std::collections::HashMap;
use dashmap::DashMap;
use std::time::{Instant, Duration};
use std::thread;
use crossbeam::channel::{unbounded, Receiver};
use std::cmp::min;
use crate::rainbow::database::{self, PersistPermStore, Persist};

const MEMORY_FRACTION: f64 = 0.5;

static N_PERMS: AtomicI64 = AtomicI64::new(0);
static CKT_CHECK: AtomicI64 = AtomicI64::new(0);
static SKIP_INV: AtomicI64 = AtomicI64::new(0);
static CKT_I: AtomicI64 = AtomicI64::new(0);
static SKIP_ID: AtomicI64 = AtomicI64::new(0);
static OWN_INV_COUNT: AtomicI64 = AtomicI64::new(0);

#[derive(Clone)]
struct PR {
    p: Permutation,
    r: String,
    canonical: bool,
}

// Take as input a bunch of circuits and output a batch of permutations
pub fn build_circuit(n: usize, m: usize, workers: usize, circuit_ch: Receiver<Vec<usize>>, base_gates: Arc<Vec<Gate>>) -> Receiver<Vec<PR>> {
    const BATCH_SIZE: usize = 16384;
    let (perm_tx, perm_rx) = unbounded(); 
    let ckt_ch = Arc::new(circuit_ch);
    println!("{} perm workers launching", workers);
    for _ in 0..workers {
        let tx = perm_tx.clone();
        let ckt_ch = Arc::clone(&ckt_ch);
        let base_gates = Arc::clone(&base_gates);

        thread::spawn(move || {
            let mut batch = Vec::with_capacity(BATCH_SIZE);
            let mut ckt_buffer = vec![Gate::default(); m];
            for cc in ckt_ch.iter() {
                CKT_I.fetch_add(1, Ordering::SeqCst);
                for (i, &g) in cc.iter().enumerate() {
                    ckt_buffer[i] = base_gates[g].clone();
                }
                let mut c = Circuit::from_gates(ckt_buffer.clone());
                c.num_wires = n; //check on this later
                c.canonicalize();
                if c.adjacent_id() {
                    SKIP_ID.fetch_add(1, Ordering::SeqCst);
                    continue;
                }

                let per = c.permutation();
                let can_per = per.canonical();
                let is_canonical = (per == can_per.perm);

                batch.push(PR {
                    p: can_per.perm,
                    r: c.repr(),
                    canonical: is_canonical,
                });

                if batch.len() >= BATCH_SIZE {
                    tx.send(batch.clone()).unwrap();
                    batch.clear();
                }
            }
            if !batch.is_empty() {
                tx.send(batch).unwrap();
            }
        });
    }
    perm_rx
}

pub fn main_rainbow(n: usize, m: usize, load: Option<String>, fresh: bool) {
    if load.is_some() == fresh {
        panic!("Specify one of --load or --new but not both");
    }

    if n<3 || m < 1 {
        panic!("Invalid circuit size");
    }

    let base_gates_vec = circuit::base_gates(n)
        .into_iter()
        .enumerate()
        .map(|(i,b)| Gate{ pins: b, control_function: 2, id: i})
        .collect::<Vec<_>>();

    let base_gates = Arc::new(base_gates_vec);
    canonical::init(n);

    let generate_new = load.is_none();
    let total_circuits: i64;
    let ch = if generate_new {
        total_circuits = (base_gates.len() as i64) * ((base_gates.len() - 1) as i64).pow((m-1) as u32);
        circuit::par_all_circuits(n,m)
    } else {
        println!("Loading existing database: {:?}", load.as_ref().unwrap());
        let store = Persist::load(n,m, load.as_ref().unwrap());
        let store = Arc::new(store);
        let ch = circuit::build_from(n,m,&store);
        let prev_count: i64 = store.values().map(|p| p.circuits.len() as i64).sum();
        total_circuits = 2*prev_count*(base_gates.len() as i64);
        ch
    };

    let circuit_store: Arc<Mutex<HashMap<String, PermStore>>> = Arc::new(Mutex::new(HashMap::new()));
    let done = Arc::new(AtomicBool::new(false));

    {
        let done: Arc<AtomicBool> = Arc::clone(&done);
        thread::spawn(move || {
            let start = Instant::now();
            let mut last_count: i64 = 0;
            loop {
                thread::sleep(Duration::from_secs(1));
                if done.load(Ordering::SeqCst) {
                    break;
                }

                let elapsed = start.elapsed().as_secs_f64();
                let ci = CKT_I.load(Ordering::SeqCst) as f64;
                if elapsed < 1.0 || ci < 10.0 || last_count == 0 {
                    last_count = ci as i64;
                    continue;
                }

                let kper_second = ci/elapsed/1000.0;
                let now_kper = (ci - last_count as f64) / 1000.0;
                let eta = ((total_circuits as f64 - ci) / kper_second / 1000.0) as i32;

                println!(
                    "@ {:.1}M, now {:.1}k/s, avg {:.1}k/s ETA: {} sec",
                    ci / 1_000_000.0,
                    now_kper,
                    kper_second,
                    eta
                );

                last_count = ci as i64
            }
        });
    }

    //build permtutations
    let perm_ch = build_circuit(n, m, num_cpus::get().saturating_sub(3).min(64), ch, base_gates);
    let work_count = 2;
    println!("{} verification workers launching", work_count);
    let mut handles = vec![];

    for _ in 0..work_count {
        let perm_ch = perm_ch.clone();
        let circuit_store: Arc<Mutex<HashMap<String, PermStore>>> = Arc::clone(&circuit_store);

        let handle = thread::spawn(move || {
            for batch in perm_ch.iter() {
                for pr in batch {
                    CKT_CHECK.fetch_add(1, Ordering::SeqCst);

                    let p = &pr.p;
                    if p.data.is_empty() {
                        println!("nil perm");
                        continue;
                    }

                    let ph = p.repr();
                    let ip = p.invert();
                    let own_inv = *p == ip;

                    if !own_inv {
                        let iph = ip.repr();
                        let store_guard = circuit_store.lock().unwrap();
                        if store_guard.contains_key(&iph) {
                            SKIP_INV.fetch_add(1, Ordering::SeqCst);
                            continue;
                        }
                    } else {
                        OWN_INV_COUNT.fetch_add(1, Ordering::SeqCst);
                    }

                    let mut store_guard = circuit_store.lock().unwrap();
                    let store = store_guard.entry(ph.clone())
                        .or_insert_with(|| PermStore::NewPermStore(p.clone()));

                    if pr.canonical {
                        if store.contains_canonical {
                            store.add_circuit(&pr.r);
                        } else {
                            store.replace(&pr.r);
                        }
                        store.contains_canonical = true;
                    } else if !store.contains_any_circuit {
                        store.add_circuit(&pr.r);
                    } else {
                        store.increment();
                    }
                    store.contains_any_circuit = true;
                }
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }

    done.store(true, Ordering::SeqCst);

    let mut save_map = HashMap::new();
    for (k,v) in circuit_store.lock().unwrap().iter() {
        save_map.insert(k.clone(), database::make_persist(v.perm.clone(),v.circuits.clone(), v.count));
    }

    println!("Saving...");
    Persist::save(n,m,&save_map);

}