use std::{
    collections::{HashMap, HashSet},
    hash::RandomState,
    sync::{
        Arc, Mutex,
        atomic::{
            AtomicI64, AtomicUsize,
            Ordering::{self, Relaxed},
        },
    },
    thread::sleep,
    time::{Duration, Instant},
    u64,
};

use clap::Parser;
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use local_mixing::{
    circuit::{
        CircuitSeq, Gate, Permutation, Polynomial, base_gates, circuit::{iter_ones, poly_to_str, polys_repr_blob},
    }, random::random_data::random_circuit,
};
use nauty_Traces_sys::{
    NAUTYVERSIONID, SETWORDSNEEDED, SparseGraph, WORDSIZE, nauty_check, optionblk, sparsenauty,
    statsblk,
};
use rand::{rng, seq::SliceRandom};
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', default_value_t = 4)]
    wires: usize,

    #[arg(short = 'm', default_value = None)]
    gates: Option<usize>,
}

fn mono_to_tag(x: u64, lsb: u32) -> u64 {
    if x == 0 {
        return u64::MAX;
    } else {
        return x << lsb;
    }
}

fn to_sparse_graph(polys: &Vec<Polynomial>) -> Vec<i32> {
    let n = polys.len();
    // How many monomials?
    let mut n_edges = 0usize;
    for p in polys {
        // Backward edges are length of poly
        n_edges += p.len();
    }

    let mut all_monos =
        FxHashSet::<u64>::with_capacity_and_hasher(n_edges, FxBuildHasher::default());

    for p in polys {
        // Backward edges are length of poly
        all_monos.extend(p);
    }

    // forward edges are total # of variables over all
    n_edges += all_monos
        .iter()
        .map(|m| m.count_ones() as usize)
        .sum::<usize>();

    let count_monos = all_monos.len();
    // println!(
    //     "# monos: {}, # edges: {}, deg: {:.2}",
    //     count_monos,
    //     n_edges,
    //     (n_edges as f64) / ((count_monos + n) as f64)
    // );

    let n_vertices = n + count_monos;

    let words = SETWORDSNEEDED(n_vertices);
    unsafe {
        nauty_check(
            WORDSIZE as i32,
            words as i32,
            n_vertices as i32,
            NAUTYVERSIONID as i32,
        );
    };

    // shift up mono indices by this many bits (there are n polys)
    // [ .... monomial spec .... ] [ .. poly .. ]
    let lsb = n.ilog2() + 1;

    // Make sure enough space
    // assert!(n + (lsb as usize) < 32);

    // from -> to
    let mut edges = FxHashMap::<u64, FxHashSet<u64>>::with_capacity_and_hasher(
        n_edges,
        FxBuildHasher::default(),
    );

    for (i, p) in polys.iter().enumerate() {
        let i = i as u64;
        for &mono in p {
            let mtag = mono_to_tag(mono, lsb);
            edges.entry(mtag).or_default().insert(i);

            for j in iter_ones(mono) {
                edges.entry(j).or_default().insert(mtag);
            }
        }
    }

    // From Nauty user guide, §3:
    //   For each vertex i = 0...n−1, d[i] is the degree (out-degree for a
    //   digraph) of that vertex. v[i] is an index into the array e such that
    //   e[v[i]],e[v[i]+1],...,e[v[i]+d[i]-1] are the vertices to which vertex
    //   i is joined.

    // map p <-> i
    let mut idx_unmap =
        FxHashMap::<usize, u64>::with_capacity_and_hasher(n_vertices, FxBuildHasher::default());
    let mut idx_map =
        FxHashMap::<u64, usize>::with_capacity_and_hasher(n_vertices, FxBuildHasher::default()); // wire-id or mtag -> vertex position

    // Pass 1: positions only, no edges yet
    for p in 0..n {
        idx_map.insert(p as u64, p);
    }
    for (i, &mono) in all_monos.iter().enumerate() {
        let mtag = mono_to_tag(mono, lsb);
        idx_map.insert(mtag, i + n);
        idx_unmap.insert(i + n, mtag);
    }

    let mut lab: Vec<i32> = (0..n_vertices as i32).collect(); // identity, always
    let mut ptn: Vec<i32> = vec![1; n_vertices];

    ptn[n - 1] = 0;
    ptn[n_vertices - 1] = 0;

    let mut v = vec![0; n_vertices];
    let mut d = vec![0i32; n_vertices];
    let mut e = Vec::<i32>::with_capacity(n_edges);

    // Pass 2: fill v/d/e, translating every target through idx_map
    for p in 0..n {
        let out_edges = &edges[&(p as u64)];
        d[p] = out_edges.len() as i32;
        v[p] = e.len();
        e.extend(out_edges.iter().map(|&mtag| idx_map[&mtag] as i32));
    }
    for (i, &mono) in all_monos.iter().enumerate() {
        let j = i + n;
        let mtag = mono_to_tag(mono, lsb);
        let out_edges = &edges[&mtag];
        d[j] = out_edges.len() as i32;
        v[j] = e.len();
        e.extend(out_edges.iter().map(|&wire| idx_map[&wire] as i32));
    }

    // Add the monomial nodes

    // println!("ptn = {:?}", ptn);
    // println!("lab = {:?}", lab);

    assert_eq!(v.len(), n_vertices);
    assert_eq!(e.len(), n_edges);

    let mut sg = SparseGraph { v, d, e };

    // println!("sg = {:?}", sg);

    let mut opt = optionblk::default_sparse();
    opt.getcanon = 0; // request canonical form if 1
    opt.digraph = 1;
    opt.defaultptn = 0;
    let mut stat = statsblk::default();

    let mut orbits = vec![0; n_vertices];

    let mut cg = SparseGraph {
        d: vec![0i32; n_vertices],
        v: vec![0; n_vertices],
        e: vec![0i32; n_edges as usize],
    };

    unsafe {
        sparsenauty(
            &mut (&mut sg).into(),
            lab.as_mut_ptr(),
            ptn.as_mut_ptr(),
            orbits.as_mut_ptr(),
            &mut opt,
            &mut stat,
            &mut (&mut cg).into(),
        );
    }

    // println!("{:#?}", stat);
    // println!("{:?}", opt);
    // println!("im => {:?}", idx_map);

    // println!("lab = {:?}", lab);
    // println!("ptn = {:?}", ptn);
    // if (stat.numorbits as usize) < n_vertices {
    //     println!("orbits = {:?}", orbits);
    // }

    // println!("cg = {:?}", cg);

    // for p in 0..n {
    //     let im = idx_map[&(p as u64)];
    //     let l = lab[im];
    //     // let um = idx_unmap[&(l as usize)];
    //     println!("w{p} -> idx {} -> lab {}", im, l);
    // }

    let mut canon_label = vec![0i32; n_vertices];
    for (pos, &orig_vertex) in lab.iter().enumerate() {
        canon_label[orig_vertex as usize] = pos as i32;
    }

    // for p in 0..n {
    //     let orig_vertex = idx_map[&(p as u64)]; // == p for wires, but keep this for generality
    //     println!("w{p} -> canonical label {}", canon_label[orig_vertex]);
    // }

    // println!("Canonical polynomial order: {:?}", canonical_polys);

    canon_label
}

fn poly_canon(mut c: &CircuitSeq, n: usize) -> Permutation {
    let m = c.gates.len();
    let now = Instant::now();
    let polys = c.to_polynomial(n, 0, m);

    if now.elapsed().as_millis() == 0 {
        let elapsed = now.elapsed().as_micros();
        // println!("... poly {} us", elapsed);
    } else {
        let elapsed = now.elapsed().as_millis();
        // println!("... poly {} ms", elapsed);
    }

    // println!("Ckt: n={n} m={m}");

    let now = Instant::now();
    let r = to_sparse_graph(&polys);
    if now.elapsed().as_millis() == 0 {
        let elapsed = now.elapsed().as_micros();
        // println!("... canon {} us", elapsed);
    } else {
        let elapsed = now.elapsed().as_millis();
        // println!("... canon {} ms", elapsed);
    }
    Permutation {
        data: r[..n].iter().map(|&x| x as usize).collect(),
    }
}

fn perm(c: &CircuitSeq, n: usize) -> Permutation {
    let N = 1 << n;

    Permutation { data: (0..N).map(|i| c.evaluate(i)).collect() }
}

fn iso_bfs(n: usize, max_m: usize) {
    let Q = Arc::new(SegQueue::<CircuitSeq>::new());
    let dist = Arc::new(DashMap::<Permutation, usize>::new());
    let dist_counts = Arc::new(Mutex::new(HashMap::<usize, usize>::new()));

    let circuits_stored = Arc::new(AtomicUsize::new(1));
    let q_size = Arc::new(AtomicI64::new(1));

    let num_threads = num_cpus::get();
    let thr_counter = Arc::new(AtomicUsize::new(num_threads));

    let base_ckt = CircuitSeq {
        gates: vec![[0u16, 1, 2]],
    };
    Q.push(base_ckt.clone());

    dist.insert(Permutation::id_perm(n), 0);
    dist.insert(perm(&base_ckt, n), 1);

    let gens = Arc::new(base_gates(n));

    // Spawn worker threads
    std::thread::scope(|s| {
        for tid in 0..num_threads {
            let Q = Q.clone();
            let dist = dist.clone();
            let dist_counts: Arc<Mutex<HashMap<usize, usize>>> = dist_counts.clone();
            let gens = gens.clone();
            let circuits_stored = circuits_stored.clone();
            let thr_counter = thr_counter.clone();

            let start = Instant::now();
            let mut last_stored = 0usize;

            s.spawn(move || {
                if Q.is_empty() {
                    sleep(Duration::from_millis(500));
                }

                let mut last_print = Instant::now();

                while let Some(g) = Q.pop() {
                    // Batch collect new circuits and their metadata
                    let mut new_circuits: Vec<(CircuitSeq, Vec<u8>, usize)> = Vec::with_capacity(gens.len());

                    for &s in gens.iter() {
                        let mut h = g.clone();
                        h.gates.push(s);

                        let rw = poly_canon(&h, n);
                        h.rewire(&rw, n);
                        let m = h.gates.len();
                        let p = polys_repr_blob(&h.to_polynomial(n, 0, m));

                        if m <= max_m {
                            new_circuits.push((h, p, m));
                        }
                    }

                    // Batch insert into shared data structures
                    let mut counts_update = HashMap::<usize, usize>::new();
                    let mut to_queue = Vec::new();

                    for (h, p, m) in new_circuits {
                        if !dist.contains_key(&p) {
                            dist.insert(p, m);
                            *counts_update.entry(m).or_default() += 1;
                            to_queue.push(h);
                        }
                    }

                    // Batch update counts
                    if !counts_update.is_empty() {
                        let mut dc = dist_counts.lock().unwrap();
                        for (m, count) in counts_update {
                            *dc.entry(m).or_default() += count;
                        }
                    }

                    // let dl = dist.len();
                    let ql = Q.len();
                    let ct = circuits_stored.fetch_add(to_queue.len(), Ordering::Relaxed);

                    // Push all new circuits to queue
                    for h in to_queue {
                        Q.push(h);
                    }
                    //     println!("Insert {} (#{}, Q{})", h.repr(), 0, ql);

                    let kper_sec = (ct as f64) / start.elapsed().as_secs_f64() / 1000.0;
                    let eta = (ql as f64 / 1000.0) / kper_sec;

                    if last_print.elapsed().as_secs_f32() > 2.0 {
                        if last_stored != ct {
                            println!("t{tid:3} st:{ct:6} Q:{ql:6}    {kper_sec:.1}k/s");
                        } else {
                            // let thr = thr_counter.load(Ordering::Relaxed);
                            println!(
                                "t{tid:3} st:{ct:6} Q:{ql:6}    {kper_sec:.1}k/s   eta {eta:.0} sec"
                            );
                        }
                        last_print = Instant::now();
                    }

                    last_stored = ct;
                }
            });
        }
    });

    println!("{}", dist.len());
    let final_counts = dist_counts.lock().unwrap().clone();
    println!("{:#?}", final_counts);
}

fn main() {
    let args = Args::parse();

    let n = args.wires;
    let m = args.gates.unwrap_or(n * (n.ilog2() + 1) as usize);

    iso_bfs(n, m);

    // for m in 0..m {
    //     for g in base_gates(n) {
    //         let mut ckt =
    //     }
    // }

    // for g in base_gates(n) {
    //     let mut ckt = CircuitSeq { gates: vec![g] };
    //     let r1 = poly_canon(&ckt, n);
    //     // println!("{r1:?}");
    //     let p1 = Permutation {
    //         data: r1.into_iter().map(|x| x as usize).collect(),
    //     };
    //     ckt.rewire(&p1, n);
    //     let poly2 = ckt.to_polynomial(n, 0, 1);
    //     // println!("== {:#?}", ;
    //     d.insert(polys_repr_blob(&poly2));
    // }

    // println!("{} / {}", d.len(), base_gates(n).len());

    // return;

    // let mut ckt = random_circuit(n as usize, m);

    // let r1 = poly_canon(&ckt, n);

    // let p1 = Permutation {
    //     data: r1.into_iter().map(|x| x as usize).collect(),
    // };
    // let mut ck2 = ckt.clone();
    // ck2.rewire(&p1, n);

    // let mut p = Permutation::id_perm(n);
    // p.data.shuffle(&mut rng());

    // ckt.rewire(&p, n);

    // let r2 = poly_canon(&ckt, n);

    // let p2 = Permutation {
    //     data: r2.into_iter().map(|x| x as usize).collect(),
    // };
    // let mut ck3 = ckt.clone();
    // ck3.rewire(&p2, n);

    // println!("{p1:?}");
    // println!("{p2:?}");

    // let com = p2.invert().compose(&p1);
    // // println!("{com:?}");
    // // println!("{p:?}");

    // assert_eq!(com, p);

    // let q2 = ck2.to_polynomial(n, 0, m);
    // let q3 = ck3.to_polynomial(n, 0, m);
    // assert_eq!(q2, q3);

    // println!("OK!");
}
