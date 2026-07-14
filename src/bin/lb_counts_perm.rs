use std::{
    collections::HashMap, ptr::null_mut, sync::{Arc, atomic::{AtomicUsize, Ordering}}, thread::sleep, time::{Duration, Instant},
};

use clap::Parser;
use crossbeam::queue::SegQueue;
use dashmap::DashMap;
use itertools::Itertools;
use local_mixing::{
    circuit::{CircuitSeq, Permutation, base_gates, circuit::iter_ones},
    random::random_data::random_circuit,
};
use nauty_Traces_sys::{SparseGraph, optionblk, sparsenauty, statsblk};
use rand::{rng, seq::SliceRandom};

fn factorial(n: usize) -> Option<usize> {
    (1..=n).try_fold(1, usize::checked_mul)
}

fn canonicalize_perm_sparse_graph(p: &Permutation) -> (Permutation, usize) {
    #[allow(non_snake_case)]
    let NN = p.data.len();
    let n = NN.ilog2() as usize;
    assert_eq!(NN, 1 << n);

    let n_vertices = n + NN;
    let n_edges = NN * n;

    let mut edge_list = vec![Vec::<i32>::new(); n_edges];

    for (x, &y) in p.data.iter().enumerate() {
        let node_tag = y + n;
        for xx in iter_ones(x) {
            edge_list[xx].push(node_tag as i32);
        }
        for yy in iter_ones(y) {
            edge_list[node_tag].push(yy as i32);
        }
    }

    let mut lab: Vec<i32> = (0..n_vertices as i32).collect(); // identity, always
    let mut ptn: Vec<i32> = vec![1; n_vertices];

    ptn[n - 1] = 0;
    ptn[n_vertices - 1] = 0;

    let mut v = vec![0; n_vertices];
    let mut d = vec![0i32; n_vertices];
    let mut e = Vec::<i32>::with_capacity(n_edges);

    for bit in 0..n {
        // Handle each bit
        d[bit] = edge_list[bit].len() as i32;
        v[bit] = e.len();
        e.extend(edge_list[bit].clone());
    }

    for j in 0..NN {
        let node_tag = j + n;
        let node = edge_list[node_tag].clone();
        d[node_tag] = node.len() as i32;
        v[node_tag] = e.len();
        e.extend(node);
    }

    assert_eq!(v.len(), n_vertices);
    assert_eq!(e.len(), n_edges);

    let mut sg = SparseGraph { v, d, e };
    let mut opt = optionblk::default_sparse();
    opt.getcanon = 0; // request canonical form if 1
    opt.digraph = 1;
    opt.defaultptn = 0; // 0 = vertices do NOT have the same color
    let mut stat = statsblk::default();
    let mut orbits = vec![0; n_vertices];

    unsafe {
        sparsenauty(
            &mut (&mut sg).into(),
            lab.as_mut_ptr(),
            ptn.as_mut_ptr(),
            orbits.as_mut_ptr(),
            &mut opt,
            &mut stat,
            null_mut(),
        );
    }

    // println!("{:#?}", stat);
    // println!("{:?}", opt);

    // println!("lab = {:?}", lab);
    // println!("ptn = {:?}", ptn);
    // if (stat.numorbits as usize) < n_vertices {
    //     println!("orbits = {:?}", orbits);
    // }
    assert!(stat.grpsize2 == 0);
    let sphere = factorial(n).unwrap() / (stat.grpsize1 as usize);

    // TODO
    // println!("sphere: {sphere_size}");

    let mut canon_label = vec![0i32; n_vertices];
    for (i, &orig) in lab.iter().enumerate() {
        canon_label[orig as usize] = i as i32;
    }

    (p.bit_shuffle(&canon_label[..n].iter().map(|&x| x as usize).collect::<Vec<_>>()), sphere)
}

#[derive(Hash, PartialEq, Eq, Clone)]
struct SmallPerm(Vec<u8>);

impl Into<Permutation> for SmallPerm {
    fn into(self) -> Permutation {
        Permutation { data: self.0.iter().map(|&x| x.into()).collect() }
    }
}

impl From<Permutation> for SmallPerm {
    fn from(value: Permutation) -> Self {
        assert!(value.data.len() <= 256);
        Self(value.data.iter().map(|&x| x as u8).collect())
    }
}

fn iso_bfs(n: usize, max_m: usize) {
    #[allow(non_snake_case)]
    let Q = Arc::new(SegQueue::<(SmallPerm, usize)>::new());
    let dist = Arc::new(DashMap::<SmallPerm, usize>::new());
    let dist_counts = Arc::new(DashMap::<usize, usize>::new());
    let spheres: Arc<DashMap<usize, usize>> = Arc::new(DashMap::<usize, usize>::new());

    let circuits_stored = Arc::new(AtomicUsize::new(1));
    let num_threads = num_cpus::get();

    let base_ckt = CircuitSeq {
        gates: vec![[0, 1, 2]],
    };
    let base_perm = base_ckt.perm(n);
    Q.push((base_perm.clone().into(), 1));
    dist.insert(Permutation::id_perm(n).into(), 0);
    dist.insert(base_perm.into(), 1);

    let gens: Arc<Vec<_>> = Arc::new(
        base_gates(n)
            .into_iter()
            .map(|g| CircuitSeq { gates: vec![g] }.perm(n))
            .collect(),
    );

    std::thread::scope(|s| {
        for tid in 0..num_threads {
            let Q = Q.clone();
            let dist = dist.clone();
            let dist_counts: Arc<DashMap<usize, usize>> = dist_counts.clone();
            let spheres: Arc<DashMap<usize, usize>> = spheres.clone();
            let gens = gens.clone();
            let circuits_stored = circuits_stored.clone();

            let start = Instant::now();
            let mut last_stored = 0usize;

            s.spawn(move || {
                if Q.is_empty() {
                    sleep(Duration::from_millis(500));
                }

                // Print every so often, but not too much
                let mut last_print = Instant::now();

                let gen_size = gens.len();

                let mut new_circuits: Vec<(Permutation, usize, usize)> =
                        Vec::with_capacity(gen_size);

                while let Some((g, mp)) = Q.pop() {
                    
                    let m = mp + 1;
                    
                    if m > max_m {
                        continue;
                    }
                    
                    // Batch collect new circuits and their metadata
                    new_circuits.clear();

                    for s in gens.iter() {
                        let h = s.compose(&g.clone().into());

                        let (canon, sp) = canonicalize_perm_sparse_graph(&h);
                        new_circuits.push((canon, m, sp));

                    }

                    // Batch insert into shared data structures
                    let mut counts_update = HashMap::<usize, usize>::new();
                    let mut new_count = 0;

                    for (p, m, sphere) in new_circuits.iter() {
                        let sp: SmallPerm = p.clone().into();

                        // NOTE: race condition here.
                        if !dist.contains_key(&sp) {
                            dist.insert(sp.clone(), *m);
                            *counts_update.entry(*m).or_default() += 1;

                            // Don't add the frontier to the queue
                            if *m != max_m {
                                Q.push((sp, *m));
                            }

                            new_count += 1;
                            *spheres.entry(*m).or_default() += sphere;
                        }
                    }

                    let ql = Q.len();
                    let ct = circuits_stored.fetch_add(new_count, Ordering::Relaxed);

                    // Batch update counts
                    if !counts_update.is_empty() {
                        for (m, count) in counts_update {
                            *dist_counts.entry(m).or_default() += count;
                        }
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

    let can_ckt = dist.len();
    let total_ckt = (n * (n - 1) * (n - 2)).pow(max_m as u32);
    let compr_ratio = total_ckt / can_ckt;

    println!("Final: {} canonical perms, {} total circuits, ({}x)", can_ckt, total_ckt, compr_ratio);

    println!("m      count     sphere");
    for m in 2..=max_m {
        let count = *dist_counts.get(&m).unwrap();
        let sp = *spheres.get(&m).unwrap();
        println!("{} {:10} {:10}", m, count, sp);
    }
    // println!("Distances: {:#?}", dist_counts);
    // println!("Spheres: {:#?}", spheres);
}

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', default_value_t = 4)]
    wires: usize,

    #[arg(short = 'm', default_value = None)]
    gates: Option<usize>,
}

fn main() {
    let args = Args::parse();

    let n = args.wires;
    let m = args.gates.unwrap_or(n * (n.ilog2() + 1) as usize);

    iso_bfs(n, m)

    // let mut ckt = random_circuit(n as usize, m);
    // // let mut c2 = ckt.clone();
    // // c2.gates.reverse();
    // // ckt.gates.append(&mut c2.gates);
    // let p = ckt.perm(n);

    // let pc = canonicalize_perm_sparse_graph(&p);

    // let mut sh = Permutation::id_perm(n);
    // sh.data.shuffle(&mut rng());
    // ckt.rewire(&sh, n);

    // let now = Instant::now();
    // let shuf_p = ckt.perm(n);
    // println!("perm: {} us", now.elapsed().as_micros());

    // let now = Instant::now();
    // let sc = canonicalize_perm_sparse_graph(&shuf_p);
    // println!("canon: {} us", now.elapsed().as_micros());

    // // println!("{:?}", p);
    // // println!("{:?}", pc);
    // // println!("{:?}", sc);

    // // assert_ne!(p, shuf_p);
    // assert_eq!(pc, sc);
}
