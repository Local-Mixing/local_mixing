use std::{
    collections::{HashMap, HashSet},
    time::Instant,
    u64,
};

use clap::Parser;
use local_mixing::{
    circuit::{CircuitSeq, Permutation, Polynomial, circuit::poly_to_str},
    random::random_data::random_circuit,
};
use nauty_Traces_sys::{
    NAUTYVERSIONID, SETWORDSNEEDED, SparseGraph, WORDSIZE, nauty_check, optionblk, sparsenauty,
    statsblk,
};
use rand::{rng, seq::SliceRandom};

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', default_value_t = 4)]
    wires: usize,

    #[arg(short = 'm', default_value = None)]
    gates: Option<usize>,
}

fn iter_ones(mut x: u64) -> impl Iterator<Item = u64> {
    std::iter::from_fn(move || {
        if x == 0 {
            None
        } else {
            let idx = x.trailing_zeros();
            x &= x - 1; // clear the lowest set bit
            Some(idx as u64)
        }
    })
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
    let mut all_monos = HashSet::<u64>::new();
    let mut n_edges = 0usize;
    for p in polys {
        all_monos.extend(p);

        // Backward edges are length of poly
        n_edges += p.len();
    }

    // forward edges are total # of variables over all
    n_edges += all_monos
        .iter()
        .map(|m| m.count_ones() as usize)
        .sum::<usize>();

    let count_monos = all_monos.len();
    println!(
        "# monos: {}, # edges: {}, deg: {:.2}",
        count_monos,
        n_edges,
        (n_edges as f64) / ((count_monos + n) as f64)
    );

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
    assert!(n + (lsb as usize) < 32);

    // from -> to
    let mut edges = HashMap::<u64, HashSet<u64>>::new();

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
    let mut idx_unmap = HashMap::<usize, u64>::new();
    let mut idx_map = HashMap::<u64, usize>::new(); // wire-id or mtag -> vertex position

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
    opt.getcanon = 1; // REQUEST CANONICAL FORM
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

    // println!("{:?}", stat);
    // println!("{:?}", opt);
    // println!("im => {:?}", idx_map);

    // println!("lab = {:?}", lab);
    // println!("ptn = {:?}", ptn);
    // println!("orbits = {:?}", orbits);

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

    for p in 0..n {
        let orig_vertex = idx_map[&(p as u64)]; // == p for wires, but keep this for generality
        println!("w{p} -> canonical label {}", canon_label[orig_vertex]);
    }

    // println!("Canonical polynomial order: {:?}", canonical_polys);

    canon_label
}

fn poly_canon(mut c: &CircuitSeq, n: usize) -> Vec<i32> {
    let m = c.gates.len();
    let polys = c.to_polynomial(n, 0, m);
    for (i, p) in polys.iter().enumerate() {
        // println!("... w{i} => {}", poly_to_str(&p, n));
    }

    println!("Ckt: n={n} m={m}");

    let now = Instant::now();
    let r = to_sparse_graph(&polys);
    let elapsed = now.elapsed().as_micros();
    println!("... {} us", elapsed);
    r[..n].to_vec()
}

fn main() {
    let args = Args::parse();

    let n = args.wires;
    let m = args.gates.unwrap_or(n * (n.ilog2() + 1) as usize);

    let mut ckt = random_circuit(n as usize, m);

    let r1 = poly_canon(&ckt, n);

    let mut p = Permutation::id_perm(n);
    p.data.shuffle(&mut rng());

    ckt.rewire(&p, n);

    let r2 = poly_canon(&ckt, n);
    
    let p1 = Permutation { data: r1.into_iter().map(|x| x as usize).collect() };
    let p2 = Permutation { data: r2.into_iter().map(|x| x as usize).collect() };

    println!("{p1:?}");
    println!("{p2:?}");

    let com = p2.invert().compose(&p1);
    println!("{com:?}");
    println!("{p:?}");

    assert_eq!(com, p);
}
