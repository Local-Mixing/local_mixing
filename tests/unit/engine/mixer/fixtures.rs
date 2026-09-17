// Shared fixtures for piecewise rounds and serial walks.

fn piece_cfg(pieces: usize, sequential: bool, threads: usize) -> PieceCfg {
    PieceCfg {
        pieces,
        sequential,
        threads,
        ..PieceCfg::default()
    }
}

fn meta_snapshot(mx: &Mixer) -> Vec<(u32, u64, bool, u32, u64, u16)> {
    mx.arena
        .ids_in_order()
        .iter()
        .map(|&id| {
            let m = mx.meta_of(id);
            (
                m.origin,
                m.event,
                m.dir == Dir::R,
                m.dgen,
                m.litter,
                m.litter_size,
            )
        })
        .collect()
}

fn rand_gate(rng: &mut StdRng, wires: u16, max_w: usize, allow_comp: bool) -> XGate {
    loop {
        let target = rng.random_range(0..wires);
        let w = rng.random_range(0..=max_w);
        let lits: Vec<(u16, bool)> = (0..w)
            .map(|_| (rng.random_range(0..wires), rng.random_bool(0.5)))
            .filter(|&(cw, _)| cw != target)
            .collect();
        if let Some(mut g) = XGate::conj(target, lits) {
            if allow_comp && g.width() == 2 && rng.random_bool(0.3) {
                g.comp = true;
            }
            return g;
        }
    }
}

fn random_g57_circuit(seed: u64, wires: u16, gates: usize) -> Vec<XGate> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..gates)
        .map(|_| {
            loop {
                let a = rng.random_range(0..wires);
                let x = rng.random_range(0..wires);
                let y = rng.random_range(0..wires);
                if a != x && a != y && x != y {
                    break XGate::from_g57([a, x, y]);
                }
            }
        })
        .collect()
}

// A conjunction-dominated circuit like real circuit_mixer input (fsplit output is
// ~90% eroded); a few g57 fossils sprinkled in.
fn random_mixed_circuit(seed: u64, wires: u16, gates: usize) -> Vec<XGate> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..gates)
        .map(|i| {
            if i % 10 == 0 {
                loop {
                    let a = rng.random_range(0..wires);
                    let x = rng.random_range(0..wires);
                    let y = rng.random_range(0..wires);
                    if a != x && a != y && x != y {
                        break XGate::from_g57([a, x, y]);
                    }
                }
            } else {
                loop {
                    let g = rand_gate(&mut rng, wires, 3, false);
                    if g.width() >= 1 {
                        break g;
                    }
                }
            }
        })
        .collect()
}
