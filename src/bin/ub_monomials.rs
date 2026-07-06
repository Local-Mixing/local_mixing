use std::{cmp::min, collections::HashSet};

use primitive_types::U256;
use clap::Parser;

#[derive(Debug)]
struct Wire {
    influence_wires: HashSet<u16>,
    monos: U256,
}

impl Wire {
    fn new(i: u16) -> Self {
        Self {
            influence_wires: [i].into_iter().collect(),
            monos: 1.into(),
        }
    }

    fn is_max(&self, n: usize) -> bool {
        // TODO: div 2 or not?
        self.monos >= (U256::one() << (n - 1))
    }

    fn update(&mut self, c1: &Wire, c2: &Wire) {
        self.influence_wires.extend(&c1.influence_wires);
        self.influence_wires.extend(&c2.influence_wires);

        let max_mono = U256::one() << self.influence_wires.len();

        self.monos = min(max_mono, U256::one() + self.monos + c1.monos + c1.monos * c2.monos);
    }
}

fn random_gate(wires: u16) -> [u16; 3] {
    let mut all: Vec<u16> = (0..wires).collect();
    fastrand::shuffle(&mut all);
    [all[0], all[1], all[2]]
}

#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    #[arg(short = 'n', default_value_t = 16)]
    wires: u16,

    #[arg(short = 'm', long)]
    gates: Option<usize>,
}

fn main() {
    let args = Args::parse();

    let mut wires: Vec<_> = (0..args.wires).map(|i| Wire::new(i)).collect();

    let mut avg = Vec::<U256>::new();

    for i in 0.. {
        let v = random_gate(args.wires);

        let Ok([w0, w1, w2]) =
            wires.get_disjoint_mut([v[0] as usize, v[1] as usize, v[2] as usize])
        else {
            continue;
        };

        w0.update(w1, w2);

        let _ar: f32 = (i as f32) / (args.wires as f32);
        let _min_mono = wires.iter().min_by_key(|w| w.monos).unwrap().monos;

        let mut avg_mono = U256::zero();
        for w in &wires {
            avg_mono += w.monos;
        }
        avg_mono /= args.wires;

        avg.push(avg_mono);

        // println!("Gate {i} ar={ar:.2}: {min_mono} {avg_mono}");
        // wires.iter().map(|w| w.monos).join(", "));

        if wires.iter().all(|w| w.is_max(args.wires.into())) {
            break;
        }
    }

    println!("{avg:?}")
}
