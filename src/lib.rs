pub mod bench_support;
pub mod circuit;
pub mod rainbow;
pub mod random;
pub mod replace;
use crate::circuit::CircuitSeq;
use primitive_types::U256 as u256;
use rayon::prelude::*;
pub use replace::main_mix::open_shard_dbs;
use std::fs::File;
use std::io::{BufReader, Read};

#[allow(unused)]
fn read_n_gates(path: &str, n: usize) -> String {
    let file = File::open(path).unwrap_or_else(|_| panic!("Failed to open {}", path));
    let mut reader = BufReader::new(file);
    let mut result = String::new();
    let mut buf = [0u8; 1];
    let mut count = 0;
    while count < n {
        if reader.read(&mut buf).unwrap_or(0) == 0 {
            break;
        }
        let c = buf[0] as char;
        result.push(c);
        if c == ';' {
            count += 1;
        }
    }
    result
}
#[inline]
fn popcount_u256(x: u256) -> u32 {
    let mut count = 0;
    for limb in x.0 {
        count += limb.count_ones();
    }
    count
}

/// Restrict the Hamming-distance computation to a half of the wires.
/// `first_half` -> low `num_wires/2` bits; `second_half` -> high bits; neither -> all bits.
/// Returns the effective bit-mask and the bit count to normalize by.
#[allow(unused)]
fn half_mask_and_width(
    num_wires: usize,
    mask: u256,
    first_half: bool,
    second_half: bool,
) -> (u256, usize) {
    let half = num_wires / 2;
    let lower = if half < 256 {
        (u256::one() << half) - u256::one()
    } else {
        u256::MAX
    };
    if first_half {
        (mask & lower, half)
    } else if second_half {
        (mask & !lower, num_wires - half)
    } else {
        (mask, num_wires)
    }
}

/// Parallel heatmap grid: for x in [x1,x2] and y in [y1,y2], computes the average (over
/// `inputs`) of the overlap between circuit_one's state after gate i1 and circuit_two's
/// state after gate i2. Returns a flat, row-major [x, y, value] buffer (3 f64 per cell,
/// one row per x) ready for `Array2::from_shape_vec((n_cells, 3), _)`.
#[allow(unused)]
fn compute_grid_parallel(
    circuit_one: &CircuitSeq,
    circuit_two: &CircuitSeq,
    inputs: &[u256],
    x1: usize,
    x2: usize,
    y1: usize,
    y2: usize,
    num_wires: usize,
    mask: u256,
    flag: bool,
    hw: bool,
    first_half: bool,
    second_half: bool,
) -> Vec<f64> {
    let num_inputs = inputs.len();
    // Restrict to a half of the wires (and renormalize) if requested.
    let (mask, num_wires) = half_mask_and_width(num_wires, mask, first_half, second_half);
    // Per-input state evolutions, computed across cores...
    let evo_one: Vec<Vec<u256>> = inputs
        .par_iter()
        .map(|&ib| circuit_one.evaluate_evolution_256(ib))
        .collect();
    let evo_two: Vec<Vec<u256>> = inputs
        .par_iter()
        .map(|&ib| circuit_two.evaluate_evolution_256(ib))
        .collect();
    // ...then transposed to [position][input] so each cell scans contiguous memory.
    let one_t: Vec<Vec<u256>> = (x1..=x2)
        .into_par_iter()
        .map(|i1| (0..num_inputs).map(|k| evo_one[k][i1]).collect())
        .collect();
    let two_t: Vec<Vec<u256>> = (y1..=y2)
        .into_par_iter()
        .map(|i2| (0..num_inputs).map(|k| evo_two[k][i2]).collect())
        .collect();
    drop(evo_one);
    drop(evo_two);

    let row_w = y2 - y1 + 1;
    let n_rows = x2 - x1 + 1;
    let nw = num_wires as f64;
    let inv = 1.0 / num_inputs as f64;
    let mut data = vec![0f64; n_rows * row_w * 3];
    // One output row (fixed i1) per task; rows are disjoint, so there is no contention.
    data.par_chunks_mut(row_w * 3)
        .enumerate()
        .for_each(|(r, row)| {
            let i1 = x1 + r;
            let e1 = &one_t[r];
            for c in 0..row_w {
                let i2 = y1 + c;
                let e2 = &two_t[c];
                let mut acc = 0f64;
                for k in 0..num_inputs {
                    let a = e1[k];
                    let b = e2[k];
                    let hamming_dist = if hw {
                        (popcount_u256(a & mask) as f64 - popcount_u256(b & mask) as f64).abs()
                    } else {
                        popcount_u256((a ^ b) & mask) as f64
                    };
                    acc += if !flag || hw {
                        hamming_dist / nw
                    } else {
                        ((2.0 * hamming_dist / nw) - 1.0).abs()
                    };
                }
                row[c * 3] = i1 as f64;
                row[c * 3 + 1] = i2 as f64;
                row[c * 3 + 2] = acc * inv;
            }
        });
    data
}

// #[pyfunction]
// fn heatmap(
//     py: Python<'_>,
//     num_wires: usize,
//     num_inputs: usize,
//     flag: bool,
//     c1: &str,
//     c2: &str,
//     canon: bool,
//     fix: usize,
//     hw: bool,
//     first_half: bool,
//     second_half: bool,
// ) -> Py<PyArray2<f64>> {
//     let mask = if num_wires < 256 {
//         (u256::one() << num_wires) - u256::one()
//     } else {
//         u256::MAX
//     };
//     println!("Running heatmap on {} inputs", num_inputs);
//     io::stdout().flush().unwrap();
//     // Load circuits
//     let circuit_one_str = fs::read_to_string(c1).expect("Failed to read butterfly_recent.txt");
//     let circuit_two_str = fs::read_to_string(c2).expect("Failed to read butterfly_recent.txt");
//     let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
//     let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
//     if canon {
//         circuit_one.canonicalize();
//         circuit_two.canonicalize();
//     }
//     let circuit_one_len = circuit_one.gates.len();
//     let circuit_two_len = circuit_two.gates.len();

//     let num_points = (circuit_one_len + 1) * (circuit_two_len + 1);
//     let mut rng = rand::rng();
//     let start_time = Instant::now();
//     let mut fixed_mask = u256::zero();
//     let positions = (0..num_wires).choose_multiple(&mut rng, fix);
//     let x0: u256 =
//         u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128) & mask;
//     for p in positions {
//         fixed_mask |= u256::from(1) << p;
//     }
//     // Random inputs (fixed bits held at x0), generated sequentially to keep RNG use deterministic.
//     let inputs: Vec<u256> = (0..num_inputs)
//         .map(|_| {
//             let r: u256 =
//                 u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128);
//             ((x0 & fixed_mask) | (r & !fixed_mask)) & mask
//         })
//         .collect();

//     let data = compute_grid_parallel(
//         &circuit_one,
//         &circuit_two,
//         &inputs,
//         0,
//         circuit_one_len,
//         0,
//         circuit_two_len,
//         num_wires,
//         mask,
//         flag,
//         hw,
//         first_half,
//         second_half,
//     );

//     println!("Time elapsed: {:?}", Instant::now() - start_time);

//     let arr2 = Array2::from_shape_vec((num_points, 3), data).expect("grid shape mismatch");
//     PyArray2::from_owned_array(py, arr2).into()
// }

// #[pyfunction]
// fn heatmap_incremental(
//     py: Python<'_>,
//     num_wires: usize,
//     num_inputs: usize,
//     flag: bool,
//     c1: &str,
//     c2: &str,
//     canon: bool,
//     _fix: usize,
//     hw: bool,
//     x0_arg: Option<u128>,
//     first_half: bool,
//     second_half: bool,
// ) -> Py<PyArray2<f64>> {
//     let mask = if num_wires < 256 {
//         (u256::one() << num_wires) - u256::one()
//     } else {
//         u256::MAX
//     };
//     println!(
//         "Running incremental heatmap on {} inputs ({} base + increments)",
//         num_inputs,
//         if x0_arg.is_some() { "chosen" } else { "random" }
//     );
//     io::stdout().flush().unwrap();
//     let circuit_one_str = fs::read_to_string(c1).expect("Failed to read c1");
//     let circuit_two_str = fs::read_to_string(c2).expect("Failed to read c2");
//     let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
//     let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
//     if canon {
//         circuit_one.canonicalize();
//         circuit_two.canonicalize();
//     }
//     let circuit_one_len = circuit_one.gates.len();
//     let circuit_two_len = circuit_two.gates.len();

//     let num_points = (circuit_one_len + 1) * (circuit_two_len + 1);
//     let mut rng = rand::rng();
//     let start_time = Instant::now();
//     // Base input x0: caller-provided if given, else random. Subsequent inputs are x0+1, x0+2, ... (mod 2^num_wires).
//     let x0: u256 = match x0_arg {
//         Some(v) => u256::from(v) & mask,
//         None => {
//             (u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128)) & mask
//         }
//     };
//     let inputs: Vec<u256> = (0..num_inputs)
//         .map(|i| x0.overflowing_add(u256::from(i as u128)).0 & mask)
//         .collect();

//     let data = compute_grid_parallel(
//         &circuit_one,
//         &circuit_two,
//         &inputs,
//         0,
//         circuit_one_len,
//         0,
//         circuit_two_len,
//         num_wires,
//         mask,
//         flag,
//         hw,
//         first_half,
//         second_half,
//     );

//     println!("Time elapsed: {:?}", Instant::now() - start_time);

//     let arr2 = Array2::from_shape_vec((num_points, 3), data).expect("grid shape mismatch");
//     PyArray2::from_owned_array(py, arr2).into()
// }

// #[pyfunction]
// fn heatmap_small(
//     py: Python<'_>,
//     num_wires: usize,
//     flag: bool,
//     c1: &str,
//     c2: &str,
//     canon: bool,
// ) -> Py<PyArray2<f64>> {
//     let mask = if num_wires < 256 {
//         (u256::one() << num_wires) - u256::one()
//     } else {
//         u256::MAX
//     };
//     println!("Running heatmap on weights 0, 1, and 2");
//     io::stdout().flush().unwrap();
//     // Load circuits
//     let circuit_one_str = fs::read_to_string(c1).expect("Failed to read butterfly_recent.txt");
//     let circuit_two_str = fs::read_to_string(c2).expect("Failed to read butterfly_recent.txt");
//     let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
//     let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
//     if canon {
//         circuit_one.canonicalize();
//         circuit_two.canonicalize();
//     }
//     let circuit_one_len = circuit_one.gates.len();
//     let circuit_two_len = circuit_two.gates.len();

//     let num_points = (circuit_one_len + 1) * (circuit_two_len + 1);
//     let mut average = vec![0f64; num_points * 3]; // flat 2D array: [x, y, value] per point
//     let start_time = Instant::now();

//     // Generate inputs of Hamming weight 0, 1, and 2
//     let mut inputs: Vec<u256> = Vec::new();

//     inputs.push(u256::from(0u128));

//     for i in 0..num_wires {
//         inputs.push(u256::one() << i);
//     }

//     for i in 0..num_wires {
//         for j in (i + 1)..num_wires {
//             inputs.push((u256::one() << i) | (u256::one() << j));
//         }
//     }

//     let effective_inputs = inputs.len() as f64;

//     for (i, &input_bits) in inputs.iter().enumerate() {
//         if i % 10 == 0 {
//             println!("{}/{}", i, inputs.len());
//             io::stdout().flush().unwrap();
//         }

//         let evolution_one = circuit_one.evaluate_evolution_256(input_bits);
//         let evolution_two = circuit_two.evaluate_evolution_256(input_bits);

//         for i1 in 0..=circuit_one_len {
//             for i2 in 0..=circuit_two_len {
//                 let diff = (evolution_one[i1] ^ evolution_two[i2]) & mask;
//                 let hamming_dist = popcount_u256(diff) as f64;
//                 let overlap = if !flag {
//                     hamming_dist / num_wires as f64
//                 } else {
//                     let tmp = (2.0 * hamming_dist / num_wires as f64) - 1.0;
//                     tmp.abs()
//                 };

//                 let index = i1 * (circuit_two_len + 1) + i2;
//                 average[index * 3] = i1 as f64;
//                 average[index * 3 + 1] = i2 as f64;
//                 average[index * 3 + 2] += overlap / effective_inputs;
//             }
//         }
//     }

//     println!("Time elapsed: {:?}", Instant::now() - start_time);

//     let mut arr2 = Array2::<f64>::zeros((num_points, 3));
//     for i in 0..num_points {
//         arr2[[i, 0]] = average[i * 3];
//         arr2[[i, 1]] = average[i * 3 + 1];
//         arr2[[i, 2]] = average[i * 3 + 2];
//     }

//     let pyarray = PyArray2::from_owned_array(py, arr2);

//     pyarray.into()
// }

// #[pyfunction]
// fn heatmap_slice(
//     py: Python<'_>,
//     num_wires: usize,
//     num_inputs: usize,
//     flag: bool,
//     x1: usize,
//     x2: usize,
//     y1: usize,
//     y2: usize,
//     c1_path: &str,
//     c2_path: &str,
//     fix: usize,
//     hw: bool,
//     first_half: bool,
//     second_half: bool,
// ) -> Py<PyArray2<f64>> {
//     println!("Running heatmap on {} inputs", num_inputs);
//     io::stdout().flush().unwrap();
//     // Load circuits
//     let mask = if num_wires < 256 {
//         (u256::one() << num_wires) - u256::one()
//     } else {
//         u256::MAX
//     };
//     let circuit_one_str = read_n_gates(c1_path, x2 + 1);
//     let circuit_two_str = read_n_gates(c2_path, y2 + 1);

//     let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
//     let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
//     circuit_one.canonicalize();
//     circuit_two.canonicalize();
//     let num_points = (x2 - x1 + 1) * (y2 - y1 + 1);
//     let mut rng = rand::rng();
//     let start_time = Instant::now();
//     let mut fixed_mask = u256::zero();
//     let positions = (0..num_wires).choose_multiple(&mut rng, fix);
//     let x0: u256 =
//         u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128) & mask;
//     for p in positions {
//         fixed_mask |= u256::from(1) << p;
//     }
//     let inputs: Vec<u256> = (0..num_inputs)
//         .map(|_| {
//             let r: u256 =
//                 u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128);
//             ((x0 & fixed_mask) | (r & !fixed_mask)) & mask
//         })
//         .collect();

//     let data = compute_grid_parallel(
//         &circuit_one,
//         &circuit_two,
//         &inputs,
//         x1,
//         x2,
//         y1,
//         y2,
//         num_wires,
//         mask,
//         flag,
//         hw,
//         first_half,
//         second_half,
//     );

//     println!("Time elapsed: {:?}", Instant::now() - start_time);

//     let arr2 = Array2::from_shape_vec((num_points, 3), data).expect("grid shape mismatch");
//     let pyarray = PyArray2::from_owned_array(py, arr2);

//     pyarray.into()
// }

// #[pyfunction]
// fn heatmap_mini_slice(
//     py: Python<'_>,
//     num_wires: usize,
//     num_inputs: usize,
//     flag: bool,
//     x1: usize,
//     x2: usize,
//     y1: usize,
//     y2: usize,
//     c1_path: &str,
//     c2_path: &str,
//     fix: usize,
// ) -> Py<PyArray2<f64>> {
//     println!("Running heatmap on {} inputs", num_inputs);
//     io::stdout().flush().unwrap();
//     let circuit_one_str = read_n_gates(c1_path, x2 + 1);
//     let circuit_two_str = read_n_gates(c2_path, y2 + 1);
//     let mask = if num_wires < 256 {
//         (u256::one() << num_wires) - u256::one()
//     } else {
//         u256::MAX
//     };

//     let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
//     let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
//     circuit_one.canonicalize();
//     circuit_two.canonicalize();
//     circuit_one.gates = circuit_one.gates[x1..=x2].to_vec();
//     circuit_two.gates = circuit_two.gates[y1..=y2].to_vec();
//     let num_points = (x2 - x1 + 1) * (y2 - y1 + 1);
//     let mut average = vec![0f64; num_points * 3]; // flat 2D array: [x, y, value] per point
//     let mut rng = rand::rng();
//     let start_time = Instant::now();
//     let mut fixed_mask = u256::zero();
//     let positions = (0..num_wires).choose_multiple(&mut rng, fix);
//     let x0: u256 =
//         u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128) & mask;
//     for p in positions {
//         fixed_mask |= u256::from(1) << p;
//     }
//     for i in 0..num_inputs {
//         if i % 10 == 0 {
//             println!("{}/{}", i, num_inputs);
//             io::stdout().flush().unwrap();
//         }
//         let r: u256 = u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128);

//         let input_bits = ((x0 & fixed_mask) | (r & !fixed_mask)) & mask;

//         let evolution_one = circuit_one.evaluate_evolution_256(input_bits);
//         let evolution_two = circuit_two.evaluate_evolution_256(input_bits);

//         for i1 in x1..=x2 {
//             for i2 in y1..=y2 {
//                 let diff = evolution_one[i1 - x1] ^ evolution_two[i2 - y1];
//                 let hamming_dist = popcount_u256(diff) as f64;
//                 let overlap = if !flag {
//                     hamming_dist / num_wires as f64
//                 } else {
//                     let tmp = (2.0 * hamming_dist / num_wires as f64) - 1.0;
//                     tmp.abs()
//                 };

//                 let rel_i1 = i1 - x1;
//                 let rel_i2 = i2 - y1;
//                 let index = rel_i1 * (y2 - y1 + 1) + rel_i2;
//                 average[index * 3] = i1 as f64;
//                 average[index * 3 + 1] = i2 as f64;
//                 average[index * 3 + 2] += overlap / num_inputs as f64;
//             }
//         }
//     }

//     println!("Time elapsed: {:?}", Instant::now() - start_time);

//     let mut arr2 = Array2::<f64>::zeros((num_points, 3));
//     for i in 0..num_points {
//         arr2[[i, 0]] = average[i * 3];
//         arr2[[i, 1]] = average[i * 3 + 1];
//         arr2[[i, 2]] = average[i * 3 + 2];
//     }

//     let pyarray = PyArray2::from_owned_array(py, arr2);

//     pyarray.into()
// }

// /// Bottom corner of the heatmap: the first up-to-5000 gates of BOTH circuits.
// /// Reads only `min(5000, gate_count)` gates from each file. Supports both random
// /// sampling (with optional fixed bits) and incremental sampling (random base, then
// /// x0+1, x0+2, ...) selected by `incremental`.
// #[pyfunction]
// fn heatmap_corner(
//     py: Python<'_>,
//     num_wires: usize,
//     num_inputs: usize,
//     flag: bool,
//     c1_path: &str,
//     c2_path: &str,
//     fix: usize,
//     hw: bool,
//     incremental: bool,
//     x0_arg: Option<u128>,
//     first_half: bool,
//     second_half: bool,
// ) -> Py<PyArray2<f64>> {
//     const CORNER: usize = 5000;
//     let mask = if num_wires < 256 {
//         (u256::one() << num_wires) - u256::one()
//     } else {
//         u256::MAX
//     };
//     if incremental {
//         println!(
//             "Running corner heatmap on {} inputs (incremental: {} base + increments)",
//             num_inputs,
//             if x0_arg.is_some() { "chosen" } else { "random" }
//         );
//     } else {
//         println!("Running corner heatmap on {} inputs", num_inputs);
//     }
//     io::stdout().flush().unwrap();

//     // Read only the first CORNER gates of each circuit (fewer if the file is shorter).
//     let circuit_one_str = read_n_gates(c1_path, CORNER);
//     let circuit_two_str = read_n_gates(c2_path, CORNER);
//     let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
//     let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
//     circuit_one.canonicalize();
//     circuit_two.canonicalize();
//     let circuit_one_len = circuit_one.gates.len();
//     let circuit_two_len = circuit_two.gates.len();

//     let num_points = (circuit_one_len + 1) * (circuit_two_len + 1);
//     let mut rng = rand::rng();
//     let start_time = Instant::now();

//     // Random fixed-bit mask + base input (base reused for incremental increments).
//     let mut fixed_mask = u256::zero();
//     let positions = (0..num_wires).choose_multiple(&mut rng, fix);
//     for p in positions {
//         fixed_mask |= u256::from(1) << p;
//     }
//     // Incremental base x0: caller-provided if given, else random (also holds the fixed bits for random mode).
//     let x0: u256 = match x0_arg {
//         Some(v) => u256::from(v) & mask,
//         None => {
//             (u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128)) & mask
//         }
//     };

//     let inputs: Vec<u256> = (0..num_inputs)
//         .map(|i| {
//             if incremental {
//                 x0.overflowing_add(u256::from(i as u128)).0 & mask
//             } else {
//                 let r: u256 =
//                     u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128);
//                 ((x0 & fixed_mask) | (r & !fixed_mask)) & mask
//             }
//         })
//         .collect();

//     let data = compute_grid_parallel(
//         &circuit_one,
//         &circuit_two,
//         &inputs,
//         0,
//         circuit_one_len,
//         0,
//         circuit_two_len,
//         num_wires,
//         mask,
//         flag,
//         hw,
//         first_half,
//         second_half,
//     );

//     println!("Time elapsed: {:?}", Instant::now() - start_time);

//     let arr2 = Array2::from_shape_vec((num_points, 3), data).expect("grid shape mismatch");
//     PyArray2::from_owned_array(py, arr2).into()
// }

// /// Any 5000-gate corner of the full (canonicalized) heatmap. Reads the FULL circuits and
// /// canonicalizes both, then windows: `x_high` selects the last 5000 positions of c1 (else the
// /// first 5000), `y_high` the last 5000 of c2 (else the first 5000). So bottom-left = (false,false),
// /// top-right = (true,true), bottom-right = (true,false), top-left = (false,true). Because both
// /// circuits are read/canonicalized in full, every corner is consistent with the full heatmap.
// /// Supports random or incremental (x0) sampling, fixed bits, and first/second-half masking.
// #[pyfunction]
// fn heatmap_corner_at(
//     py: Python<'_>,
//     num_wires: usize,
//     num_inputs: usize,
//     flag: bool,
//     c1_path: &str,
//     c2_path: &str,
//     fix: usize,
//     hw: bool,
//     incremental: bool,
//     x0_arg: Option<u128>,
//     first_half: bool,
//     second_half: bool,
//     x_high: bool,
//     y_high: bool,
// ) -> Py<PyArray2<f64>> {
//     const CORNER: usize = 5000;
//     let mask = if num_wires < 256 {
//         (u256::one() << num_wires) - u256::one()
//     } else {
//         u256::MAX
//     };
//     println!(
//         "Running corner-at heatmap on {} inputs (x_high={}, y_high={}, {} base{})",
//         num_inputs,
//         x_high,
//         y_high,
//         if x0_arg.is_some() { "chosen" } else { "random" },
//         if incremental { ", incremental" } else { "" }
//     );
//     io::stdout().flush().unwrap();

//     let circuit_one_str = fs::read_to_string(c1_path).expect("Failed to read c1");
//     let circuit_two_str = fs::read_to_string(c2_path).expect("Failed to read c2");
//     let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
//     let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
//     circuit_one.canonicalize();
//     circuit_two.canonicalize();
//     let circuit_one_len = circuit_one.gates.len();
//     let circuit_two_len = circuit_two.gates.len();

//     let (x1, x2) = if x_high {
//         (circuit_one_len.saturating_sub(CORNER), circuit_one_len)
//     } else {
//         (0, CORNER.min(circuit_one_len))
//     };
//     let (y1, y2) = if y_high {
//         (circuit_two_len.saturating_sub(CORNER), circuit_two_len)
//     } else {
//         (0, CORNER.min(circuit_two_len))
//     };

//     let mut rng = rand::rng();
//     let start_time = Instant::now();
//     let mut fixed_mask = u256::zero();
//     let positions = (0..num_wires).choose_multiple(&mut rng, fix);
//     for p in positions {
//         fixed_mask |= u256::from(1) << p;
//     }
//     let x0: u256 = match x0_arg {
//         Some(v) => u256::from(v) & mask,
//         None => {
//             (u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128)) & mask
//         }
//     };
//     let inputs: Vec<u256> = (0..num_inputs)
//         .map(|i| {
//             if incremental {
//                 x0.overflowing_add(u256::from(i as u128)).0 & mask
//             } else {
//                 let r: u256 =
//                     u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128);
//                 ((x0 & fixed_mask) | (r & !fixed_mask)) & mask
//             }
//         })
//         .collect();

//     let data = compute_grid_parallel(
//         &circuit_one,
//         &circuit_two,
//         &inputs,
//         x1,
//         x2,
//         y1,
//         y2,
//         num_wires,
//         mask,
//         flag,
//         hw,
//         first_half,
//         second_half,
//     );

//     println!("Time elapsed: {:?}", Instant::now() - start_time);

//     let num_points = (x2 - x1 + 1) * (y2 - y1 + 1);
//     let arr2 = Array2::from_shape_vec((num_points, 3), data).expect("grid shape mismatch");
//     PyArray2::from_owned_array(py, arr2).into()
// }

// /// Memory-light subsampled mixing heatmap.
// /// Picks `n1`/`n2` evenly spaced positions along c1/c2 (including 0 and len),
// /// streams each circuit per input snapshotting only those positions, and
// /// returns (pos1, pos2, avg_hamming_fraction) for every sampled cell.
// #[pyfunction]
// fn heatmap_subsampled(
//     py: Python<'_>,
//     num_wires: usize,
//     num_inputs: usize,
//     c1: &str,
//     c2: &str,
//     n1: usize,
//     n2: usize,
// ) -> Py<PyArray2<f64>> {
//     let mask = if num_wires < 256 {
//         (u256::one() << num_wires) - u256::one()
//     } else {
//         u256::MAX
//     };
//     let s1 = fs::read_to_string(c1).expect("Failed to read c1");
//     let s2 = fs::read_to_string(c2).expect("Failed to read c2");
//     let circuit_one = CircuitSeq::from_string(&s1);
//     let circuit_two = CircuitSeq::from_string(&s2);
//     let len1 = circuit_one.gates.len();
//     let len2 = circuit_two.gates.len();

//     let sample_positions = |len: usize, n: usize| -> Vec<usize> {
//         if n <= 1 {
//             return vec![0, len];
//         }
//         let mut v: Vec<usize> = (0..n).map(|k| (k * len) / (n - 1)).collect();
//         v.dedup();
//         v
//     };
//     let p1 = sample_positions(len1, n1);
//     let p2 = sample_positions(len2, n2);
//     let m1 = p1.len();
//     let m2 = p2.len();

//     let mut rng = rand::rng();
//     let inputs: Vec<u256> = (0..num_inputs)
//         .map(|_| {
//             let r: u256 =
//                 u256::from(rng.random::<u128>()) | (u256::from(rng.random::<u128>()) << 128);
//             r & mask
//         })
//         .collect();

//     let snap = |gates: &Vec<[u16; 3]>, pos: &[usize], ib: u256| -> Vec<u256> {
//         let mut out = Vec::with_capacity(pos.len());
//         let mut state = ib;
//         let mut gi = 0usize;
//         for &target in pos {
//             while gi < target {
//                 state = crate::circuit::Gate::evaluate_index_256(state, gates[gi]);
//                 gi += 1;
//             }
//             out.push(state);
//         }
//         out
//     };

//     let start_time = Instant::now();
//     // Per-input snapshots (small: num_inputs * (m1+m2) u256), computed in parallel.
//     let snaps: Vec<(Vec<u256>, Vec<u256>)> = inputs
//         .par_iter()
//         .map(|&ib| {
//             (
//                 snap(&circuit_one.gates, &p1, ib),
//                 snap(&circuit_two.gates, &p2, ib),
//             )
//         })
//         .collect();
//     println!("Subsampled snapshots in {:?}", Instant::now() - start_time);

//     let nw = num_wires as f64;
//     let inv = 1.0 / num_inputs as f64;
//     let mut data = vec![0f64; m1 * m2 * 3];
//     data.par_chunks_mut(m2 * 3)
//         .enumerate()
//         .for_each(|(a, row)| {
//             for b in 0..m2 {
//                 let mut acc = 0f64;
//                 for s in &snaps {
//                     let x = s.0[a];
//                     let y = s.1[b];
//                     acc += popcount_u256((x ^ y) & mask) as f64 / nw;
//                 }
//                 row[b * 3] = p1[a] as f64;
//                 row[b * 3 + 1] = p2[b] as f64;
//                 row[b * 3 + 2] = acc * inv;
//             }
//         });

//     let arr2 = Array2::from_shape_vec((m1 * m2, 3), data).expect("grid shape mismatch");
//     PyArray2::from_owned_array(py, arr2).into()
// }

// #[pymodule]
// fn local_mixing(module: &Bound<'_, PyModule>) -> PyResult<()> {
//     // wrap the function, passing the module `m`
//     module.add_function(wrap_pyfunction!(heatmap, module)?)?;
//     module.add_function(wrap_pyfunction!(heatmap_subsampled, module)?)?;
//     module.add_function(wrap_pyfunction!(heatmap_incremental, module)?)?;
//     module.add_function(wrap_pyfunction!(heatmap_small, module)?)?;
//     module.add_function(wrap_pyfunction!(heatmap_slice, module)?)?;
//     module.add_function(wrap_pyfunction!(heatmap_mini_slice, module)?)?;
//     module.add_function(wrap_pyfunction!(heatmap_corner, module)?)?;
//     module.add_function(wrap_pyfunction!(heatmap_corner_at, module)?)?;
//     Ok(())
// }
