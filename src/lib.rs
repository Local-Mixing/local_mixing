pub mod circuit;
pub mod replace;
pub mod rainbow;
pub mod random;
use pyo3::prelude::*;
use numpy::PyArray2;
use std::fs;
use crate::circuit::CircuitSeq;
use std::time::Instant;
use rand::Rng;
use numpy::ndarray::Array2;
use std::io::{self, Write};

#[pyfunction]
fn heatmap(py: Python<'_>, num_wires: usize, num_inputs: usize, flag: bool, c1: &str, c2: &str, canon: bool) -> Py<PyArray2<f64>> {
    let mask: u128 = if num_wires < u128::BITS as usize {
        (1 << num_wires) - 1
    } else {
        u128::MAX
    };
    println!("Running heatmap on {} inputs", num_inputs);
    io::stdout().flush().unwrap();
    // Load circuits
    let circuit_one_str = fs::read_to_string(c1)
        .expect("Failed to read butterfly_recent.txt");
    let circuit_two_str = fs::read_to_string(c2)
        .expect("Failed to read butterfly_recent.txt");
    let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
    let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
    if canon {
        circuit_one.canonicalize();
        circuit_two.canonicalize();
    }
    let circuit_one_len = circuit_one.gates.len();
    let circuit_two_len = circuit_two.gates.len();

    let num_points = (circuit_one_len + 1) * (circuit_two_len + 1);
    let mut average = vec![0f64; num_points * 3]; // flat 2D array: [x, y, value] per point
    let mut rng = rand::rng();
    let start_time = Instant::now();

    for i in 0..num_inputs {
        if i % 10 == 0 {
            println!("{}/{}", i, num_inputs);
            io::stdout().flush().unwrap();
        }
        let input_bits: u128 = if num_wires < u128::BITS as usize {
            rng.random_range(0..(1u128 << num_wires))
        } else {
            rng.random_range(0..=u128::MAX)
        };

        let evolution_one = circuit_one.evaluate_evolution_128(input_bits);
        let evolution_two = circuit_two.evaluate_evolution_128(input_bits);

        for i1 in 0..=circuit_one_len {
            for i2 in 0..=circuit_two_len {
                let diff = (evolution_one[i1] ^ evolution_two[i2]) & mask;
                let hamming_dist = diff.count_ones() as f64;
                let overlap = if !flag {
                    hamming_dist / num_wires as f64
                } else {
                    let tmp = (2.0 * hamming_dist / num_wires as f64) - 1.0;
                    tmp.abs()
                };

                let index = i1 * (circuit_two_len + 1) + i2;
                average[index * 3] = i1 as f64;
                average[index * 3 + 1] = i2 as f64;
                average[index * 3 + 2] += overlap / num_inputs as f64;
            }
        }
    }

    println!("Time elapsed: {:?}", Instant::now() - start_time);

    let mut arr2 = Array2::<f64>::zeros((num_points, 3));
    for i in 0..num_points {
        arr2[[i, 0]] = average[i * 3];
        arr2[[i, 1]] = average[i * 3 + 1];
        arr2[[i, 2]] = average[i * 3 + 2];
    }

    let pyarray = PyArray2::from_owned_array(py, arr2);

    pyarray.into()
}

#[pyfunction]
fn heatmap_small(py: Python<'_>, num_wires: usize, flag: bool, c1: &str, c2: &str, canon: bool) -> Py<PyArray2<f64>> {
    let mask: u128 = if num_wires < u128::BITS as usize {
        (1 << num_wires) - 1
    } else {
        u128::MAX
    };
    println!("Running heatmap on weights 0, 1, and 2");
    io::stdout().flush().unwrap();
    // Load circuits
    let circuit_one_str = fs::read_to_string(c1)
        .expect("Failed to read butterfly_recent.txt");
    let circuit_two_str = fs::read_to_string(c2)
        .expect("Failed to read butterfly_recent.txt");
    let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
    let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
    if canon {
        circuit_one.canonicalize();
        circuit_two.canonicalize();
    }
    let circuit_one_len = circuit_one.gates.len();
    let circuit_two_len = circuit_two.gates.len();

    let num_points = (circuit_one_len + 1) * (circuit_two_len + 1);
    let mut average = vec![0f64; num_points * 3]; // flat 2D array: [x, y, value] per point
    let start_time = Instant::now();

    // Generate inputs of Hamming weight 0, 1, and 2
    let mut inputs: Vec<u128> = Vec::new();

    inputs.push(0);

    for i in 0..num_wires {
        inputs.push(1u128 << i);
    }

    for i in 0..num_wires {
        for j in (i + 1)..num_wires {
            inputs.push((1u128 << i) | (1u128 << j));
        }
    }

    let effective_inputs = inputs.len() as f64;

    for (i, &input_bits) in inputs.iter().enumerate() {
        if i % 10 == 0 {
            println!("{}/{}", i, inputs.len());
            io::stdout().flush().unwrap();
        }

        let evolution_one = circuit_one.evaluate_evolution_128(input_bits);
        let evolution_two = circuit_two.evaluate_evolution_128(input_bits);

        for i1 in 0..=circuit_one_len {
            for i2 in 0..=circuit_two_len {
                let diff = (evolution_one[i1] ^ evolution_two[i2]) & mask;
                let hamming_dist = diff.count_ones() as f64;
                let overlap = if !flag {
                    hamming_dist / num_wires as f64
                } else {
                    let tmp = (2.0 * hamming_dist / num_wires as f64) - 1.0;
                    tmp.abs()
                };

                let index = i1 * (circuit_two_len + 1) + i2;
                average[index * 3] = i1 as f64;
                average[index * 3 + 1] = i2 as f64;
                average[index * 3 + 2] += overlap / effective_inputs;
            }
        }
    }

    println!("Time elapsed: {:?}", Instant::now() - start_time);

    let mut arr2 = Array2::<f64>::zeros((num_points, 3));
    for i in 0..num_points {
        arr2[[i, 0]] = average[i * 3];
        arr2[[i, 1]] = average[i * 3 + 1];
        arr2[[i, 2]] = average[i * 3 + 2];
    }

    let pyarray = PyArray2::from_owned_array(py, arr2);

    pyarray.into()
}

#[pyfunction]
fn heatmap_slice(py: Python<'_>, num_wires: usize, num_inputs: usize, flag: bool, x1: usize, x2: usize, y1: usize, y2: usize, c1_path: &str,
    c2_path: &str) -> Py<PyArray2<f64>> {
    println!("Running heatmap on {} inputs", num_inputs);
    io::stdout().flush().unwrap();
    // Load circuits
    let circuit_one_str = fs::read_to_string(c1_path)
        .unwrap_or_else(|_| panic!("Failed to read circuit file: {}", c1_path));
    let circuit_two_str = fs::read_to_string(c2_path)
        .unwrap_or_else(|_| panic!("Failed to read circuit file: {}", c2_path));

    let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
    let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
    circuit_one.canonicalize();
    circuit_two.canonicalize();
    circuit_one.gates = circuit_one.gates[..=x2].to_vec();
    circuit_two.gates = circuit_two.gates[..=y2].to_vec();
    let num_points = (x2 - x1 + 1) * (y2 - y1 + 1);
    let mut average = vec![0f64; num_points * 3]; // flat 2D array: [x, y, value] per point
    let mut rng = rand::rng();
    let start_time = Instant::now();

    for i in 0..num_inputs {
        if i % 10 == 0 {
            println!("{}/{}", i, num_inputs);
            io::stdout().flush().unwrap();
        }
        let input_bits: u128 = if num_wires < u128::BITS as usize {
            rng.random_range(0..(1u128 << num_wires))
        } else {
            rng.random_range(0..=u128::MAX)
        };

        let evolution_one = circuit_one.evaluate_evolution_128(input_bits);
        let evolution_two = circuit_two.evaluate_evolution_128(input_bits);

        for i1 in x1..=x2 {
            for i2 in y1..=y2 {
                let diff = evolution_one[i1] ^ evolution_two[i2];
                let hamming_dist = diff.count_ones() as f64;
                let overlap = if !flag {
                    hamming_dist / num_wires as f64
                } else {
                    let tmp = (2.0 * hamming_dist / num_wires as f64) - 1.0;
                    tmp.abs()
                };

                let rel_i1 = i1 - x1;
                let rel_i2 = i2 - y1;
                let index = rel_i1 * (y2 - y1 + 1) + rel_i2;
                average[index * 3] = i1 as f64;
                average[index * 3 + 1] = i2 as f64;
                average[index * 3 + 2] += overlap / num_inputs as f64;
            }
        }
    }

    println!("Time elapsed: {:?}", Instant::now() - start_time);

    let mut arr2 = Array2::<f64>::zeros((num_points, 3));
    for i in 0..num_points {
        arr2[[i, 0]] = average[i * 3];
        arr2[[i, 1]] = average[i * 3 + 1];
        arr2[[i, 2]] = average[i * 3 + 2];
    }

    let pyarray = PyArray2::from_owned_array(py, arr2);

    pyarray.into()
}

#[pyfunction]
fn heatmap_mini_slice(py: Python<'_>, num_wires: usize, num_inputs: usize, flag: bool, x1: usize, x2: usize, y1: usize, y2: usize, c1_path: &str,
    c2_path: &str) -> Py<PyArray2<f64>> {
    println!("Running heatmap on {} inputs", num_inputs);
    io::stdout().flush().unwrap();
    // Load circuits
    let circuit_one_str = fs::read_to_string(c1_path)
        .unwrap_or_else(|_| panic!("Failed to read circuit file: {}", c1_path));
    let circuit_two_str = fs::read_to_string(c2_path)
        .unwrap_or_else(|_| panic!("Failed to read circuit file: {}", c2_path));

    let mut circuit_one = CircuitSeq::from_string(&circuit_one_str);
    let mut circuit_two = CircuitSeq::from_string(&circuit_two_str);
    circuit_one.canonicalize();
    circuit_two.canonicalize();
    circuit_one.gates = circuit_one.gates[x1..=x2].to_vec();
    circuit_two.gates = circuit_two.gates[y1..=y2].to_vec();
    let num_points = (x2 - x1 + 1) * (y2 - y1 + 1);
    let mut average = vec![0f64; num_points * 3]; // flat 2D array: [x, y, value] per point
    let mut rng = rand::rng();
    let start_time = Instant::now();

    for i in 0..num_inputs {
        if i % 10 == 0 {
            println!("{}/{}", i, num_inputs);
            io::stdout().flush().unwrap();
        }
        let input_bits: u128 = if num_wires < u128::BITS as usize {
            rng.random_range(0..(1u128 << num_wires))
        } else {
            rng.random_range(0..=u128::MAX)
        };

        let evolution_one = circuit_one.evaluate_evolution_128(input_bits);
        let evolution_two = circuit_two.evaluate_evolution_128(input_bits);

        for i1 in x1..=x2 {
            for i2 in y1..=y2 {
                let diff = evolution_one[i1] ^ evolution_two[i2];
                let hamming_dist = diff.count_ones() as f64;
                let overlap = if !flag {
                    hamming_dist / num_wires as f64
                } else {
                    let tmp = (2.0 * hamming_dist / num_wires as f64) - 1.0;
                    tmp.abs()
                };

                let rel_i1 = i1 - x1;
                let rel_i2 = i2 - y1;
                let index = rel_i1 * (y2 - y1 + 1) + rel_i2;
                average[index * 3] = i1 as f64;
                average[index * 3 + 1] = i2 as f64;
                average[index * 3 + 2] += overlap / num_inputs as f64;
            }
        }
    }

    println!("Time elapsed: {:?}", Instant::now() - start_time);

    let mut arr2 = Array2::<f64>::zeros((num_points, 3));
    for i in 0..num_points {
        arr2[[i, 0]] = average[i * 3];
        arr2[[i, 1]] = average[i * 3 + 1];
        arr2[[i, 2]] = average[i * 3 + 2];
    }

    let pyarray = PyArray2::from_owned_array(py, arr2);

    pyarray.into()
}

#[pymodule]
fn local_mixing(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // wrap the function, passing the module `m`
    module.add_function(wrap_pyfunction!(heatmap, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap_small, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap_slice, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap_mini_slice, module)?)?;
    Ok(())
}
