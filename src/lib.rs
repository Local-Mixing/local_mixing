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
fn heatmap(py: Python<'_>, num_wires: usize, num_inputs: usize, flag: bool) -> Py<PyArray2<f64>> {
    println!("Running heatmap on {} inputs", num_inputs);
    io::stdout().flush().unwrap();
    // Load circuits
    let contents = fs::read_to_string("butterfly_recent.txt")
        .expect("Failed to read butterfly_recent.txt");

    let (circuit_one_str, circuit_two_str) = contents
        .split_once(':')
        .expect("Invalid format in butterfly_recent.txt");

    let mut circuit_one = CircuitSeq::from_string(circuit_one_str);
    let mut circuit_two = CircuitSeq::from_string(circuit_two_str);
    circuit_one.canonicalize();
    circuit_two.canonicalize();
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
        let input_bits: usize = if num_wires < usize::BITS as usize {
            rng.random_range(0..(1usize << num_wires))
        } else {
            rng.random_range(0..=usize::MAX)
        };

        let evolution_one = circuit_one.evaluate_evolution(input_bits);
        let evolution_two = circuit_two.evaluate_evolution(input_bits);

        for i1 in 0..=circuit_one_len {
            for i2 in 0..=circuit_two_len {
                let diff = evolution_one[i1] ^ evolution_two[i2];
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
fn heatmap_2(py: Python<'_>, num_wires: usize, num_inputs: usize, flag: bool) -> Py<PyArray2<f64>> {
    println!("Running heatmap on {} inputs", num_inputs);
    io::stdout().flush().unwrap();
    // Load circuits
    let contents = fs::read_to_string("butterfly_recent.txt")
        .expect("Failed to read butterfly_recent.txt");

    let (circuit_one_str, circuit_two_str) = contents
        .split_once(':')
        .expect("Invalid format in butterfly_recent.txt");

    let mut circuit_one = CircuitSeq::from_string(circuit_one_str);
    let mut circuit_two = CircuitSeq::from_string(circuit_two_str);
    circuit_one.canonicalize();
    circuit_two.canonicalize();
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
        let input_bits: usize = if num_wires < usize::BITS as usize {
            rng.random_range(0..(1usize << num_wires))
        } else {
            rng.random_range(0..=usize::MAX)
        };

        let evolution_one = circuit_one.evaluate_evolution(input_bits);
        let evolution_two = circuit_two.evaluate_evolution(input_bits);

        for i1 in 0..=circuit_one_len/2 {
            for i2 in 0..=circuit_two_len/2 {
                let diff = evolution_one[i1] ^ evolution_two[i2];
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
fn heatmap_4(py: Python<'_>, num_wires: usize, num_inputs: usize, flag: bool) -> Py<PyArray2<f64>> {
    println!("Running heatmap on {} inputs", num_inputs);
    io::stdout().flush().unwrap();
    // Load circuits
    let contents = fs::read_to_string("butterfly_recent.txt")
        .expect("Failed to read butterfly_recent.txt");

    let (circuit_one_str, circuit_two_str) = contents
        .split_once(':')
        .expect("Invalid format in butterfly_recent.txt");

    let mut circuit_one = CircuitSeq::from_string(circuit_one_str);
    let mut circuit_two = CircuitSeq::from_string(circuit_two_str);
    circuit_one.canonicalize();
    circuit_two.canonicalize();
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
        let input_bits: usize = if num_wires < usize::BITS as usize {
            rng.random_range(0..(1usize << num_wires))
        } else {
            rng.random_range(0..=usize::MAX)
        };

        let evolution_one = circuit_one.evaluate_evolution(input_bits);
        let evolution_two = circuit_two.evaluate_evolution(input_bits);

        for i1 in 0..=circuit_one_len/2 {
            for i2 in 0..=circuit_two_len/2 {
                let diff = evolution_one[i1] ^ evolution_two[i2];
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

#[pymodule]
fn local_mixing(module: &Bound<'_, PyModule>) -> PyResult<()> {
    // wrap the function, passing the module `m`
    module.add_function(wrap_pyfunction!(heatmap, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap_2, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap_4, module)?)?;
    Ok(())
}
