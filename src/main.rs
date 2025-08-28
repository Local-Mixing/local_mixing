use local_mixing::{
                rainbow::constants::{self, CONTROL_FUNC_TABLE},
                circuit::{Circuit, Gate, Permutation},
                };
use local_mixing::rainbow::rainbow::main_rainbow;
fn main () {
    // Example arguments
    let n = 4;
    let m = 6;
    let load = None; // or Some("db_file".to_string())
    let fresh = true;

    main_rainbow(n, m, load, fresh);
}


// fn main() {
//     let c = Circuit::random_circuit(4,3, &mut rand::rng());
//     let perm = c.permutation();
//     println!("{}", c.to_string());
//     for n in &perm.data {
//         println!("{:0width$b}", n, width = c.num_wires); // pad to num_wires bits
//     }
// }


// fn find_circuit_no_pin_last_wire(n: usize) -> () {
//     let mut count = 0;
//     loop {
//         let rand_circ = Circuit::random_circuit(10,5, &mut rand::rng());
//         let mut found = true;
//         for gate in &rand_circ.gates {
//             for pins in gate.pins {
//                 if pins >= rand_circ.num_wires-n {
//                 found = false;
//                 break;
//                 }
//             } 
//         }
//         if found {
//             println!("Circuits tested: {}", count);
//             println!("{}", rand_circ.to_string());
//             break;
//         }
//         count += 1;
//     }
// }

// fn test_equiv_circuits() {
//     let circuit_one = Circuit::new
//     (3, vec![
//         Gate::new(1,2,0,0), 
//         Gate::new(1,2,0,0)]);
    
//     let circuit_two = Circuit::new
//     (3, vec![
//         Gate::new(1,2,0,15), 
//         Gate::new(1,2,0,0)]);
    
//     println!("{}", circuit_one.to_string());
//     println!("{}", circuit_two.to_string());
//     match circuit_one.probably_equal(&circuit_two, 2) {
//         Ok(()) => println!("The circuits are probably equal. No tests failed"),
//         _ => println!("The circuits are not equal. Test has failed."),
//     }
// }
