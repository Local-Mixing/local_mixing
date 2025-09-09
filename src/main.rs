use local_mixing::{
                rainbow::constants::{self, CONTROL_FUNC_TABLE},
                circuit::{Circuit, Gate, Permutation},
                };
use local_mixing::rainbow::rainbow::main_rainbow;
use local_mixing::rainbow::explore::explore_db;
use clap::{Arg, Command};

fn main() {
    let matches = Command::new("rainbow")
        .about("Rainbow circuit generator")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("new")
                .about("Build a new database")
                .arg(
                    Arg::new("n")
                        .short('n')
                        .long("n")
                        .required(true)
                        .value_parser(clap::value_parser!(usize)),
                )
                .arg(
                    Arg::new("m")
                        .short('m')
                        .long("m")
                        .required(true)
                        .value_parser(clap::value_parser!(usize)),
                ),
        )
        .subcommand(
            Command::new("load")
                .about("Load an existing database")
                .arg(
                    Arg::new("n")
                        .short('n')
                        .long("n")
                        .required(true)
                        .value_parser(clap::value_parser!(usize)),
                )
                .arg(
                    Arg::new("m")
                        .short('m')
                        .long("m")
                        .required(true)
                        .value_parser(clap::value_parser!(usize)),
                ),
        )
        .subcommand(
            Command::new("explore")
                .about("Explore an existing database")
                .arg(
                    Arg::new("n")
                        .short('n')
                        .long("n")
                        .required(true)
                        .value_parser(clap::value_parser!(usize)),
                )
                .arg(
                    Arg::new("m")
                        .short('m')
                        .long("m")
                        .required(true)
                        .value_parser(clap::value_parser!(usize)),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("new", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            main_rainbow(n, m, None, true);
        }
        Some(("load", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            main_rainbow(n, m, Some("./db".to_string()), false);
        }
        Some(("explore", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            explore_db(n, m);
        }
        _ => unreachable!(),
    }
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
