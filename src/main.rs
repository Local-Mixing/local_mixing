use local_mixing::random::random_data::main_random;
use local_mixing::rainbow::rainbow::{main_rainbow_generate, main_rainbow_load};
use local_mixing::rainbow::explore::explore_db;
use clap::{Arg, ArgAction, Command};

fn main() {
    let matches = Command::new("rainbow")
        .about("Rainbow circuit generator")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(
            Command::new("new")
                .about("Build a new database")
                .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
                .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize))),
        )
        .subcommand(
            Command::new("load")
                .about("Load an existing database")
                .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
                .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize))),
        )
        .subcommand(
            Command::new("explore")
                .about("Explore an existing database")
                .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
                .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize))),
        )
        .subcommand(
            Command::new("random")
                .about("Generate random circuits and store in DB")
                .arg(Arg::new("n").short('n').long("n").required(true).value_parser(clap::value_parser!(usize)))
                .arg(Arg::new("m").short('m').long("m").required(true).value_parser(clap::value_parser!(usize)))
                .arg(
                    Arg::new("count")
                        .short('c')
                        .long("count")
                        .value_parser(clap::value_parser!(usize))
                        .conflicts_with("sliding"),
                )
                .arg(
                    Arg::new("sliding")
                        .short('C')
                        .long("sliding")
                        .action(ArgAction::SetTrue)
                        .conflicts_with("count"),
                ),
        )
        .get_matches();

    match matches.subcommand() {
        Some(("new", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            main_rainbow_generate(n, m);
        }
        Some(("load", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            main_rainbow_load(n, m, "./db");
        }
        Some(("explore", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();
            explore_db(n, m);
        }
        Some(("random", sub)) => {
            let n: usize = *sub.get_one("n").unwrap();
            let m: usize = *sub.get_one("m").unwrap();

            if let Some(count) = sub.get_one::<usize>("count") {
                // Fixed-count mode
                main_random(n, m, *count, false);
            } else if sub.get_flag("sliding") {
                // Sliding-window fail-rate mode
                main_random(n, m, 0, true);
            } else {
                panic!("You must provide either -c <count> or -C for sliding-window mode");
            }
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
