use local_mixing::{
                temp::constants::{self, CONTROL_FUNC_TABLE},
                circuit::{Circuit, Gate},
                };

fn main() {
    println!("{}", Circuit::random_circuit(5,10,&mut rand::rng()).to_string());
}
