pub mod circuit;
pub mod control_functions;
pub use circuit::{Circuit, Gate, Permutation, CircuitSeq};
pub use self::circuit::base_gates;
pub use self::circuit::par_all_circuits;