//! General gate tape with an explicit physical wire count.
use super::XGate;

#[derive(Clone, Debug)]
pub struct Circuit {
    pub gates: Vec<XGate>,
    pub num_wires: usize,
}

/// Historical name retained for callers; the tape supports general X gates.
pub use Circuit as CnotCircuit;
