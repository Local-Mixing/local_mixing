//! General gate tape with an explicit physical wire count.
use super::XGate;

#[derive(Clone, Debug)]
pub struct Circuit {
    pub gates: Vec<XGate>,
    pub num_wires: usize,
}
