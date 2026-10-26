//! TEMPORARY API STUB (synthesis-half branch): the declared interface of the
//! analysis half, implemented on the analysis branch. Every body is
//! `todo!()`; the merge replaces this file.
use crate::circuit::xgate::XGate;
use std::collections::BTreeSet;
use std::path::Path;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BoundaryRow {
    pub source_gate: usize,
    pub source_wire: u16,
    pub input_quadratic: bool,
    pub output_quadratic: bool,
    pub is_output: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Boundary {
    pub rows: Vec<BoundaryRow>,
    pub excluded: BTreeSet<usize>,
    pub probe_inputs: usize,
}

pub const DEFAULT_PROBE_INPUTS: usize = 8192;

pub fn classify(_c: &[XGate], _n: usize, _probe_inputs: usize) -> Boundary {
    todo!("analysis half: boundary::classify")
}

pub fn write_file(_b: &Boundary, _path: &Path) -> std::io::Result<()> {
    todo!("analysis half: boundary::write_file")
}

pub fn read_file(_path: &Path) -> std::io::Result<Boundary> {
    todo!("analysis half: boundary::read_file")
}
