//! TEMPORARY API STUB (synthesis-half branch): the declared interface of the
//! analysis half, implemented on the analysis branch. Every body is
//! `todo!()`; the merge replaces this file.
use super::types::{Census, LeakageMetrics, Source};
use crate::circuit::xgate::XGate;
use std::collections::BTreeSet;
use std::path::Path;

pub const DEFAULT_BATCHES: usize = 96;

pub fn census(_source: &Source, _circuit: &[XGate], _width: usize, _batches: usize) -> Census {
    todo!("analysis half: census")
}

pub fn write_dir(_c: &Census, _dir: &Path) -> std::io::Result<()> {
    todo!("analysis half: census::write_dir")
}

pub fn read_dir(_dir: &Path) -> std::io::Result<Census> {
    todo!("analysis half: census::read_dir")
}

impl Census {
    pub fn metrics(&self, _source: &Source, _validated_only: bool) -> LeakageMetrics {
        todo!("analysis half: Census::metrics")
    }

    pub fn coverage(&self) -> Vec<i32> {
        todo!("analysis half: Census::coverage")
    }

    pub fn interior_sets(&self, _edges: &[usize], _source: &Source) -> Vec<BTreeSet<usize>> {
        todo!("analysis half: Census::interior_sets")
    }
}
