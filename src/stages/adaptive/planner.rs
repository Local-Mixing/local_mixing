//! TEMPORARY API STUB (synthesis-half branch): the declared interface of the
//! analysis half, implemented on the analysis branch. Every body is
//! `todo!()`; the merge replaces this file.
use super::types::{Census, Plan, PlanParams, Source};
use crate::circuit::xgate::XGate;
use std::path::Path;

pub fn plan(
    _circuit: &[XGate],
    _width: usize,
    _census: &Census,
    _source: &Source,
    _params: &PlanParams,
    _prev_bounds: &[usize],
) -> Plan {
    todo!("analysis half: planner::plan")
}

pub struct Profile;

impl Profile {
    pub fn new(_census: &Census, _source: &Source) -> Profile {
        todo!("analysis half: Profile::new")
    }

    pub fn split(&self, _lo: usize, _hi: usize) -> usize {
        todo!("analysis half: Profile::split")
    }
}

pub fn write_plan(_p: &Plan, _dir: &Path) -> std::io::Result<()> {
    todo!("analysis half: planner::write_plan")
}

pub fn read_bounds(_path: &Path) -> std::io::Result<Vec<usize>> {
    todo!("analysis half: planner::read_bounds")
}
