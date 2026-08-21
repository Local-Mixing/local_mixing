pub mod circuit;
pub mod db_generation;
pub mod db_mixing;
pub mod engine;
pub mod experimental;
pub mod postprocessing;
pub mod preprocessing;

// Compatibility path for downstream callers using the pre-reorganization
// module name. New code should use `postprocessing`.
#[doc(hidden)]
pub use postprocessing as fragmentation;

// The PyO3 heatmap surface is red-team analysis code. Keep its Python API
// registered here for compatibility while its implementation lives outside
// the production source tree.
#[path = "../red_team_tests/heatmap.rs"]
mod heatmap;

use pyo3::prelude::*;

#[pymodule]
fn local_mixing(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(heatmap::heatmap, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap::heatmap_subsampled, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap::heatmap_incremental, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap::heatmap_small, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap::heatmap_slice, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap::heatmap_mini_slice, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap::heatmap_corner, module)?)?;
    module.add_function(wrap_pyfunction!(heatmap::heatmap_corner_at, module)?)?;
    Ok(())
}
