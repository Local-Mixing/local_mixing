//! Optional Python analysis extension; retains the local_mixing module API.
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
