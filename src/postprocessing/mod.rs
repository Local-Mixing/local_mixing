//! Compatibility exports for callers predating the stage ownership layout.
//! New code uses `stages::post_processing`.
pub use crate::stages::post_processing::compression as compress;
pub use crate::stages::post_processing::compression::downhill;
pub use crate::stages::post_processing::{crossing as cross_walk, splitting};
