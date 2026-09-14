//! Stage 2: quadratic masking and native nonlinear291 preprocessing.

pub mod construct;
pub mod nonlinear291;
pub mod quadratic_masking;
pub mod slice_guards;
pub mod types;
pub mod verify;
pub use construct::preprocess_sandwich;
pub use types::{PreprocessingMode, PreprocessingOutput, PreprocessingParams};
