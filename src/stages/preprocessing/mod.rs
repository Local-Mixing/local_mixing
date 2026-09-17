//! Stage 2: embedded masking and native nonlinear291 preprocessing.

pub mod construct;
pub mod embedded_masking;
pub mod nonlinear291;
pub mod preprocessing_shuffling;
pub mod slice_guards;
pub mod types;
pub mod verify;
pub use construct::preprocess_sandwich;
pub use types::{PreprocessingMode, PreprocessingOutput, PreprocessingParams};
