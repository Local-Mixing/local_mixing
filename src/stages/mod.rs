//! TDP stage ownership: sandwich (1), preprocessing (2), database mixing (3),
//! post-processing (4 split, 5 crossing walk), packing (6: windowed ANF
//! delivery) and unpacking (7: resynthesis from the delivered ANF windows).
pub mod adaptive;
pub mod db_mixing;
pub mod packing;
#[path = "post-processing/mod.rs"]
pub mod post_processing;
pub mod preprocessing;
pub mod sandwich;
pub mod unpacking;
