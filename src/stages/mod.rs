//! TDP stage ownership: sandwich, preprocessing and database mixing. Old
//! post-processing stages 4 (split), 5 (crossing walk) and 6 (fcompress) are
//! historic: see branch ran/archive/curated-and-postmix.
pub mod db_mixing;
pub mod preprocessing;
pub mod sandwich;
