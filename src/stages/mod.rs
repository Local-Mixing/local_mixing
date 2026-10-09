//! TDP stage ownership: sandwich, preprocessing, database mixing and packing
//! (stage 4). Old post-processing stages 4 (split), 5 (crossing walk) and 6
//! (fcompress) are historic: see branch ran/archive/curated-and-postmix.
pub mod db_mixing;
pub mod packing;
pub mod preprocessing;
pub mod sandwich;
