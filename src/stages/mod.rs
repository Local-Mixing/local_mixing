//! TDP stage ownership: sandwich, preprocessing, database mixing, packing
//! (windowed ANF delivery) and unpacking (stage 7: resynthesis from the
//! delivered ANF windows). Old post-processing stages 4 (split), 5
//! (crossing walk) and 6 (fcompress) are historic: see branch
//! ran/archive/curated-and-postmix.
pub mod db_mixing;
pub mod packing;
pub mod preprocessing;
pub mod sandwich;
pub mod unpacking;
