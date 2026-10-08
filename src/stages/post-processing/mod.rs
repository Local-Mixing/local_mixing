//! Old stages 4-5 (split, crossing) share a namespace while retaining separate algorithms and state contracts.
//! Old stage 6 (fcompress) is historic: see branch ran/archive/curated-and-postmix.
pub mod crossing;
pub mod splitting;
