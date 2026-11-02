//! Old stages 4-5 (split, crossing) share a namespace while retaining separate algorithms and state contracts.
//! `compression` is the fcompress engine (gather/reduce + Toffoli-sliding transport + conjugation
//! downhill), revived from branch ran/archive/curated-and-postmix for stage 6's multi-scale descent.
pub mod compression;
pub mod crossing;
pub mod splitting;
