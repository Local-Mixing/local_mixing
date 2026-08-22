pub mod compress;
#[cfg(feature = "legacy-db-tools")]
pub mod db_generation;
#[cfg(not(feature = "legacy-db-tools"))]
pub mod db_generation {
    fn unavailable() -> ! {
        panic!("regular database generation requires --features legacy-db-tools")
    }

    pub fn run_rocksdb_1(_: &clap::ArgMatches) {
        unavailable()
    }

    pub fn run_rocksdb_2(_: &clap::ArgMatches) {
        unavailable()
    }

    pub fn run_rocks_to_lmdb(_: &clap::ArgMatches) {
        unavailable()
    }
}
pub mod equal;
pub mod evaluate;
pub mod genran;
pub mod gss;
pub mod shoot;
pub mod shuffle;
pub mod ssg;
pub mod sss;
