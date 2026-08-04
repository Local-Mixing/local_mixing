pub mod compress;
pub mod equal;
pub mod evaluate;
pub mod genran;
#[cfg(feature = "legacy-db-tools")]
pub mod rainbow_table;
#[cfg(not(feature = "legacy-db-tools"))]
pub mod rainbow_table {
    fn unavailable() -> ! {
        panic!("legacy database tooling requires --features legacy-db-tools")
    }

    pub fn run_rocksdb_1(_: &clap::ArgMatches) {
        unavailable()
    }

    pub fn run_rocksdb_2(_: &clap::ArgMatches) {
        unavailable()
    }

    pub fn run_combine_rocks(_: &clap::ArgMatches) {
        unavailable()
    }

    pub fn run_rocks_to_lmdb(_: &clap::ArgMatches) {
        unavailable()
    }
}
pub mod r_ssg;
pub mod shoot;
pub mod shuffle;
pub mod sss;
