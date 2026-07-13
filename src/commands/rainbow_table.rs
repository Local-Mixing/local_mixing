use std::sync::Arc;

use local_mixing::rainbow_table::{
    build_from_2rocks, build_from_rocks, build_m1, combine_rocks_dbs, open_db_for_read,
    open_db_for_write, rocks_to_lmdb,
};

/// `rocksdb_1`: build the m-gate rainbow DB by extending the (m-1)-gate DB one
/// gate at a time (m == 1 builds the base case). Reads ./rocks_db_m{m-1},
/// writes ./test_rocks_db_m{m}.
pub fn run_rocksdb_1(sub: &clap::ArgMatches) {
    let m: usize = *sub.get_one("m").expect("Missing -m <gates>");
    let min_n: usize = *sub.get_one("min_n").unwrap_or(&0);
    let max_n: usize = *sub.get_one("max_n").unwrap_or(&0);
    let no_rule_l: bool = *sub.get_one::<bool>("no_L").unwrap_or(&false);
    let new_db = Arc::new(open_db_for_write(m));
    if m == 1 {
        build_m1(&new_db).expect("build_m1 failed");
    } else {
        let old_db = Arc::new(open_db_for_read(m - 1));
        build_from_rocks(&old_db, &new_db, m, min_n, max_n, no_rule_l)
            .expect("build_from_rocks failed");
    }
}

/// `rocksdb_2`: build the (m1+m2)-gate rainbow DB by combining the m1 and m2
/// DBs over all wire overlaps. Reads ./rocks_db_m{m1} and ./rocks_db_m{m2},
/// writes ./test_rocks_db_m{m1+m2}.
pub fn run_rocksdb_2(sub: &clap::ArgMatches) {
    let m1: usize = *sub.get_one("m1").expect("Missing --m1 <gates>");
    let m2: usize = *sub.get_one("m2").expect("Missing --m2 <gates>");
    let min_n: usize = *sub.get_one("min_n").unwrap_or(&0);
    let new_db = Arc::new(open_db_for_write(m1 + m2));
    let old_db1 = Arc::new(open_db_for_read(m1));
    let old_db2 = Arc::new(open_db_for_read(m2));
    build_from_2rocks(&old_db1, &old_db2, &new_db, m1, m2, min_n)
        .expect("build_from_2rocks failed");
}

/// `combine_rocks`: merge ./rocks_db_m1..=m9 into a single keyed RocksDB.
pub fn run_combine_rocks(sub: &clap::ArgMatches) {
    let path: &String = sub.get_one("path").expect("Missing -p <path>");
    combine_rocks_dbs(path).expect("combine_rocks_dbs failed");
}

/// `rocks_to_lmdb`: convert a combined RocksDB into the sharded LMDB store the
/// mixing code reads.
pub fn run_rocks_to_lmdb(sub: &clap::ArgMatches) {
    let source: &String = sub.get_one("source").expect("Missing -s <source>");
    let path: &String = sub.get_one("path").expect("Missing -p <path>");
    if let Err(e) = rocks_to_lmdb(source, path) {
        let msg = format!("rocks_to_lmdb failed: {}", e);
        eprintln!("{}", msg);
        let _ = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("error.txt")
            .and_then(|mut f| {
                use std::io::Write;
                writeln!(f, "{}", msg)
            });
        std::process::exit(1);
    }
}
