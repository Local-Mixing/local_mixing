use std::sync::Arc;

use local_mixing::db_generation::regular::{
    MAX_REGULAR_GATES, build_from_2rocks, build_from_rocks, build_m1, open_db_for_read,
    open_db_for_write, rocks_to_lmdb,
};

fn validate_rocksdb_1_bounds(m: usize, min_n: usize, max_n: usize) -> Result<(), String> {
    if !(1..=MAX_REGULAR_GATES).contains(&m) {
        return Err(format!("m must be in 1..={MAX_REGULAR_GATES}"));
    }
    if max_n != 0 && min_n > max_n {
        return Err(format!("min_n ({min_n}) cannot exceed max_n ({max_n})"));
    }
    let wire_limit = 3 * m;
    if min_n > wire_limit {
        return Err(format!("min_n ({min_n}) cannot exceed 3*m ({wire_limit})"));
    }
    if max_n > wire_limit {
        return Err(format!("max_n ({max_n}) cannot exceed 3*m ({wire_limit})"));
    }
    if m == 1 && max_n != 0 && max_n < 3 {
        return Err(
            "m=1 always uses exactly 3 wires, so a nonzero max_n must be at least 3".into(),
        );
    }
    Ok(())
}

fn validate_rocksdb_2_bounds(m1: usize, m2: usize, min_n: usize) -> Result<(), String> {
    if m1 == 0 || m2 == 0 {
        return Err("m1 and m2 must both be nonzero".to_string());
    }
    let total = m1
        .checked_add(m2)
        .ok_or_else(|| "m1 + m2 overflowed".to_string())?;
    if total > MAX_REGULAR_GATES {
        return Err(format!(
            "m1 + m2 ({total}) cannot exceed {MAX_REGULAR_GATES}"
        ));
    }
    let wire_limit = 3 * total;
    if min_n > wire_limit {
        return Err(format!(
            "min_n ({min_n}) cannot exceed 3*(m1+m2) ({wire_limit})"
        ));
    }
    Ok(())
}

fn invalid_arguments(error: String) -> ! {
    eprintln!("invalid regular database generation arguments: {error}");
    std::process::exit(2)
}

/// `rocksdb_1`: build the m-gate rainbow DB by extending the (m-1)-gate DB one
/// gate at a time (m == 1 builds the base case). Reads ./rocks_db_m{m-1},
/// writes ./test_rocks_db_m{m}.
pub fn run_rocksdb_1(sub: &clap::ArgMatches) {
    let m: usize = *sub.get_one("m").expect("Missing -m <gates>");
    let min_n: usize = *sub.get_one("min_n").unwrap_or(&0);
    let max_n: usize = *sub.get_one("max_n").unwrap_or(&0);
    let no_rule_l: bool = *sub.get_one::<bool>("no_L").unwrap_or(&false);
    validate_rocksdb_1_bounds(m, min_n, max_n).unwrap_or_else(|error| invalid_arguments(error));
    let new_db = Arc::new(open_db_for_write(m).expect("open regular DB output"));
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
    validate_rocksdb_2_bounds(m1, m2, min_n).unwrap_or_else(|error| invalid_arguments(error));
    let new_db = Arc::new(open_db_for_write(m1 + m2).expect("open regular DB output"));
    let old_db1 = Arc::new(open_db_for_read(m1));
    let old_db2 = Arc::new(open_db_for_read(m2));
    build_from_2rocks(&old_db1, &old_db2, &new_db, m1, m2, min_n)
        .expect("build_from_2rocks failed");
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocksdb_1_bounds_match_the_value_abi() {
        assert!(validate_rocksdb_1_bounds(1, 0, 0).is_ok());
        assert!(validate_rocksdb_1_bounds(21, 63, 63).is_ok());
        assert!(validate_rocksdb_1_bounds(0, 0, 0).is_err());
        assert!(validate_rocksdb_1_bounds(22, 0, 0).is_err());
        assert!(validate_rocksdb_1_bounds(7, 16, 15).is_err());
        assert!(validate_rocksdb_1_bounds(7, 22, 0).is_err());
        assert!(validate_rocksdb_1_bounds(1, 0, 2).is_err());
    }

    #[test]
    fn rocksdb_2_bounds_match_the_value_abi() {
        assert!(validate_rocksdb_2_bounds(10, 11, 63).is_ok());
        assert!(validate_rocksdb_2_bounds(0, 1, 0).is_err());
        assert!(validate_rocksdb_2_bounds(11, 11, 0).is_err());
        assert!(validate_rocksdb_2_bounds(2, 3, 16).is_err());
    }
}
