use std::sync::Arc;

use local_mixing::db_generation::regular::{
    MAX_REGULAR_GATES, build_from_2rocks, build_from_rocks, build_m1, build_wide_from_rocks,
    open_db_for_read, open_db_for_write, open_wide_db_for_write, rocks_to_lmdb,
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

/// `rocksdb_wide`: extend band m with one 3-control conjunction gate per
/// candidate (S7 of the wide-gate design). Reads ./rocks_db_m{m}, writes
/// ./test_wide_db_m{m} with MPX1 values.
pub fn run_rocksdb_wide(sub: &clap::ArgMatches) {
    let m: usize = *sub.get_one("m").expect("Missing -m <gates>");
    let min_n: usize = *sub.get_one("min_n").expect("min_n has a default");
    let max_n: usize = *sub.get_one("max_n").expect("max_n has a default");
    if !(1..=MAX_REGULAR_GATES).contains(&m) {
        invalid_arguments(format!("m must be in 1..={MAX_REGULAR_GATES}"));
    }
    if max_n != 0 && min_n > max_n {
        invalid_arguments(format!("min_n ({min_n}) cannot exceed max_n ({max_n})"));
    }
    let old_db = Arc::new(open_db_for_read(m));
    let new_db = Arc::new(open_wide_db_for_write(m).unwrap_or_else(|e| {
        eprintln!("open wide output: {e}");
        std::process::exit(2)
    }));
    if let Err(e) = build_wide_from_rocks(&old_db, &new_db, min_n, max_n) {
        eprintln!("wide build failed: {e}");
        std::process::exit(1);
    }
}

/// `rocksdb_wide2`: extend the wide band m with a SECOND 3-control conjunction
/// gate. Reads ./wide_db_m{m} (MPX1), writes ./test_wide2_db_m{m}.
pub fn run_rocksdb_wide2(sub: &clap::ArgMatches) {
    use local_mixing::db_generation::regular::{
        build_wide2_from_wide, open_wide_db_for_read, open_wide2_db_for_write,
    };
    let m: usize = *sub.get_one("m").expect("Missing -m <gates>");
    let min_n: usize = *sub.get_one("min_n").expect("min_n has a default");
    let max_n: usize = *sub.get_one("max_n").expect("max_n has a default");
    if !(1..=MAX_REGULAR_GATES).contains(&m) {
        invalid_arguments(format!("m must be in 1..={MAX_REGULAR_GATES}"));
    }
    if max_n != 0 && min_n > max_n {
        invalid_arguments(format!("min_n ({min_n}) cannot exceed max_n ({max_n})"));
    }
    let old_db = Arc::new(
        open_wide_db_for_read(&format!("wide_db_m{m}")).unwrap_or_else(|e| {
            eprintln!("open wide source: {e}");
            std::process::exit(2)
        }),
    );
    let new_db = Arc::new(open_wide2_db_for_write(m).unwrap_or_else(|e| {
        eprintln!("open wide2 output: {e}");
        std::process::exit(2)
    }));
    if let Err(e) = build_wide2_from_wide(&old_db, &new_db, min_n, max_n) {
        eprintln!("wide2 build failed: {e}");
        std::process::exit(1);
    }
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
#[path = "../tests/db_gen/commands/tests.rs"]
mod tests;

use clap::{Arg, Command};
fn parse_regular_gate_count(raw: &str) -> Result<usize, String> {
    let value = raw
        .parse::<usize>()
        .map_err(|_| format!("expected an integer in 1..=21, got {raw:?}"))?;
    if !(1..=21).contains(&value) {
        return Err(format!(
            "gate count must be in 1..=21 so 3m fits u64 monomials, got {value}"
        ));
    }
    Ok(value)
}

// Grouped database builder arguments; aliases retain the historical names.

pub fn command() -> Command {
    Command::new("db").about("Build and export replacement databases").subcommand_required(true).arg_required_else_help(true)
        .subcommand(
            Command::new("build-regular").visible_alias("rocksdb_1")

                .about("Build an m-gate RocksDB by extending the (m-1)-gate DB")
                .arg(
                    Arg::new("m")
                        .short('m')
                        .long("m")
                        .required(true)
                        .value_parser(parse_regular_gate_count)
                        .help("Number of gates (1..=21; 3m must fit u64 monomials)"),
                )
                .arg(
                    Arg::new("min_n")
                        .long("min_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Minimum used-wire count (0 = no lower bound)"),
                )
                .arg(
                    Arg::new("max_n")
                        .long("max_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Maximum used-wire count (0 = no upper bound)"),
                )
                .arg(
                    Arg::new("no_L")
                        .long("no_L")
                        .action(clap::ArgAction::SetTrue)
                        .help("Skip candidates whose canonicalization requires Rule L (no effect for m=1)"),
                ),
        )
        .subcommand(
            Command::new("combine-regular").visible_alias("rocksdb_2")

                .about("Build an (m1+m2)-gate RocksDB by combining two source DBs")
                .arg(
                    Arg::new("m1")
                        .long("m1")
                        .required(true)
                        .value_parser(parse_regular_gate_count)
                        .help("Gate count of the first source DB"),
                )
                .arg(
                    Arg::new("m2")
                        .long("m2")
                        .required(true)
                        .value_parser(parse_regular_gate_count)
                        .help("Gate count of the second source DB"),
                )
                .arg(
                    Arg::new("min_n")
                        .long("min_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Minimum used-wire count (0 = no lower bound)"),
                ),
        )
        .subcommand(
            Command::new("build-wide").visible_alias("rocksdb_wide")

                .about("Extend band m with one 3-control conjunction gate (wide sidecar store)")
                .arg(
                    Arg::new("m")
                        .short('m')
                        .required(true)
                        .value_parser(parse_regular_gate_count)
                        .help("Source band gate count (reads ./rocks_db_m{m})"),
                )
                .arg(
                    Arg::new("min_n")
                        .long("min_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Minimum used-wire count after the wide gate (0 = no bound)"),
                )
                .arg(
                    Arg::new("max_n")
                        .long("max_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Maximum used-wire count after the wide gate (0 = no bound)"),
                ),
        )
        .subcommand(
            Command::new("build-wide2").visible_alias("rocksdb_wide2")

                .about("Extend wide band m with a SECOND 3-control conjunction gate (wide2 sidecar store)")
                .arg(
                    Arg::new("m")
                        .short('m')
                        .required(true)
                        .value_parser(parse_regular_gate_count)
                        .help("Source wide band gate count (reads ./wide_db_m{m})"),
                )
                .arg(
                    Arg::new("min_n")
                        .long("min_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Minimum used-wire count after the second wide gate (0 = no bound)"),
                )
                .arg(
                    Arg::new("max_n")
                        .long("max_n")
                        .required(false)
                        .default_value("0")
                        .value_parser(clap::value_parser!(usize))
                        .help("Maximum used-wire count after the second wide gate (0 = no bound)"),
                ),
        )
        .subcommand(
            Command::new("export-lmdb").visible_alias("rocks_to_lmdb")

                .about("Convert a combined RocksDB into 256 sharded LMDB databases")
                .arg(
                    Arg::new("source")
                        .short('s')
                        .long("source")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Source RocksDB path"),
                )
                .arg(
                    Arg::new("path")
                        .short('p')
                        .long("path")
                        .required(true)
                        .value_parser(clap::value_parser!(String))
                        .help("Output LMDB directory"),
                ),
        )
}

pub fn run(sub: &clap::ArgMatches) {
    match sub.subcommand() {
        Some(("build-regular", m)) => run_rocksdb_1(m),
        Some(("combine-regular", m)) => run_rocksdb_2(m),
        Some(("build-wide", m)) => run_rocksdb_wide(m),
        Some(("build-wide2", m)) => run_rocksdb_wide2(m),
        Some(("export-lmdb", m)) => run_rocks_to_lmdb(m),
        _ => unreachable!(),
    }
}
