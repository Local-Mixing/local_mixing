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
