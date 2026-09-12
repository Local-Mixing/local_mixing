use super::*;
use std::collections::BTreeSet;

struct TestDir(PathBuf);

impl TestDir {
    fn new(label: &str) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let serial = NEXT.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "local-mixing-{label}-{}-{serial}",
            std::process::id()
        ));
        assert!(!path.exists());
        std::fs::create_dir(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn exact_composite_to_lmdb_preserves_three_hundred_candidates() {
    let root = TestDir::new("curated-full-e2e");
    let rocks_path = root.0.join("composite");
    let lmdb_path = root.0.join("lmdb");
    let key = [0x5au8; FUNCTION_KEY_BYTES];
    let mut expected = BTreeSet::new();

    {
        let database = DB::open(&composite_options(true), path_text(&rocks_path).unwrap())
            .expect("create composite test store");
        put_format_marker(&database).unwrap();
        for i in 0..300u16 {
            let blob = vec![
                (i >> 8) as u8,
                i as u8,
                1,
                9,
                (i.wrapping_mul(17) >> 8) as u8,
                i.wrapping_mul(17) as u8,
            ];
            expected.insert(blob.clone());
            database
                .put(composite_key(&key, &blob).unwrap(), [])
                .unwrap();
        }
        finalize_composite(&database, "test").unwrap();
    }

    audit(&rocks_path).unwrap();
    to_lmdb(&rocks_path, &lmdb_path, 1).unwrap();
    validate_lmdb(&rocks_path, &lmdb_path).unwrap();

    let (environment, databases) = open_input_lmdb(&lmdb_path).unwrap();
    let transaction = environment.begin_ro_txn().unwrap();
    let value = transaction.get(databases[key[0] as usize], &key).unwrap();
    let actual: BTreeSet<Vec<u8>> = decode_legacy_value(value).unwrap().into_iter().collect();
    assert_eq!(actual, expected);
    assert_eq!(actual.len(), 300);
    assert!(value.len() > 512);
}

#[test]
fn interrupted_composite_store_is_rejected() {
    let root = TestDir::new("curated-full-partial");
    let rocks_path = root.0.join("partial");
    {
        let database = DB::open(&composite_options(true), path_text(&rocks_path).unwrap())
            .expect("create partial test store");
        put_format_marker(&database).unwrap();
        database.flush().unwrap();
    }
    let error = match open_composite_read(&rocks_path) {
        Ok(_) => panic!("partial composite store was accepted"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("interrupted partial build"));
}

#[test]
fn regular_shortcut_enumerates_source_friend_after_twenty() {
    let friends: Vec<u8> = (0..21).collect();
    let mut visited = BTreeSet::new();
    for_each_ordered_source_pair(&friends, |left, _, right, _| {
        visited.insert((left, right));
        Ok(())
    })
    .unwrap();

    assert_eq!(visited.len(), 21 * 20);
    assert!(visited.contains(&(20, 0)));
    assert!(visited.contains(&(0, 20)));
}
