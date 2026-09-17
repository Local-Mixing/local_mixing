use super::*;

#[test]
fn explicit_cache_preserves_namespaces_eviction_and_disabled_lookup() {
    let db = FrozenDb::empty();
    let cache = LookupCache::new(&db, 2 * (17 + 64));
    let key = [7; 16];
    assert!(cache.get_regular(&key).is_none());
    assert!(cache.get_curated(&key).is_none());
    assert_eq!(cache.map.as_ref().unwrap().len(), 2);
    assert_eq!(cache.bytes.load(Ordering::Relaxed), 2 * (17 + 64));
    // Repeating a cached miss must neither allocate a second entry nor evict.
    assert!(cache.get_regular(&key).is_none());
    assert_eq!(cache.map.as_ref().unwrap().len(), 2);
    assert!(cache.get_regular(&[8; 16]).is_none());
    assert_eq!(cache.map.as_ref().unwrap().len(), 1);
    assert_eq!(cache.bytes.load(Ordering::Relaxed), 17 + 64);

    let disabled = LookupCache::new(&db, 0);
    assert!(disabled.get_regular(&key).is_none());
    assert!(disabled.get_curated(&key).is_none());
    assert!(disabled.map.is_none());
    assert_eq!(disabled.bytes.load(Ordering::Relaxed), 0);
}
