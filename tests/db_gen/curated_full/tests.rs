use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[test]
fn malformed_legacy_value_is_not_partially_accepted() {
    let err = decode_legacy_value(&[3, 0, 1]).unwrap_err();
    assert!(matches!(err, CuratedError::MalformedValue { .. }));
}

#[test]
fn checked_blob_rejects_each_legacy_format_overflow() {
    let too_many_gates = CircuitSeq {
        gates: vec![[0, 1, 2]; 86],
    };
    assert!(matches!(
        checked_blob(&too_many_gates),
        Err(CuratedError::CircuitTooLong { gates: 86, .. })
    ));
    let wide_wire = CircuitSeq {
        gates: vec![[256, 1, 2]],
    };
    assert_eq!(
        checked_blob(&wide_wire),
        Err(CuratedError::WireTooLarge { wire: 256 })
    );
}

#[test]
fn composite_records_preserve_more_than_all_historical_caps() {
    let key = [0x5au8; FUNCTION_KEY_BYTES];
    let mut records = BTreeSet::new();
    for i in 0..300u16 {
        let blob = vec![
            (i >> 8) as u8,
            i as u8,
            1,
            9,
            (i.wrapping_mul(17) >> 8) as u8,
            i.wrapping_mul(17) as u8,
        ];
        records.insert(composite_key(&key, &blob).unwrap());
    }
    assert_eq!(records.len(), 300);

    let mut grouped: BTreeMap<[u8; FUNCTION_KEY_BYTES], Vec<u8>> = BTreeMap::new();
    for record in records {
        let (record_key, blob) = split_composite_key(&record).unwrap();
        grouped
            .entry(record_key)
            .or_default()
            .extend(encode_legacy_record(blob).unwrap());
    }
    let value = &grouped[&key];
    assert_eq!(decode_legacy_value(value).unwrap().len(), 300);
    assert!(value.len() > 512);
}

#[test]
fn identity_split_candidates_rehash_to_their_emitted_keys() {
    // A B A B is an identity because A and B act on disjoint wires and
    // therefore commute; neither adjacent pair is equal in this spelling.
    let identity = CircuitSeq {
        gates: vec![[0, 1, 2], [3, 4, 5], [0, 1, 2], [3, 4, 5]],
    };
    let mut records = BTreeSet::new();
    let emitted = derive_identity_candidates(&identity, |key, blob| {
        records.insert(composite_key(&key, &blob)?);
        Ok(())
    })
    .unwrap();
    assert_eq!(emitted, 2 * 2 * 4 * 3);
    assert!(!records.is_empty());
    for record in records {
        let (key, blob) = split_composite_key(&record).unwrap();
        assert_eq!(canonical_key(&CircuitSeq::from_blob(blob)).unwrap(), key);
    }
}
