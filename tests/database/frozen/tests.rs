use super::*;

fn qc_test_store(tables: Tables) -> Frozen {
    Frozen {
        shards: Vec::new(),
        tables,
        filters: None,
        big_cache: std::sync::Mutex::new(std::collections::HashMap::new()),
        swap_ctrls: false,
    }
}

fn qc_raw_tables() -> Tables {
    Tables {
        header: rebuild_canonical(vec![(ESC, 1)]),
        gates: Vec::new(),
    }
}

#[test]
fn qc_cached_lookup_copies_only_requested_records_and_preserves_sampler_cache() {
    let store = qc_test_store(qc_raw_tables());
    let key = [0; 16];
    let value = [3, 10, 1, 2].repeat(1000);
    store
        .big_cache
        .lock()
        .unwrap()
        .insert(key, std::sync::Arc::new(value.clone()));
    let db = FrozenDb {
        regular: Some(store),
        curated: None,
    };
    assert_eq!(db.get_regular_qc(&key, 2, 8), Ok(Some(value[..8].to_vec())));
    assert_eq!(
        db.get_regular_qc(&key, 2, 7),
        Err(QcLookupLimit::DecodeBytes { limit: 7 })
    );
    assert_eq!(db.get_regular(&key), Some(value));
    assert_eq!(db.get_curated_qc(&key, 2, 8), Ok(None));
}

#[test]
fn qc_bucket_limit_is_checked_before_reading_or_allocating_bucket() {
    let path = std::env::temp_dir().join(format!("frozen_qc_bucket_{}", std::process::id()));
    let file = std::fs::File::create(&path).unwrap();
    drop(file);
    let file = std::fs::File::open(&path).unwrap();
    let mut store = qc_test_store(qc_raw_tables());
    // Only bucket zero's two offsets are needed; no 256-shard fixture.
    let mut offsets = vec![0; 13];
    offsets[5..10].copy_from_slice(&4096u64.to_le_bytes()[..5]);
    store.shards.push(FrozenShard {
        file,
        offs_raw: offsets,
        data_base: 0,
    });
    assert_eq!(
        store.get_qc(&[0; 16], 2, 128),
        Err(QcLookupLimit::BucketBytes {
            bytes: 4096,
            limit: 128
        })
    );
    store.shards[0].offs_raw[5..10].copy_from_slice(&127u64.to_le_bytes()[..5]);
    assert_eq!(
        store.get_qc(&[0; 16], 2, 128),
        Err(QcLookupLimit::ReadFailure)
    );
    drop(store);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn qc_decode_bounds_raw_prefix_and_predecessor_work() {
    let tables = qc_raw_tables();
    let raw = [121, 8, 0, 3, 10, 1, 2, 3, 20, 3, 4];
    let encoded = [raw.as_slice(), raw.as_slice()].concat();
    let mut prefix = QcReader::new(&encoded, 4);
    assert_eq!(
        decode_value_qc(&tables, &mut prefix, 1, true),
        Ok(vec![3, 10, 1, 2])
    );
    let mut reader = QcReader::new(&encoded, 8);
    assert_eq!(
        decode_value_qc(&tables, &mut reader, usize::MAX, false),
        Ok(vec![])
    );
    assert_eq!(
        decode_value_qc(&tables, &mut reader, 1, true),
        Err(QcLookupLimit::DecodeBytes { limit: 8 })
    );
    assert_eq!(
        decode_value_qc(&tables, &mut QcReader::new(&raw[..6], 100), 2, true),
        Err(QcLookupLimit::InvalidData("truncated bit stream"))
    );
}

#[test]
fn qc_decode_terminates_zero_bit_header_chains_at_work_budget() {
    let tables = Tables {
        header: rebuild_canonical(vec![(1, 1)]),
        gates: Vec::new(),
    };
    assert_eq!(
        decode_value_qc(&tables, &mut QcReader::new(&[], 8), usize::MAX, false),
        Err(QcLookupLimit::DecodeBytes { limit: 8 })
    );
}

#[test]
fn split_key_recovers_frozen_address_fields() {
    let shard = 0xabu64;
    let bucket = 0x54321u64;
    let tail = 0x1234_5678_9abcu64;
    let hi = (shard << 56) | (bucket << 36) | (tail >> 12);
    let lo = (tail & 0xfff) << 52;
    let mut key = [0u8; 16];
    key[..8].copy_from_slice(&hi.to_be_bytes());
    key[8..].copy_from_slice(&lo.to_be_bytes());

    assert_eq!(split_key(&key), (shard as usize, bucket as u32, tail));
}

#[test]
fn runtime_handle_is_thread_shareable() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FrozenDb>();
}

// legacy-swapped-controls: bytes 1 and 2 of every 3-byte gate swap, per
// circuit chunk of the [len][len bytes]* value chain, targets untouched.
#[test]
fn swap_value_controls_swaps_per_gate_within_chunks() {
    let mut v = vec![
        6, /* two gates */ 10, 1, 2, 20, 3, 4, //
        3, /* one gate */ 30, 5, 6,
    ];
    swap_value_controls(&mut v);
    assert_eq!(v, vec![6, 10, 2, 1, 20, 4, 3, 3, 30, 6, 5]);
}

// Reference implementation of the pre-LUT canonical walk; the LUT fast
// path must decode identical symbols and consume identical bit counts.
fn reference_decode(t: &HuffTable, r: &mut BitReader) -> u32 {
    if t.single {
        return t.syms[0];
    }
    let mut code = 0u64;
    for len in 1..=MAXLEN {
        code = (code << 1) | r.get1();
        if t.count[len] > 0 {
            let fc = t.first_code[len];
            if code >= fc && code < fc + t.count[len] as u64 {
                return t.syms[t.first_idx[len] + (code - fc) as usize];
            }
        }
    }
    panic!("corrupt reference stream");
}

struct BitWriter {
    bytes: Vec<u8>,
    acc: u64,
    nbits: u32,
}
impl BitWriter {
    fn new() -> Self {
        BitWriter {
            bytes: Vec::new(),
            acc: 0,
            nbits: 0,
        }
    }
    fn push_bit(&mut self, b: u64) {
        self.acc |= (b & 1) << self.nbits;
        self.nbits += 1;
        if self.nbits == 8 {
            self.bytes.push(self.acc as u8);
            self.acc = 0;
            self.nbits = 0;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.bytes.push(self.acc as u8);
        }
        self.bytes
    }
}

#[test]
fn qc_huffman_value_matches_legacy_decode_for_chained_escape_forms() {
    fn put(writer: &mut BitWriter, value: u64, bits: usize) {
        for bit in 0..bits {
            writer.push_bit((value >> bit) & 1);
        }
    }
    let header_last = (1 << 7) | (3 << 1);
    let tables = Tables {
        header: rebuild_canonical(vec![(header_last, 1), (header_last | 1, 1)]),
        gates: (0..396)
            .map(|_| rebuild_canonical(vec![(ESC, 1)]))
            .collect(),
    };
    let mut writer = BitWriter::new();
    put(&mut writer, 1, 1); // chained header
    put(&mut writer, 0, 1); // packed 15-bit gate escape
    put(&mut writer, (5 << 10) | (7 << 5) | 11, 15);
    put(&mut writer, 0, 1); // last header
    put(&mut writer, 1, 1); // raw 24-bit gate escape
    put(&mut writer, 200, 8);
    put(&mut writer, 201, 8);
    put(&mut writer, 202, 8);
    let encoded = writer.finish();
    let mut legacy = Vec::new();
    decode_value(
        &tables,
        &mut BitReader::new(&encoded),
        &mut legacy,
        usize::MAX,
    );
    assert_eq!(legacy, [3, 5, 7, 11, 3, 200, 201, 202]);
    assert_eq!(
        decode_value_qc(&tables, &mut QcReader::new(&encoded, 8), 3, true),
        Ok(legacy.clone())
    );
    assert_eq!(
        decode_value_qc(&tables, &mut QcReader::new(&encoded, 4), 1, true),
        Ok(legacy[..4].to_vec())
    );
}

#[test]
fn opt_equiv_lut_decode_matches_reference_walk() {
    // Complete canonical codes, including lengths beyond LUT_BITS so the
    // fallback path is exercised.
    let mut deep: Vec<(u32, u32)> = (0..12u32).map(|i| (100 + i, i + 1)).collect();
    deep.push((900, 13));
    deep.push((901, 13));
    let alphabets: Vec<Vec<(u32, u32)>> = vec![
        vec![(7, 1), (8, 2), (9, 3), (10, 3)],
        vec![(1, 2), (2, 2), (3, 2), (4, 2)],
        deep,
    ];
    for sym_lens in alphabets {
        let table = rebuild_canonical(sym_lens.clone());
        // Recompute (code, len) per symbol exactly as rebuild_canonical does.
        let mut sorted = sym_lens.clone();
        sorted.sort_by_key(|&(s, l)| (l, s));
        let mut codes = Vec::new();
        let mut code = 0u64;
        let mut prev_len = sorted[0].1;
        for &(s, l) in &sorted {
            if l > prev_len {
                code <<= l - prev_len;
                prev_len = l;
            }
            codes.push((s, code, l));
            code += 1;
        }
        // Deterministic pseudo-random symbol sequence.
        let mut state = 0x1234_5678_9abc_def0u64;
        let mut seq = Vec::new();
        for _ in 0..500 {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            seq.push(codes[(state >> 33) as usize % codes.len()]);
        }
        let mut w = BitWriter::new();
        for &(_, c, l) in &seq {
            for i in (0..l).rev() {
                w.push_bit((c >> i) & 1);
            }
        }
        let stream = w.finish();
        let mut fast = BitReader::new(&stream);
        let mut refr = BitReader::new(&stream);
        for &(s, _, _) in &seq {
            assert_eq!(table.decode(&mut fast), s);
            assert_eq!(reference_decode(&table, &mut refr), s);
            // peek refills eagerly, so raw (pos, nbits) may differ while
            // the logical bit offset must not.
            assert_eq!(
                fast.pos * 8 - fast.nbits as usize,
                refr.pos * 8 - refr.nbits as usize,
                "bit positions diverged"
            );
        }
    }
}

#[test]
fn bounded_raw_decode_stops_on_a_circuit_boundary() {
    let tables = Tables {
        header: rebuild_canonical(vec![(ESC, 1)]),
        gates: Vec::new(),
    };
    // Raw marker, eight bytes, then two one-gate circuit chunks.
    let encoded = [121, 8, 0, 3, 10, 1, 2, 3, 20, 3, 4];
    let mut reader = BitReader::new(&encoded);
    let mut output = Vec::new();
    decode_value(&tables, &mut reader, &mut output, 1);
    assert_eq!(output, [3, 10, 1, 2]);
}

#[test]
fn unbounded_raw_decode_returns_more_than_the_old_curated_cap() {
    let tables = Tables {
        header: rebuild_canonical(vec![(ESC, 1)]),
        gates: Vec::new(),
    };
    let mut value = Vec::new();
    for i in 0..300u16 {
        value.extend_from_slice(&[3, (i & 0xff) as u8, 1, 2]);
    }
    let mut encoded = Vec::with_capacity(value.len() + 3);
    encoded.push(121);
    encoded.extend_from_slice(&(value.len() as u16).to_le_bytes());
    encoded.extend_from_slice(&value);

    let mut reader = BitReader::new(&encoded);
    let mut output = Vec::new();
    decode_value(&tables, &mut reader, &mut output, usize::MAX);
    assert_eq!(output, value);
    assert_eq!(output.chunks_exact(4).len(), 300);
}

#[test]
fn explicit_lookup_caches_keep_distinct_store_values_separate() {
    use crate::database::lookup_cache::LookupCache;
    let key = [3; 16];
    let build = |value: Vec<u8>| {
        let store = qc_test_store(qc_raw_tables());
        store
            .big_cache
            .lock()
            .unwrap()
            .insert(key, std::sync::Arc::new(value));
        FrozenDb {
            regular: Some(store),
            curated: None,
        }
    };
    let first = build(vec![3, 0, 1, 2]);
    let second = build(vec![3, 0, 2, 1]);
    let first_cache = LookupCache::new(&first, 4096);
    let second_cache = LookupCache::new(&second, 4096);
    for _ in 0..3 {
        assert_eq!(
            first_cache.get_regular(&key).as_deref(),
            Some([3, 0, 1, 2].as_slice())
        );
        assert_eq!(
            second_cache.get_regular(&key).as_deref(),
            Some([3, 0, 2, 1].as_slice())
        );
    }
}
