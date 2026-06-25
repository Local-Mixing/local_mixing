use std::{marker::PhantomData, ptr, slice};

use libc::c_uint;

use lmdb::{Cursor, RoCursor};

extern crate lmdb_sys;
use lmdb_sys as ffi;

// Old iterator method for cursor fails if the given key is not found
// This does not unwrap a None value in that case
pub struct Iter<'txn> {
    cursor: *mut ffi::MDB_cursor,
    op: c_uint,
    next_op: c_uint,
    finished: bool,
    _marker: PhantomData<&'txn ()>,
}

impl<'txn> Iter<'txn> {
    pub fn new(cursor: *mut ffi::MDB_cursor, op: c_uint, next_op: c_uint) -> Self {
        Self {
            cursor,
            op,
            next_op,
            finished: false,
            _marker: PhantomData,
        }
    }
}

impl<'txn> Iterator for Iter<'txn> {
    type Item = (&'txn [u8], &'txn [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }

        unsafe {
            let mut key = ffi::MDB_val {
                mv_size: 0,
                mv_data: ptr::null_mut(),
            };
            let mut data = ffi::MDB_val {
                mv_size: 0,
                mv_data: ptr::null_mut(),
            };

            let rc = ffi::mdb_cursor_get(self.cursor, &mut key, &mut data, self.op);
            self.op = self.next_op;

            if rc == ffi::MDB_NOTFOUND {
                self.finished = true;
                return None;
            } else if rc != ffi::MDB_SUCCESS {
                panic!("LMDB error: {}", rc);
            }

            let key_slice = slice::from_raw_parts(key.mv_data as *const u8, key.mv_size);
            let data_slice = slice::from_raw_parts(data.mv_data as *const u8, data.mv_size);
            Some((key_slice, data_slice))
        }
    }
}

pub trait RoCursorExt<'txn> {
    fn iter_from_safe<K>(&mut self, key: K) -> Iter<'txn>
    where
        K: AsRef<[u8]>;
}

impl<'txn> RoCursorExt<'txn> for RoCursor<'txn> {
    fn iter_from_safe<K>(&mut self, key: K) -> Iter<'txn>
    where
        K: AsRef<[u8]>,
    {
        let rc = unsafe {
            let mut key_val = lmdb_sys::MDB_val {
                mv_size: key.as_ref().len(),
                mv_data: key.as_ref().as_ptr() as *mut _,
            };
            lmdb_sys::mdb_cursor_get(
                self.cursor(),
                &mut key_val,
                std::ptr::null_mut(),
                lmdb_sys::MDB_SET_RANGE,
            )
        };

        if rc == lmdb_sys::MDB_NOTFOUND {
            Iter {
                cursor: self.cursor(),
                op: lmdb_sys::MDB_GET_CURRENT,
                next_op: lmdb_sys::MDB_NEXT,
                finished: true,
                _marker: std::marker::PhantomData,
            }
        } else if rc != lmdb_sys::MDB_SUCCESS {
            panic!("LMDB error: {}", rc);
        } else {
            Iter::new(self.cursor(), lmdb_sys::MDB_GET_CURRENT, lmdb_sys::MDB_NEXT)
        }
    }
}

// Timing variables for benchmarking
// static DB_NAME_TIME: AtomicU64 = AtomicU64::new(0);
// static DB_LOOKUP_TIME: AtomicU64 = AtomicU64::new(0);
// static TXN_BEGIN_TIME: AtomicU64 = AtomicU64::new(0);
// static SERIALIZE_KEY_TIME: AtomicU64 = AtomicU64::new(0);
// static LMDB_GET_TIME: AtomicU64 = AtomicU64::new(0);
// static DESERIALIZE_LIST_TIME: AtomicU64 = AtomicU64::new(0);
// static RNG_CHOOSE_TIME: AtomicU64 = AtomicU64::new(0);

// Creates an identity with the first part limited to 16..=28 wires (exclude wires 29, 30, 31), the middle part spanning all 0..=31, and the last part spanning 0..=12 wires (exclude wires 13, 14, 15)
// returns the identity, the number of transpositions of the first part, and the number of transpositions of the second part
