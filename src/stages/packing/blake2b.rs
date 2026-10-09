//! BLAKE2b (RFC 7693), unkeyed, any digest length 1..=64
//! (docs/STAGE4_RUST_DESIGN.md §4, §4.7).
//!
//! The packer keys a value exactly as `window_pack3.py` does:
//! `hashlib.blake2b(int.to_bytes(128, "little"), digest_size=16)`, so value
//! identity, including any (astronomically unlikely) collision, behaves
//! identically in Rust and in the oracle.

const IV: [u64; 8] = [
    0x6a09_e667_f3bc_c908,
    0xbb67_ae85_84ca_a73b,
    0x3c6e_f372_fe94_f82b,
    0xa54f_f53a_5f1d_36f1,
    0x510e_527f_ade6_82d1,
    0x9b05_688c_2b3e_6c1f,
    0x1f83_d9ab_fb41_bd6b,
    0x5be0_cd19_137e_2179,
];

const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

#[inline(always)]
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}

fn compress(h: &mut [u64; 8], block: &[u8; 128], t: u128, last: bool) {
    let mut m = [0u64; 16];
    for (i, w) in m.iter_mut().enumerate() {
        *w = u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap());
    }
    let mut v = [0u64; 16];
    v[..8].copy_from_slice(h);
    v[8..].copy_from_slice(&IV);
    v[12] ^= t as u64;
    v[13] ^= (t >> 64) as u64;
    if last {
        v[14] = !v[14];
    }
    for s in &SIGMA {
        g(&mut v, 0, 4, 8, 12, m[s[0]], m[s[1]]);
        g(&mut v, 1, 5, 9, 13, m[s[2]], m[s[3]]);
        g(&mut v, 2, 6, 10, 14, m[s[4]], m[s[5]]);
        g(&mut v, 3, 7, 11, 15, m[s[6]], m[s[7]]);
        g(&mut v, 0, 5, 10, 15, m[s[8]], m[s[9]]);
        g(&mut v, 1, 6, 11, 12, m[s[10]], m[s[11]]);
        g(&mut v, 2, 7, 8, 13, m[s[12]], m[s[13]]);
        g(&mut v, 3, 4, 9, 14, m[s[14]], m[s[15]]);
    }
    for i in 0..8 {
        h[i] ^= v[i] ^ v[i + 8];
    }
}

/// Incremental unkeyed BLAKE2b with an `nn`-byte digest.
pub struct Blake2b {
    h: [u64; 8],
    buf: [u8; 128],
    buf_len: usize,
    t: u128,
    nn: usize,
}

impl Blake2b {
    pub fn new(nn: usize) -> Self {
        assert!((1..=64).contains(&nn), "blake2b digest length {nn}");
        let mut h = IV;
        h[0] ^= 0x0101_0000 ^ nn as u64;
        Blake2b {
            h,
            buf: [0; 128],
            buf_len: 0,
            t: 0,
            nn,
        }
    }

    pub fn update(&mut self, mut data: &[u8]) {
        while !data.is_empty() {
            // A full buffer is compressed only once more input follows (the
            // last block must be compressed with the final flag).
            if self.buf_len == 128 {
                self.t += 128;
                let block = self.buf;
                compress(&mut self.h, &block, self.t, false);
                self.buf_len = 0;
            }
            let take = (128 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
        }
    }

    pub fn finalize(mut self) -> Vec<u8> {
        self.t += self.buf_len as u128;
        self.buf[self.buf_len..].fill(0);
        let block = self.buf;
        compress(&mut self.h, &block, self.t, true);
        let mut out = Vec::with_capacity(64);
        for w in &self.h {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out.truncate(self.nn);
        out
    }
}

/// `hashlib.blake2b(data, digest_size=nn).digest()`.
pub fn blake2b(data: &[u8], nn: usize) -> Vec<u8> {
    let mut h = Blake2b::new(nn);
    h.update(data);
    h.finalize()
}

/// `blake2b(<16 little-endian u64 lanes>, digest_size=16)`: one 128-byte
/// block, compressed directly (the packer's hot path).
pub fn blake2b16_lanes(lanes: &[u64; 16]) -> [u8; 16] {
    let mut h = IV;
    h[0] ^= 0x0101_0000 ^ 16;
    let mut block = [0u8; 128];
    for (i, l) in lanes.iter().enumerate() {
        block[8 * i..8 * i + 8].copy_from_slice(&l.to_le_bytes());
    }
    compress(&mut h, &block, 128, true);
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&h[0].to_le_bytes());
    out[8..].copy_from_slice(&h[1].to_le_bytes());
    out
}

/// Lowercase hex of a digest.
pub fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
#[path = "../../../tests/unit/stages/packing/blake2b.rs"]
mod tests;
