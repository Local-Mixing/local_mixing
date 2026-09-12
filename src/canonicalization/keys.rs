//! Byte representation used by persisted function keys.
use super::Polynomial;
// Display polynomials

pub fn polys_repr_blob(polys: &Vec<Polynomial>) -> Vec<u8> {
    let total: usize = polys.iter().map(|p| p.len() + 1).sum();
    let mut bytes = Vec::with_capacity(total * 8);
    let mut scratch: Vec<u64> = Vec::new();
    for poly in polys {
        // Canonical polys are already monomial-sorted; only re-sort when a
        // caller hands us an unsorted polynomial.
        if poly.is_sorted() {
            for m in poly {
                bytes.extend_from_slice(&m.to_le_bytes());
            }
        } else {
            scratch.clear();
            scratch.extend_from_slice(poly);
            scratch.sort_unstable();
            for m in &scratch {
                bytes.extend_from_slice(&m.to_le_bytes());
            }
        }
        bytes.extend_from_slice(&u64::MAX.to_le_bytes()); // separator
    }
    bytes
}
