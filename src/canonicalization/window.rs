//! G57 polynomial composition and canonical lookup preparation.
#[cfg(feature = "legacy-tools")]
use super::legacy_environment::bench_canon_enabled;
use super::options::*;
use super::{cache::*, canonicalize::*, keys::*, polynomial::*};
use crate::circuit::{CircuitSeq, Permutation};
use std::sync::atomic::Ordering;
use std::time::Instant;

impl CircuitSeq {
    pub fn to_polynomial(&self, n: usize, start: usize, end: usize) -> Vec<Polynomial> {
        let gates = &self.gates[start..end];
        // Wire i starts as degree 1 monomial
        let mut polys: Vec<Polynomial> = (0..n).map(|i| vec![1u64 << i]).collect();

        // Ping-pong scratch buffers reused across the gate loop: `term`
        // holds the AND-NOT product, `merged` receives the XOR merge and is
        // then swapped into polys[a], recycling the displaced allocation.
        let mut term: Vec<Monomial> = Vec::new();
        let mut merged: Vec<Monomial> = Vec::new();

        for &[a, b, c] in gates {
            // evaluate() toggles on b OR NOT(c), which is 1 + c*NOT(b) over GF(2).
            poly_and_not_into(&polys[c as usize], &polys[b as usize], &mut term);
            poly_xor_merge_into(&polys[a as usize], &term, &mut merged);
            std::mem::swap(&mut polys[a as usize], &mut merged);
            toggle_monomial(&mut polys[a as usize], 0u64);
        }

        // XOR each wire with its initial value x_i so unchanged wires become 0
        // for i in 0..n {
        //     let xi = vec![1u64 << i];
        //     polys[i] = poly_xor(polys[i].clone(), xi);
        // }

        polys
    }

    /// Like `to_polynomial`, but returns `None` when polynomial growth exceeds
    /// `cap`. The early product guard avoids allocating an intermediate AND
    /// whose raw cross product is already far beyond the useful limit.
    pub fn to_polynomial_capped(
        &self,
        n: usize,
        start: usize,
        end: usize,
        cap: usize,
    ) -> Option<Vec<Polynomial>> {
        let gates = &self.gates[start..end];
        let mut polys: Vec<Polynomial> = (0..n).map(|i| vec![1u64 << i]).collect();

        // Same reused scratch buffers as `to_polynomial`; the budget checks
        // below are unchanged and see identical lengths.
        let mut term: Vec<Monomial> = Vec::new();
        let mut merged: Vec<Monomial> = Vec::new();

        for &[a, b, c] in gates {
            if polys[b as usize]
                .len()
                .saturating_mul(polys[c as usize].len())
                > cap.saturating_mul(16)
            {
                return None;
            }

            // Keep the executor's g57 convention: a += NOT(b)*c + 1.
            poly_and_not_into(&polys[c as usize], &polys[b as usize], &mut term);
            poly_xor_merge_into(&polys[a as usize], &term, &mut merged);
            std::mem::swap(&mut polys[a as usize], &mut merged);
            toggle_monomial(&mut polys[a as usize], 0u64);
            if polys[a as usize].len() > cap {
                return None;
            }
        }

        Some(polys)
    }

    /// Compute canonical polynomials for one direction only (forward or reversed).
    /// Returns (canonical_polys, final_order, used_wires).
    /// Used by frozen compression to try forward first, then reverse on miss.
    pub fn canonicalize_polys_single(
        &self,
        reversed: bool,
    ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
        self.canonicalize_polys_single_using(reversed, None)
    }

    /// Explicit limits, without environment reads or the process-wide legacy cache.
    /// The supplied options are independent between calls and between threads.
    pub fn canonicalize_polys_single_with_options(
        &self,
        reversed: bool,
        options: &G57CanonicalizationOptions,
    ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
        self.canonicalize_polys_single_using(reversed, Some(options))
    }

    fn canonicalize_polys_single_using(
        &self,
        reversed: bool,
        options: Option<&G57CanonicalizationOptions>,
    ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
        let used = self.used_wires();
        match self.canonicalize_polys_single_inner(reversed, &used, options) {
            CanonSingleInner::Skip => (Vec::new(), Permutation { data: Vec::new() }, used),
            CanonSingleInner::Cached(entry) => (
                entry.polys.clone(),
                Permutation {
                    data: entry.order.clone(),
                },
                used,
            ),
            CanonSingleInner::Fresh(polys, order, _) => (polys, Permutation { data: order }, used),
        }
    }

    /// Like `canonicalize_polys_single`, but returns the frozen-DB lookup key
    /// (`xxh3_128(polys_repr_blob(polys)).to_le_bytes()`) instead of the
    /// canonical polynomials, so cache hits skip the deep clone of the
    /// polynomial vector and the re-serialize/re-hash on the caller side.
    /// `None` corresponds exactly to the empty-polys skip outcome of
    /// `canonicalize_polys_single` (oversized window, monomial-cap skip, or
    /// Rule-L budget skip).
    pub fn canonicalize_polys_single_hashed(
        &self,
        reversed: bool,
    ) -> (Option<[u8; 16]>, Permutation, Vec<u16>) {
        self.canonicalize_polys_single_hashed_using(reversed, None)
    }

    /// Explicit limits, without environment reads or the process-wide legacy cache.
    /// The supplied options are independent between calls and between threads.
    pub fn canonicalize_polys_single_hashed_with_options(
        &self,
        reversed: bool,
        options: &G57CanonicalizationOptions,
    ) -> (Option<[u8; 16]>, Permutation, Vec<u16>) {
        self.canonicalize_polys_single_hashed_using(reversed, Some(options))
    }

    fn canonicalize_polys_single_hashed_using(
        &self,
        reversed: bool,
        options: Option<&G57CanonicalizationOptions>,
    ) -> (Option<[u8; 16]>, Permutation, Vec<u16>) {
        let used = self.used_wires();
        match self.canonicalize_polys_single_inner(reversed, &used, options) {
            CanonSingleInner::Skip => (None, Permutation { data: Vec::new() }, used),
            CanonSingleInner::Cached(entry) => (
                Some(entry.polys_key),
                Permutation {
                    data: entry.order.clone(),
                },
                used,
            ),
            CanonSingleInner::Fresh(polys, order, key) => {
                let key = key.unwrap_or_else(|| {
                    xxhash_rust::xxh3::xxh3_128(&polys_repr_blob(&polys)).to_le_bytes()
                });
                (Some(key), Permutation { data: order }, used)
            }
        }
    }

    fn canonicalize_polys_single_inner(
        &self,
        reversed: bool,
        used: &[u16],
        options: Option<&G57CanonicalizationOptions>,
    ) -> CanonSingleInner {
        // A u64 monomial cannot distinguish x_64 from lower variables. Treat
        // oversized lookup windows as clean misses rather than constructing an
        // overflow-aliased key.
        if used.len() > 64 {
            OVERSIZED_CANON_SKIPS.fetch_add(1, Ordering::Relaxed);
            return CanonSingleInner::Skip;
        }
        let wire_map = dense_wire_map(used);
        let mut c = CircuitSeq {
            gates: self
                .gates
                .iter()
                .map(|&[t, c1, c2]| {
                    [
                        wire_map[t as usize],
                        wire_map[c1 as usize],
                        wire_map[c2 as usize],
                    ]
                })
                .collect(),
        };
        if reversed {
            c.gates.reverse();
        }
        c.canonicalize();

        // Canonicalization is pure and windows repeat heavily in the
        // compress/expand games, so an exact process-wide cache keyed on the
        // dense-remapped, gate-canonicalized window is a straight win. The
        // reversed direction needs no key flag: it produces a different gate
        // sequence (and when it doesn't, the results coincide anyway).
        let cache = if options.is_none() {
            canon_cache()
        } else {
            None
        };
        let cache_key: Option<Box<[u16]>> = cache.map(|_| {
            let mut key = Vec::with_capacity(c.gates.len() * 3);
            for g in &c.gates {
                key.extend_from_slice(g);
            }
            key.into_boxed_slice()
        });
        if let (Some(cache), Some(key)) = (cache, cache_key.as_ref()) {
            CANON_CACHE_QUERIES.fetch_add(1, Ordering::Relaxed);
            if let Some(entry) = cache.get(key) {
                CANON_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                return CanonSingleInner::Cached(std::sync::Arc::clone(entry.value()));
            }
        }

        let n = c.max_wire() as usize + 1;
        let polys = match options.map_or_else(canon_monomial_cap, |value| value.monomial_cap) {
            Some(cap) => match c.to_polynomial_capped(n, 0, c.gates.len(), cap) {
                Some(polys) => polys,
                None => {
                    CANON_CAP_SKIPS.fetch_add(1, Ordering::Relaxed);
                    return CanonSingleInner::Skip;
                }
            },
            None => c.to_polynomial(n, 0, c.gates.len()),
        };

        #[cfg(feature = "legacy-tools")]
        let bench_polys = if options.is_none() && bench_canon_enabled() {
            Some(polys.clone())
        } else {
            None
        };

        let t4 = Instant::now();
        let canon = match canonicalize_polys_4_using(
            polys,
            true,
            options.map(|value| &value.canonicalization),
        ) {
            Ok(canon) => canon,
            Err(()) => {
                CANON_RULE_L_SKIPS.fetch_add(1, Ordering::Relaxed);
                return CanonSingleInner::Skip;
            }
        };
        let canon_elapsed = t4.elapsed();
        CANON4_CORE_TIME.fetch_add(canon_elapsed.as_nanos() as u64, Ordering::Relaxed);
        if trace_enabled(options.map(|value| &value.canonicalization))
            && canon_elapsed.as_millis()
                >= trace_threshold_ms(options.map(|value| &value.canonicalization))
        {
            eprintln!(
                "[compress-trace] slow canonicalize direction={} gates={} used_wires={} elapsed_ms={}",
                if reversed { "reverse" } else { "forward" },
                self.gates.len(),
                used.len(),
                canon_elapsed.as_millis()
            );
        }

        #[cfg(feature = "legacy-tools")]
        if let Some(polys) = bench_polys {
            let tp = Instant::now();
            let perm = crate::experimental::poly_canon_graph::canonicalize_graph(&polys, n);
            let _form = crate::experimental::poly_canon_graph::canonical_form(&polys, &perm);
            POLYCANON_CORE_TIME.fetch_add(tp.elapsed().as_nanos() as u64, Ordering::Relaxed);
            CANON_BENCH_CALLS.fetch_add(1, Ordering::Relaxed);
        }

        // All cap exits return above, so the exact cache can contain only
        // complete, valid canonical keys—never a clean-miss sentinel.
        if let (Some(cache), Some(key)) = (cache, cache_key) {
            let polys_key = xxhash_rust::xxh3::xxh3_128(&polys_repr_blob(&canon.0)).to_le_bytes();
            let entry_bytes = (96
                + key.len() * 2
                + canon.0.iter().map(|p| 24 + p.len() * 8).sum::<usize>()
                + canon.1.data.len() * 8) as u64;
            if CANON_CACHE_BYTES.fetch_add(entry_bytes, Ordering::Relaxed) + entry_bytes
                > canon_cache_cap_bytes()
            {
                // Wholesale epoch reset, same policy as the frozen lookup cache.
                cache.clear();
                CANON_CACHE_BYTES.store(entry_bytes, Ordering::Relaxed);
            }
            cache.insert(
                key,
                std::sync::Arc::new(CanonCacheEntry {
                    polys: canon.0.clone(),
                    order: canon.1.data.clone(),
                    polys_key,
                }),
            );
            return CanonSingleInner::Fresh(canon.0, canon.1.data, Some(polys_key));
        }

        CanonSingleInner::Fresh(canon.0, canon.1.data, None)
    }

    /// Like `canonicalize_polys_single`, but absorbs pending NOTs on input wires by substituting
    /// x_w -> x_w + 1 in the polynomial form before canonicalization. This supports Stage-F-style
    /// curated lookups where the replacement consumes the pending NOT instead of materializing a
    /// standalone NOT gadget.
    pub fn canonicalize_polys_single_neg(
        &self,
        negated_inputs: &[u16],
    ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
        self.canonicalize_polys_single_neg_using(negated_inputs, None)
    }

    /// Input-negation canonicalization with explicit limits and no legacy cache.
    pub fn canonicalize_polys_single_neg_with_options(
        &self,
        negated_inputs: &[u16],
        options: &G57CanonicalizationOptions,
    ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
        self.canonicalize_polys_single_neg_using(negated_inputs, Some(options))
    }

    fn canonicalize_polys_single_neg_using(
        &self,
        negated_inputs: &[u16],
        options: Option<&G57CanonicalizationOptions>,
    ) -> (Vec<Polynomial>, Permutation, Vec<u16>) {
        let used = self.used_wires();
        if used.len() > 64 {
            OVERSIZED_CANON_SKIPS.fetch_add(1, Ordering::Relaxed);
            return (Vec::new(), Permutation { data: Vec::new() }, used);
        }
        let wire_map = dense_wire_map(&used);
        let mut c = CircuitSeq {
            gates: self
                .gates
                .iter()
                .map(|&[t, c1, c2]| {
                    [
                        wire_map[t as usize],
                        wire_map[c1 as usize],
                        wire_map[c2 as usize],
                    ]
                })
                .collect(),
        };
        c.canonicalize();

        // Sorted mapped negation multiset. Substitutions of distinct input
        // variables commute and a repeated variable is an involution applied
        // twice, so the sorted list fully determines the outcome and doubles
        // as the cache-key component.
        let mut mapped_negs: Vec<u16> = negated_inputs
            .iter()
            .filter_map(|&w| match wire_map.get(w as usize) {
                Some(&m) if m != u16::MAX => Some(m),
                _ => None,
            })
            .collect();
        mapped_negs.sort_unstable();

        // Same exact result-cache treatment as the plain path. The leading
        // u16::MAX tag namespaces neg entries away from plain window keys
        // (whose first element is a dense wire index < 64, given the
        // used.len() <= 64 guard above), and the negation-count element keeps
        // the encoding prefix-unambiguous.
        let cache = if options.is_none() {
            canon_cache()
        } else {
            None
        };
        let cache_key: Option<Box<[u16]>> = cache.map(|_| {
            let mut key = Vec::with_capacity(2 + mapped_negs.len() + c.gates.len() * 3);
            key.push(u16::MAX);
            key.push(mapped_negs.len() as u16);
            key.extend_from_slice(&mapped_negs);
            for g in &c.gates {
                key.extend_from_slice(g);
            }
            key.into_boxed_slice()
        });
        if let (Some(cache), Some(key)) = (cache, cache_key.as_ref()) {
            CANON_CACHE_QUERIES.fetch_add(1, Ordering::Relaxed);
            if let Some(entry) = cache.get(key) {
                CANON_CACHE_HITS.fetch_add(1, Ordering::Relaxed);
                return (
                    entry.polys.clone(),
                    Permutation {
                        data: entry.order.clone(),
                    },
                    used,
                );
            }
        }

        let n = c.max_wire() as usize + 1;
        let mut polys = match options.map_or_else(canon_monomial_cap, |value| value.monomial_cap) {
            Some(cap) => match c.to_polynomial_capped(n, 0, c.gates.len(), cap) {
                Some(polys) => polys,
                None => {
                    CANON_CAP_SKIPS.fetch_add(1, Ordering::Relaxed);
                    return (Vec::new(), Permutation { data: Vec::new() }, used);
                }
            },
            None => c.to_polynomial(n, 0, c.gates.len()),
        };
        for &mapped in &mapped_negs {
            let mapped = mapped as usize;
            if mapped >= 64 {
                return (Vec::new(), Permutation { data: Vec::new() }, used);
            }
            for p in polys.iter_mut() {
                substitute_input_negation(p, mapped);
            }
        }

        let canon = match canonicalize_polys_4_using(
            polys,
            true,
            options.map(|value| &value.canonicalization),
        ) {
            Ok(canon) => canon,
            Err(()) => {
                CANON_RULE_L_SKIPS.fetch_add(1, Ordering::Relaxed);
                return (Vec::new(), Permutation { data: Vec::new() }, used);
            }
        };

        // Only complete successes are inserted, mirroring the plain path: a
        // hit therefore always replays a deterministic success.
        if let (Some(cache), Some(key)) = (cache, cache_key) {
            let polys_key = xxhash_rust::xxh3::xxh3_128(&polys_repr_blob(&canon.0)).to_le_bytes();
            let entry_bytes = (96
                + key.len() * 2
                + canon.0.iter().map(|p| 24 + p.len() * 8).sum::<usize>()
                + canon.1.data.len() * 8) as u64;
            if CANON_CACHE_BYTES.fetch_add(entry_bytes, Ordering::Relaxed) + entry_bytes
                > canon_cache_cap_bytes()
            {
                cache.clear();
                CANON_CACHE_BYTES.store(entry_bytes, Ordering::Relaxed);
            }
            cache.insert(
                key,
                std::sync::Arc::new(CanonCacheEntry {
                    polys: canon.0.clone(),
                    order: canon.1.data.clone(),
                    polys_key,
                }),
            );
        }
        (canon.0, canon.1, used)
    }
}
/// Dense old-wire -> compact-index map for a sorted `used_wires` list.
/// Entries for unused wires are `u16::MAX`; used wires map to their position.
/// Replaces per-call `HashMap<u16, u16>` construction on canonicalization hot paths.
pub(super) fn dense_wire_map(used: &[u16]) -> Vec<u16> {
    let len = used.last().map_or(0, |&w| w as usize + 1);
    let mut map = vec![u16::MAX; len];
    for (i, &w) in used.iter().enumerate() {
        map[w as usize] = i as u16;
    }
    map
}
