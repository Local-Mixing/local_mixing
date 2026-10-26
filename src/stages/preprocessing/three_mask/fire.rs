//! Literal-factored masked products. Helpers are arbitrary dirty values.
//!
//! Ported verbatim from the `stage2_compute_pilot` generator (`src/fire.rs`).

use crate::circuit::XGate;

pub type Literal = (u16, bool);
pub type Mask = [u16; 3];

/// A reversible multiple-control NOT using arbitrary borrowed helpers.
/// Every helper is restored; no clean-wire assumption is made.
pub fn borrowed_mcx(t: u16, l: &[Literal], helpers: &[u16]) -> Vec<XGate> {
    assert!(l.iter().all(|x| x.0 != t && !helpers.contains(&x.0)));
    assert!(!helpers.contains(&t));
    if l.len() <= 2 {
        return vec![XGate::conj(t, l.iter().copied()).unwrap()];
    }
    if l.len() == 4 {
        let h = helpers[0];
        let k = helpers[1];
        let a = XGate::conj(h, l[..2].iter().copied()).unwrap();
        let b = XGate::conj(k, l[2..].iter().copied()).unwrap();
        let c = XGate::conj(t, [(h, true), (k, true)]).unwrap();
        return vec![
            c.clone(),
            a.clone(),
            c.clone(),
            b.clone(),
            c.clone(),
            a,
            c,
            b,
        ];
    }
    let h = helpers[0];
    let split = l.len().div_ceil(2);
    let a = borrowed_mcx(h, &l[..split], &helpers[1..]);
    let mut right = vec![(h, true)];
    right.extend_from_slice(&l[split..]);
    let b = borrowed_mcx(t, &right, &helpers[1..]);
    [a.clone(), b.clone(), a, b].concat()
}

/// Dirty v-chain: 4*k-8 Toffolis for k>=3, using k-2 borrowed wires.
/// V toggles each helper by a control prefix; B V B V cancels its unknown
/// initial value from the target and restores every borrowed wire.
pub fn borrowed_mcx_chain(t: u16, l: &[Literal], helpers: &[u16]) -> Vec<XGate> {
    assert!(l.iter().all(|x| x.0 != t && !helpers.contains(&x.0)));
    assert!(!helpers.contains(&t));
    if l.len() <= 2 {
        return vec![XGate::conj(t, l.iter().copied()).unwrap()];
    }
    let k = l.len();
    assert!(helpers.len() >= k - 2);
    let mut v = vec![];
    for j in (1..k - 2).rev() {
        v.push(XGate::conj(helpers[j], [(helpers[j - 1], true), l[j + 1]]).unwrap());
    }
    v.push(XGate::conj(helpers[0], l[..2].iter().copied()).unwrap());
    for j in 1..k - 2 {
        v.push(XGate::conj(helpers[j], [(helpers[j - 1], true), l[j + 1]]).unwrap());
    }
    let b = XGate::conj(t, [(helpers[k - 3], true), l[k - 1]]).unwrap();
    [vec![b.clone()], v.clone(), vec![b], v].concat()
}

/// B V B applies the same target update as B V B V, leaving arbitrary
/// garbage only in unreferenced growing-bank helpers. V's last helper changes
/// by the needed control prefix, independently of its initial dirty contents.
pub fn borrowed_mcx_chain_dirty(t: u16, l: &[Literal], helpers: &[u16]) -> Vec<XGate> {
    let mut out = borrowed_mcx_chain(t, l, helpers);
    if l.len() >= 3 {
        out.truncate(2 * l.len() - 3);
    }
    out
}

pub fn cubic_mask_value(m: Mask, s: &[u64]) -> u64 {
    s[m[0] as usize] & s[m[1] as usize] & s[m[2] as usize]
}

pub fn emit_cubic_mask(t: u16, m: Mask, helpers: &[u16]) -> Vec<XGate> {
    borrowed_mcx(t, &m.map(|w| (w, true)), helpers)
}

pub fn compile_cubic(
    g: &XGate,
    masks: &[Vec<Mask>],
    helpers: &[u16],
    stats: &mut FireStats,
) -> Vec<Vec<XGate>> {
    compile_cubic_with(g, masks, helpers, stats, false)
}

pub fn compile_cubic_with(
    g: &XGate,
    masks: &[Vec<Mask>],
    helpers: &[u16],
    stats: &mut FireStats,
    chain: bool,
) -> Vec<Vec<XGate>> {
    compile_cubic_options(g, masks, helpers, stats, chain, false)
}

/// Keep a negative logical-control constant inside its data literal, rather
/// than expanding it into standalone mask terms.
pub fn compile_cubic_options(
    g: &XGate,
    masks: &[Vec<Mask>],
    helpers: &[u16],
    stats: &mut FireStats,
    chain: bool,
    fold_polarity: bool,
) -> Vec<Vec<XGate>> {
    compile_cubic_options_dirty(g, masks, helpers, stats, chain, fold_polarity, false)
}

pub fn compile_cubic_options_dirty(
    g: &XGate,
    masks: &[Vec<Mask>],
    helpers: &[u16],
    stats: &mut FireStats,
    chain: bool,
    fold_polarity: bool,
    discard_helpers: bool,
) -> Vec<Vec<XGate>> {
    let mut products = vec![vec![]];
    for &(w, pos) in &g.ctrls {
        let mut op = vec![vec![(w, if fold_polarity { pos } else { true })]];
        for &m in &masks[w as usize] {
            op.push(m.map(|x| (x, true)).to_vec());
        }
        if !pos && !fold_polarity {
            op.push(vec![]);
        }
        products = products
            .into_iter()
            .flat_map(|a| {
                op.iter().map(move |b| {
                    let mut x = a.clone();
                    x.extend(b);
                    x
                })
            })
            .collect();
    }
    if g.comp {
        products.push(vec![]);
    }
    let labels: Vec<_> = masks
        .iter()
        .flatten()
        .map(|m| {
            let mut k = *m;
            k.sort();
            k
        })
        .collect();
    let mut units = vec![];
    for (pi, p) in products.into_iter().enumerate() {
        stats.products += 1;
        let helpers: &[u16] = if helpers.len() > 4 {
            let off = (pi * 4) % (helpers.len() - 3);
            &helpers[off..off + 4]
        } else {
            helpers
        };
        let Some(norm) = XGate::conj(g.target, p) else {
            stats.contradictory_products += 1;
            continue;
        };
        let mut l = norm.ctrls.to_vec();
        // Do not make a dirty helper hold exactly a live cubic mask when
        // splitting a five/six-control term. All selected values remain dirty.
        if l.len() >= 5 {
            let mut chosen = None;
            'pick: for a in 0..l.len() {
                for b in a + 1..l.len() {
                    for c in b + 1..l.len() {
                        let mut key = [l[a].0, l[b].0, l[c].0];
                        key.sort();
                        if !labels.contains(&key) {
                            chosen = Some([a, b, c]);
                            break 'pick;
                        }
                    }
                }
            }
            let pick = chosen.expect("a mixed triple exists");
            let mut reordered = pick.map(|i| l[i]).to_vec();
            reordered.extend(
                l.iter()
                    .enumerate()
                    .filter(|(i, _)| !pick.contains(i))
                    .map(|(_, x)| *x),
            );
            l = reordered;
        }
        units.push(if discard_helpers {
            borrowed_mcx_chain_dirty(g.target, &l, helpers)
        } else if chain {
            borrowed_mcx_chain(g.target, &l, helpers)
        } else {
            borrowed_mcx(g.target, &l, helpers)
        });
    }
    units
}

/// XOR old_product XOR new_product into t without materializing either cubic.
/// A telescoping coordinate replacement computes (a XOR d)*b*c, then
/// d*(b XOR e)*c, then d*e*(c XOR f). The temporary band-coordinate XORs
/// are restored. An unreferenced growing-bank helper may keep its garbage.
pub fn emit_cubic_delta(
    t: u16,
    old: Mask,
    new: Mask,
    helpers: &[u16],
    keep_garbage: bool,
) -> Vec<XGate> {
    assert!(old.iter().all(|w| !new.contains(w)));
    let h = helpers[0];
    assert!(!old.contains(&h) && !new.contains(&h) && h != t);
    let mut cur = old;
    let mut out = vec![];
    for i in 0..3 {
        let others: Vec<_> = cur
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, w)| *w)
            .collect();
        let mix = XGate::cnot(cur[i], new[i]);
        let a = XGate::conj(h, [(others[0], true), (others[1], true)]).unwrap();
        let b = XGate::conj(t, [(h, true), (cur[i], true)]).unwrap();
        out.push(mix.clone());
        out.extend([b.clone(), a.clone(), b]);
        if !keep_garbage {
            out.push(a);
        }
        out.push(mix);
        cur[i] = new[i];
    }
    out
}

pub fn emit_mask(t: u16, m: Mask) -> Vec<XGate> {
    vec![
        XGate::x_gate(t),
        XGate::conj(t, [(m[0], false), (m[1], true)]).unwrap(),
        XGate::cnot(t, m[2]),
    ]
}

pub fn mask_value(m: Mask, s: &[u64]) -> u64 {
    !((!s[m[0] as usize]) & s[m[1] as usize]) ^ s[m[2] as usize]
}

pub fn emit_mask_compact(t: u16, m: Mask) -> Vec<XGate> {
    let mut nonlinear = XGate::conj(t, [(m[0], false), (m[1], true)]).unwrap();
    nonlinear.comp = true;
    vec![nonlinear, XGate::cnot(t, m[2])]
}

fn operand(w: u16, pos: bool, masks: &[Mask], factored: bool) -> Vec<Vec<Literal>> {
    let mut out = vec![vec![(w, true)]];
    for &[x, y, z] in masks {
        if factored {
            out.push(vec![(x, false), (y, true)]);
        } else {
            out.push(vec![(x, true), (y, true)]);
            out.push(vec![(y, true)]);
        }
        out.push(vec![(z, true)]);
    }
    if (masks.len() % 2 == 1) ^ !pos {
        out.push(vec![]);
    }
    out
}

#[derive(Default, Clone, Debug)]
pub struct FireStats {
    pub products: usize,
    pub contradictory_products: usize,
    pub pair_guard_fallbacks: usize,
    pub cubic_units: usize,
    pub quartic_units: usize,
    pub omitted_restores: usize,
}

/// Each returned unit has its promised target update at its end. With
/// keep_garbage, only h/k may differ, and no later unit assumes their value.
pub fn compile(
    g: &XGate,
    masks: &[Vec<Mask>],
    helpers: [u16; 2],
    factored: bool,
    keep_garbage: bool,
    stats: &mut FireStats,
) -> Vec<Vec<XGate>> {
    assert!(g.ctrls.len() <= 2);
    let [h, k] = helpers;
    assert_ne!(h, k);
    let mut products = vec![vec![]];
    for &(w, pos) in &g.ctrls {
        products = products
            .into_iter()
            .flat_map(|a| {
                operand(w, pos, &masks[w as usize], factored)
                    .into_iter()
                    .map(move |b| {
                        let mut c = a.clone();
                        c.extend(b);
                        c
                    })
            })
            .collect();
    }
    if g.comp {
        products.push(vec![]);
    }
    let forbidden: Vec<_> = masks
        .iter()
        .flatten()
        .map(|m| (m[0].min(m[1]), m[0].max(m[1])))
        .collect();
    let np = masks.len();
    let safe = |a: Literal, b: Literal| {
        // Same-color band pairs also avoid temporary cover/fresh masks that
        // are not operands, and any later mask using this fixed partition.
        let partition_safe = (a.0 as usize) < np || (b.0 as usize) < np || a.0 % 2 == b.0 % 2;
        partition_safe && !forbidden.contains(&(a.0.min(b.0), a.0.max(b.0)))
    };
    let mut units = Vec::new();
    for p in products {
        stats.products += 1;
        let Some(normalized) = XGate::conj(g.target, p) else {
            stats.contradictory_products += 1;
            continue;
        };
        let l = &normalized.ctrls;
        assert!(l.iter().all(|(w, _)| *w != h && *w != k));
        let mut unit = vec![];
        let mut emit = |t, literals: Vec<Literal>| unit.push(XGate::conj(t, literals).unwrap());
        match l.len() {
            0..=2 => unit.push(normalized),
            3 => {
                stats.cubic_units += 1;
                let choices = [(0, 1, 2), (0, 2, 1), (1, 2, 0)];
                let (a, b, c) = choices
                    .iter()
                    .copied()
                    .find(|&(a, b, _)| safe(l[a], l[b]))
                    .unwrap_or_else(|| {
                        stats.pair_guard_fallbacks += 1;
                        choices[0]
                    });
                emit(g.target, vec![(h, true), l[c]]);
                emit(h, vec![l[a], l[b]]);
                emit(g.target, vec![(h, true), l[c]]);
                if !keep_garbage {
                    emit(h, vec![l[a], l[b]]);
                } else {
                    stats.omitted_restores += 1;
                }
            }
            4 => {
                stats.quartic_units += 1;
                let choices = [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)];
                let (a, b, c, d) = choices
                    .iter()
                    .copied()
                    .find(|&(a, b, c, d)| safe(l[a], l[b]) && safe(l[c], l[d]))
                    .unwrap_or_else(|| {
                        stats.pair_guard_fallbacks += 1;
                        choices[0]
                    });
                emit(g.target, vec![(h, true), (k, true)]);
                emit(h, vec![l[a], l[b]]);
                emit(g.target, vec![(h, true), (k, true)]);
                emit(k, vec![l[c], l[d]]);
                emit(g.target, vec![(h, true), (k, true)]);
                emit(h, vec![l[a], l[b]]);
                emit(g.target, vec![(h, true), (k, true)]);
                if !keep_garbage {
                    emit(k, vec![l[c], l[d]]);
                } else {
                    stats.omitted_restores += 1;
                }
            }
            _ => unreachable!(),
        }
        units.push(unit);
    }
    units
}
