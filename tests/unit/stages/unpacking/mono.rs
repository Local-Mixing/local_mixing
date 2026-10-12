use super::*;

#[test]
fn bit_operations_and_var_iteration() {
    let m = Mono::<2>::from_vars(&[3, 64, 127]);
    assert_eq!(m.degree(), 3);
    assert!(m.contains(64) && !m.contains(65));
    assert_eq!(m.to_vec(), vec![3, 64, 127]);
    assert_eq!(m.first_var(), Some(3));
    assert_eq!(m.without(64).to_vec(), vec![3, 127]);
    assert_eq!(m.with(5).degree(), 4);
    let o = Mono::<2>::from_vars(&[64, 70]);
    assert_eq!(m.or(&o).to_vec(), vec![3, 64, 70, 127]);
    assert_eq!(m.and(&o).to_vec(), vec![64]);
    assert_eq!(m.minus(&o).to_vec(), vec![3, 127]);
    assert!(Mono::<2>::var(64).is_subset(&m) && !o.is_subset(&m));
    assert!(Mono::<2>::ONE.is_one() && !m.is_one());
    assert_eq!(Mono::<2>::ONE.first_var(), None);
    assert_eq!(Mono::<2>::VARS, 128);
    assert_eq!(format!("{:?}", m), "x3*x64*x127");
    assert_eq!(format!("{:?}", Mono::<1>::ONE), "1");
}

#[test]
fn order_is_total_and_distinguishes_words() {
    let a = Mono::<2>::var(1);
    let b = Mono::<2>::var(64);
    assert!(a != b);
    assert!(a.cmp(&b) != std::cmp::Ordering::Equal);
    assert_eq!(
        Mono::<1>::var(2).cmp(&Mono::var(3)),
        std::cmp::Ordering::Less
    );
}

#[test]
fn rng_is_deterministic_and_below_is_bounded() {
    let mut a = Rng::new(42);
    let mut b = Rng::new(42);
    for _ in 0..100 {
        assert_eq!(a.next_u64(), b.next_u64());
    }
    let mut c = Rng::new(43);
    assert_ne!(Rng::new(42).next_u64(), c.next_u64());
    for _ in 0..1000 {
        assert!(c.below(7) < 7);
    }
    assert_eq!(c.below(1), 0);
    assert_eq!(c.below(0), 0);
}
