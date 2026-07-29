//! Ask whether the CURATED store's answers are equivalent to the window they
//! are stored under.
//!
//! The normal lookup path only ever verifies the one candidate it picks, so a
//! failure there cannot distinguish "this entry is bad" from "every entry under
//! this key is bad". This decodes EVERY candidate for a window, from both
//! stores and both canonical directions, and checks each exhaustively.
//!
//! All curated candidates failing points at the frozen CONVERSION (a key/value
//! misalignment would put circuits under permutations they do not implement);
//! a mixture points at the source data.
//!
//! Usage: db_curated_probe   (window is the observed failing case, hardcoded)
use local_mixing::postmix::db_replace::db_probe;
use local_mixing::postmix::rules::verify_rewrite;
use local_mixing::postmix::xgate::XGate;
use local_mixing::postmix::xpoly::XPolyBudget;
use local_mixing::replace::frozen::FrozenDb;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn main() {
    let num_wires = 64;
    // The window that failed verification in the curated arm.
    let window = vec![
        XGate::conj(13, [(27u16, true)]).expect("g0"),
        {
            let mut g = XGate::conj(13, [(5u16, true), (27u16, false)]).expect("g1");
            g.comp = true;
            g
        },
    ];
    println!("window: {window:?}");
    let db = FrozenDb::from_env();
    let mut rng = StdRng::seed_from_u64(1);
    let cands = db_probe(&window, num_wires, &db, XPolyBudget::default(), &mut rng);
    println!("decoded {} candidates", cands.len());
    let (mut cok, mut cbad, mut rok, mut rbad) = (0, 0, 0, 0);
    for (g, curated, reversed) in &cands {
        let ok = verify_rewrite(&window, g);
        match (curated, ok) {
            (true, true) => cok += 1,
            (true, false) => cbad += 1,
            (false, true) => rok += 1,
            (false, false) => rbad += 1,
        }
        if !ok {
            println!("  FAIL curated={curated} reversed={reversed} len={} {:?}", g.len(), g);
        }
    }
    println!("curated: {cok} equivalent, {cbad} NOT equivalent");
    println!("regular: {rok} equivalent, {rbad} NOT equivalent");
}
