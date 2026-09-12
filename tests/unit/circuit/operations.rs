use super::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT: AtomicU64 = AtomicU64::new(0);
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "circuit_operations_{}_{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
    fn load(&self, name: &str, text: &str) -> CircuitSource {
        let path = self.0.join(name);
        fs::write(&path, text).unwrap();
        CircuitSource::read(path.to_str().unwrap()).unwrap()
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

struct Ones;
impl RngCore for Ones {
    fn next_u32(&mut self) -> u32 {
        u32::MAX
    }
    fn next_u64(&mut self) -> u64 {
        u64::MAX
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        dest.fill(u8::MAX);
    }
}

#[test]
fn loaded_evaluation_matches_existing_kernels_across_width_boundaries() {
    let scratch = Scratch::new();
    for width in [3, 64, 65, 128, 129, 256, 257, 512, 1024] {
        let circuit = CircuitSeq {
            gates: vec![[0, 1, width - 1], [width - 1, 0, 1]],
        };
        let source = scratch.load("g57", &circuit.repr());
        assert_eq!(source.wire_count(), width as usize);
        for initial in [[u64::MAX; 16], [0x5aa55aa55aa55aa5; 16], [0; 16]] {
            let mut actual = initial;
            source.evaluate(&mut actual);
            assert_eq!(
                actual,
                circuit.evaluate_1024(U1024(initial)).0,
                "width {width}"
            );
        }
        // The same function loaded through the general-gate reader also matches.
        let path = scratch.0.join("general");
        let gates: Vec<_> = circuit
            .gates
            .iter()
            .map(|&gate| XGate::from_g57(gate))
            .collect();
        format::write_mpmct(path.to_str().unwrap(), &gates, width as usize).unwrap();
        let general = CircuitSource::read(path.to_str().unwrap()).unwrap();
        let comparison = source
            .compare_sampled(&general, 1024, 8, &mut Ones)
            .unwrap();
        assert_eq!(comparison.wires, 1024);
        assert!(comparison.counterexample.is_none());
    }
}

#[test]
fn comparison_samples_auxiliary_inputs_instead_of_fixing_them_to_zero() {
    let scratch = Scratch::new();
    let identity = scratch.load("identity", "mpmct1 301 0\n");
    let controlled = scratch.load("controlled", "mpmct1 301 1\n0 0 1 300 1\n");
    let comparison = identity
        .compare_sampled(&controlled, 1, 1, &mut Ones)
        .unwrap();
    assert_eq!(comparison.wires, 301);
    let mismatch = comparison
        .counterexample
        .expect("high input must be sampled");
    assert_eq!(mismatch.sample, 1);
    assert_eq!(mismatch.input[4] >> 44, 1);
    assert!(mismatch.input[5..].iter().all(|&limb| limb == 0));
    // On the zero auxiliary-input slice they agree; the full functions differ.
    let mut zero = [0; 16];
    controlled.evaluate(&mut zero);
    assert_eq!(zero, [0; 16]);
}

#[test]
fn comparison_checks_auxiliary_outputs_even_when_requested_width_is_small() {
    let scratch = Scratch::new();
    let identity = scratch.load("identity", "mpmct1 301 0\n");
    let flip = scratch.load("flip", "mpmct1 301 1\n300 0 0\n");
    let comparison = identity.compare_sampled(&flip, 1, 1, &mut Ones).unwrap();
    let mismatch = comparison
        .counterexample
        .expect("high output must be compared");
    let mut output = mismatch.input;
    flip.evaluate(&mut output);
    assert_eq!(output[0], mismatch.input[0]);
    assert_ne!(output[4], mismatch.input[4]);
}

#[test]
fn empty_circuits_preserve_state_and_oversized_files_are_rejected() {
    let scratch = Scratch::new();
    let identity = scratch.load("empty", "");
    assert_eq!(identity.wire_count(), 0);
    let mut state = [u64::MAX; 16];
    identity.evaluate(&mut state);
    assert_eq!(state, [u64::MAX; 16]);
    let path = scratch.0.join("too-wide");
    fs::write(&path, "mpmct1 1025 0\n").unwrap();
    let error = CircuitSource::read(path.to_str().unwrap()).err().unwrap();
    assert!(error.contains("needs 1025 wires"));
}

#[test]
fn comparison_rejects_invalid_widths_and_vacuous_sample_counts() {
    let scratch = Scratch::new();
    let identity = scratch.load("identity", "mpmct1 1 0\n");
    for (width, samples) in [(0, 1), (1025, 1), (1, 0)] {
        assert!(
            identity
                .compare_sampled(&identity, width, samples, &mut Ones)
                .is_err()
        );
    }
}
