//! G57 scalar, fixed-limb and bit-sliced evaluation kernels.
use super::{CircuitSeq, Gate};
use primitive_types::{U256 as u256, U512 as u512};
use rand::RngCore;
use uint::construct_uint;
construct_uint! {
    pub struct U1024(16);
}

impl Gate {
    // Evaluate a bit string after a single gate under gate r57
    #[inline(always)]
    pub fn evaluate_index(state: usize, gate: [u16; 3]) -> usize {
        let c1 = (state >> gate[1]) & 1;
        let c2 = (state >> gate[2]) & 1;
        state ^ (c1 | ((!c2) & 1)) << gate[0]
    }

    // Evaluate up to 128 bits
    #[inline(always)]
    pub fn evaluate_index_128(state: u128, gate: [u16; 3]) -> u128 {
        let c1 = (state >> gate[1]) & 1;
        let c2 = (state >> gate[2]) & 1;
        state ^ ((c1 | (1 ^ c2)) << gate[0])
    }

    // Evaluate up to 256 bits. Direct limb indexing: the bignum shift chains
    // cost ~6 full-width ops per gate where 3 u64 ops suffice. Wires must be
    // in range (the old full-width shifts silently evaluated out-of-range
    // wires as zero; dispatchers pick the kernel by max touched wire).
    #[inline(always)]
    pub fn evaluate_index_256(mut state: u256, gate: [u16; 3]) -> u256 {
        debug_assert!(gate[0] < 256 && gate[1] < 256 && gate[2] < 256);
        let c1 = (state.0[(gate[1] >> 6) as usize] >> (gate[1] & 63)) & 1;
        let c2 = (state.0[(gate[2] >> 6) as usize] >> (gate[2] & 63)) & 1;
        state.0[(gate[0] >> 6) as usize] ^= (c1 | (1 ^ c2)) << (gate[0] & 63);
        state
    }

    #[inline(always)]
    pub fn evaluate_index_512(mut state: u512, gate: [u16; 3]) -> u512 {
        debug_assert!(gate[0] < 512 && gate[1] < 512 && gate[2] < 512);
        let c1 = (state.0[(gate[1] >> 6) as usize] >> (gate[1] & 63)) & 1;
        let c2 = (state.0[(gate[2] >> 6) as usize] >> (gate[2] & 63)) & 1;
        state.0[(gate[0] >> 6) as usize] ^= (c1 | (1 ^ c2)) << (gate[0] & 63);
        state
    }

    #[inline(always)]
    pub fn evaluate_index_1024(mut state: U1024, gate: [u16; 3]) -> U1024 {
        debug_assert!(gate[0] < 1024 && gate[1] < 1024 && gate[2] < 1024);
        let c1 = (state.0[(gate[1] >> 6) as usize] >> (gate[1] & 63)) & 1;
        let c2 = (state.0[(gate[2] >> 6) as usize] >> (gate[2] & 63)) & 1;
        state.0[(gate[0] >> 6) as usize] ^= (c1 | (1 ^ c2)) << (gate[0] & 63);
        state
    }

    // Evaluate a list of gates
    #[inline(always)]
    pub fn evaluate_index_list(state: usize, gates: &[[u16; 3]]) -> usize {
        let mut current_wires = state;
        for g in gates {
            current_wires = Self::evaluate_index(current_wires, *g);
        }
        current_wires
    }

    /// Evaluate a g57 gate list on a single u64 state word (up to 64 wires).
    ///
    /// The narrowest kernel: the whole state lives in one register, so the
    /// walk is a pure dependency chain with no memory traffic beyond the gate
    /// stream. Dispatchers pick it when every touched wire is below 64.
    #[inline(always)]
    pub fn evaluate_index_list_64(state: u64, gates: &[[u16; 3]]) -> u64 {
        let mut s = state;
        for &g in gates {
            debug_assert!(g[0] < 64 && g[1] < 64 && g[2] < 64);
            let c1 = (s >> (g[1] & 63)) & 1;
            let c2 = (s >> (g[2] & 63)) & 1;
            s ^= (c1 | (1 ^ c2)) << (g[0] & 63);
        }
        s
    }

    #[inline(always)]
    pub fn evaluate_index_list_128(state: u128, gates: &[[u16; 3]]) -> u128 {
        let mut limbs = [state as u64, (state >> 64) as u64];
        eval_limbs::<2>(&mut limbs, gates);
        (limbs[0] as u128) | ((limbs[1] as u128) << 64)
    }

    // The per-gate work is three u64 accesses; taking/returning the whole
    // bignum by value made the loop copy 32/64/128 bytes per gate on top of
    // that. Running the limb array in place instead keeps one stack slot live
    // for the entire walk (measured 11.2 -> 3.4 ns/gate at 256 bits).
    #[inline(always)]
    pub fn evaluate_index_list_256(mut state: u256, gates: &[[u16; 3]]) -> u256 {
        eval_limbs::<4>(&mut state.0, gates);
        state
    }

    #[inline(always)]
    pub fn evaluate_index_list_512(mut state: u512, gates: &[[u16; 3]]) -> u512 {
        eval_limbs::<8>(&mut state.0, gates);
        state
    }

    #[inline(always)]
    pub fn evaluate_index_list_1024(mut state: U1024, gates: &[[u16; 3]]) -> U1024 {
        eval_limbs::<16>(&mut state.0, gates);
        state
    }

    /// Bit-sliced g57 evaluation: `state[w]` carries one bit per sample lane,
    /// so a single walk evaluates 64 independent inputs.
    ///
    /// Transposing the state this way turns "one wide word per sample" into
    /// "one word per wire", which is what makes multi-input work cheap: the
    /// per-gate cost is the same handful of u64 ops the scalar kernel pays,
    /// but it now covers 64 samples instead of one.
    ///
    /// `t ^= pos OR NOT neg` becomes `state[t] ^= state[pos] | !state[neg]`,
    /// which reproduces the scalar kernel lane by lane — including the X-gate
    /// case `pos == neg`, where `s | !s` is all ones and the target toggles
    /// unconditionally.
    ///
    /// `state.len()` must be a power of two greater than every wire index the
    /// circuit touches; size it with [`lane_state_len`].
    #[inline]
    pub fn eval_lanes_index_list(gates: &[[u16; 3]], state: &mut [u64]) {
        debug_assert!(state.len().is_power_of_two() && !state.is_empty());
        // Masking with a power-of-two length lets the bounds checks fold away
        // without unsafe; the debug_assert pins the in-range contract.
        let m = state.len() - 1;
        for &[t, x, y] in gates {
            debug_assert!((t as usize) <= m && (x as usize) <= m && (y as usize) <= m);
            let fire = state[(x as usize) & m] | !state[(y as usize) & m];
            state[(t as usize) & m] ^= fire;
        }
    }
}

///
/// Rounded up to a power of two so the kernels can mask instead of
/// bounds-check; the slack is a few hundred bytes at most.
pub fn lane_state_len(wires: usize) -> usize {
    wires.max(1).next_power_of_two()
}

/// One g57 gate against a fixed-size limb array, in place.
///
/// `t ^= pos OR NOT neg`, with every wire index split into (limb, bit) exactly
/// as the fixed-width kernels above do. `L` is a const parameter so the limb
/// index masks fold to constants and the array stays in one stack slot.
#[inline(always)]
fn apply_limbs<const L: usize>(state: &mut [u64; L], gate: [u16; 3]) {
    debug_assert!(
        (gate[0] as usize) < L * 64 && (gate[1] as usize) < L * 64 && (gate[2] as usize) < L * 64
    );
    // `& (L - 1)` where L is a power of two lets LLVM drop the bounds check
    // without unsafe; the debug_assert above pins the in-range contract that
    // every dispatcher already guarantees.
    const {
        assert!(L.is_power_of_two(), "limb count must be a power of two");
    }
    let c1 = (state[((gate[1] >> 6) as usize) & (L - 1)] >> (gate[1] & 63)) & 1;
    let c2 = (state[((gate[2] >> 6) as usize) & (L - 1)] >> (gate[2] & 63)) & 1;
    state[((gate[0] >> 6) as usize) & (L - 1)] ^= (c1 | (1 ^ c2)) << (gate[0] & 63);
}

/// Walk a gate list against a limb array in place.
#[inline(always)]
fn eval_limbs<const L: usize>(state: &mut [u64; L], gates: &[[u16; 3]]) {
    for &g in gates {
        apply_limbs::<L>(state, g);
    }
}

impl CircuitSeq {
    // Evaluate the entire circuit with a starting input
    pub fn evaluate(&self, input: usize) -> usize {
        Gate::evaluate_index_list(input, &self.gates)
    }

    // Evaluate the circuit on a 64-bit input state (one bit per wire).
    pub fn evaluate_64(&self, input: u64) -> u64 {
        Gate::evaluate_index_list_64(input, &self.gates)
    }

    // Evaluate the circuit on a 128-bit input state (one bit per wire).
    pub fn evaluate_128(&self, input: u128) -> u128 {
        Gate::evaluate_index_list_128(input, &self.gates)
    }

    // Evaluate the circuit on a 512-bit input state (one bit per wire).
    pub fn evaluate_512(&self, input: u512) -> u512 {
        Gate::evaluate_index_list_512(input, &self.gates)
    }

    // Evaluate the circuit on a 256-bit input state (one bit per wire).
    pub fn evaluate_256(&self, input: u256) -> u256 {
        Gate::evaluate_index_list_256(input, &self.gates)
    }

    // Evaluate the circuit on a 1024-bit input state (one bit per wire).
    pub fn evaluate_1024(&self, input: U1024) -> U1024 {
        Gate::evaluate_index_list_1024(input, &self.gates)
    }

    pub fn evaluate_evolution_1024(&self, input: U1024) -> Vec<U1024> {
        let mut state = input;
        let mut evolution = Vec::with_capacity(self.gates.len() + 1);
        evolution.push(state);

        for gate in &self.gates {
            state = Gate::evaluate_index_1024(state, *gate);
            evolution.push(state);
        }

        evolution
    }

    // Probablistic check on circuit equality
    pub fn probably_equal(
        &self,
        other_circuit: &Self,
        num_wires: usize,
        num_inputs: usize,
    ) -> Result<(), String> {
        use rayon::prelude::*;

        // Arithmetic width must cover every wire either circuit TOUCHES, not
        // just the num_wires input/compare contract: primitive_types shifts
        // >= the type width silently return 0, so evaluating a circuit wider
        // than the chosen type corrupts every access to the high wires (e.g.
        // a 512-wire gadgetized circuit checked against its 256-wire source
        // was evaluated in u256 and reported non-equivalent). The num_wires
        // mask below is unchanged: inputs are drawn on num_wires bits and
        // outputs compared on num_wires bits.
        let eval_wires = num_wires
            .max(self.max_wire() + 1)
            .max(other_circuit.max_wire() + 1);

        if eval_wires > 1024 {
            // Retained from the fixed-width implementation. The bit-sliced
            // kernel below has no width ceiling of its own, so this is now an
            // artificial limit rather than a representational one.
            return Err("probabilistic equality supports up to 1024 wires".to_string());
        }
        if num_inputs == 0 {
            return Ok(());
        }

        // Bit-sliced: one walk per 64 inputs instead of one walk per input.
        // The old code drew a single wide word per sample and re-walked both
        // gate lists for each, so an m-gate check on k inputs streamed 2*m*k
        // gates; transposing to one word per wire makes that 2*m*ceil(k/64)
        // for the same per-gate cost. `num_inputs` therefore rounds up to a
        // multiple of 64 — the extra samples are free and only sharpen the
        // test.
        //
        // Input/compare contract is unchanged: samples are drawn on the low
        // `num_wires` wires (everything above starts at zero, which is what
        // masking the old wide input to `num_wires` bits did) and only those
        // wires are compared.
        let len = lane_state_len(eval_wires);
        let batches = num_inputs.div_ceil(64);

        (0..batches).into_par_iter().try_for_each(|_| {
            let mut rng = rand::rng();
            let mut mine = vec![0u64; len];
            for lane in mine[..num_wires].iter_mut() {
                *lane = rng.next_u64();
            }
            let mut theirs = mine.clone();

            Gate::eval_lanes_index_list(&self.gates, &mut mine);
            Gate::eval_lanes_index_list(&other_circuit.gates, &mut theirs);

            if mine[..num_wires] != theirs[..num_wires] {
                Err("Circuits are not equal".to_string())
            } else {
                Ok(())
            }
        })
    }
}
