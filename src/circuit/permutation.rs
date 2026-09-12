//! Wire permutations and composition.
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
// Permutations are all the possible outputs of a circuit
// On n wires permutation length is 1 << n
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Permutation {
    pub data: Vec<usize>,
}

impl Permutation {
    pub fn new(data: Vec<usize>) -> Permutation {
        Permutation { data }
    }

    // Compose two permutations: (self ∘ other)[i] = self[other[i]].
    pub fn compose(&self, other: &Permutation) -> Permutation {
        if self.data.len() != other.data.len() {
            panic!("Permutation length mismatch in compose");
        }
        let data = (0..self.data.len())
            .map(|i| self.data[other.data[i]])
            .collect();
        Permutation { data }
    }
    pub fn is_perm(&self) -> bool {
        let mut temp_perm = self.clone();
        temp_perm.data.sort_unstable();
        temp_perm == Permutation::id_perm(self.data.len())
    }

    pub fn id_perm(n: usize) -> Permutation {
        let temp_data = (0..n).collect();
        Permutation { data: temp_data }
    }

    // n is the length of the permutation. For a random permutation on n bits, do 1 << n
    pub fn rand_perm(n: usize) -> Permutation {
        let mut p = Permutation::id_perm(n);
        let mut rng = rand::rng();
        p.data.shuffle(&mut rng);
        p
    }

    pub fn invert(&self) -> Permutation {
        let mut inv = vec![0; self.data.len()];
        self.data
            .iter()
            .enumerate()
            .for_each(|(i, &val)| inv[val] = i);
        Permutation { data: inv }
    }
}
