use crate::circuit::{Permutation};

use bincode;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, BufWriter},
};

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PersistPermStore {
    pub perm: Permutation,
    pub circuits: Vec<Vec<u8>>,
    pub count: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Persist {
    pub wires: usize,
    pub gates: usize,
    pub store: HashMap<Vec<u8>, Vec<Vec<u8>>>,
}

impl Persist {
    pub fn save(n: usize, m: usize, store: HashMap<Vec<u8>, Vec<Vec<u8>>>) {
        let file = File::create(format!("./db/n{}m{}.bin", n, m))
            .expect("Failed to create file");
        let writer = BufWriter::new(file);
        let persist = Persist {
            wires: n,
            gates: m,
            store: store,
        };
        bincode::serialize_into(writer, &persist)
            .expect("Failed to serialize Persist");
    }

    //correctly loads the file for m-1 db
    pub fn load(
        n: usize,
        m: usize,
    ) -> HashMap<Vec<u8>, Vec<Vec<u8>>> {
        let path = format!("./db/n{n}m{}.bin", m-1);
        let mut reader = std::io::BufReader::new(
            std::fs::File::open(&path).unwrap()
        );

        let mut map = HashMap::new();

        loop {
            match bincode::deserialize_from::<_, (Vec<u8>, Vec<Vec<u8>>)>(&mut reader) {
                Ok((k, v)) => {
                    map.insert(k, v);
                }
                Err(e) => {
                    if let bincode::ErrorKind::Io(ref io_err) = *e {
                        if io_err.kind() == std::io::ErrorKind::UnexpectedEof {
                            break;
                        }
                    }
                    panic!("Deserialization error: {:?}", e);
                }
            }
        }

        map
    }

    //correctly loads the file for m-1 db
    pub fn load_use(
        n: usize,
        m: usize,
    ) -> HashMap<Vec<u8>, Vec<Vec<u8>>> {
        let path = format!("./db/n{n}m{}.bin", m);
        let mut reader = std::io::BufReader::new(
            std::fs::File::open(&path).unwrap()
        );

        let mut map = HashMap::new();

        loop {
            match bincode::deserialize_from::<_, (Vec<u8>, Vec<Vec<u8>>)>(&mut reader) {
                Ok((k, v)) => {
                    map.insert(k, v);
                }
                Err(e) => {
                    if let bincode::ErrorKind::Io(ref io_err) = *e {
                        if io_err.kind() == std::io::ErrorKind::UnexpectedEof {
                            break;
                        }
                    }
                    panic!("Deserialization error: {:?}", e);
                }
            }
        }

        map
    }

}

pub fn make_persist(perm: Permutation, circuits: HashMap<Vec<u8>, bool>, count: usize) -> PersistPermStore {
    let keys: Vec<Vec<u8>> = circuits.into_keys().collect();
    PersistPermStore{ perm, circuits: keys, count, }
}