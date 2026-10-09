//! Executable adapters: argument resolution, environment translation and artifact I/O.
//! Stage algorithms accept typed parameters and live under `stages`.
pub mod circuit_mixer;
pub mod gen_sandwich_gadget;
pub mod tdp_packing;

/// The largest seed file accepted (one decimal u64 plus whitespace).
const MAX_SEED_FILE_BYTES: u64 = 64;

/// Read a seed handed over as a file (gen's `@PATH` positionals, the mixer's
/// `--seed-file PATH`) so the value never appears on argv, where `ps` shows it
/// to every local user. The managed driver passes `/dev/fd/N` (a here-string
/// on an inherited descriptor), so nothing is written to disk. The file holds
/// one decimal u64; surrounding whitespace is ignored. Error messages never
/// quote the content.
pub(crate) fn read_seed_file(path: &str) -> Result<u64, String> {
    use std::io::Read;
    let mut text = String::new();
    std::fs::File::open(path)
        .and_then(|file| file.take(MAX_SEED_FILE_BYTES + 1).read_to_string(&mut text))
        .map_err(|error| format!("cannot read seed file {path}: {error}"))?;
    if text.len() as u64 > MAX_SEED_FILE_BYTES {
        return Err(format!(
            "seed file {path} is larger than {MAX_SEED_FILE_BYTES} bytes"
        ));
    }
    text.trim()
        .parse::<u64>()
        .map_err(|_| format!("seed file {path} does not hold one decimal seed"))
}
