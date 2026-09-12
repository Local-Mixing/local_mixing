//! MPX1: the mixed-alphabet DB value codec for wide (multi-control,
//! polarity-explicit) gates.
//!
//! The legacy value chain is `[len_u8][len bytes]` with `len % 3 == 0` and
//! 3 bytes per g57 gate. MPX1 keeps the same outer chunk shape so RocksDB
//! merge concatenation (`dedup_concat`) and the frozen raw-block container
//! work unchanged, but encodes gates as
//!
//! ```text
//! GATE_RECORD := [hdr][target][lit]*k    hdr = (comp << 7) | k
//!                                        lit = (wire << 1) | pol
//! CIRCUIT_CHUNK := [len_u8][records]     one chunk per stored circuit
//! ```
//!
//! A serving/freeze path may prepend the tag chunk `0x02 0xFE 0x01`: its
//! length byte is 2, and `2 % 3 != 0`, so every legacy `%3` decoder breaks
//! out on the first byte and yields zero circuits rather than misparsing.
//! (`0xFF` would NOT have this property: 255 % 3 == 0.)
//!
//! RocksDB-stored values are UNTAGGED chunk chains — the wide store lives in
//! its own directory that legacy consumers never open, and untagged chains
//! merge-concatenate without mid-stream tag skipping. `decode_value` accepts
//! and skips tag chunks wherever they appear, so both forms decode.

use crate::circuit::xgate::{Lits, XGate, sort_lits};
use std::io;

/// Tag chunk prepended by serving/freeze paths: `[len=2][0xFE][version=1]`.
pub const MPX1_TAG: [u8; 3] = [0x02, 0xFE, 0x01];

/// Maximum controls per gate record (6 bits of header, minus safety).
pub const MPX1_MAX_CONTROLS: usize = 63;

fn err(msg: impl Into<String>) -> io::Error {
    io::Error::other(msg.into())
}

/// Encode one circuit as a single MPX1 chunk (`[len][records]`).
///
/// Errors if any wire id needs more than 7 bits, a gate has more than
/// [`MPX1_MAX_CONTROLS`] controls, or the record stream exceeds one chunk
/// (255 bytes — 50 three-control gates).
pub fn encode_circuit(gates: &[XGate]) -> io::Result<Vec<u8>> {
    let mut body = Vec::with_capacity(gates.len() * 5);
    for g in gates {
        if g.ctrls.len() > MPX1_MAX_CONTROLS {
            return Err(err(format!("mpx1: gate has {} controls", g.ctrls.len())));
        }
        if g.target >= 128 {
            return Err(err(format!("mpx1: target wire {} needs >7 bits", g.target)));
        }
        body.push(((g.comp as u8) << 7) | g.ctrls.len() as u8);
        body.push(g.target as u8);
        for &(w, p) in &g.ctrls {
            if w >= 128 {
                return Err(err(format!("mpx1: control wire {w} needs >7 bits")));
            }
            body.push(((w as u8) << 1) | p as u8);
        }
    }
    if body.len() > u8::MAX as usize {
        return Err(err(format!(
            "mpx1: circuit encodes to {} bytes (> 255 per chunk)",
            body.len()
        )));
    }
    let mut out = Vec::with_capacity(1 + body.len());
    out.push(body.len() as u8);
    out.extend_from_slice(&body);
    Ok(out)
}

/// Encode a list of circuits as an untagged chunk chain (RocksDB value form).
pub fn encode_value(circuits: &[Vec<XGate>]) -> io::Result<Vec<u8>> {
    let mut out = Vec::new();
    for c in circuits {
        out.extend_from_slice(&encode_circuit(c)?);
    }
    Ok(out)
}

/// Prepend the MPX1 tag chunk (serving/freeze form).
pub fn tag_value(untagged: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(3 + untagged.len());
    out.extend_from_slice(&MPX1_TAG);
    out.extend_from_slice(untagged);
    out
}

/// True if the value starts with the MPX1 tag chunk.
pub fn is_mpx1(value: &[u8]) -> bool {
    value.len() >= 3 && value[..3] == MPX1_TAG
}

/// Decode an MPX1 value (tagged or untagged chunk chain) into circuits.
///
/// Strict: truncation, header/record mismatch, out-of-range wires, a control
/// on the gate's own target, or a duplicate/contradictory control wire are
/// all hard errors — this decoder guards the XGate invariant at the trust
/// boundary exactly like the validating `read_mpmct`.
pub fn decode_value(value: &[u8]) -> io::Result<Vec<Vec<XGate>>> {
    let mut circuits = Vec::new();
    let mut i = 0usize;
    while i < value.len() {
        let len = value[i] as usize;
        i += 1;
        if len == 0 {
            return Err(err("mpx1: zero-length chunk"));
        }
        if i + len > value.len() {
            return Err(err("mpx1: truncated chunk"));
        }
        let chunk = &value[i..i + len];
        i += len;
        // Tag chunks may appear anywhere after merges; skip them.
        if len == 2 && chunk == &MPX1_TAG[1..] {
            continue;
        }
        circuits.push(decode_chunk(chunk)?);
    }
    Ok(circuits)
}

fn decode_chunk(chunk: &[u8]) -> io::Result<Vec<XGate>> {
    let mut gates = Vec::new();
    let mut j = 0usize;
    while j < chunk.len() {
        if j + 2 > chunk.len() {
            return Err(err("mpx1: truncated gate header"));
        }
        let hdr = chunk[j];
        let target = chunk[j + 1] as u16;
        let comp = hdr & 0x80 != 0;
        let k = (hdr & 0x7f) as usize;
        j += 2;
        if j + k > chunk.len() {
            return Err(err("mpx1: truncated control list"));
        }
        let mut ctrls: Lits = Lits::new();
        for &b in &chunk[j..j + k] {
            ctrls.push(((b >> 1) as u16, b & 1 != 0));
        }
        j += k;
        sort_lits(&mut ctrls);
        for idx in 0..ctrls.len() {
            if ctrls[idx].0 == target {
                return Err(err("mpx1: control on the gate's own target"));
            }
            if idx > 0 && ctrls[idx - 1].0 == ctrls[idx].0 {
                return Err(err("mpx1: duplicate or contradictory control wire"));
            }
        }
        gates.push(XGate {
            target,
            comp,
            ctrls,
        });
    }
    Ok(gates)
}

#[cfg(test)]
#[path = "../../tests/db_gen/support/mpx1/tests.rs"]
mod tests;
