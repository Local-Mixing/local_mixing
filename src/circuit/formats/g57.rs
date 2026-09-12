//! Historical base-83 and raw G57 encodings.
use crate::circuit::CircuitSeq;
const SEMI: u8 = 0xFD;
const TILDE: u8 = 0xFE;

/// Byte -> base-83 wire digit, with the gate separator and the `~` overflow
/// prefix folded in as sentinels above the digit range and `0xFF` for every
/// byte that is not part of the grammar. Built at compile time from the same
/// alphabet `repr()` emits, so the table cannot drift from the encoder.
const fn build_base83_decode() -> [u8; 256] {
    const ALPHABET: &[u8; 83] =
        b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?";
    let mut table = [0xFFu8; 256];
    let mut i = 0;
    while i < 83 {
        table[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    table[b';' as usize] = SEMI;
    table[b'~' as usize] = TILDE;
    table
}

/// Close one `;`-delimited segment, rejecting the same malformed shapes the
/// segment-at-a-time parser did.
#[inline]
fn finish_gate(
    gates: &mut Vec<[u16; 3]>,
    wires: &[u16; 3],
    count: usize,
    overflow: u32,
    seg: &[u8],
) {
    if overflow != 0 {
        panic!("Expected wire character after ~");
    }
    if count != 3 {
        panic!(
            "Each gate must have exactly 3 wires: {:?}",
            String::from_utf8_lossy(seg)
        );
    }
    gates.push(*wires);
}

impl CircuitSeq {
    /// Reconstruct CircuitSeq from a BLOB
    pub fn from_blob(blob: &[u8]) -> Self {
        assert!(blob.len() % 3 == 0, "Invalid blob length");
        let gates: Vec<[u16; 3]> = blob
            .chunks(3)
            .map(|chunk| [chunk[0] as u16, chunk[1] as u16, chunk[2] as u16])
            .collect();
        CircuitSeq { gates }
    }

    // Representing circuit as a string
    pub fn repr(&self) -> String {
        fn wire_to_char(w: u8) -> char {
            match w {
                0..=9 => (b'0' + w) as char,          // 0-9
                10..=35 => (b'a' + (w - 10)) as char, // a-z
                36..=61 => (b'A' + (w - 36)) as char, // A-Z
                // Special characters 62..=71
                62 => '!',
                63 => '@',
                64 => '#',
                65 => '$',
                66 => '%',
                67 => '^',
                68 => '&',
                69 => '*',
                70 => '(',
                71 => ')',
                // Special characters 72..=82
                72 => '-',
                73 => '_',
                74 => '=',
                75 => '+',
                76 => '[',
                77 => ']',
                78 => '{',
                79 => '}',
                80 => '<',
                81 => '>',
                82 => '?',
                _ => panic!("Invalid wire index: {}", w),
            }
        }

        const BASE: u32 = 83; // 0..82 is base

        // Append in place: the previous version built a fresh `String` per
        // wire, i.e. three heap allocations per gate.
        fn encode_wire_into(out: &mut String, w: u32) {
            let tildes = w / BASE;
            for _ in 0..tildes {
                out.push('~');
            }
            out.push(wire_to_char((w - tildes * BASE) as u8));
        }

        // Four bytes per gate is the floor (three wire characters plus the
        // separator); wide circuits add `~` prefixes on top.
        let mut s = String::with_capacity(self.gates.len() * 4);
        for gate in &self.gates {
            for &wire in gate {
                encode_wire_into(&mut s, wire as u32);
            }
            s.push(';'); // gate separator
        }
        s
    }

    pub fn from_string(s: &str) -> Self {
        Self::from_bytes(s.as_bytes())
    }

    /// Parse the base-83 `repr()` encoding straight from bytes.
    ///
    /// The encoding is ASCII, so this is the same grammar `from_string` accepts
    /// without the UTF-8 round trip a caller would otherwise pay to read a
    /// multi-megabyte circuit file. A flat byte scan over a `;`-separated
    /// stream also avoids the per-gate `Vec` and `chars().peekable()` the
    /// previous implementation built (measured 65 -> 5 ns/gate on a 13.5M-gate
    /// circuit, i.e. ~0.9 s off a single `evaluate` invocation).
    pub fn from_bytes(raw: &[u8]) -> Self {
        const BASE: u32 = 83;

        // Byte -> wire digit, with `;` and `~` as sentinels above the digit
        // range and 0xFF for everything outside the grammar.
        static DECODE: [u8; 256] = build_base83_decode();

        // Mirror `s.trim()`: the old parser trimmed Unicode whitespace, but the
        // only characters that can appear around a base-83 body are ASCII
        // spaces and newlines, and every wire character is a non-whitespace
        // ASCII byte.
        let mut body = raw;
        while let Some((first, rest)) = body.split_first() {
            if first.is_ascii_whitespace() {
                body = rest;
            } else {
                break;
            }
        }
        while let Some((last, rest)) = body.split_last() {
            if last.is_ascii_whitespace() {
                body = rest;
            } else {
                break;
            }
        }

        // `repr()` emits one `;` per gate, so the separator count is the exact
        // gate count for anything it wrote. The `+ 1` covers a hand-written
        // body whose last segment is unterminated: without it that one extra
        // gate would double the whole buffer to hold it.
        let mut gates: Vec<[u16; 3]> =
            Vec::with_capacity(body.iter().filter(|&&b| b == b';').count() + 1);

        // One flat pass with a single table lookup per byte. The separator and
        // the `~` overflow prefix are encoded in the same table as the wire
        // digits, so the loop has one dispatch instead of the nested
        // scan-tildes / check-separator / decode structure a segment-at-a-time
        // reader needs.
        let mut wires = [0u16; 3];
        let mut count = 0usize;
        let mut overflow: u32 = 0;
        let mut seg_start = 0usize;

        for (i, &b) in body.iter().enumerate() {
            let d = DECODE[b as usize];
            if d < BASE as u8 {
                if count < 3 {
                    wires[count] = (d as u32 + overflow * BASE) as u16;
                }
                count += 1;
                overflow = 0;
            } else if d == TILDE {
                overflow += 1;
            } else if d == SEMI {
                if count != 0 || overflow != 0 {
                    finish_gate(&mut gates, &wires, count, overflow, &body[seg_start..i]);
                }
                count = 0;
                overflow = 0;
                seg_start = i + 1;
            } else {
                panic!("Invalid wire char: {}", b as char);
            }
        }
        // A body that does not end in a separator still carries a final gate,
        // matching `split(';')` on an unterminated last segment.
        if count != 0 || overflow != 0 {
            finish_gate(&mut gates, &wires, count, overflow, &body[seg_start..]);
        }
        CircuitSeq { gates }
    }

    // Gives a "pretty" circuit representation. Does not support over 83 wires
    pub fn to_string(&self, num_wires: usize) -> String {
        let mut result = String::new();

        // Local character map (0-9, a-z, A-Z)
        let wire_map_chars: Vec<char> =
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!@#$%^&*()-_=+[]{}<>?"
                .chars()
                .collect();

        // --- Pretty circuit diagram ---
        for wire in 0..num_wires {
            result += &format!("{:<2} --", wire);
            for gate in &self.gates {
                if gate[0] == wire as u16 {
                    result += "( )";
                } else if gate[1] == wire as u16 {
                    result += "-●-";
                } else if gate[2] == wire as u16 {
                    result += "-○-";
                } else {
                    result += "-|-";
                }
                result.push_str("---");
            }
            result.push('\n');
        }

        // Compact circuit string (like "123;124;213;")
        let compact: String = self
            .gates
            .iter()
            .map(|g| {
                g.iter()
                    .map(|&x| wire_map_chars.get(x as usize).unwrap_or(&'?').to_string())
                    .collect::<String>()
                    + ";"
            })
            .collect();

        result.push_str("\n");
        result.push_str(&compact);

        result
    }
}
