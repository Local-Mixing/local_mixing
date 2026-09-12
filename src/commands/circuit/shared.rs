//! Shared command-line wire validation and display formatting.
use local_mixing::circuit::operations::CircuitState;

pub(super) fn parse_wires(raw: &str) -> Result<usize, String> {
    let n = raw
        .parse::<usize>()
        .map_err(|_| "wires must be an integer".to_owned())?;
    if !(1..=1024).contains(&n) {
        return Err("wires must be in 1..=1024".into());
    }
    Ok(n)
}

pub(super) fn format_bits(state: &CircuitState, n: usize) -> String {
    let bits: String = (0..n)
        .map(|i| {
            if (state[i >> 6] >> (i & 63)) & 1 == 1 {
                '1'
            } else {
                '0'
            }
        })
        .collect();
    let needed = n.div_ceil(8);
    let hex: String = (0..needed)
        .rev()
        .map(|byte| format!("{:02x}", (state[byte >> 3] >> ((byte & 7) * 8)) as u8))
        .collect();
    format!("{} (0x{})", bits, hex)
}
