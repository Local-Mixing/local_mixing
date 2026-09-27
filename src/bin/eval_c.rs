//! Evaluate a run's stored source C on a fresh random input, printing ONLY
//! the output.
//!
//!   eval_c <source_c.g57> <x_out_path> [n=128]
//!
//! Draws a 64-bit x from the OS CSPRNG, evaluates y = C(x || 0^(n-64)) — x on
//! the low 64 wires, zeros above — writes x to <x_out_path> (mode 600) and
//! prints y to stdout. x is NEVER printed: it is a plaintext of the secret
//! computation and stays wherever this runs (see docs/GSS_MIX.md, "seeds").
use local_mixing::circuit::U1024;
use local_mixing::postmix::format::read_g57_file;
use local_mixing::postmix::xgate::eval_u1024;
use std::io::Read;

fn hex_of(val: U1024, n: usize) -> String {
    let bytes = val.to_little_endian();
    let needed = n.div_ceil(8);
    bytes[..needed].iter().rev().map(|b| format!("{b:02x}")).collect()
}

fn main() {
    let mut a = std::env::args().skip(1);
    let c_path = a.next().expect("usage: eval_c <source_c.g57> <x_out_path> [n]");
    let x_path = a.next().expect("usage: eval_c <source_c.g57> <x_out_path> [n]");
    let n: usize = a.next().and_then(|s| s.parse().ok()).unwrap_or(128);

    let gates = read_g57_file(&c_path).expect("read source C");

    // x from the OS CSPRNG, low 64 wires; the upper n-64 wires are zero.
    let mut xb = [0u8; 8];
    std::fs::File::open("/dev/urandom")
        .expect("open urandom")
        .read_exact(&mut xb)
        .expect("read urandom");
    let x = u64::from_le_bytes(xb);
    let mut inb = [0u8; 128];
    inb[..8].copy_from_slice(&xb);
    let input = U1024::from_little_endian(&inb);

    let y = eval_u1024(&gates, input);

    // x to disk, owner-only; never to stdout.
    {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;
        let mut f = std::fs::OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&x_path)
            .expect("open x out");
        writeln!(f, "{x:016x}").expect("write x");
    }

    println!("gates={} n={} y=0x{}", gates.len(), n, hex_of(y, n));
}
