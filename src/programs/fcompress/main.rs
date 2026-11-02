//! Executable boundary for final compression and canonical packing.
use clap::Parser;

fn main() {
    let args = local_mixing::programs::fcompress::Args::parse();
    local_mixing::programs::fcompress::run(args);
}
