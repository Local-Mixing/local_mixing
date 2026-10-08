//! Executable boundary for the database-mixing and crossing program.
use clap::CommandFactory;

fn main() {
    let matches = local_mixing::programs::circuit_mixer::Args::command().get_matches();
    local_mixing::programs::circuit_mixer::run(matches);
}
