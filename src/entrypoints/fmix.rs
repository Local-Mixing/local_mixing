//! Executable boundary for the database-mixing, splitting and crossing program.
use clap::CommandFactory;

fn main() {
    let matches = local_mixing::programs::fmix::Args::command().get_matches();
    local_mixing::programs::fmix::run(matches);
}
