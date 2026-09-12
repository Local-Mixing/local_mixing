//! Grouped circuit command registration and dispatch.
use clap::{ArgMatches, Command};

pub mod compare;
pub mod evaluate;
pub mod generate;
mod shared;

pub fn command() -> Command {
    Command::new("circuit")
        .about("Generate, evaluate and compare circuits")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(generate::command())
        .subcommand(evaluate::command())
        .subcommand(compare::command())
}

pub fn run(sub: &ArgMatches) -> Result<(), String> {
    match sub.subcommand() {
        Some(("generate", m)) => generate::run(m),
        Some(("evaluate", m)) => evaluate::run(m),
        Some(("compare", m)) => compare::run(m),
        _ => unreachable!(),
    }
}
