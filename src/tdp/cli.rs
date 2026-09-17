use clap::{Arg, ArgAction, Command};
use std::path::PathBuf;

pub fn command() -> Command {
    Command::new("tdp_gen")
        .about("Generate a TDP circuit using the pipeline recipe in configs/tdp.toml")
        .arg(
            Arg::new("config")
                .long("config")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf))
                .help("TOML recipe (default: configs/tdp.toml)"),
        )
        .arg(
            Arg::new("dry_run")
                .long("dry-run")
                .action(ArgAction::SetTrue)
                .help("Resolve and validate the config without building or running the pipeline"),
        )
}
