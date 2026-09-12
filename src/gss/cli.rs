use clap::{Arg, ArgAction, Command};
use std::path::PathBuf;

pub fn command() -> Command {
    Command::new("gss")
        .visible_alias("gss-mix")
        .about("Mix a circuit using the GSS recipe in configs/gss.toml")
        .arg(
            Arg::new("config")
                .long("config")
                .value_name("PATH")
                .value_parser(clap::value_parser!(PathBuf))
                .help("TOML recipe (default: configs/gss.toml); existing marked Markdown recipes remain readable"),
        )
        .arg(
            Arg::new("dry_run")
                .long("dry-run")
                .action(ArgAction::SetTrue)
                .help("Resolve and validate the config without building or running the pipeline"),
        )
}
