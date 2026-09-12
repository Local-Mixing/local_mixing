use clap::Command;
mod commands;

fn main() {
    let command = Command::new("local_mixing")
        .about("GSS mixing and circuit utilities")
        .subcommand_required(true)
        .arg_required_else_help(true)
        .subcommand(commands::gss::command())
        .subcommand(commands::circuit::command());
    #[cfg(feature = "legacy-db-tools")]
    let command = command.subcommand(commands::db::command());
    let matches = command.get_matches();
    let result = match matches.subcommand() {
        Some(("gss", sub)) => {
            commands::gss::run(sub);
            Ok(())
        }
        Some(("circuit", sub)) => commands::circuit::run(sub),
        #[cfg(feature = "legacy-db-tools")]
        Some(("db", sub)) => {
            commands::db::run(sub);
            Ok(())
        }
        _ => unreachable!(),
    };
    if let Err(error) = result {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
