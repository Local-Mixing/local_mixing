//! GSS application: resolve a recipe, preserve run identity, and execute stages.
mod cli;
pub(crate) mod config;
pub use cli::command;
mod manifest;
pub(crate) mod paths;
mod runner;
use clap::ArgMatches;
use config::*;
use manifest::*;
use paths::*;
use runner::*;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::{OsStr, OsString};
use std::fs;
use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::process::{Command as ProcessCommand, ExitStatus};
use std::time::{SystemTime, UNIX_EPOCH};
use xxhash_rust::xxh3::Xxh3;

#[derive(Debug)]
pub(crate) struct GssError {
    pub(crate) message: String,
    pub(crate) exit_code: i32,
}

impl GssError {
    pub(crate) fn config(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            exit_code: 2,
        }
    }

    pub(crate) fn io(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            exit_code: 1,
        }
    }

    pub(crate) fn child(label: &str, status: ExitStatus) -> Self {
        let exit_code = status.code().unwrap_or(1);
        Self {
            message: format!("{label} exited with status {status}"),
            exit_code,
        }
    }
}

pub fn run(sub: &ArgMatches) {
    if let Err(error) = run_inner(sub) {
        eprintln!("[gss] FATAL: {}", error.message);
        std::process::exit(error.exit_code);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/gss.rs"]
mod tests;
