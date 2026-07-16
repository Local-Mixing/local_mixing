//! Compact source-parent tracking for fragment-mixing experiments.
//!
//! A mark is either one original input-g57 index or [`MIXED_SOURCE`].  Once a
//! rewrite combines gates from distinct parents, the result stays mixed.  This
//! deliberately records interaction history rather than attempting algebraic
//! cancellation of ancestry, which is exactly the distinction needed to tell
//! an original sibling pair returning home from a genuinely mixed synthesis.

use super::xgate::XGate;
use std::io;

pub const MIXED_SOURCE: u32 = u32::MAX;
pub const UNKNOWN_SOURCE: u32 = u32::MAX - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceClass {
    /// The gate has one pristine parent and is structurally identical to it.
    ReturnedToParent,
    /// The gate has one pristine parent but is a different g57 triple.
    NewSameParent,
    /// At least two distinct original parents influenced the gate.
    NewMixedParents,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SourceClassCounts {
    pub returned_to_parent: u64,
    pub new_same_parent: u64,
    pub new_mixed_parents: u64,
    pub unknown: u64,
}

impl SourceClassCounts {
    pub fn record(&mut self, mark: u32, gate: &XGate, sources: &[XGate]) {
        match classify_source(mark, gate, sources) {
            Some(SourceClass::ReturnedToParent) => self.returned_to_parent += 1,
            Some(SourceClass::NewSameParent) => self.new_same_parent += 1,
            Some(SourceClass::NewMixedParents) => self.new_mixed_parents += 1,
            None => self.unknown += 1,
        }
    }

    pub fn known_total(self) -> u64 {
        self.returned_to_parent + self.new_same_parent + self.new_mixed_parents
    }

    pub fn new_total(self) -> u64 {
        self.new_same_parent + self.new_mixed_parents
    }
}

#[inline]
pub fn merge_source(left: u32, right: u32) -> u32 {
    if left == UNKNOWN_SOURCE || right == UNKNOWN_SOURCE {
        UNKNOWN_SOURCE
    } else if left == right {
        left
    } else {
        MIXED_SOURCE
    }
}

pub fn merge_sources(marks: impl IntoIterator<Item = u32>) -> u32 {
    let mut marks = marks.into_iter();
    let Some(mut merged) = marks.next() else {
        return UNKNOWN_SOURCE;
    };
    for mark in marks {
        merged = merge_source(merged, mark);
    }
    merged
}

pub fn classify_source(mark: u32, gate: &XGate, sources: &[XGate]) -> Option<SourceClass> {
    if mark == UNKNOWN_SOURCE {
        return None;
    }
    if mark == MIXED_SOURCE {
        return Some(SourceClass::NewMixedParents);
    }
    let parent = sources.get(mark as usize)?;
    Some(if gate == parent {
        SourceClass::ReturnedToParent
    } else {
        SourceClass::NewSameParent
    })
}

/// Text sidecar aligned with an mpmct1 output tape.
///
/// ```text
/// fsource1 <num_original_g57s> <num_output_gates>
/// <parent-index | mixed>
/// ...
/// ```
pub fn write_source_marks(path: &str, marks: &[u32], num_sources: usize) -> io::Result<()> {
    assert!(
        num_sources < UNKNOWN_SOURCE as usize,
        "source-parent ids exhausted u32 space"
    );
    let mut out = String::with_capacity(marks.len() * 8 + 32);
    out.push_str(&format!("fsource1 {num_sources} {}\n", marks.len()));
    for &mark in marks {
        if mark == MIXED_SOURCE {
            out.push_str("mixed\n");
        } else {
            assert!(
                mark != UNKNOWN_SOURCE && (mark as usize) < num_sources,
                "invalid source mark {mark} for {num_sources} parents"
            );
            out.push_str(&format!("{mark}\n"));
        }
    }
    std::fs::write(path, out)
}

pub fn read_source_marks(path: &str, expected_gates: usize) -> io::Result<(Vec<u32>, usize)> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines();
    let header = lines
        .next()
        .ok_or_else(|| io::Error::other("empty fsource sidecar"))?;
    let fields: Vec<&str> = header.split_whitespace().collect();
    if fields.len() != 3 || fields[0] != "fsource1" {
        return Err(io::Error::other("invalid fsource1 header"));
    }
    let num_sources: usize = fields[1]
        .parse()
        .map_err(|_| io::Error::other("invalid fsource source count"))?;
    let declared_gates: usize = fields[2]
        .parse()
        .map_err(|_| io::Error::other("invalid fsource gate count"))?;
    if declared_gates != expected_gates {
        return Err(io::Error::other(format!(
            "fsource gate count mismatch: sidecar {declared_gates}, circuit {expected_gates}"
        )));
    }
    let mut marks = Vec::with_capacity(declared_gates);
    for line in lines.filter(|line| !line.trim().is_empty()) {
        let mark = if line.trim() == "mixed" {
            MIXED_SOURCE
        } else {
            let mark: u32 = line
                .trim()
                .parse()
                .map_err(|_| io::Error::other("invalid fsource parent id"))?;
            if mark as usize >= num_sources {
                return Err(io::Error::other(format!(
                    "fsource parent {mark} outside 0..{num_sources}"
                )));
            }
            mark
        };
        marks.push(mark);
    }
    if marks.len() != declared_gates {
        return Err(io::Error::other(format!(
            "fsource row count mismatch: header {declared_gates}, parsed {}",
            marks.len()
        )));
    }
    Ok((marks, num_sources))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_merge_is_monotone() {
        assert_eq!(merge_source(7, 7), 7);
        assert_eq!(merge_source(7, 8), MIXED_SOURCE);
        assert_eq!(merge_source(MIXED_SOURCE, 7), MIXED_SOURCE);
        assert_eq!(merge_source(UNKNOWN_SOURCE, 7), UNKNOWN_SOURCE);
    }

    #[test]
    fn source_class_distinguishes_return_new_and_mixed() {
        let sources = vec![XGate::from_g57([0, 1, 2])];
        assert_eq!(
            classify_source(0, &sources[0], &sources),
            Some(SourceClass::ReturnedToParent)
        );
        assert_eq!(
            classify_source(0, &XGate::from_g57([0, 2, 1]), &sources),
            Some(SourceClass::NewSameParent)
        );
        assert_eq!(
            classify_source(MIXED_SOURCE, &sources[0], &sources),
            Some(SourceClass::NewMixedParents)
        );
    }
}
