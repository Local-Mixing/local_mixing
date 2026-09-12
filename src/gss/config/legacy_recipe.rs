//! Historical mode names and version-specific recipe spellings.
//! These variants let recorded runs retain modes that are unavailable to fresh runs.
use super::*;
use crate::stages::preprocessing::PreprocessingMode;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecipePreprocessingMode {
    QuadraticMasking,
    Product2223,
    Nonlinear193,
    Nonlinear291,
}

impl RecipePreprocessingMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::QuadraticMasking => PreprocessingMode::QuadraticMasking.canonical_name(),
            Self::Product2223 => "product-2223",
            Self::Nonlinear193 => "nonlinear193",
            Self::Nonlinear291 => PreprocessingMode::Nonlinear291.canonical_name(),
        }
    }

    pub(crate) fn supported(self) -> Option<PreprocessingMode> {
        match self {
            Self::QuadraticMasking => Some(PreprocessingMode::QuadraticMasking),
            Self::Nonlinear291 => Some(PreprocessingMode::Nonlinear291),
            Self::Product2223 | Self::Nonlinear193 => None,
        }
    }

    pub(crate) fn for_recipe(self, version: u8) -> &'static str {
        if version <= 6 && self == Self::QuadraticMasking {
            "ran-balanced"
        } else {
            self.as_str()
        }
    }

    pub(crate) fn nonlinear(self) -> Option<NonlinearGssMode> {
        match self {
            Self::QuadraticMasking | Self::Product2223 => None,
            #[cfg(feature = "legacy-tools")]
            Self::Nonlinear193 => Some(NonlinearGssMode::Nonlinear193),
            #[cfg(not(feature = "legacy-tools"))]
            Self::Nonlinear193 => None,
            Self::Nonlinear291 => Some(NonlinearGssMode::Nonlinear291),
        }
    }
}

/// Parse retained modes before the runner knows whether this is a saved recipe.
pub(crate) fn parse_gadgetization_mode(raw: &RawConfig) -> Result<RecipePreprocessingMode, String> {
    let value = raw
        .value("gadgetization_mode")
        .unwrap_or(if raw.legacy_markdown {
            "product-2223"
        } else {
            "quadratic-masking"
        });
    if let Some(mode) = PreprocessingMode::parse(value) {
        return Ok(match mode {
            PreprocessingMode::QuadraticMasking => RecipePreprocessingMode::QuadraticMasking,
            PreprocessingMode::Nonlinear291 => RecipePreprocessingMode::Nonlinear291,
        });
    }
    match value {
        "product-2223" | "2223" => Ok(RecipePreprocessingMode::Product2223),
        "nonlinear193" => Ok(RecipePreprocessingMode::Nonlinear193),
        other => Err(config_error(
            raw,
            "gadgetization_mode",
            &format!(
                "expected quadratic-masking or nonlinear291 (historical mode names remain readable for saved runs); got {other:?}"
            ),
        )),
    }
}
