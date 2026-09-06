//! Serializable analysis output.
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Serialize, Deserialize)]
pub struct Report {
    pub version: u32,
    pub design: String,
    pub results: Vec<ResultReport>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct ResultReport {
    pub name: String,
    pub model: String,
    pub n_complete: usize,
    pub missing_runs: Vec<String>,
    pub estimability: EstimabilityReport,
    pub fit: FitReport,
    pub anova: Vec<Anova>,
    pub strata: Vec<Stratum>,
    pub coefficients: Vec<Coefficient>,
    pub predictions: Vec<Prediction>,
    pub recommendation: Recommendation,
    pub warnings: Vec<String>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct EstimabilityReport {
    pub n_params: usize,
    pub rank: usize,
    pub residual_df: usize,
    pub terms: Vec<TermReport>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct TermReport {
    pub term: String,
    pub df: usize,
    pub estimable: bool,
    pub aliased_with: Vec<String>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct FitReport {
    pub sigma: Option<f64>,
    pub rss: f64,
    pub r_squared: Option<f64>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Anova {
    pub term: String,
    pub stratum: String,
    pub df: usize,
    pub ss: f64,
    pub ms: Option<f64>,
    pub f: Option<f64>,
    pub p: Option<f64>,
    pub significant: bool,
    pub pooled: bool,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Stratum {
    pub name: String,
    pub df: usize,
    pub ms: Option<f64>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Coefficient {
    pub label: String,
    pub estimate: Option<f64>,
    pub se: Option<f64>,
    pub ci_low: Option<f64>,
    pub ci_high: Option<f64>,
    pub boot_low: Option<f64>,
    pub boot_high: Option<f64>,
    pub estimable: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    pub factors: BTreeMap<String, String>,
    pub mean: f64,
    pub se: Option<f64>,
    pub estimable: bool,
    pub measured: bool,
    pub observed_mean: Option<f64>,
    pub boot_low: Option<f64>,
    pub boot_high: Option<f64>,
}
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Recommendation {
    pub maximize: Option<Optimum>,
    pub minimize: Option<Optimum>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct Optimum {
    pub factors: BTreeMap<String, String>,
    pub mean: f64,
    pub ci_low: Option<f64>,
    pub ci_high: Option<f64>,
    pub measured: bool,
}
