use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

mod analyze;
mod construct;
mod factor;
mod oa;

#[derive(Parser)]
#[command(
    name = "taguchi",
    about = "Orthogonal-array experiment design and ANOVA analysis.",
    version
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Interactively define factors and result columns, then emit a Taguchi/orthogonal-array CSV.
    ///
    /// Factor spec syntax (case-sensitive):
    ///   d[1,2-4,9]             discrete ints (range with dash)
    ///   d["off","on"]          discrete strings
    ///   uniform[low,high,n]    n evenly-spaced points in [low,high]
    ///   normal[low,high,n,std] n quantile points of N((low+high)/2, std), clamped
    ///   logLow[low,high,n]     geometric spacing, dense near low
    ///   logHigh[low,high,n]    geometric spacing, dense near high
    Construct {
        /// Output CSV path. A sidecar <stem>.design.json is also written.
        #[arg(short, long, default_value = "experiments.csv")]
        output: PathBuf,
        /// Read factors and results from a file instead of prompting.
        /// File format: 'name = spec' lines under [factors], result names under [results].
        #[arg(short, long)]
        from: Option<PathBuf>,
    },
    /// Read a filled-in experiment CSV and decompose factor impact via ANOVA.
    Analyze {
        /// Input CSV (must have factor and result columns matching the design).
        csv: PathBuf,
        /// Override design sidecar path (defaults to <csv-stem>.design.json).
        #[arg(short, long)]
        design: Option<PathBuf>,
        /// Comma-separated result columns to optimize by maximization.
        #[arg(long, value_delimiter = ',')]
        maximize: Vec<String>,
        /// Comma-separated result columns to optimize by minimization.
        #[arg(long, value_delimiter = ',')]
        minimize: Vec<String>,
        /// Report effects whose level-range exceeds this noise floor.
        /// Without this flag, the F-test (α=0.05) is the gate and inconclusive
        /// results are refused with a non-zero exit.
        #[arg(long)]
        tolerate_noise: Option<f64>,
        /// Significance level when --tolerate-noise is not used.
        #[arg(long, default_value_t = 0.05)]
        alpha: f64,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Construct { output, from } => match from {
            Some(path) => construct::run_construct_from_file(path, output),
            None => construct::run_construct(output),
        },
        Cmd::Analyze {
            csv,
            design,
            maximize,
            minimize,
            tolerate_noise,
            alpha,
        } => analyze::run_analyze(analyze::AnalyzeArgs {
            csv_path: csv,
            design_path: design,
            maximize,
            minimize,
            tolerate_noise,
            alpha,
        }),
    }
}
