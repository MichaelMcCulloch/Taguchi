use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use taguchi::{analyze, construct, design_select};

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
        /// Model formula, e.g. 'y ~ a*b + c'. Repeatable; appended after the
        /// file's [model] lines.
        #[arg(long = "model")]
        models: Vec<String>,
        /// Replicate role as NAME=SPEC, e.g. 'seed=5'. Repeatable; appended
        /// after the file's [replicates] lines.
        #[arg(long = "replicate")]
        replicates: Vec<String>,
        /// Experimental unit as NAME=key1,key2,… ; keys are factor or
        /// replicate-role names.
        #[arg(long)]
        unit: Option<String>,
        /// Randomization seed. Default: nanoseconds since the epoch, recorded
        /// in the manifest.
        #[arg(long)]
        seed: Option<u64>,
        /// Refuse designs above this many runs, counted after replication.
        #[arg(long)]
        max_runs: Option<usize>,
        /// Require at least this many residual degrees of freedom.
        #[arg(long, default_value_t = 0)]
        min_residual_df: usize,
        /// Write the design even when a requested term is not estimable.
        #[arg(long)]
        allow_aliased: bool,
        /// Execution order = design order (the seed is still recorded).
        #[arg(long)]
        no_shuffle: bool,
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
        Cmd::Construct {
            output,
            from,
            models,
            replicates,
            unit,
            seed,
            max_runs,
            min_residual_df,
            allow_aliased,
            no_shuffle,
        } => {
            let opts = construct::ConstructOpts {
                models,
                replicates,
                unit,
                seed,
                max_runs,
                min_residual_df,
                allow_aliased,
                no_shuffle,
            };
            let result = match from {
                Some(path) => construct::run_construct_from_file(path, output, &opts),
                None => construct::run_construct(output, &opts),
            };
            // A refused model is exit 3; the report is already in the payload.
            if let Err(e) = &result
                && let Some(refusal) = e.downcast_ref::<design_select::Refusal>()
            {
                eprintln!("{}", refusal);
                std::process::exit(3);
            }
            result
        }
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
