use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal};
use std::fmt;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Factor {
    pub name: String,
    pub spec: String,
    pub kind: FactorKind,
    pub values: Vec<FactorValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FactorKind {
    Discrete,
    Uniform { low: f64, high: f64 },
    Normal { low: f64, high: f64, mean: f64, std: f64 },
    LogLow { low: f64, high: f64 },
    LogHigh { low: f64, high: f64 },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum FactorValue {
    Int(i64),
    Float(f64),
    Text(String),
}

impl fmt::Display for FactorValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FactorValue::Int(n) => write!(f, "{}", n),
            FactorValue::Float(x) => {
                if x.fract() == 0.0 && x.abs() < 1e15 {
                    write!(f, "{}", *x as i64)
                } else {
                    let s = format!("{:.6}", x);
                    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
                    write!(f, "{}", trimmed)
                }
            }
            FactorValue::Text(s) => write!(f, "{}", s),
        }
    }
}

impl FactorValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            FactorValue::Int(n) => Some(*n as f64),
            FactorValue::Float(x) => Some(*x),
            FactorValue::Text(_) => None,
        }
    }
}

impl Factor {
    pub fn level_count(&self) -> usize {
        self.values.len()
    }

    pub fn parse(name: &str, spec: &str) -> Result<Factor> {
        let spec_trim = spec.trim();
        let open = spec_trim
            .find('[')
            .ok_or_else(|| anyhow!("factor spec '{}': missing '['", spec_trim))?;
        if !spec_trim.ends_with(']') {
            bail!("factor spec '{}': must end with ']'", spec_trim);
        }
        let kind = spec_trim[..open].trim();
        let args = &spec_trim[open + 1..spec_trim.len() - 1];

        let mut factor = match kind {
            "d" | "discrete" => parse_discrete(name, args)?,
            "uniform" => parse_uniform(name, args)?,
            "normal" => parse_normal(name, args)?,
            "logLow" | "log_low" => parse_log_low(name, args)?,
            "logHigh" | "log_high" => parse_log_high(name, args)?,
            other => bail!("unknown factor kind '{}'. expected one of: d, uniform, normal, logLow, logHigh", other),
        };
        factor.spec = spec_trim.to_string();
        if factor.values.len() < 2 {
            bail!("factor '{}': must have ≥ 2 levels (got {})", name, factor.values.len());
        }
        Ok(factor)
    }
}

fn parse_discrete(name: &str, args: &str) -> Result<Factor> {
    let mut values = Vec::new();
    for raw in args.split(',') {
        let part = raw.trim();
        if part.is_empty() {
            continue;
        }
        if part.starts_with('"') && part.ends_with('"') && part.len() >= 2 {
            values.push(FactorValue::Text(part[1..part.len() - 1].to_string()));
            continue;
        }
        if let Some((lo_s, hi_s)) = split_range(part) {
            let lo: i64 = lo_s
                .parse()
                .with_context(|| format!("range lower bound '{}'", lo_s))?;
            let hi: i64 = hi_s
                .parse()
                .with_context(|| format!("range upper bound '{}'", hi_s))?;
            if lo > hi {
                bail!("invalid range {}..{}", lo, hi);
            }
            for n in lo..=hi {
                values.push(FactorValue::Int(n));
            }
            continue;
        }
        if part.contains('.') || part.contains('e') || part.contains('E') {
            let f: f64 = part
                .parse()
                .with_context(|| format!("float '{}'", part))?;
            values.push(FactorValue::Float(f));
        } else {
            let n: i64 = part
                .parse()
                .with_context(|| format!("integer '{}'", part))?;
            values.push(FactorValue::Int(n));
        }
    }
    Ok(Factor {
        name: name.into(),
        spec: String::new(),
        kind: FactorKind::Discrete,
        values,
    })
}

fn split_range(s: &str) -> Option<(&str, &str)> {
    for (i, c) in s.char_indices() {
        if i == 0 || c != '-' {
            continue;
        }
        let left = &s[..i];
        let right = &s[i + 1..];
        if left.parse::<i64>().is_ok() && right.parse::<i64>().is_ok() {
            return Some((left, right));
        }
    }
    None
}

fn parse_uniform(name: &str, args: &str) -> Result<Factor> {
    let p = parse_args_f64(args, 3, 3, "uniform")?;
    let low = p[0];
    let high = p[1];
    let count = p[2] as usize;
    if count < 2 {
        bail!("uniform count must be ≥ 2");
    }
    if high <= low {
        bail!("uniform: high must be > low");
    }
    let values: Vec<FactorValue> = (0..count)
        .map(|i| {
            let t = i as f64 / (count - 1) as f64;
            FactorValue::Float(low + t * (high - low))
        })
        .collect();
    Ok(Factor {
        name: name.into(),
        spec: String::new(),
        kind: FactorKind::Uniform { low, high },
        values,
    })
}

fn parse_normal(name: &str, args: &str) -> Result<Factor> {
    let p = parse_args_f64(args, 4, 5, "normal")?;
    let low = p[0];
    let high = p[1];
    let count = p[2] as usize;
    let std = p[3];
    if count < 2 {
        bail!("normal count must be ≥ 2");
    }
    if high <= low {
        bail!("normal: high must be > low");
    }
    if std <= 0.0 {
        bail!("normal: std must be > 0");
    }
    let mean = (low + high) / 2.0;
    let dist = Normal::new(mean, std).context("invalid normal parameters")?;
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let q = (i as f64 + 0.5) / count as f64;
        let mut x = dist.inverse_cdf(q);
        if x < low {
            x = low;
        }
        if x > high {
            x = high;
        }
        values.push(FactorValue::Float(x));
    }
    Ok(Factor {
        name: name.into(),
        spec: String::new(),
        kind: FactorKind::Normal {
            low,
            high,
            mean,
            std,
        },
        values,
    })
}

fn parse_log_low(name: &str, args: &str) -> Result<Factor> {
    let p = parse_args_f64(args, 3, 4, "logLow")?;
    let low = p[0];
    let high = p[1];
    let count = p[2] as usize;
    if count < 2 {
        bail!("logLow count must be ≥ 2");
    }
    if low <= 0.0 || high <= low {
        bail!("logLow: require 0 < low < high");
    }
    let ratio = high / low;
    let values: Vec<FactorValue> = (0..count)
        .map(|i| {
            let t = i as f64 / (count - 1) as f64;
            FactorValue::Float(low * ratio.powf(t))
        })
        .collect();
    Ok(Factor {
        name: name.into(),
        spec: String::new(),
        kind: FactorKind::LogLow { low, high },
        values,
    })
}

fn parse_log_high(name: &str, args: &str) -> Result<Factor> {
    let p = parse_args_f64(args, 3, 4, "logHigh")?;
    let low = p[0];
    let high = p[1];
    let count = p[2] as usize;
    if count < 2 {
        bail!("logHigh count must be ≥ 2");
    }
    if low <= 0.0 || high <= low {
        bail!("logHigh: require 0 < low < high");
    }
    let ratio = high / low;
    let values: Vec<FactorValue> = (0..count)
        .map(|i| {
            let t = i as f64 / (count - 1) as f64;
            FactorValue::Float(high + low - low * ratio.powf(1.0 - t))
        })
        .collect();
    Ok(Factor {
        name: name.into(),
        spec: String::new(),
        kind: FactorKind::LogHigh { low, high },
        values,
    })
}

fn parse_args_f64(args: &str, min: usize, max: usize, kind: &str) -> Result<Vec<f64>> {
    let parts: Vec<&str> = args.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    if parts.len() < min || parts.len() > max {
        bail!(
            "{} expects {}..{} args, got {}",
            kind,
            min,
            max,
            parts.len()
        );
    }
    parts
        .iter()
        .enumerate()
        .map(|(i, p)| {
            p.parse::<f64>()
                .with_context(|| format!("{} arg {}: '{}'", kind, i + 1, p))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discrete_range() {
        let f = Factor::parse("x", "d[1,2-4,9]").unwrap();
        assert_eq!(f.values.len(), 5);
        assert_eq!(f.values[0], FactorValue::Int(1));
        assert_eq!(f.values[1], FactorValue::Int(2));
        assert_eq!(f.values[4], FactorValue::Int(9));
    }

    #[test]
    fn discrete_negative() {
        let f = Factor::parse("x", "d[-5--3,0]").unwrap();
        assert_eq!(f.values.len(), 4);
        assert_eq!(f.values[0], FactorValue::Int(-5));
        assert_eq!(f.values[3], FactorValue::Int(0));
    }

    #[test]
    fn discrete_strings() {
        let f = Factor::parse("mode", r#"d["off","soft","aggressive"]"#).unwrap();
        assert_eq!(f.values.len(), 3);
    }

    #[test]
    fn uniform_spacing() {
        let f = Factor::parse("x", "uniform[0, 10, 5]").unwrap();
        assert_eq!(f.values[0].as_f64().unwrap(), 0.0);
        assert!((f.values[2].as_f64().unwrap() - 5.0).abs() < 1e-9);
        assert!((f.values[4].as_f64().unwrap() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn log_low_geometric() {
        let f = Factor::parse("x", "logLow[1, 1000, 4]").unwrap();
        let v: Vec<f64> = f.values.iter().map(|x| x.as_f64().unwrap()).collect();
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 10.0).abs() < 1e-6);
        assert!((v[2] - 100.0).abs() < 1e-6);
        assert!((v[3] - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn log_high_endpoints() {
        let f = Factor::parse("x", "logHigh[1, 1000, 4]").unwrap();
        let v: Vec<f64> = f.values.iter().map(|x| x.as_f64().unwrap()).collect();
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[3] - 1000.0).abs() < 1e-6);
        // dense at high end → spacing at low end > spacing at high end
        assert!((v[1] - v[0]) > (v[3] - v[2]));
    }
}
