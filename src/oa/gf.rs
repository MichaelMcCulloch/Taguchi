//! Finite field arithmetic for GF(p^k) where p prime, k ≥ 1.
//!
//! For k=1 this is just modular arithmetic. For k≥2 we represent elements as
//! polynomials of degree < k over GF(p), encoded base-p (element index i has
//! polynomial coefficients i_0 + i_1·p + …). Multiplication reduces modulo a
//! hardcoded primitive irreducible polynomial.

use anyhow::{Result, anyhow};

#[derive(Debug, Clone)]
pub struct Gf {
    pub p: usize,
    pub k: usize,
    pub q: usize,
    // Multiplication and addition tables (q × q). Indexed [a * q + b].
    add: Vec<usize>,
    mul: Vec<usize>,
}

impl Gf {
    pub fn new(q: usize) -> Result<Self> {
        let (p, k, irr) = factor_prime_power(q)
            .ok_or_else(|| anyhow!("{} is not a prime power", q))?;
        if k == 1 {
            return Ok(Gf::prime(p));
        }
        Ok(Gf::extension(p, k, irr))
    }

    fn prime(p: usize) -> Self {
        let q = p;
        let mut add = vec![0; q * q];
        let mut mul = vec![0; q * q];
        for a in 0..q {
            for b in 0..q {
                add[a * q + b] = (a + b) % p;
                mul[a * q + b] = (a * b) % p;
            }
        }
        Gf { p, k: 1, q, add, mul }
    }

    /// Build GF(p^k) where `irr` is a length-(k+1) coefficient list of the
    /// monic irreducible polynomial (constant term first, leading 1 last).
    fn extension(p: usize, k: usize, irr: &[usize]) -> Self {
        let q = (p as u64).pow(k as u32) as usize;
        let mut add = vec![0; q * q];
        let mut mul = vec![0; q * q];

        for a in 0..q {
            let pa = to_poly(a, p, k);
            for b in 0..q {
                let pb = to_poly(b, p, k);

                // Addition: coefficient-wise mod p.
                let sum_poly: Vec<usize> = (0..k).map(|i| (pa[i] + pb[i]) % p).collect();
                add[a * q + b] = from_poly(&sum_poly, p);

                // Multiplication: polynomial mult, reduce mod irr.
                let mut prod = vec![0usize; 2 * k - 1];
                for i in 0..k {
                    for j in 0..k {
                        prod[i + j] = (prod[i + j] + pa[i] * pb[j]) % p;
                    }
                }
                // Reduce: for each high-degree term from 2k-2 down to k, subtract
                // its leading coefficient times the irreducible polynomial.
                for deg in (k..2 * k - 1).rev() {
                    let coef = prod[deg];
                    if coef == 0 {
                        continue;
                    }
                    // x^deg ≡ -(irr_low_coeffs · x^(deg-k)) since irr is monic.
                    for i in 0..k {
                        let neg = (p - irr[i] % p) % p;
                        prod[deg - k + i] = (prod[deg - k + i] + coef * neg) % p;
                    }
                    prod[deg] = 0;
                }
                let reduced: Vec<usize> = prod[..k].to_vec();
                mul[a * q + b] = from_poly(&reduced, p);
            }
        }
        Gf { p, k, q, add, mul }
    }

    #[inline]
    pub fn add(&self, a: usize, b: usize) -> usize {
        self.add[a * self.q + b]
    }
    #[inline]
    pub fn mul(&self, a: usize, b: usize) -> usize {
        self.mul[a * self.q + b]
    }
}

fn to_poly(mut n: usize, p: usize, k: usize) -> Vec<usize> {
    let mut out = vec![0usize; k];
    for i in 0..k {
        out[i] = n % p;
        n /= p;
    }
    out
}

fn from_poly(poly: &[usize], p: usize) -> usize {
    let mut n = 0usize;
    for &c in poly.iter().rev() {
        n = n * p + c;
    }
    n
}

/// If q = p^k for prime p and k ≥ 1, return (p, k, monic_irreducible_coeffs).
/// Coefficients are constant-term-first; leading 1 implicit (not in the list).
fn factor_prime_power(q: usize) -> Option<(usize, usize, &'static [usize])> {
    if q < 2 {
        return None;
    }
    for p in [2usize, 3, 5, 7, 11, 13, 17, 19, 23] {
        let mut n = q;
        let mut k = 0;
        while n % p == 0 {
            n /= p;
            k += 1;
        }
        if n == 1 && k >= 1 {
            return match (p, k) {
                // k=1: irreducible is just x (constant 0), unused.
                (_, 1) => Some((p, 1, &[])),
                // GF(4) = GF(2)[x]/(x^2 + x + 1)
                (2, 2) => Some((2, 2, &[1, 1])),
                // GF(8) = GF(2)[x]/(x^3 + x + 1)
                (2, 3) => Some((2, 3, &[1, 1, 0])),
                // GF(16) = GF(2)[x]/(x^4 + x + 1)
                (2, 4) => Some((2, 4, &[1, 1, 0, 0])),
                // GF(32) = GF(2)[x]/(x^5 + x^2 + 1)
                (2, 5) => Some((2, 5, &[1, 0, 1, 0, 0])),
                // GF(9) = GF(3)[x]/(x^2 + 1)  (since -1 is non-residue mod 3)
                (3, 2) => Some((3, 2, &[1, 0])),
                // GF(27) = GF(3)[x]/(x^3 + 2x + 1)
                (3, 3) => Some((3, 3, &[1, 2, 0])),
                // GF(25) = GF(5)[x]/(x^2 + 2)
                (5, 2) => Some((5, 2, &[2, 0])),
                // GF(49) = GF(7)[x]/(x^2 + 1)  (-1 is non-residue mod 7)
                (7, 2) => Some((7, 2, &[1, 0])),
                _ => None,
            };
        }
    }
    None
}

pub fn is_prime_power(q: usize) -> bool {
    factor_prime_power(q).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_field(gf: &Gf) {
        let q = gf.q;
        // 0 is additive identity, 1 is multiplicative identity.
        for a in 0..q {
            assert_eq!(gf.add(a, 0), a, "a+0 != a at a={}", a);
            assert_eq!(gf.mul(a, 1), a, "a*1 != a at a={}", a);
            assert_eq!(gf.mul(a, 0), 0, "a*0 != 0 at a={}", a);
        }
        // Commutativity.
        for a in 0..q {
            for b in 0..q {
                assert_eq!(gf.add(a, b), gf.add(b, a));
                assert_eq!(gf.mul(a, b), gf.mul(b, a));
            }
        }
        // Every nonzero has a multiplicative inverse.
        for a in 1..q {
            let mut found = false;
            for b in 1..q {
                if gf.mul(a, b) == 1 {
                    found = true;
                    break;
                }
            }
            assert!(found, "no inverse for {} in GF({})", a, q);
        }
        // Distributivity (spot check).
        for a in 0..q {
            for b in 0..q {
                for c in 0..q {
                    let left = gf.mul(a, gf.add(b, c));
                    let right = gf.add(gf.mul(a, b), gf.mul(a, c));
                    assert_eq!(left, right);
                }
            }
        }
    }

    #[test]
    fn gf3_prime() {
        check_field(&Gf::new(3).unwrap());
    }

    #[test]
    fn gf4() {
        check_field(&Gf::new(4).unwrap());
    }

    #[test]
    fn gf8() {
        check_field(&Gf::new(8).unwrap());
    }

    #[test]
    fn gf9() {
        check_field(&Gf::new(9).unwrap());
    }

    #[test]
    fn gf16() {
        check_field(&Gf::new(16).unwrap());
    }

    #[test]
    fn gf25() {
        check_field(&Gf::new(25).unwrap());
    }

    #[test]
    fn gf27() {
        check_field(&Gf::new(27).unwrap());
    }

    #[test]
    fn rejects_non_prime_power() {
        assert!(Gf::new(6).is_err());
        assert!(Gf::new(10).is_err());
        assert!(Gf::new(12).is_err());
    }
}
