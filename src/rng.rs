//! Reproducible splitmix64 and Fisher–Yates randomization.
pub struct Rng {
    state: u64,
}
impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0, "random range must be nonempty");
        let threshold = n.wrapping_neg() % n;
        loop {
            let r = self.next_u64();
            if r >= threshold {
                return r % n;
            }
        }
    }
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            v.swap(i, self.below(i as u64 + 1) as usize);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn reference_splitmix64() {
        let mut rng = Rng::new(42);
        assert_eq!(rng.next_u64(), 0xbdd732262feb6e95);
        assert_eq!(rng.next_u64(), 0x28efe333b266f103);
        assert_eq!(rng.next_u64(), 0x47526757130f9f52);
    }
    #[test]
    fn shuffle_and_ranges() {
        let mut a: Vec<_> = (0..30).collect();
        let mut b = a.clone();
        Rng::new(42).shuffle(&mut a);
        Rng::new(42).shuffle(&mut b);
        assert_eq!(a, b);
        assert_ne!(a, (0..30).collect::<Vec<_>>());
        a.sort_unstable();
        assert_eq!(a, (0..30).collect::<Vec<_>>());
        let mut rng = Rng::new(0);
        for n in [1, 2, 3, u64::MAX / 2 + 2, u64::MAX] {
            for _ in 0..100 {
                assert!(rng.below(n) < n);
            }
        }
        rng.shuffle::<u8>(&mut []);
        rng.shuffle(&mut [1]);
    }
    #[test]
    #[should_panic(expected = "random range must be nonempty")]
    fn empty_range() {
        Rng::new(0).below(0);
    }
}
