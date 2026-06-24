/// SplitMix64 — extremely fast non-cryptographic PRNG, suitable for the
/// high-frequency RNG calls in the TM inner training loop.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    /// Create a new RNG from `seed`, mixing in the golden ratio constant to avoid state 0.
    pub fn new(seed: u64) -> Self {
        // Avoid degenerate state 0 by mixing in the golden constant.
        Rng(seed.wrapping_add(0x9E3779B97F4A7C15))
    }

    /// Advance the state and return the next pseudorandom 64-bit value.
    #[inline(always)]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Uniform float in [0, 1).
    #[inline(always)]
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Uniform usize in [0, n). Panics if n == 0.
    #[inline(always)]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Raw internal state word. Used by serialisation to restore the exact
    /// stream position (so a reloaded model resumes identically).
    pub(crate) fn state(&self) -> u64 {
        self.0
    }

    /// Reconstruct an RNG from a raw state word produced by [`Rng::state`].
    /// Unlike [`Rng::new`], this does **not** mix the value, so the stream
    /// continues exactly where it left off.
    pub(crate) fn from_state(state: u64) -> Self {
        Rng(state)
    }
}
