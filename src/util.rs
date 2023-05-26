#[inline(always)]
pub fn weighted_average(frac: f64, a: f64, b: f64) -> f64 {
    frac * a + (1.0-frac) * b
}
