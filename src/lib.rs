pub mod ch2;
pub mod ch3;

pub mod data;
pub mod neuron;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ch2_run() {
        ch2::run();
    }

    #[test]
    fn ch3_run() {
        ch3::run();
    }
}
