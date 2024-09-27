use candle_core::{Tensor, DType};

use crate::select_device;

pub fn create_zero_gpu() {
    let n = 10000;
    let dev = select_device().unwrap();
    for _ in 0 .. n {
        let _ = Tensor::zeros((1024, 8192), DType::F32, &dev);
    }
}

pub fn create_ones_gpu() {
    let n = 10000;
    let dev = select_device().unwrap();
    for _ in 0 .. n {
        let _ = Tensor::ones((1024, 8192), DType::F32, &dev);
    }
}

#[cfg(test)]
mod tests {
    use crate::simple_timer;

    use super::*;

    #[test]
    fn test_create_tensor() {
        let t = simple_timer(create_zero_gpu);
        println!("Zeroes: {t}s {}s/iter", t/ 10000. );

        let t = simple_timer(create_ones_gpu);
        println!("Ones: {t}s {}s/iter", t/ 10000. );
    }
}