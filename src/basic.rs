use candle_core::{DType, Device, Tensor};

const N: usize = 64;

pub fn create_zero_gpu(dev: &Device) {
    for _ in 0 .. N {
        let _ = Tensor::zeros((1024, 8192), DType::F32, dev);
    }
}

pub fn create_ones_gpu(dev: &Device) {
    for _ in 0 .. N {
        let _ = Tensor::ones((1024, 8192), DType::F32, dev);
    }
}

pub fn create_full_gpu(dev: &Device) {
    for i in 0 .. N {
        let _ = Tensor::full( i as f32, (1024, 8192), dev);
    }
}

#[cfg(test)]
mod tests {
    use crate::{select_device, simple_timer};

    use super::*;

    #[test]
    fn test_create_tensor() {
        let dev = select_device().unwrap();
        let t = simple_timer(&dev, create_zero_gpu);
        println!("Zeroes: {t}s {}s/iter", t/ N as f32);

        let t = simple_timer(&dev, create_ones_gpu);
        println!("Ones: {t}s {}s/iter", t/ N as f32);

        let t = simple_timer(&dev, create_full_gpu);
        println!("Full: {t}s {}s/iter", t/ N as f32);
    }
}