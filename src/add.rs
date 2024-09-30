use candle_core_opt::{Device, Tensor};

const N: usize = 64;

pub fn add(dev: &Device, t1: &Tensor, t2: &Tensor) {
    for _ in 0 .. N {
        let _ = t1.add(t2);
    }
}

#[cfg(test)]
mod tests {
    use crate::{select_device, simple_timer_with_tensor2};

    use super::*;

    #[test]
    fn test_broadcast_add() {
        let tn1 = Tensor::rand(0f32, 1., (32, 630, 12, 32), &Device::Cpu).unwrap();
        let tn2 = Tensor::rand(0f32, 1., (32, 630, 12, 32), &Device::Cpu).unwrap();

        let t = simple_timer_with_tensor2(&Device::Cpu, add, &tn1, &tn2);
        println!("Broadcase add Cpu: {t}s {}s/iter", t/ N as f32);

        let dev = select_device().unwrap();
        let tn1 = Tensor::rand(0f32, 1., (32, 630, 12, 32), &dev).unwrap();
        let tn2 = Tensor::rand(0f32, 1., (32, 630, 12, 32), &dev).unwrap();

        let t = simple_timer_with_tensor2(&dev, add, &tn1, &tn2);
        println!("Broadcase add Gpu: {t}s {}s/iter", t/ N as f32);
    }
}