use std::time::Instant;

// use candle_core::{utils::{cuda_is_available, metal_is_available}, Device, Tensor};
use candle_core_opt::{utils::{cuda_is_available, metal_is_available}, Device, Tensor};

mod add;
mod basic;

pub use basic::*;

pub fn select_device() -> Result<Device, String> {
    if metal_is_available() {
        Ok(Device::new_metal(0).map_err(|e| e.to_string())?)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0).map_err(|e| e.to_string())?)
    } else {
        Ok(Device::Cpu)
    }
}

pub fn simple_timer(dev: &Device, f: fn(d: &Device) -> ()) -> f32 {
    let s = Instant::now();
    f(dev);
    dev.synchronize().unwrap();
    (Instant::now() - s).as_secs_f32()
}

pub fn simple_timer_with_tensor2(dev: &Device, f: fn(d: &Device, t1: &Tensor, t2: &Tensor) -> (), t1: &Tensor, t2: &Tensor) -> f32 {
    let s = Instant::now();
    f(dev, t1, t2);
    dev.synchronize().unwrap();
    (Instant::now() - s).as_secs_f32()
}