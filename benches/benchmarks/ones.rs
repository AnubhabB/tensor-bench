use std::time::Instant;

use crate::benchmarks::BenchDevice;
use candle_core::{DType, Device, Tensor};
use candle_core_opt::{Tensor as TensorOpt, Device as DeviceOpt, DType as DTypeOpt};
use criterion::{black_box, criterion_group, Criterion, Throughput};

fn run_orig(dev: &Device, dtype: DType, shape: (usize, usize, usize)) {
    Tensor::ones(shape, dtype, dev).unwrap();
}

fn run_opt(dev: &DeviceOpt, dtype: DTypeOpt, shape: (usize, usize, usize)) {
    TensorOpt::ones(shape, dtype, dev).unwrap();
}

fn run_ones_orig(c: &mut Criterion, device: &Device, dtype: DType, name: &str) {
    let b = 1;
    let m = 1024;
    let k = 1024;

    let flops = b * m * k * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b_| {
        b_.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_orig(black_box(device), black_box(dtype), black_box((b, m, k)));
            }
            device.sync();
            start.elapsed()
        })
    });
    group.finish();
}

fn run_ones_opt(c: &mut Criterion, device: &DeviceOpt, dtype: DTypeOpt, name: &str) {
    let b = 1;
    let m = 1024;
    let k = 1024;

    let flops = b * m * k * dtype.size_in_bytes();

    let mut group = c.benchmark_group(device.bench_name(name));
    group.throughput(Throughput::Bytes(flops as u64));
    group.bench_function("iter", move |b_| {
        b_.iter_custom(|iters| {
            let start = Instant::now();
            for _i in 0..iters {
                run_opt(black_box(device), black_box(dtype), black_box((b, m, k)));
            }
            device.sync();
            start.elapsed()
        })
    });
    group.finish();
}

fn criterion_benchmark(c: &mut Criterion) {
    for d in [Device::Cpu, Device::new_metal(0).unwrap()] {
        run_ones_orig(c, &d, DType::F32, "orig_ones_f32");
        run_ones_orig(c, &d, DType::BF16, "orig_ones_bf16");
        run_ones_orig(c, &d, DType::U32, "orig_ones_u32");
        run_ones_orig(c, &d, DType::F16, "orig_ones_f16");
        run_ones_orig(c, &d, DType::U8, "orig_ones_u8");
        run_ones_orig(c, &d, DType::I64, "orig_ones_i64");
    }

    let d = DeviceOpt::new_metal(0).unwrap();
    run_ones_opt(c, &d, DTypeOpt::F32, "opt_ones_f32");
    run_ones_opt(c, &d, DTypeOpt::BF16, "opt_ones_bf16");
    run_ones_opt(c, &d, DTypeOpt::U32, "opt_ones_u32");
    run_ones_opt(c, &d, DTypeOpt::F16, "opt_ones_f16");
    run_ones_opt(c, &d, DTypeOpt::U8, "opt_ones_u8");
    run_ones_opt(c, &d, DTypeOpt::I64, "opt_ones_i64");
}

criterion_group!(benches, criterion_benchmark);