use candle_core::Device;
use candle_core_opt::Device as DeviceOpt;

pub(crate) mod ones;

pub(crate) trait BenchDevice {
    fn sync(&self);

    fn bench_name<S: Into<String>>(&self, name: S) -> String;
}

impl BenchDevice for Device {
    fn sync(&self) {
        match self {
            Device::Cpu => {},
            Device::Cuda(_) => unimplemented!(),
            Device::Metal(device) => {
                device.wait_until_completed().unwrap();
            }
        }
    }

    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        match self {
            Device::Cpu =>  format!("cpu_{}", name.into()),
            Device::Cuda(_) => format!("cuda_{}", name.into()),
            Device::Metal(_) => format!("metal_{}", name.into()),
        }
    }
}

impl BenchDevice for DeviceOpt {
    fn sync(&self) {
        match self {
            DeviceOpt::Cpu => {},
            DeviceOpt::Cuda(_) => unimplemented!(),
            DeviceOpt::Metal(device) => {
                device.wait_until_completed().unwrap();
            }
        }
    }

    fn bench_name<S: Into<String>>(&self, name: S) -> String {
        match self {
            DeviceOpt::Cpu =>  format!("cpu_{}", name.into()),
            DeviceOpt::Cuda(_) => format!("cuda_{}", name.into()),
            DeviceOpt::Metal(_) => format!("metal_{}", name.into()),
        }
    }
}

// struct BenchDeviceHandler {
//     devices: Vec<Device>,
// }

// impl BenchDeviceHandler {
//     pub fn new() -> Result<Self> {
//         let mut devices = Vec::new();
//         // if cfg!(feature = "metal") {
//         devices.push(Device::new_metal(0)?);
//         // } else if cfg!(feature = "cuda") {
//         //     devices.push(Device::new_cuda(0)?);
//         // }
//         devices.push(Device::Cpu);
//         Ok(Self { devices })
//     }
// }