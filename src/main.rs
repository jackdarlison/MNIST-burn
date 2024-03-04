
mod models;
mod data;
mod training;
use burn::optim::AdamConfig;
use burn::backend::{Autodiff, Wgpu, wgpu::AutoGraphicsApi};
use crate::models::conv::ConvModelConfig;
use crate::models::linear::LinearModelConfig;

fn main() {
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    crate::training::train::<MyAutodiffBackend>(
        "out/conv/",
        crate::training::TrainingConfig::new(ConvModelConfig::new(10, 256), AdamConfig::new()),
        // crate::training::TrainingConfig::new(LinearModelConfig::new(28*28, 128, 128, 10), AdamConfig::new()),
        device,
    );
}
