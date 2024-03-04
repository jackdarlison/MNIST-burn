use burn::{config::Config, module::Module, nn::{loss::CrossEntropyLossConfig, Linear, LinearConfig, ReLU}, tensor::{backend::{AutodiffBackend, Backend}, Int, Tensor}, train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep}};

use crate::data::MNISTBatch;


#[derive(Module, Debug)]
pub struct LinearModel<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    linear3: Linear<B>,
    activation: ReLU,
}

#[derive(Config, Debug)]
pub struct LinearModelConfig {
    pub input_size: usize,
    pub hidden_size1: usize,
    pub hidden_size2: usize,
    pub output_size: usize,
}

impl LinearModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinearModel<B> {
        LinearModel {
            linear1: LinearConfig::new(self.input_size, self.hidden_size1).init(device),
            linear2: LinearConfig::new(self.hidden_size1, self.hidden_size2).init(device),
            linear3: LinearConfig::new(self.hidden_size2, self.output_size).init(device),
            activation: ReLU::new(),
        }
    }
}

impl<B: Backend> LinearModel<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // flatten the images
        let images = images.reshape([batch_size, height * width]);

        let x = self.linear1.forward(images);
        let x = self.activation.forward(x);
        let x = self.linear2.forward(x);
        let x = self.activation.forward(x);
        let x = self.linear3.forward(x);
        x
    }
}

impl<B: Backend> LinearModel<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new().init(&output.device()).forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for LinearModel<B> {
    fn step(&self, batch: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for LinearModel<B> {
    fn step(&self, batch: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}