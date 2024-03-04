use burn::{config::Config, module::Module, nn::{conv::{Conv2d, Conv2dConfig}, loss::CrossEntropyLossConfig, pool::{MaxPool2d, MaxPool2dConfig}, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d, ReLU}, tensor::{backend::{AutodiffBackend, Backend}, Int, Tensor}, train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep}};

use crate::data::MNISTBatch;

#[derive(Module, Debug)]
pub struct ConvModel<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    pool1: MaxPool2d,
    conv3: Conv2d<B>,
    conv4: Conv2d<B>,
    pool2: MaxPool2d,
    dropout: Dropout,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: ReLU,
}

#[derive(Config, Debug)]
pub struct ConvModelConfig {
    pub output_size: usize,
    pub hidden_size: usize,
    #[config(default = "0.5")]
    pub dropout: f64, 
}

impl ConvModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> ConvModel<B> {
        ConvModel {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv2: Conv2dConfig::new([8, 8], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            pool1: MaxPool2dConfig::new([2, 2]).init(),
            conv3: Conv2dConfig::new([8, 16], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            conv4: Conv2dConfig::new([16, 16], [3, 3]).with_padding(PaddingConfig2d::Same).init(device),
            pool2: MaxPool2dConfig::new([2, 2]).init(),
            activation: ReLU::new(),
            linear1: LinearConfig::new(16 * 7 * 7, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.output_size).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> ConvModel<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, height, width] = images.dims();

        // Create channel
        let images = images.reshape([batch_size, 1, height, width]);

        let x = self.conv1.forward(images.clone());
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv2.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool1.forward(x);

        let x = self.conv3.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.conv4.forward(x);
        let x = self.dropout.forward(x);
        let x = self.activation.forward(x);

        let x = self.pool2.forward(x);

        let x = x.reshape([batch_size, 16 * 7 * 7]);

        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);

        let x = self.linear2.forward(x);
        x
    }
}

impl<B: Backend> ConvModel<B> {
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

impl<B: AutodiffBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for ConvModel<B> {
    fn step(&self, batch: MNISTBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for ConvModel<B> {
    fn step(&self, batch: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}