use crate::nn::NeuralNetwork;
use ndarray::{Array1, Axis};
use std::ops::Add;

impl NeuralNetwork {
    pub fn forward(&self, context: [usize; 2]) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // Embedding Layer:
        // Get flat vector for both context word indexes to create the input
        let embed_1 = self.embedding_table.row(context[0]);
        let embed_2 = self.embedding_table.row(context[1]);
        let input = ndarray::concatenate(Axis(0), &[embed_1, embed_2]).unwrap();
        let output_input = input.clone();

        // Hidden Layer:
        // Create the hidden vector by multiplying the input array by the first weights
        let hidden = input.dot(&self.weights_1);
        let hidden_bias = hidden.add(&self.bias_1);
        let hidden_layer = hidden_bias.mapv(|x| x.max(0.0));
        let output_hidden_layer = hidden_layer.clone();

        // Output Layer:
        // Multiply the hidden bias relu result by the second set of weights
        let hidden_dot_by_weights_2 = hidden_layer.dot(&self.weights_2);
        let output = hidden_dot_by_weights_2.add(&self.bias_2);

        // Softmax - calculate probablities. exp(x_i) / sum(exp(x_j) for all j)
        let element_exp = output.mapv(|x| x.exp());
        let element_sum = element_exp.sum();
        let output_exp_sum = element_exp.mapv(|x| x / element_sum);

        // Finally, return the exp_sum, hidden_layer and input.
        return (output_exp_sum, output_hidden_layer, output_input);
    }
}
