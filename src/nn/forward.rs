use crate::nn::NeuralNetwork;
use ndarray::{Array1, ArrayBase, Axis, Dim, ViewRepr};
use std::ops::Add;

impl NeuralNetwork {
    pub fn forward(&self, context: &Vec<usize>) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // Embedding Layer:
        // Get flat vector for both context word indexes to create the input
        let mut arrays: Vec<ArrayBase<ViewRepr<&f32>, Dim<[usize; 1]>>> = Vec::new();
        for i in 0..context.len() {
            arrays.push(self.embedding_table.row(context[i]));
        }
        let input = ndarray::concatenate(Axis(0), &arrays).unwrap();
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
        let max = output.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let element_exp = output.mapv(|x| (x - max).exp());
        let element_sum = element_exp.sum();
        let probabilities = element_exp.mapv(|x| x / element_sum);

        // Finally, return the probabilities, hidden_layer and input.
        return (probabilities, output_hidden_layer, output_input);
    }
}
