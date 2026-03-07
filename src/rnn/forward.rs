use std::ops::Add;

use crate::rnn::RecurrentNeuralNetwork;
use ndarray::Array1;

impl RecurrentNeuralNetwork {
    pub fn forward(&self, token: usize, hidden: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let x = self.embedding_table.row(token);
        let x_weights = x.dot(&self.input_weights);
        let h_weights = hidden.dot(&self.hidden_weights);
        let x_h_weights = x_weights.add(h_weights).add(&self.hidden_bias);
        let new_hidden = x_h_weights.mapv(|y| y.tanh());

        let y = new_hidden.dot(&self.output_weights).add(&self.output_bias);

        // Softmax - calculate probablities. exp(x_i) / sum(exp(x_j) for all j)
        let max = y.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let element_exp = y.mapv(|x| (x - max).exp());
        let element_sum = element_exp.sum();
        let probabilities = element_exp.mapv(|x| x / element_sum);

        (probabilities, new_hidden)
    }
}
