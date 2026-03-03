use crate::nn::gradients::Gradients;
use crate::nn::NeuralNetwork;
use ndarray::{s, Array1, Axis};
use std::collections::HashMap;

impl NeuralNetwork {
    pub fn compute_gradients(
        &self,
        context: &Vec<usize>,
        target: usize,
        probs: &Array1<f32>,
        hidden_layer: &Array1<f32>,
        input: &Array1<f32>,
    ) -> Gradients {
        let mut d_output = probs.clone();
        d_output[target] -= 1.0;

        // Compute output weight 2 layer gradients
        let d_weights_2 = hidden_layer
            .view()
            .insert_axis(Axis(1))
            .dot(&d_output.view().insert_axis(Axis(0)));
        let d_bias_2 = d_output.clone();
        let d_hidden = d_output.dot(&self.weights_2.t());

        // The ReLU graident
        let d_hidden_relu = d_hidden * hidden_layer.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });

        // Compute output weight 1 layer gradients
        let d_weights_1 = input
            .view()
            .insert_axis(Axis(1))
            .dot(&d_hidden_relu.view().insert_axis(Axis(0)));
        let d_bias_1 = d_hidden_relu.clone();
        let d_input = d_hidden_relu.dot(&self.weights_1.t());

        let mut embedding_table: Vec<(usize, Array1<f32>)> = Vec::new();
        let embedding_dim = self.embedding_table.ncols();
        for i in 0..context.len() {
            let start = i * embedding_dim;
            let end = start + embedding_dim;
            let slice = d_input.slice(s![start..end]);
            embedding_table.push((context[i], slice.to_owned()));
        }

        return Gradients {
            weights_1: d_weights_1,
            bias_1: d_bias_1,
            weights_2: d_weights_2,
            bias_2: d_bias_2,
            embedding_table: embedding_table,
        };
    }

    pub fn apply_gradients(&mut self, gradients: Gradients, learning_rate: f32) {
        let mut embedding_map: HashMap<usize, (Array1<f32>, usize)> = HashMap::new();

        for (token_index, grad) in &gradients.embedding_table {
            if embedding_map.contains_key(&token_index) {
                let value = embedding_map.get_mut(&token_index).unwrap();
                value.0 += grad;
                value.1 += 1;
            } else {
                embedding_map.insert(token_index.clone(), (grad.clone(), 1));
            }
        }

        for (token_index, (summed_grad, count)) in &embedding_map {
            let averaged_grad = summed_grad / *count as f32;
            self.embedding_table
                .row_mut(token_index.clone())
                .scaled_add(-learning_rate, &averaged_grad);
        }

        self.weights_2
            .scaled_add(-learning_rate, &gradients.weights_2);
        self.bias_2.scaled_add(-learning_rate, &gradients.bias_2);
        self.weights_1
            .scaled_add(-learning_rate, &gradients.weights_1);
        self.bias_1.scaled_add(-learning_rate, &gradients.bias_1);
    }
}
