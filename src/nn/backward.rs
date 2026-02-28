use crate::nn::NeuralNetwork;
use ndarray::{s, Array1, Axis};

impl NeuralNetwork {
    pub fn backward(
        &mut self,
        context: [usize; 2],
        target: usize,
        probs: &Array1<f32>,
        hidden_layer: &Array1<f32>,
        input: &Array1<f32>,
        learning_rate: f32,
    ) -> f32 {
        // Compute d_output:
        let loss = -probs[target].ln();
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

        // Get the first half and second half of the input, since forward joined them together
        let embedding_dim = self.embedding_table.ncols();
        let first_half = d_input.slice(s![0..embedding_dim]);
        let second_half = d_input.slice(s![embedding_dim..]);

        // Update embedding table
        self.embedding_table
            .row_mut(context[0])
            .scaled_add(-learning_rate, &first_half);
        self.embedding_table
            .row_mut(context[1])
            .scaled_add(-learning_rate, &second_half);

        // Update all weights
        self.weights_2.scaled_add(-learning_rate, &d_weights_2);
        self.bias_2.scaled_add(-learning_rate, &d_bias_2);
        self.weights_1.scaled_add(-learning_rate, &d_weights_1);
        self.bias_1.scaled_add(-learning_rate, &d_bias_1);

        return loss;
    }
}
