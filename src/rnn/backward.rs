use crate::rnn::{gradients::Gradients, RecurrentNeuralNetwork};
use ndarray::{Array1, Array2, Axis};
use std::collections::HashMap;

impl RecurrentNeuralNetwork {
    pub fn backward(
        &self,
        tokens: &[usize],
        targets: &[usize],
        hidden: &Vec<Array1<f32>>,
        probs: &Vec<Array1<f32>>,
    ) -> Gradients {
        let mut d_input_weights = Array2::zeros(self.input_weights.dim());
        let mut d_hidden_weights = Array2::zeros(self.hidden_weights.dim());
        let mut d_hidden_bias = Array1::zeros(self.hidden_bias.dim());
        let mut d_output_weights = Array2::zeros(self.output_weights.dim());
        let mut d_output_bias = Array1::zeros(self.output_bias.dim());
        let mut d_embeddings: Vec<(usize, Array1<f32>)> = Vec::new();
        let mut d_hidden_next = Array1::<f32>::zeros(self.hidden_bias.len());

        for t in (0..tokens.len()).rev() {
            // Step 1 — Output gradient:
            //  d_output = probs[t].clone()
            //  d_output[targets[t]] -= 1.0
            //  Same as your MLP — subtract 1 from the target token's probability.
            let mut d_output = probs[t].clone();
            d_output[targets[t]] -= 1.0;

            //  Step 2 — Output weight gradients:
            //  d_output_weights += hidden[t+1] outer_product d_output
            //  d_output_bias += d_output
            //  Note hidden[t+1] not hidden[t] — because hiddens[0] is the initial zero state, so the hidden state after processing
            //  token t is at index t+1.

            d_output_weights += &hidden[t + 1]
                .view()
                .insert_axis(Axis(1))
                .dot(&d_output.view().insert_axis(Axis(0)));
            d_output_bias += &d_output;

            //  Step 3 — Gradient flowing back into hidden state:
            //  d_hidden = d_output dot output_weights.T + d_hidden_next dot hidden_weights.T
            //  This combines two sources of gradient — from the output at this time step, and from the future time step via
            //  d_hidden_next.
            let mut d_hidden = d_output.dot(&self.output_weights.t())
                + d_hidden_next.dot(&self.hidden_weights.t());

            //  Step 4 — tanh gradient:
            //  d_hidden = d_hidden * (1 - hidden[t+1]^2)
            //  The derivative of tanh is 1 - tanh(x)^2, and since hidden[t+1] is already the tanh output, you can use it directly.
            let tanh_grad = hidden[t + 1].mapv(|x| 1.0 - x * x);
            d_hidden *= &tanh_grad;

            //  Step 5 — Input/hidden weight gradients:
            //  d_input_weights += x[t] outer_product d_hidden
            //  d_hidden_weights += hidden[t] outer_product d_hidden
            //  d_hidden_bias += d_hidden
            //  Where x[t] is the embedding for tokens[t].
            let x = self.embedding_table.row(tokens[t]);
            d_input_weights += &x
                .view()
                .insert_axis(Axis(1))
                .dot(&d_hidden.view().insert_axis(Axis(0)));
            d_hidden_weights += &hidden[t]
                .view()
                .insert_axis(Axis(1))
                .dot(&d_hidden.view().insert_axis(Axis(0)));
            d_hidden_bias += &d_hidden;

            //  Step 6 — Embedding gradient:
            //  d_embeddings.push((tokens[t], d_hidden dot input_weights.T))
            let d_embedding = d_hidden.dot(&self.input_weights.t());
            d_embeddings.push((tokens[t], d_embedding));

            //  Step 7 — Pass gradient to previous time step:
            //  d_hidden_next = d_hidden
            d_hidden_next = d_hidden;
        }

        return Gradients {
            input_weights: d_input_weights,
            hidden_weights: d_hidden_weights,
            hidden_bias: d_hidden_bias,
            output_weights: d_output_weights,
            output_bias: d_output_bias,
            embedding_table: d_embeddings,
        };
    }

    pub fn apply_gradients(&mut self, gradients: Gradients, learning_rate: f32) {
        // Embedding table — accumulate and average gradients per token
        let mut embedding_map: HashMap<usize, (Array1<f32>, usize)> = HashMap::new();

        for (token_index, grad) in &gradients.embedding_table {
            if embedding_map.contains_key(token_index) {
                let value = embedding_map.get_mut(token_index).unwrap();
                value.0 += grad;
                value.1 += 1;
            } else {
                embedding_map.insert(*token_index, (grad.clone(), 1));
            }
        }

        for (token_index, (summed_grad, count)) in &embedding_map {
            let averaged_grad = summed_grad / *count as f32;
            self.embedding_table
                .row_mut(*token_index)
                .scaled_add(-learning_rate, &averaged_grad);
        }

        // Weight matrices and biases
        self.input_weights
            .scaled_add(-learning_rate, &gradients.input_weights);
        self.hidden_weights
            .scaled_add(-learning_rate, &gradients.hidden_weights);
        self.hidden_bias
            .scaled_add(-learning_rate, &gradients.hidden_bias);
        self.output_weights
            .scaled_add(-learning_rate, &gradients.output_weights);
        self.output_bias
            .scaled_add(-learning_rate, &gradients.output_bias);
    }
}
