use crate::rnn::gradients::Gradients;
use crate::rnn::RecurrentNeuralNetwork;
use ndarray::Array1;
use rayon::prelude::*;
use std::io::{self, Write};
use std::time::Instant;

impl RecurrentNeuralNetwork {
    pub fn train(
        &mut self,
        training_data: Vec<(Vec<usize>, Vec<usize>)>,
        epocs: i32,
        learning_rate: f32,
        batch_size: usize,
        max_norm: f32,
    ) {
        for epoc in 0..epocs {
            let start = Instant::now();
            let system_time = chrono::Local::now().format("%H:%M:%S");
            print!("[{}]: Epoc: {} ", system_time.to_string(), epoc);
            io::stdout().flush().expect("Failed to flush.");

            let mut total_loss: f32 = 0.0;
            for batch in training_data.chunks(batch_size) {
                let all_gradients: Vec<(Gradients, f32)> = batch
                    .par_iter()
                    .map(|(tokens, targets)| {
                        let mut hidden_states = vec![Array1::zeros(self.hidden_bias.len())];
                        let mut probs_vec = Vec::new();

                        let mut loss = 0.0f32;
                        for (t, token) in tokens.iter().enumerate() {
                            let (probs, new_hidden) =
                                self.forward(*token, hidden_states.last().unwrap());
                            loss += -probs[targets[t]].ln();
                            probs_vec.push(probs);
                            hidden_states.push(new_hidden);
                        }

                        let gradients =
                            self.backward(&tokens, &targets, &hidden_states, &probs_vec);

                        (gradients, loss)
                    })
                    .collect();

                let (gradients_vec, losses): (Vec<Gradients>, Vec<f32>) =
                    all_gradients.into_iter().unzip();
                total_loss += losses.iter().sum::<f32>();

                let mut accumulated = gradients_vec
                    .into_iter()
                    .reduce(|mut a, b| {
                        a.add(&b);
                        a
                    })
                    .unwrap();

                let norm = accumulated.norm();
                if norm > max_norm {
                    accumulated.scale(max_norm / norm);
                }

                //accumulated.scale(1.0 / batch_size as f32); // average

                self.apply_gradients(accumulated, learning_rate);
            }

            let elapsed = start.elapsed();
            let avg_loss = total_loss / training_data.len() as f32;
            eprint!("-> Avg Loss: {} | Elapsed: {:.2?}", avg_loss, elapsed);
            eprintln!();
        }
    }
}
