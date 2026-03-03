use crate::nn::gradients::Gradients;
use crate::nn::NeuralNetwork;
use rayon::prelude::*;
use std::io::{self, Write};
use std::time::Instant;

impl NeuralNetwork {
    pub fn train(
        &mut self,
        training_data: &Vec<(Vec<usize>, usize)>,
        epocs: i32,
        learning_rate: f32,
        batch_size: usize,
        max_norm: f32,
        print_epoc: bool,
    ) {
        for epoc in 0..epocs {
            let start = Instant::now();
            if print_epoc {
                let system_time = chrono::Local::now().format("%H:%M:%S");
                print!("[{}]: Epoc: {} ", system_time.to_string(), epoc);
                io::stdout().flush().expect("Failed to flush.");
            }
            let mut total_loss: f32 = 0.0;
            for batch in training_data.chunks(batch_size) {
                let all_gradients: Vec<(Gradients, f32)> = batch
                    .par_iter()
                    .map(|(context, target)| {
                        let forward_data = self.forward(context);
                        let gradients = self.compute_gradients(
                            context,
                            *target,
                            &forward_data.0,
                            &forward_data.1,
                            &forward_data.2,
                        );

                        let loss = -&forward_data.0[*target].ln();
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
            if print_epoc {
                let elapsed = start.elapsed();
                let avg_loss = total_loss / training_data.len() as f32;
                eprint!("-> Avg Loss: {} | Elapsed: {:.2?}", avg_loss, elapsed);
                eprintln!();
            }
        }
    }
}
