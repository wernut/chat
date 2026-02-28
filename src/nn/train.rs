use crate::nn::NeuralNetwork;
use ndarray::Array1;

impl NeuralNetwork {
    pub fn train(
        &mut self,
        training_data: &Vec<([usize; 2], usize)>,
        epocs: i32,
        learning_rate: f32,
        print_epoc: bool,
    ) {
        for epoc in 0..epocs {
            let mut total_loss: f32 = 0.0;
            for (context, target) in training_data.iter() {
                let forward_data: (Array1<f32>, Array1<f32>, Array1<f32>) = self.forward(*context);
                let backward_data = self.backward(
                    *context,
                    *target,
                    &forward_data.0,
                    &forward_data.1,
                    &forward_data.2,
                    learning_rate,
                );
                total_loss += backward_data;
            }

            if print_epoc {
                let avg_loss = total_loss / training_data.len() as f32;
                eprintln!("Epoc: {} | Avg Loss: {}", epoc, avg_loss);
            }
        }
    }
}
