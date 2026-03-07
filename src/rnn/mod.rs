use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

pub mod backward;
pub mod build;
pub mod forward;
pub mod gradients;
pub mod load;
pub mod save;
pub mod train;

#[derive(Serialize, Deserialize, Debug)]
pub struct RecurrentNeuralNetwork {
    input_weights: Array2<f32>,
    hidden_weights: Array2<f32>,
    hidden_bias: Array1<f32>,
    output_weights: Array2<f32>,
    output_bias: Array1<f32>,
    embedding_table: Array2<f32>,
}

impl RecurrentNeuralNetwork {
    pub fn new(vocab_size: usize, embedding_dim: usize, hidden_size: usize) -> Self {
        RecurrentNeuralNetwork {
            input_weights: Array2::random((embedding_dim, hidden_size), Uniform::new(-0.01, 0.01)),
            hidden_weights: Array2::random((hidden_size, hidden_size), Uniform::new(-0.01, 0.01)),
            hidden_bias: Array1::zeros(hidden_size),
            output_weights: Array2::random((hidden_size, vocab_size), Uniform::new(-0.01, 0.01)),
            output_bias: Array1::zeros(vocab_size),
            embedding_table: Array2::random((vocab_size, embedding_dim), Uniform::new(-0.01, 0.01)),
        }
    }
}
