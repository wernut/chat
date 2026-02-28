use ndarray::{Array1, Array2};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use serde::{Deserialize, Serialize};

pub mod backward;
pub mod chat;
pub mod forward;
pub mod load;
pub mod predict;
pub mod save;
pub mod train;

#[derive(Serialize, Deserialize, Debug)]
pub struct NeuralNetwork {
    embedding_table: Array2<f32>,
    weights_1: Array2<f32>,
    bias_1: Array1<f32>,
    weights_2: Array2<f32>,
    bias_2: Array1<f32>,
}

impl NeuralNetwork {
    pub fn new(vocab_size: usize, embedding_dim: usize, hidden_size: usize) -> Self {
        NeuralNetwork {
            embedding_table: Array2::random((vocab_size, embedding_dim), Uniform::new(-0.1, 0.1)),
            weights_1: Array2::random((embedding_dim * 2, hidden_size), Uniform::new(-0.1, 0.1)),
            bias_1: Array1::zeros(hidden_size),
            weights_2: Array2::random((hidden_size, vocab_size), Uniform::new(-0.1, 0.1)),
            bias_2: Array1::zeros(vocab_size),
        }
    }
}
