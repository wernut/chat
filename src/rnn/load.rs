use crate::rnn::RecurrentNeuralNetwork;
use std::fs;

impl RecurrentNeuralNetwork {
    pub fn load(path: &str) -> Result<RecurrentNeuralNetwork, String> {
        let contents = fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&contents).map_err(|e| e.to_string())
    }
}
