use crate::nn::NeuralNetwork;
use std::fs;

impl NeuralNetwork {
    pub fn load(path: &str) -> Result<NeuralNetwork, String> {
        let contents = fs::read_to_string(path).map_err(|e| e.to_string())?;
        serde_json::from_str(&contents).map_err(|e| e.to_string())
    }
}
