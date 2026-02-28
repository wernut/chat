use crate::nn::NeuralNetwork;
use std::fs::File;
use std::io::Write;

impl NeuralNetwork {
    pub fn save(&self, path: &str) -> Result<(), String> {
        let json = serde_json::to_string(self).map_err(|e| e.to_string())?;
        let mut file = File::create(path).map_err(|e| e.to_string())?;
        file.write_all(json.as_bytes()).map_err(|e| e.to_string())
    }
}
