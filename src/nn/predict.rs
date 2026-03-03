use crate::{nn::NeuralNetwork, vocabulary::Vocabulary};
use rand;

impl NeuralNetwork {
    pub fn predict(
        &self,
        words: &Vec<String>,
        vocabulary: &Vocabulary,
        probability_cut_off: f32,
        temperature: f32,
    ) -> Result<String, String> {
        let mut word_indexes: Vec<usize> = Vec::new();
        for word in words {
            match vocabulary.get_index(&word) {
                Some(index) => word_indexes.push(index),
                None => {
                    return Err(
                        format!("Could not find word '{}' in the vocabulary.", word).to_string()
                    )
                }
            };
        }

        let forward_pass = self.forward(&word_indexes);

        let temps: Vec<f32> = forward_pass
            .0
            .iter()
            .map(|p| p.powf(1.0 / temperature))
            .collect();
        let sum: f32 = temps.iter().sum();
        let probs: Vec<f32> = temps.iter().map(|p| p / sum).collect();

        let random = rand::random::<f32>();
        let mut cumulative = 0.0;
        let mut predicted_index: usize = probs.len() - 1;
        for (i, prob) in probs.iter().enumerate() {
            cumulative += prob;
            if cumulative > random {
                predicted_index = i;
                break;
            }
        }

        if probs[predicted_index] < probability_cut_off {
            return Ok("".to_string());
        }

        let predicted_word = match vocabulary.get_word(predicted_index) {
            Some(value) => value,
            None => {
                return Err(format!(
                    "Predicted index does not exist in the vocabulary: {}",
                    predicted_index
                ))
            }
        };
        return Ok(predicted_word.clone());
    }
}
