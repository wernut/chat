use crate::{nn::NeuralNetwork, vocabulary::Vocabulary};

impl NeuralNetwork {
    pub fn predict(
        &self,
        word1: &str,
        word2: &str,
        vocabulary: &Vocabulary,
    ) -> Result<String, String> {
        let first_word_index = match vocabulary.get_index(word1) {
            Some(value) => value,
            None => return Err("Could not find first word in the vocabulary.".to_string()),
        };

        let second_word_index = match vocabulary.get_index(word2) {
            Some(value) => value,
            None => return Err("Could not find second word in the vocabulary.".to_string()),
        };

        let forward_pass = self.forward([first_word_index, second_word_index]);
        let predicted_index = forward_pass
            .0
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();

        if *predicted_index.1 < 0.2f32 {
            return Ok("".to_string());
        }

        let predicted_word = match vocabulary.get_word(predicted_index.0) {
            Some(value) => value,
            None => {
                return Err(format!(
                    "Predicted index does not exist in the vocabulary: {}",
                    predicted_index.0
                ))
            }
        };
        return Ok(predicted_word.clone());
    }
}
