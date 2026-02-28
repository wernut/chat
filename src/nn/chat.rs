use crate::{nn::NeuralNetwork, vocabulary::Vocabulary};
use std::io::{self, Write};

impl NeuralNetwork {
    pub fn chat(&self, vocabulary: &Vocabulary) {
        loop {
            print!("> ");
            io::stdout().flush().expect("Failed to flush.");
            let mut input_string = String::new();
            io::stdin()
                .read_line(&mut input_string)
                .expect("Failed to read the line.");

            let input_string_parts: Vec<String> = input_string
                .split_whitespace()
                .map(|w| w.to_string().to_lowercase())
                .collect();

            if input_string_parts.len() != 2 {
                println!("Input needs to be exactly 2 words split by a space.");
                break;
            }

            let mut first_word: String = input_string_parts[0].clone();
            let mut second_word: String = input_string_parts[1].clone();

            let mut words: Vec<String> =
                vec![input_string_parts[0].clone(), input_string_parts[1].clone()];

            'word_gen: loop {
                match self.predict(first_word.as_str(), second_word.as_str(), &vocabulary) {
                    Ok(predicted_word) => {
                        if predicted_word.is_empty() {
                            break 'word_gen;
                        }
                        first_word = second_word;
                        second_word = predicted_word.clone();
                        words.push(predicted_word);
                        if words.len() > 20 {
                            break 'word_gen;
                        }
                    }
                    Err(e) => {
                        println!("Error: {}", e);
                        break 'word_gen;
                    }
                }
            }

            let generated_sentence = words.join(" ");
            println!("Output: {}", generated_sentence);
        }
    }
}
