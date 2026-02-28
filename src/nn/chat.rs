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
                .map(|w| w.to_string())
                .collect();

            if input_string_parts.len() != 2 {
                println!("Input needs to be exactly 2 words split by a space.");
                break;
            }

            let mut first_token: String = input_string_parts[0].clone();
            let mut second_token: String = input_string_parts[1].clone();

            let mut tokens: Vec<String> = vec![
                input_string_parts[0].clone(),
                " ".to_string(),
                input_string_parts[1].clone(),
            ];

            'word_gen: loop {
                match self.predict(first_token.as_str(), second_token.as_str(), &vocabulary) {
                    Ok(predicted_token) => {
                        if predicted_token.is_empty() {
                            break 'word_gen;
                        }

                        if Self::is_punc(&predicted_token) {
                            tokens.push(predicted_token.clone());
                        } else {
                            if !Self::is_joiner(&second_token) {
                                tokens.push(" ".to_string());
                            }
                            tokens.push(predicted_token.clone());
                        }

                        if tokens.len() > 20 {
                            break 'word_gen;
                        }

                        first_token = second_token;
                        second_token = predicted_token.clone();
                    }
                    Err(e) => {
                        println!("Error: {}", e);
                        break 'word_gen;
                    }
                }
            }

            let generated_sentence = tokens.join("");
            println!("Output: {}", generated_sentence);
        }
    }

    fn is_punc(token: &String) -> bool {
        for c in token.chars() {
            if c.is_ascii_punctuation() {
                return true;
            }
        }
        return false;
    }

    fn is_joiner(token: &String) -> bool {
        for c in token.chars() {
            if c == '\'' || c == '-' {
                return true;
            }
        }
        return false;
    }
}
