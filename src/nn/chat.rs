use crate::{nn::NeuralNetwork, tokenizer::Tokenizer, vocabulary::Vocabulary};
use std::io::{self, Write};

impl NeuralNetwork {
    pub fn chat(&self, vocabulary: &Vocabulary, context_size: usize) {
        loop {
            print!("> ");
            io::stdout().flush().expect("Failed to flush.");
            let mut input_string = String::new();
            io::stdin()
                .read_line(&mut input_string)
                .expect("Failed to read the line.");

            let tokenized_input = Tokenizer::tokenize_sequence(&input_string);

            if tokenized_input.len() != context_size {
                println!(
                    "Input needs to be exactly {} tokens split by a space.",
                    context_size
                );
                continue;
            }

            let mut generated_tokens: Vec<String> = Vec::new();
            let mut last_input_token: String = String::new();
            for i in 0..tokenized_input.len() {
                if Self::is_punc(&tokenized_input[i]) {
                    generated_tokens.push(tokenized_input[i].clone());
                } else {
                    if !Self::is_joiner(&last_input_token) {
                        generated_tokens.push(" ".to_string());
                    }
                    generated_tokens.push(tokenized_input[i].clone());
                }
                last_input_token = tokenized_input[i].clone();
            }

            let mut sliding_context_window: Vec<String> = Vec::new();
            sliding_context_window.extend(tokenized_input.clone());

            'word_gen: loop {
                let last_context_token: String =
                    sliding_context_window[sliding_context_window.len() - 1].clone();

                match self.predict(&sliding_context_window, &vocabulary, 0.0001, 0.7) {
                    Ok(predicted_token) => {
                        if predicted_token.is_empty() {
                            break 'word_gen;
                        }

                        if Self::is_punc(&predicted_token) {
                            generated_tokens.push(predicted_token.clone());
                        } else {
                            if !Self::is_joiner(&last_context_token) {
                                generated_tokens.push(" ".to_string());
                            }
                            generated_tokens.push(predicted_token.clone());
                        }

                        if generated_tokens.len() > 20 {
                            break 'word_gen;
                        }

                        sliding_context_window = sliding_context_window.split_off(1);
                        sliding_context_window.push(predicted_token);
                    }
                    Err(e) => {
                        println!("Error: {}", e);
                        break 'word_gen;
                    }
                }
            }

            let generated_sentence = generated_tokens.join("");
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
