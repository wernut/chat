use crate::tokenizer::Tokenizer;
use std::collections::HashMap;

pub struct Vocabulary {
    word_to_index: HashMap<String, usize>,
    index_to_word: Vec<String>,
}

impl Vocabulary {
    pub fn new(sequences: &[&str], min_frequency: usize) -> Self {
        let data: (HashMap<String, usize>, Vec<String>) =
            Vocabulary::construct_data(sequences, min_frequency);
        Vocabulary {
            word_to_index: data.0,
            index_to_word: data.1,
        }
    }

    // Construct the word_to_index and index_to_word variables
    fn construct_data(
        sequences: &[&str],
        min_frequency: usize,
    ) -> (HashMap<String, usize>, Vec<String>) {
        let mut token_frequency_map: HashMap<String, usize> = HashMap::new();
        for sequence in sequences {
            let tokens = Tokenizer::tokenize_sequence(sequence);
            for token in tokens {
                match token_frequency_map.get_mut(&token) {
                    Some(value) => *value += 1,
                    None => {
                        token_frequency_map.insert(token.clone(), 1);
                    }
                }
            }
        }

        let mut sorted_tokens: Vec<String> = token_frequency_map
            .into_iter()
            .filter(|(_, freq)| *freq >= min_frequency)
            .map(|(token, _)| token)
            .collect();

        sorted_tokens.sort();

        let mut word_to_index: HashMap<String, usize> = HashMap::new();
        let mut index_to_word: Vec<String> = Vec::new();

        for token in sorted_tokens {
            let index = word_to_index.len();
            word_to_index.insert(token.clone(), index);
            index_to_word.push(token);
        }

        return (word_to_index, index_to_word);
    }

    pub fn get_index(&self, word: &str) -> Option<usize> {
        self.word_to_index.get(word).copied()
    }

    pub fn get_word(&self, index: usize) -> Option<&String> {
        self.index_to_word.get(index)
    }

    pub fn size(&self) -> usize {
        self.word_to_index.len()
    }
}
