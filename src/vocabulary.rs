use crate::tokenizer::Tokenizer;
use std::collections::HashMap;

pub struct Vocabulary {
    word_to_index: HashMap<String, usize>,
    index_to_word: Vec<String>,
}

impl Vocabulary {
    pub fn new(sequences: &[&str]) -> Self {
        let data: (HashMap<String, usize>, Vec<String>) = Vocabulary::construct_data(sequences);
        Vocabulary {
            word_to_index: data.0,
            index_to_word: data.1,
        }
    }

    // Construct the word_to_index and index_to_word variables
    fn construct_data(sequences: &[&str]) -> (HashMap<String, usize>, Vec<String>) {
        let mut word_to_index: HashMap<String, usize> = HashMap::new();
        let mut index_to_word: Vec<String> = Vec::new();
        for sequence in sequences {
            let tokens = Tokenizer::tokenize_sequence(sequence);
            for token in tokens {
                if !word_to_index.contains_key(&token) {
                    let index: usize = word_to_index.len();
                    word_to_index.insert(token.clone(), index);
                    index_to_word.push(token.clone());
                }
            }
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
