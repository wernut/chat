use crate::{tokenizer::Tokenizer, vocabulary::Vocabulary};

pub fn build(sequences: &[&str], vocabulary: &Vocabulary) -> Vec<([usize; 2], usize)> {
    let mut training_data: Vec<([usize; 2], usize)> = Vec::new();

    for sequence in sequences {
        let tokens = Tokenizer::tokenize_sequence(sequence);
        for (i, curr_token) in tokens.iter().enumerate() {
            if i + 2 >= tokens.len() {
                break;
            }
            let next_token = &tokens[i + 1];
            let last_token = &tokens[i + 2];

            let curr_token_index: usize;
            match vocabulary.get_index(curr_token) {
                Some(value) => {
                    curr_token_index = value;
                }
                None => {
                    continue;
                }
            }

            let next_token_index: usize;
            match vocabulary.get_index(next_token) {
                Some(value) => {
                    next_token_index = value;
                }
                None => {
                    continue;
                }
            }

            let last_token_index: usize;
            match vocabulary.get_index(last_token) {
                Some(value) => {
                    last_token_index = value;
                }
                None => {
                    continue;
                }
            }

            training_data.push(([curr_token_index, next_token_index], last_token_index));
        }
    }

    return training_data;
}
