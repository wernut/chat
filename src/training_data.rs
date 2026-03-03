use crate::{tokenizer::Tokenizer, vocabulary::Vocabulary};

pub fn build(
    sequences: &[&str],
    vocabulary: &Vocabulary,
    context_size: usize,
) -> Vec<(Vec<usize>, usize)> {
    let mut training_data: Vec<(Vec<usize>, usize)> = Vec::new();

    for sequence in sequences {
        let tokens = Tokenizer::tokenize_sequence(sequence);
        for i in 0..tokens.len() {
            if i + context_size >= tokens.len() {
                break;
            }

            match build_token_index_entry(i, context_size, &tokens, vocabulary) {
                Some(token_index_entry) => training_data.push(token_index_entry),
                None => continue,
            };
        }
    }
    return training_data;
}

fn build_token_index_entry(
    i: usize,
    context_size: usize,
    tokens: &[String],
    vocabulary: &Vocabulary,
) -> Option<(Vec<usize>, usize)> {
    let mut context_token_indexes: Vec<usize> = Vec::new();
    for j in 0..context_size {
        let context_token = &tokens[i + j];
        let context_token_index: usize = match vocabulary.get_index(context_token) {
            Some(value) => value,
            None => {
                return None;
            }
        };
        context_token_indexes.push(context_token_index);
    }

    let target_token_index = match vocabulary.get_index(&tokens[i + context_size]) {
        Some(value) => value,
        None => {
            return None;
        }
    };

    Some((context_token_indexes, target_token_index))
}
