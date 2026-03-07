use crate::rnn::RecurrentNeuralNetwork;
use crate::tokenizer::Tokenizer;
use crate::vocabulary::Vocabulary;

impl RecurrentNeuralNetwork {
    pub fn build(sequences: &[&str], vocabulary: &Vocabulary) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut training_data: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        for sequence in sequences {
            let tokens = Tokenizer::tokenize_sequence(sequence);
            if tokens.len() < 2 {
                continue;
            }

            let indices: Vec<usize> = tokens
                .iter()
                .filter_map(|t| vocabulary.get_index(t))
                .collect();

            if indices.len() < 2 {
                continue;
            }

            let inputs = indices[..indices.len() - 1].to_vec();
            let targets = indices[1..].to_vec();
            training_data.push((inputs, targets));
        }

        training_data
    }
}
