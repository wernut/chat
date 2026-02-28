use crate::vocabulary::Vocabulary;
use std::fs;
use std::io::{self, Write};

// Register modules
mod nn;
mod tokenizer;
mod training_data;
mod version_helper;
mod vocabulary;

// Load a bunch of sentences from a text file
fn load_sentences(path: &str) -> Vec<String> {
    let contents = fs::read_to_string(path).expect("Could not read the file from path");
    let contents_split = contents.split('\n');
    return contents_split.map(|s| s.to_string()).collect();
}

fn main() {
    println!("Loading vocabulary...");
    let sentences = load_sentences("C:/Users/werne/source/repos/wernut/chat/chat_input.text");
    let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    let vocabulary: Vocabulary = Vocabulary::new(&sentence_refs);

    print!("Train new model? (Y/N): ");
    io::stdout().flush().expect("Failed to flush.");
    let mut input_string = String::new();
    io::stdin()
        .read_line(&mut input_string)
        .expect("Failed to read the line.");

    let input = input_string.trim().to_lowercase();

    let path = "C:/Users/werne/source/repos/wernut/chat/models/";
    let new_version = version_helper::get_next_model_version_index(path);

    let neural_network = if input == "y" {
        eprintln!("Building training data for: {}", new_version);
        let training_data = training_data::build(&sentence_refs, &vocabulary);

        eprintln!("Initializing the neural network...");
        let mut nn = nn::NeuralNetwork::new(vocabulary.size(), 32, 128);

        eprintln!("Starting training process...");
        nn.train(&training_data, 500, 0.01, true);

        eprintln!("Saving model...");
        let model_path = format!("{}model_{}.bin", path, new_version);
        match nn.save(&model_path) {
            Ok(()) => {
                println!("Succesfully trained and saved the model at: {}", model_path)
            }
            Err(e) => println!("Failed to save the model after training, error: {}", e),
        };
        nn
    } else if input == "n" {
        let model_path = format!("{}model_{}.bin", path, (new_version - 1).to_string());
        println!("Loading model: {}", model_path);
        match nn::NeuralNetwork::load(model_path.as_str()) {
            Ok(nn) => nn,
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                return;
            }
        }
    } else {
        eprintln!("Invalid input.");
        return;
    };

    neural_network.chat(&vocabulary);
}
