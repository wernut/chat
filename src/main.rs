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
    let sentences = load_sentences("./chat_input.text");
    let sentence_refs: Vec<&str> = sentences.iter().map(|s| s.as_str()).collect();
    let vocabulary: Vocabulary = Vocabulary::new(&sentence_refs, 5);

    print!("Would you like to train a new model? (Y/N): ");
    io::stdout().flush().expect("Failed to flush.");
    let mut input_string = String::new();
    io::stdin()
        .read_line(&mut input_string)
        .expect("Failed to read the line.");

    let input = input_string.trim();

    let path = "./models/";
    let new_version = version_helper::get_next_model_version_index(path);

    let context_size = 5;

    let neural_network = if input == "y" {
        eprintln!("Building training data for model version: {}", new_version);
        let training_data = training_data::build(&sentence_refs, &vocabulary, context_size);

        eprintln!("Initializing the neural network...");
        let mut nn = nn::NeuralNetwork::new(vocabulary.size(), 16, 64, context_size);

        eprintln!("Starting training process...");
        nn.train(&training_data, 10, 0.001, 256, 10.0, true);

        eprintln!("Saving model...");
        let model_path = format!("{}model_{}.json", path, new_version);
        match nn.save(&model_path) {
            Ok(()) => {
                println!("Succesfully trained and saved the model at: {}", model_path)
            }
            Err(e) => println!("Failed to save the model after training, error: {}", e),
        };
        nn
    } else if input == "n" {
        let model_path = format!("{}model_{}.json", path, (new_version - 1).to_string());
        println!("Loading model: {}", model_path);
        match nn::NeuralNetwork::load(model_path.as_str()) {
            Ok(mut nn) => {
                print!("Would you like to continue training this model? (Y/N): ");
                io::stdout().flush().expect("Failed to flush.");
                let mut input_string = String::new();
                io::stdin()
                    .read_line(&mut input_string)
                    .expect("Failed to read the line.");
                let input = input_string.trim();

                if input == "y" {
                    eprintln!("Building training data for model version: {}", new_version);
                    let training_data =
                        training_data::build(&sentence_refs, &vocabulary, context_size);

                    eprintln!("Continuing to train model...");
                    nn.train(&training_data, 10, 0.001, 256, 10.0, true);

                    match nn.save(&model_path) {
                        Ok(()) => {
                            println!("Succesfully trained and saved the model at: {}", model_path)
                        }
                        Err(e) => println!("Failed to save the model after training, error: {}", e),
                    };
                }

                nn
            }
            Err(e) => {
                eprintln!("Failed to load model: {}", e);
                return;
            }
        }
    } else {
        eprintln!("Invalid input.");
        return;
    };

    neural_network.chat(&vocabulary, context_size);
}
