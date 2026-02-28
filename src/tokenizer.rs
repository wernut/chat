pub struct Tokenizer {}

impl Tokenizer {
    // Tokenize a sequence: "Hello friend, how are you going?" -> ["Hello", "friend", ",", "how", "are", "you", "going", "?"]
    pub fn tokenize_sequence(sequence: &str) -> Vec<String> {
        let words: Vec<String> = sequence.split_whitespace().map(|w| w.to_string()).collect();
        let mut tokens: Vec<String> = Vec::new();
        for word in &words {
            tokens.extend(Tokenizer::tokenize_word(word));
        }
        return tokens;
    }

    // Tokenize a word: "Hello!" -> ["Hello", "!"]
    fn tokenize_word(word: &str) -> Vec<String> {
        let mut tokens: Vec<String> = Vec::new();
        let mut current = String::new();
        for c in word.chars() {
            if c.is_alphabetic() {
                current.push(c);
            } else if c.is_ascii_punctuation() {
                if !current.is_empty() {
                    tokens.push(current.clone());
                    current.clear();
                }
                tokens.push(c.to_string());
            }
        }
        if !current.is_empty() {
            tokens.push(current);
        }
        tokens.retain(|t| !t.is_empty());
        return tokens;
    }
}
