use ndarray::{Array1, Array2};

pub struct Gradients {
    pub input_weights: Array2<f32>,
    pub output_weights: Array2<f32>,
    pub output_bias: Array1<f32>,
    pub hidden_weights: Array2<f32>,
    pub hidden_bias: Array1<f32>,
    pub embedding_table: Vec<(usize, Array1<f32>)>,
}

impl Gradients {
    pub fn add(&mut self, other: &Gradients) {
        self.input_weights += &other.input_weights;
        self.hidden_weights += &other.hidden_weights;
        self.hidden_bias += &other.hidden_bias;
        self.output_weights += &other.output_weights;
        self.output_bias += &other.output_bias;
        for (index, grad) in &other.embedding_table {
            self.embedding_table.push((*index, grad.clone()));
        }
    }

    pub fn scale(&mut self, factor: f32) {
        self.input_weights *= factor;
        self.hidden_weights *= factor;
        self.hidden_bias *= factor;
        self.output_weights *= factor;
        self.output_bias *= factor;
        for (_, grad) in &mut self.embedding_table {
            *grad *= factor;
        }
    }

    pub fn norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        sum_sq += self.input_weights.mapv(|x| x * x).sum();
        sum_sq += self.hidden_weights.mapv(|x| x * x).sum();
        sum_sq += self.hidden_bias.mapv(|x| x * x).sum();
        sum_sq += self.output_weights.mapv(|x| x * x).sum();
        sum_sq += self.output_bias.mapv(|x| x * x).sum();
        for (_, grad) in &self.embedding_table {
            sum_sq += grad.mapv(|x| x * x).sum();
        }
        sum_sq.sqrt()
    }
}
