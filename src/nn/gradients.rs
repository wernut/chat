use ndarray::{Array1, Array2};

pub struct Gradients {
    pub weights_1: Array2<f32>,
    pub bias_1: Array1<f32>,
    pub weights_2: Array2<f32>,
    pub bias_2: Array1<f32>,
    pub embedding_table: Vec<(usize, Array1<f32>)>,
}

impl Gradients {
    pub fn add(&mut self, other: &Gradients) {
        self.weights_1 += &other.weights_1;
        self.bias_1 += &other.bias_1;
        self.weights_2 += &other.weights_2;
        self.bias_2 += &other.bias_2;
        for (index, grad) in &other.embedding_table {
            self.embedding_table.push((*index, grad.clone()));
        }
    }

    pub fn scale(&mut self, factor: f32) {
        self.weights_1 *= factor;
        self.bias_1 *= factor;
        self.weights_2 *= factor;
        self.bias_2 *= factor;
        for (_, grad) in &mut self.embedding_table {
            *grad *= factor;
        }
    }

    pub fn norm(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        sum_sq += self.weights_1.mapv(|x| x * x).sum();
        sum_sq += self.bias_1.mapv(|x| x * x).sum();
        sum_sq += self.weights_2.mapv(|x| x * x).sum();
        sum_sq += self.bias_2.mapv(|x| x * x).sum();
        for (_, grad) in &self.embedding_table {
            sum_sq += grad.mapv(|x| x * x).sum();
        }
        sum_sq.sqrt()
    }
}
