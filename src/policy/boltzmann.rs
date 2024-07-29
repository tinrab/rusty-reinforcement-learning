use rand::Rng;

use super::PolicyLike;

pub struct BoltzmannPolicy {
    pub temperature: f32,
}

impl BoltzmannPolicy {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }
}

impl PolicyLike for BoltzmannPolicy {
    fn select_action(&self, estimates: &[f32]) -> usize {
        let mut probabilities = Vec::with_capacity(estimates.len());
        let mut probability_sum = 0.0f32;
        for &estimate in estimates {
            let probability = (estimate / self.temperature).exp();
            probabilities.push(probability);
            probability_sum += probability;
        }

        if probability_sum == 0.0f32 || !probability_sum.is_normal() {
            return estimates
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap()
                .0;
        }

        let r: f32 = rand::thread_rng().gen();
        let mut cumulative_probability = 0.0f32;
        for (i, &probability) in probabilities.iter().enumerate() {
            cumulative_probability += probability / probability_sum;
            if r <= cumulative_probability {
                return i;
            }
        }

        estimates.len() - 1
    }
}
