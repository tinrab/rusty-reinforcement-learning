use rand::Rng;

use super::PolicyLike;

pub struct EpsilonGreedyPolicy {
    pub epsilon: f32,
}

impl EpsilonGreedyPolicy {
    pub fn new(epsilon: f32) -> Self {
        Self { epsilon }
    }
}

impl PolicyLike for EpsilonGreedyPolicy {
    fn select_action(&self, estimates: &[f32]) -> usize {
        let r: f32 = rand::thread_rng().gen();
        if r < self.epsilon {
            return rand::thread_rng().gen_range(0..estimates.len());
        }

        estimates
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0
    }
}
