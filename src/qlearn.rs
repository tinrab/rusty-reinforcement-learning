use crate::policy::{Policy, PolicyLike};

pub struct QLearn {
    table: Vec<Vec<f32>>,
    action_space: usize,
    state_space: usize,
    learning_rate: f32,
    discount_factor: f32,
    policy: Policy,

    current_state: usize,
    selected_action: usize,
    fitness: f32,
}

impl QLearn {
    pub fn new(
        action_space: usize,
        state_space: usize,
        learning_rate: f32,
        discount_factor: f32,
        policy: Policy,
    ) -> Self {
        assert!(action_space > 0, "action_space must be greater than 0");
        assert!(state_space > 0, "state_space must be greater than 0");
        Self {
            table: vec![vec![0.0f32; action_space]; state_space],
            action_space,
            state_space,
            learning_rate,
            discount_factor,
            policy,
            current_state: 0,
            selected_action: 0,
            fitness: 0.0f32,
        }
    }

    pub fn randomize_table(&mut self) {
        for i in 0..self.state_space {
            for j in 0..self.action_space {
                self.table[i][j] = rand::random::<f32>() - 0.5f32;
            }
        }
    }

    pub fn start(&mut self, state: usize) {
        self.current_state = state;
        self.selected_action = self.select_action(state);
        self.fitness = 0.0f32;
    }

    pub fn step(&mut self, reward: f32, next_state: usize) {
        let mut best_next = self.table[next_state][0];

        for &q in &self.table[next_state] {
            if q > best_next {
                best_next = q;
            }
        }

        let target = reward + self.discount_factor * best_next;
        let delta = target - self.table[self.current_state][self.selected_action];
        self.table[self.current_state][self.selected_action] += self.learning_rate * delta;

        self.current_state = next_state;
        self.selected_action = self.select_action(next_state);

        self.fitness += reward;
    }

    pub fn select_action(&self, state: usize) -> usize {
        self.policy.select_action(&self.table[state])
    }

    pub fn fitness(&self) -> f32 {
        self.fitness
    }
}

#[cfg(test)]
mod tests {
    use crate::policy::epsilon_greedy::EpsilonGreedyPolicy;

    use super::*;

    #[test]
    fn it_works() {
        let mut qlearn = QLearn::new(
            2,
            10,
            0.9f32,
            0.1f32,
            EpsilonGreedyPolicy::new(0.1f32).into(),
        );
        let environment: Vec<i32> = (0..10).collect();

        // qlearn.randomize_table();
        let mut total_fitness = 0.0f32;

        for _epoch in 0..20 {
            qlearn.start(0);

            let mut current_state = qlearn.current_state;
            while current_state < 9 {
                let selected_action = qlearn.selected_action;

                let reward = if environment[current_state] % 2 == 0 {
                    if selected_action == 0 {
                        1.0f32
                    } else {
                        0.0f32
                    }
                } else {
                    if selected_action == 1 {
                        1.0f32
                    } else {
                        0.0f32
                    }
                };

                let next_state = current_state + 1;
                qlearn.step(reward, next_state);
                current_state = next_state;
            }

            total_fitness += qlearn.fitness;
        }

        assert!(total_fitness > 10.0f32);
    }
}
