use boltzmann::BoltzmannPolicy;
use enum_dispatch::enum_dispatch;
use epsilon_greedy::EpsilonGreedyPolicy;

pub mod boltzmann;
pub mod epsilon_greedy;

#[enum_dispatch]
pub trait PolicyLike {
    fn select_action(&self, estimates: &[f32]) -> usize;
}

#[enum_dispatch(PolicyLike)]
pub enum Policy {
    Boltzmann(BoltzmannPolicy),
    EpsilonGreedy(EpsilonGreedyPolicy),
}
