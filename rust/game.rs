use crate::neural_network::NeuralNetwork;
use crate::topology::Topology;

/// Trait to implement in order to use Train
pub trait Game {
    /// Run a game round
    fn run_generation(&mut self) -> Vec<f64>;

    /// Resets the neural networks
    ///
    /// # Arguments
    ///
    /// `nets` - A vector containing the last generation of neural networks
    fn reset_players(&mut self, nets: &[NeuralNetwork]);

    /// Function to be run at the end of the training
    ///
    /// # Arguments
    ///
    /// `net` - The best historical network
    fn post_training(&mut self, history: &[Topology]);
}