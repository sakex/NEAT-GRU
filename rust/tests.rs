use std::fs;
use crate::neural_network::NeuralNetwork;
use crate::train::Train;
use crate::game::Game;
use crate::topology::Topology;

#[test]
pub fn test_import_network() {
    let serialized: String = fs::read_to_string("topology_test.json")
        .expect("Something went wrong reading the file");

    let mut net = NeuralNetwork::from_string(&serialized);
    let input_1: Vec<f64> = vec![0.5, 0.5, 0.1, -0.2];
    let input_2: Vec<f64> = vec![-0.5, -0.5, -0.1, 0.2];

    let output_1 = net.compute(&input_1, 2);
    let output_2 = net.compute(&input_2, 2);
    let output_3 = net.compute(&input_1, 2);

    // 1 and 2 should by definition be different
    assert_ne!(output_1, output_2);
    assert_ne!(output_1, output_3);
    //Because of GRU gates, giving the same input twice won't yield the same output
    assert_ne!(output_2, output_3);

    // Reset
    net.reset_state();
    let output_4 = net.compute(&input_1, 2);

    // After resetting, giving the same input sequence should yield the same results
    assert_eq!(output_1, output_4);
}


struct TestGame {
    nets: Vec<NeuralNetwork>
}

impl TestGame {
    pub fn new() -> TestGame {
        TestGame {
            nets: Vec::new()
        }
    }
}

impl Game for TestGame {
    fn run_generation(&mut self) -> Vec<f64> {
        vec![0f64; self.nets.len()]
    }

    fn reset_players(&mut self, nets: &[NeuralNetwork]) {
        self.nets = Vec::from(nets);
    }

    fn post_training(&mut self, history: &[Topology]) {
        let generated = &history[history.len() - 1];
        let as_str = generated.to_string();
        let deserialized = Topology::from_string(&*as_str);
        assert!(generated == &deserialized);
    }
}

#[test]
pub fn test_train() {
    let game = TestGame::new();
    let mut runner: Train<TestGame> = Train::new();
    runner.simulation(game)
        .iterations(10)
        .max_individuals(5)
        .inputs(5)
        .outputs(1);
    runner.start();
}