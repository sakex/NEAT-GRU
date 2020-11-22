use std::fs;
use crate::neural_network::NeuralNetwork;
use crate::train::Train;
use crate::game::Game;
use crate::topology::Topology;
use crate::ffi::network_from_topology;

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
        self.nets.iter().map(|network| {
            let inputs = vec![0.1, 0.2, 0.3, 0.4, 0.5];
            let out = network.compute(&*inputs, 5);
            let mut diff = 0f64;
            inputs.iter().zip(out.iter()).for_each(|(a, b)| {
                diff -= (a - b).abs();
            });
            diff
        }).collect()
    }

    fn reset_players(&mut self, nets: &[NeuralNetwork]) {
        self.nets = Vec::from(nets);
    }

    fn post_training(&mut self, history: &[Topology]) {
        for top in history {
            let as_str = top.to_string();
            let deserialized = Topology::from_string(&*as_str);
            assert!(top == &deserialized);
            let net1: NeuralNetwork = top.into();
            let net2 = NeuralNetwork::from_string(&*as_str);
            assert!(net1 == net2);
        }
    }
}

#[test]
pub fn test_train() {
    let game = TestGame::new();
    let mut runner: Train<TestGame> = Train::new();
    runner.simulation(game)
        .iterations(100)
        .max_individuals(50)
        .max_species(5)
        .inputs(5)
        .outputs(5);
    runner.start();
}