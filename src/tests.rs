use std::fs;
use crate::neural_network::NeuralNetwork;

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