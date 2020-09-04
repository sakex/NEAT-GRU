//
// Created by alexandre on 03.09.20.
//

#include "GpuGame.h"

GpuGame::GpuGame(Dim dims) : compute_instance(dims), dim(dims) {
    size_t size = dims.x * dims.y * dims.z;
    auto *dataset = (double*)malloc(size * sizeof(double));
    for (size_t i = 0; i < size; ++i) {
        dataset[i] = utils::Random::random_number(1);
    }
    for (size_t i = 0; i < size; i += dims.x) {
        double sum = 0;
        for (size_t j = i; j < i + dims.x; ++j) sum += dataset[j];
        outputs.push_back(sum > (dims.x / 2));
    }
    update_dataset_gpu_instance(&compute_instance, dataset);
    free(dataset);
}

std::vector<double> GpuGame::do_run_generation() {
    compute_gpu_instance(&compute_instance, 1);
    double *output = compute_instance.h_output;
    std::vector<double> scores;
    for (size_t it = 0; it < player_count; ++it) {
        double sum = 0;
        for (size_t j = 0; j < dim.y * dim.z; ++j) {
            sum -= std::abs(output[it * dim.y * dim.z + j] - outputs[j % (dim.y * dim.z)]);
        }
        scores.push_back(sum);
    }
    return scores;
}

void GpuGame::do_reset_players(NeuralNetworkCuda::NN *nets, size_t count) {
    set_networks_gpu_instance(&compute_instance, nets, count);
    player_count = count;
}

void GpuGame::do_post_training(const NeuralNetwork::Topology *, size_t) {
    std::cout << "DONE" << std::endl;
}
