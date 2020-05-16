//
// Created by alexandre on 16.05.20.
//

#include "bindings.h"

GameBinding::GameBinding(Simulation _sim) : sim(_sim) {
}

std::vector<double> GameBinding::do_run_generation() {
    double * results = (*sim.run_generation)();
    std::vector<double> ret(results, results + _size);
    return ret;
}

void GameBinding::do_reset_players(NN *brains, size_t size) {
    _size = size;
    (*sim.reset_players)(brains, static_cast<unsigned >(size));
}

::Game::Player *GameBinding::do_post_training(Topology_ptr) {
    return nullptr;
}

double * compute_network(NN* net, const double * inputs) {
    std::vector<double> outputs = net->compute(inputs);
    return outputs.data();
}

void train(Simulation const sim, int const iterations, int const max_individuals, int const inputs, int const outputs) {
    auto *binding = new GameBinding(sim);
    Train::Train train(binding, iterations, max_individuals, inputs, outputs);
    train.start();
}
