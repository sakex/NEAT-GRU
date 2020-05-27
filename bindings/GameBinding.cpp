//
// Created by alexandre on 20.05.20.
//

#include "GameBinding.h"

GameBinding::GameBinding(Simulation * _sim) : sim(_sim), _size(0) {
}

void GameBinding::do_reset_players(NN *brains, size_t size) {
    _size = size;
    auto *wrappers = new NetWrapper[size];
    for (std::size_t i = 0; i < size; ++i) {
        void *ptr = &brains[i];
        wrappers[i].net = ptr;
    }
    (*sim->reset_players)(sim->context, wrappers, static_cast<unsigned>(size));
    delete[] wrappers;
}

std::vector<double> GameBinding::do_run_generation() {
    double *results = (*sim->run_generation)(sim->context);
    std::vector<double> ret(results, results + _size);
    return ret;
}

::Game::Player *GameBinding::do_post_training(Topology_ptr) {
    return nullptr;
}