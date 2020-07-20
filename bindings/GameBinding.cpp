//
// Created by alexandre on 20.05.20.
//

#include "GameBinding.h"

GameBinding::GameBinding(Simulation *_sim) : sim(_sim), _size(0) {
}

void GameBinding::do_reset_players(NN *brains, size_t size) {
    _size = size;
    void *void_brains = brains;
    (*sim->reset_players)(sim->context, void_brains, sizeof(NN), static_cast<unsigned>(size));
}

std::vector<double> GameBinding::do_run_generation() {
    double *results = (*sim->run_generation)(sim->context);
    std::vector<double> ret(results, results + _size);
    free(results);
    return ret;
}

void GameBinding::do_post_training(Topology const * history, size_t size) {
    (*sim->post_training)(sim->context, history, sizeof(Topology), size);
}