//
// Created by alexandre on 16.05.20.
//

#ifndef NEAT_BINDINGS_H
#define NEAT_BINDINGS_H

#include "Train.h"
#include "../Game/Game.h"

extern "C"
struct NetWrapper {
    void* net;
};

extern "C"
struct Simulation {
    double * (*run_generation)();

    void (*reset_players)(NetWrapper *, unsigned);
};

class GameBinding : public Game::Game {
public:
    explicit GameBinding(Simulation);

protected:
    std::vector<double> do_run_generation() final;

    void do_reset_players(NN *, size_t) final;

    ::Game::Player *do_post_training(Topology_ptr) final;

private:
    Simulation const sim;
    size_t _size = 0;
};

extern "C"
double * compute_network(NetWrapper, const double *);

extern "C" void train(Simulation, int, int, int, int);

#endif //NEAT_BINDINGS_H
