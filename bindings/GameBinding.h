//
// Created by alexandre on 20.05.20.
//

#ifndef NEAT_GRU_GAMEBINDING_H
#define NEAT_GRU_GAMEBINDING_H


#include "../Game/Game.h"
#include "../NeuralNetwork/Topology.h"
#include "structs.h"


class GameBinding : public Game::Game {
public:
    explicit GameBinding(Simulation _sim);

    ~GameBinding() override = default;

private:
    void do_reset_players(NN *brains, size_t size) override;

    std::vector<double> do_run_generation() override;

    ::Game::Player *do_post_training(Topology_ptr) override;

private:
    Simulation const sim;
    size_t _size;
};


#endif //NEAT_GRU_GAMEBINDING_H
