//
// Created by sakex on 02.06.20.
//

#ifndef NEAT_GRU_MEMORY_H
#define NEAT_GRU_MEMORY_H

#include "../Game/Game.h"
#include "MemoryPlayer.h"
#include "../Threading/multithreaded_methods.h"

class Memory : public Game::Game {
public:
    Memory();

    std::vector<double> do_run_generation() override;

    void do_reset_players(NN *nets, size_t count) override;

    void do_post_training(Topology_ptr topology) override;

private:
    std::vector<MemoryPlayer> players;
};


#endif //NEAT_GRU_MEMORY_H