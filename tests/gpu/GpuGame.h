//
// Created by alexandre on 03.09.20.
//

#ifndef NEAT_GRU_GPUGAME_H
#define NEAT_GRU_GPUGAME_H
#include "Game.h"
#include "../GPU/ComputeInstance.cuh"


class GpuGame: public Game::Game {
public:
    explicit GpuGame(Dim dims);

private:
    std::vector<double> do_run_generation() override;
    void do_reset_players(NN * nets, size_t count) override;
    void do_post_training(Topology const * history, size_t size) override;

private:
    ComputeInstance compute_instance;
    Dim dim;
    std::vector<double> outputs;
    size_t player_count;
};


#endif //NEAT_GRU_GPUGAME_H
