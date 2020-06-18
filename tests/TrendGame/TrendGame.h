//
// Created by sakex on 02.06.20.
//

#ifndef NEAT_GRU_TRENDGAME_H
#define NEAT_GRU_TRENDGAME_H

#include "Game.h"
#include "multithreaded_methods.h"
#include "../../Private/Random.h"

constexpr int DIFFERENT_NUMBERS = 5;

struct Player {
    NN *network;
    long score;
};

struct Dataset {
    std::vector<int> data;
    std::vector<int> most_frequent;
};

class TrendGame : public Game::Game {
public:
    void generate_dataset();

public:
    TrendGame();

    std::vector<float> do_run_generation() override;

    void do_reset_players(NN *nets, size_t count) override;

    void do_post_training(NN *net) override;

private:
    std::vector<Player> players;
    std::vector<Dataset> datasets;
};


#endif //NEAT_GRU_TRENDGAME_H
