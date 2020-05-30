/*
 * Game.cpp
 *
 *  Created on: Aug 16, 2019
 *      Author: sakex
 */

#include "Game.h"

namespace Game {

    std::vector<double> Game::run_generation() {
        return do_run_generation();
    }


    void Game::reset_players(NN * brains, size_t const size) {
        do_reset_players(brains, size);
    }

    Player *Game::post_training(Topology_ptr top) {
        return do_post_training(std::move(top));
    }
}