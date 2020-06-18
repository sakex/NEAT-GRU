/*
 * Game.cpp
 *
 *  Created on: Aug 16, 2019
 *      Author: sakex
 */

#include "Game.h"

namespace Game {

    std::vector<float> Game::run_generation() {
        return do_run_generation();
    }


    void Game::reset_players(NN * brains, size_t const size) {
        do_reset_players(brains, size);
    }

    void Game::post_training(NN* network) {
        do_post_training(network);
    }
}
