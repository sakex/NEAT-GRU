/*
 * Game.cpp
 *
 *  Created on: Aug 16, 2019
 *      Author: sakex
 */

#include "Game.h"

namespace Game {

    void Game::run_generation() {
        do_run_generation();
    }

    void Game::set_last_results() {
        do_set_last_results();
    }


    void Game::reset_players(std::vector<Topology_ptr> &topologies) {
        do_reset_players(topologies);
    }

    Player *Game::post_training(Topology_ptr top) {
        return do_post_training(std::move(top));
    }
}
