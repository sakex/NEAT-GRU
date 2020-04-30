/*
 * Game.h
 *
 *  Created on: Aug 14, 2019
 *      Author: sakex
 */

#ifndef GAME_GAME_H_
#define GAME_GAME_H_

#include <vector>
#include <memory>
#include "neat/NeuralNetwork/Topology.h"
#include "Player.h"

using namespace NeuralNetwork;
namespace Game {

    class Game {
    public:
        virtual ~Game() = default;

        void run_generation();

        void set_last_results();

        void reset_players(std::vector<Topology_ptr> &);

        Player *post_training(Topology_ptr);

    private:
        virtual void do_run_generation() = 0;

        virtual void do_set_last_results() = 0;

        virtual void do_reset_players(std::vector<Topology_ptr> &) = 0;

        virtual Player *do_post_training(Topology_ptr) = 0;

    };
}

#endif /* GAME_GAME_H_ */
