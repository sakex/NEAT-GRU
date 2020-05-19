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
#include "../NeuralNetwork/Topology.h"
#include "Player.h"

using namespace NeuralNetwork;
namespace Game {

    class Game {
    public:
        virtual ~Game() = default;

        std::vector<double> run_generation();

        void reset_players(NN *, size_t);

        Player *post_training(Topology_ptr);

    private:
        virtual std::vector<double> do_run_generation() = 0;

        virtual void do_reset_players(NN *, size_t) = 0;

        virtual Player *do_post_training(Topology_ptr) = 0;

    };
}

#endif /* GAME_GAME_H_ */
