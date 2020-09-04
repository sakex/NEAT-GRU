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

#ifndef CUDA_ENABLED
#include "../NeuralNetwork/NN.h"
using namespace NeuralNetwork;
#else
#include "../GPU/NN.cuh"
using namespace NeuralNetworkCuda;
#endif

/// Namespace for the Game abstract classes
namespace Game {

    /// Abstract class to implement to run a simulation to train on
    class Game {
    public:
        virtual ~Game() = default;

        /**
         * Run a generation, which means play a round of the game
         *
         * @return vector of scores of the players
         */
        std::vector<double> run_generation();


        /**
         * Reset the players between two generations
         *
         * @param nets, A C array of NeuralNetworks to pass to the players
         * @param count, An unsigned int that tells how many elements are in nets
         */
        void reset_players(NN *nets, size_t count);

        /**
         * Action to run after the training is done
         *
         * @param history The historic of the best topologies
         * @param size The size of the array of history
         */
        void post_training(NeuralNetwork::Topology const * history, size_t size);

    private:

        /**
         * Function to be implemented to run the generation
         *
         * @return vector of scores of the players
         */
        virtual std::vector<double> do_run_generation() = 0;

        /**
         * Function to be implemented to reset the players
         *
         * @param nets, A C array of NeuralNetworks to pass to the players
         * @param count, An unsigned int that tells how many elements are in nets
         */
        virtual void do_reset_players(NN * nets, size_t count) = 0;

        /**
         * Function to be implemented for the action to be run after the training
         *
         * @param history The historic of the best topologies
         * @param size The size of the array of history
         */
        virtual void do_post_training(NeuralNetwork::Topology const * history, size_t size) = 0;

    };
}

#endif /* GAME_GAME_H_ */
