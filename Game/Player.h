/*
 * Player.h
 *
 *  Created on: Aug 14, 2019
 *      Author: sakex
 */

#ifndef GAME_PLAYER_H_
#define GAME_PLAYER_H_

#include "../NeuralNetwork/Topology.h"
#ifdef CUDA_ENABLED
#include "../GPU/NN.cuh"
#else
#include "../NeuralNetwork/NN.h"
#endif

/// Namespace for the Game abstract classes
namespace Game {

    /// Abstract class to implement for the players of the simulation
    class Player {
    public:
        /**
         * Constructor with a NeuralNetwork as parameter
         * @param network, The network to be passed
         */
        explicit Player(NeuralNetwork::NN * network);

        virtual ~Player() = default;

        /// Decide action to take
        void decide();

        /**
         * Reset with a new network
         *
         * @param network New network to reset
         */
        void reset(NeuralNetwork::NN * network);

        /**
         * Getter to get results
         * @return Loss function output
         */
        double get_result();

    protected:
        /// The NeuralNetwork of the player
        NeuralNetwork::NN * brain;

    private:
        /// Function to override to make decision
        virtual void do_decide() = 0;

        /**
         * Function to override to reset with a new network
         *
         * @param network New network to reset
         */
        virtual void do_reset(NeuralNetwork::NN * network) = 0;

        /**
         * Function to implement for the getter to get results
         * @return Loss function output
         */
        virtual double do_get_result() = 0;
    };
}

#endif /* GAME_PLAYER_H_ */
