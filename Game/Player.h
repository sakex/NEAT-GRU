/*
 * Player.h
 *
 *  Created on: Aug 14, 2019
 *      Author: sakex
 */

#ifndef GAME_PLAYER_H_
#define GAME_PLAYER_H_

#include "../NeuralNetwork/Topology.h"

#ifndef CUDA_ENABLED

#include "../NeuralNetwork/NN.h"

#else
#include "../NeuralNetwork/CUDA/NN.cuh"
#endif

namespace Game {

    class Player {
    public:
        explicit Player(NeuralNetwork::NN *);

        virtual ~Player() = default;

        void decide();

        void reset(NeuralNetwork::NN *);

        long double get_result();

    protected:
        NeuralNetwork::NN * brain;

    private:
        virtual void do_decide() = 0;

        virtual void do_reset(NeuralNetwork::NN *) = 0;

        virtual long double do_get_result() = 0;
    };
}

#endif /* GAME_PLAYER_H_ */
