/*
 * Player.h
 *
 *  Created on: Aug 14, 2019
 *      Author: sakex
 */

#ifndef GAME_PLAYER_H_
#define GAME_PLAYER_H_

#include "neat/NeuralNetwork/Topology.h"

#ifndef CUDA_ENABLED

#include "neat/NeuralNetwork/NN.h"

#else
#include "../NeuralNetwork/CUDA/NN.cuh"
#endif

namespace Game {

    class Player {
    public:
        explicit Player(NeuralNetwork::Topology_ptr &);

        Player(Player &base);

        virtual ~Player();

        void decide();

        void set_last_result();

        void reset(NeuralNetwork::Topology_ptr &);

        NeuralNetwork::Topology_ptr get_topology() const;

        long double get_result();

        Player &operator=(Player const &base);

    protected:
        NeuralNetwork::NN *brain;
        NeuralNetwork::Topology_ptr topology;

    private:
        virtual void do_decide() = 0;

        virtual void do_reset(NeuralNetwork::Topology_ptr &) = 0;

        virtual long double do_get_result() = 0;

        virtual void do_set_last_result() = 0;
    };
}

#endif /* GAME_PLAYER_H_ */
