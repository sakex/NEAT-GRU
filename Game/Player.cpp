/*
 * Player.cpp
 *
 *  Created on: Aug 16, 2019
 *      Author: sakex
 */

#include "Player.h"

namespace Game {

    Player::Player(NeuralNetwork::NN &_brain): brain(_brain) {
    }

    void Player::decide() {
        do_decide();
    }

    void Player::reset(NeuralNetwork::NN &new_brain) {
        do_reset(new_brain);
    }

    long double Player::get_result() {
        return do_get_result();
    }

}
