/*
 * Player.cpp
 *
 *  Created on: Aug 16, 2019
 *      Author: sakex
 */

#include "Player.h"

namespace Game {

    Player::Player(NeuralNetwork::Topology_ptr &brain_topology) :
            topology(brain_topology) {
        this->brain = new NeuralNetwork::NN(topology);
    }

    Player::Player(Player &base) {
        brain = new NeuralNetwork::NN(*base.brain);
        topology = base.topology;
    }

    Player &Player::operator=(Player const &base) {
        if (this != &base) {
            delete brain;
            brain = new NeuralNetwork::NN(*base.brain);
            topology = base.topology;
        }
        return *this;
    }

    Player::~Player() {
        delete brain;
    }

    void Player::decide() {
        do_decide();
    }

    void Player::reset(NeuralNetwork::Topology_ptr &brain_topology) {
        do_reset(brain_topology);
    }

    long double Player::get_result() {
        return do_get_result();
    }

    NeuralNetwork::Topology_ptr Player::get_topology() const {
        return topology;
    }

    void Player::set_last_result() {
        do_set_last_result();
    }

}
