/*
 * Train.h
 *
 *  Created on: May 26, 2019
 *      Author: sakex
 */

#ifndef TRAIN_TRAIN_H_
#define TRAIN_TRAIN_H_

#include <unordered_map>
#include <vector>
#include <memory>
#include <algorithm>
#include <functional>
#include <mutex>

#include "../Timer.h"
#include "../Threading/multithreaded_methods.h"
#include "../NeuralNetwork/Topology.h"
#include "../Private/Species.h"
#include "../Private/Generation.h"
#include "../Game/Game.h"
#include "../Game/Player.h"
#include "../Serializer/Serializer.hpp"
#include "../Private/Random.h"
#include "static.h"

#ifdef CUDA_ENABLED
#include "../NeuralNetwork/NN.cuh"
#else
#include "../NeuralNetwork/NN.h"
#endif

/// Namespace that solely countains the Train class
namespace Train {
    using namespace NeuralNetwork;

    /// Class to train topologies on a game
    class Train {
    public:

        /**
         * Constructor without a base Topology
         *
         * @param _game The game on which we train our networks
         * @param _iterations Number of iterations until we end, set to less than 0 for infinite training
         * @param _max_individuals Maximum number of players for a given generation
         * @param _max_layers Maximum number of layers in a given network
         * @param _max_per_layer Maximum number of neurons per layer in a given network
         * @param inputs Number of input Neurons on the first layer of the Neural Networks
         * @param outputs Number of output Neurons on the last layer of the Neural Networks
         */
        Train(Game::Game *_game, int _iterations, int _max_individuals, int _max_layers, int _max_per_layer, int inputs,
              int outputs);

        /**
         * Constructor with a base Topology to continue training it
         *
         * @param _game The game on which we train our networks
         * @param _iterations Number of iterations until we end, set to less than 0 for infinite training
         * @param _max_individuals Maximum number of players for a given generation
         * @param _max_layers Maximum number of layers in a given network
         * @param _max_per_layer Maximum number of neurons per layer in a given network
         * @param inputs Number of input Neurons on the first layer of the Neural Networks
         * @param outputs Number of output Neurons on the last layer of the Neural Networks
         * @param top The pretrained topology to continue training
         */
        Train(Game::Game *_game, int _iterations, int _max_individuals, int _max_layers, int _max_per_layer, int inputs,
              int outputs, Topology_ptr top);

        ~Train();

        /// Starts training
        void start();

    public:
        Topology_ptr get_best() const;

    private:
        // Members
        Game::Game *game;
        std::vector<Species_ptr> species;
        Topology_ptr best_historical_topology;
        int iterations;  // iterations < 0 -> run forever
        int inputs_count;
        int outputs_count;
        int max_individuals;
        bool new_best = false;
        std::vector<Topology_ptr> last_topologies;

        NN *brains;

    private:
        /// Generate random species for the first generation
        void random_new_species();

        /// Calls reassign_species
        void reset_species();

        /**
         * Reset players by deleting former Networks and assigning new ones
         * Inits the new Networks with the topologies
         */
        void reset_players();

    private:
        /**
         * Pairs topologies and their results returned by the Simulation
         * @param results the results from the simulation's run_generation()
         */
        void assign_results(std::vector<double> const &results);

        /**
         * Calls run_generation on the Game
         * @return a vector of results
         */
        std::vector<double> run_generation();

        /// Do the natural selection
        void natural_selection();

        /// Updates best historical topology
        void update_best();

        /**
         * Checks if topologies belong to the same species and reassigns them
         * @param topologies Vector of topologies
         */
        void reassign_species(std::vector<Topology_ptr> &topologies);

        /// Extincts species if we have more than 20
        void extinct_species();

    private:
        /**
         * Returns a vector with all topologies
         * @return Vector of topologies
         */
        std::vector<Topology_ptr> topologies_vector();

    private:
        /// Action to be executed after the training
        void post_training() const;

    };
}

#endif /* TRAIN_TRAIN_H_ */
