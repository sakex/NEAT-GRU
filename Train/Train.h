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
#include "Random.h"

#ifndef CUDA_ENABLED

#include "neat/NeuralNetwork/NN.h"

#else
#include "../NeuralNetwork/CUDA/NN.cuh"
#endif

namespace Train {
    using namespace NeuralNetwork;

    class Train {
    public:
        Train(Game::Game *, int, int, int, int);

        Train(Game::Game *, int, int, int, int, Topology_ptr);

        ~Train();

        void start();

    public:
        Topology_ptr get_best() const;

    private:
        Game::Game *game;
        std::vector<Species_ptr> species;
        Topology_ptr best_historical_topology;
        int iterations;  // iterations < 0 -> run forever
        int inputs_count;
        int outputs_count;
        int max_individuals;
        bool new_best = false;
        std::vector<Topology_ptr> last_topologies;

        NN * brains;

    private:
        void random_new_species();

        void reset_species();

        void reset_players();

    private:
        void assign_results(std::vector<double> const &);

        std::vector<double> run_dataset();

        void natural_selection();

        void update_best();

        void reassign_species(std::vector<Topology_ptr> &);

        void extinct_species();

    private:
        std::vector<Topology_ptr> topologies_vector();

    private:
        void plot_best() const;

    };
}

#endif /* TRAIN_TRAIN_H_ */
