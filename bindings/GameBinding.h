//
// Created by alexandre on 20.05.20.
//

#ifndef NEAT_GRU_GAMEBINDING_H
#define NEAT_GRU_GAMEBINDING_H


#include "../Game/Game.h"
#include "../NeuralNetwork/Topology.h"
#include "structs.h"

#ifndef CUDA_ENABLED
#include "../NeuralNetwork/NN.h"
using namespace NeuralNetwork;
#else
#include "../GPU/NN.cuh"
using namespace NeuralNetworkCuda;
#endif

#if __EMSCRIPTEN__
#include <emscripten.h>
#define EMSCRIPTEN_EXPORT EMSCRIPTEN_KEEPALIVE
#else
#define EMSCRIPTEN_EXPORT
#endif

/// Wrapper around the struct Simulation to enable the training on a simulation
class GameBinding : public Game::Game {
public:
    /**
     * Constructor that takes a simulation
     *
     * @param _sim C struct simulation to be run
     */
    explicit GameBinding(Simulation *_sim);

    ~GameBinding() override = default;

private:
    /**
     * Will be called during the reset player phase a call the underlying simulation's reset_players()
     * @param brains Thet networks
     * @param size The number of networks
     */
    void do_reset_players(NN *brains, size_t size) override;

    /**
     * We be called during the run generation phase of the training
     * @return A vector of results
     */
    std::vector<double> do_run_generation() override;

    /**
     * Action to run at the end of the training period
     *
     * @param history - Historically best topologies
     * @param size - Size of the topologies
     */
    void do_post_training(NeuralNetwork::Topology const * history, size_t size) override;

private:
    /// The simulation to be run
    Simulation *sim;

    /// Number of networks in the last generation (bookkeeping)
    size_t _size;
};


#endif //NEAT_GRU_GAMEBINDING_H
