//
// Created by sakex on 02.06.20.
//

#ifndef NEAT_GRU_MEMORYPLAYER_H
#define NEAT_GRU_MEMORYPLAYER_H

#ifdef CUDA_ENABLED
#include "../../GPU/NN.cuh"
#else
#include "NN.h"
#endif
#include "MemoryGrid.h"

class MemoryPlayer {
public:
    explicit MemoryPlayer(NeuralNetwork::NN * net);

    void play_rounds(std::vector<numbers_list> const & datasets, bool showing = false);

    long score() const;

private:
    NeuralNetwork::NN * network;
    MemoryGrid grid;
    long _score = 0;

    long play(bool showing);
};


#endif //NEAT_GRU_MEMORYPLAYER_H
