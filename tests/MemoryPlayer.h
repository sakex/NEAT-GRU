//
// Created by sakex on 02.06.20.
//

#ifndef NEAT_GRU_MEMORYPLAYER_H
#define NEAT_GRU_MEMORYPLAYER_H

#include "../NeuralNetwork/NN.h"
#include "MemoryGrid.h"

class MemoryPlayer {
public:
    explicit MemoryPlayer(NeuralNetwork::NN * net);

    void play_rounds(std::size_t rounds);

    long score() const;

private:
    NeuralNetwork::NN * network;
    MemoryGrid grid;
    long _score = 0;

    long play();
};


#endif //NEAT_GRU_MEMORYPLAYER_H
