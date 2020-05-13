//
// Created by alexandre on 13.05.20.
//

#ifndef TRADING_LAYER_H
#define TRADING_LAYER_H

#include "Neuron.h"

namespace NeuralNetwork {
    class Layer {
    public:
        Layer();
        ~Layer();

    public:
        void set_size(unsigned);

        unsigned size() const;

        Neuron* operator[](unsigned);

    private:
        unsigned _size;
        Neuron *neurons;
    };
}

#endif //TRADING_LAYER_H
