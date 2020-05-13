//
// Created by alexandre on 13.05.20.
//

#include "Layer.h"

namespace NeuralNetwork {
    Layer::Layer(): _size(0), neurons(nullptr) {

    }

    Layer::~Layer() {
        delete[] neurons;
    }

    void Layer::set_size(unsigned int const new_size) {
        _size = new_size;
        neurons = new Neuron[new_size];
    }

    unsigned Layer::size() const {
        return _size;
    }

    Neuron* Layer::operator[](unsigned int const it) {
        return &neurons[it];
    }

}