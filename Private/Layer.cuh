//
// Created by alexandre on 16.06.20.
//

#ifndef NEAT_GRU_LAYER_CUH
#define NEAT_GRU_LAYER_CUH


#include "Neuron.cuh"

namespace NeuralNetwork {
    class Layer {
    public:
        __device__ __host__ Layer();
        ~Layer();

    public:
        __host__ void set_size(unsigned);

        unsigned size() const;

        __host__ void set_input(double const * inputs);

        __host__ void feed_forward();

        __host__ double * get_result();

        __host__ Neuron* raw();

    private:
        unsigned _size;
        Neuron *neurons;
    };
}


#endif //NEAT_GRU_LAYER_CUH
