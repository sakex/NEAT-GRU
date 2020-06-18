//
// Created by alexandre on 16.06.20.
//

#ifndef NEAT_GRU_LAYER_CUH
#define NEAT_GRU_LAYER_CUH

namespace NeuralNetwork {
    class Neuron;

    class Layer {
    public:
        Layer();
        ~Layer();

    public:
        void set_size(unsigned);

        unsigned size() const;

        void set_input(float const * inputs);

        void feed_forward();

        float * get_result();

        Neuron* raw();

        void free_neurons();

    private:
        unsigned _size;
        Neuron *neurons;
    };
}


#endif //NEAT_GRU_LAYER_CUH
