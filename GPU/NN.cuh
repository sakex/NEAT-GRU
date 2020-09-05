//
// Created by alexandre on 16.06.20.
//

#ifndef NEAT_GRU_NN_CUH
#define NEAT_GRU_NN_CUH

#include <vector>
#include <iostream>
#include "CudaGene.cuh"
#include "../NeuralNetwork/Topology.h"
#include <cuda.h>
#include <cuda_runtime.h>

/// Namespace containing the different classes relevant for the neural network
namespace NeuralNetworkCuda {

    class Neuron;

    /// Class Neural Network with GRU gates
    class NN {
    public:
        static size_t current_id;

        NN();

        /**
         * Constructor with a topology as input
         * @param topology Topology from which we create the Neural Network
         */
        explicit NN(NeuralNetwork::Topology &topology);

        virtual ~NN();

        /**
         * Compute the Neural Network with given inputs
         *
         * @param inputs_vector C double array of inputs
         * @return a vector of weights
         */
        __device__ void compute(
                const double *inputs_vector,
                int from,
                int to,
                int output_size,
                double *out,
                int write_from
        );

        /**
         * Inits the Network from a topology
         * @param topology The input topology
         */
        void init_topology(NeuralNetwork::Topology &topology);

        /// Resets the hidden state to 0
        __device__ void reset_state();

    private:
        int neurons_count;
        Neuron *layers;
        int layer_count;
        int *layer_addresses;
        cudaStream_t stream;
        size_t id;

    private:
        /**
         * Sets the inputs on the first layer
         * @param inputs_vector Array of doubles to initiate the inputs
         */
        __device__ void set_inputs(const double *inputs_vector, int from, int to);

        /// Delete data
        void delete_layers();
    };

} /* namespace NeuralNetwork */


#endif //NEAT_GRU_NN_CUH
