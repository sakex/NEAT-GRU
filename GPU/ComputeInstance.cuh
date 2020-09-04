//
// Created by alexandre on 03.09.20.
//

#ifndef NEAT_GRU_COMPUTEINSTANCE_H
#define NEAT_GRU_COMPUTEINSTANCE_H


#include "NN.cuh"

extern "C" {
struct Dim {
    unsigned long int x;
    unsigned long int y;
    unsigned long int z;
};


__global__ void compute_kernel(Dim dim, NeuralNetworkCuda::NN *networks, double *data, unsigned long int networks_count, unsigned long int output_size);

struct ComputeInstance {
    Dim dim;
    double *data{nullptr};
    double *h_output{nullptr};
    double *d_output{nullptr};
    unsigned long int networks_count = 0;
    NeuralNetworkCuda::NN *networks{nullptr};
};

void compute_gpu_instance(ComputeInstance *instance, const unsigned int output_size);
void update_dataset_gpu_instance(ComputeInstance *instance, double const *host_data);
void set_networks_gpu_instance(ComputeInstance *instance, NeuralNetworkCuda::NN* nets, unsigned long int count);

}


#endif //NEAT_GRU_COMPUTEINSTANCE_H
