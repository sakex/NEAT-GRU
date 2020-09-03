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


__global__ void compute_kernel(Dim dim, NeuralNetwork::NN *networks, double *data, unsigned long int networks_count, unsigned long int output_size);

struct ComputeInstance {
public:
    explicit ComputeInstance(Dim dim);
    void set_networks(NeuralNetwork::NN* nets, unsigned long int count);
    void update_dataset(double *host_data);
    void compute(unsigned int output_size);
    double *get_output() const;

private:
    Dim dim;
    double *data{nullptr};
    double *h_output{nullptr};
    double *d_output{nullptr};
    unsigned long int networks_count = 0;
    NeuralNetwork::NN *networks{nullptr};
};

ComputeInstance * create_compute_instance(Dim dim);
}


#endif //NEAT_GRU_COMPUTEINSTANCE_H
