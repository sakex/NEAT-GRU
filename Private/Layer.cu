//
// Created by alexandre on 16.06.20.
//

#include "Layer.cuh"

namespace NeuralNetwork {
    __global__ void set_input_kernel(Neuron *layer, double const *inputs) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        layer[tid].set_input_value(inputs[tid]);
    }

    __global__ void feed_forward_kernel(Neuron *layer) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        layer[tid].feed_forward();
    }

    __global__ void get_result_kernel(Neuron *layer, double * arr) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        arr[tid] = layer[tid].get_value();
        layer[tid].set_value(0);
    }


    __device__ __host__ Layer::Layer(): _size(0), neurons(nullptr) {

    }

    Layer::~Layer() {
        cudaFree(neurons);
    }

    __host__ void Layer::set_size(unsigned int const new_size) {
        _size = new_size;
        cudaMalloc(&neurons, sizeof(Neuron)*new_size);
    }

    unsigned Layer::size() const {
        return _size;
    }

    __host__ void Layer::set_input(double const *inputs) {
        set_input_kernel<<<1, _size>>>(neurons, inputs);
    }

    __host__ void Layer::feed_forward() {
        feed_forward_kernel<<<1, _size>>>(neurons);
    }
    __host__ double * Layer::get_result() {
        auto *output = new double[_size];
        double *dev_output;
        cudaMalloc(&dev_output, _size*sizeof(double));
        get_result_kernel<<<1, _size>>>(neurons, dev_output);
        cudaMemcpy(output, dev_output, _size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(dev_output);
        return output;
    }
}