//
// Created by alexandre on 16.06.20.
//


#include "Neuron.cuh"
#include "Layer.cuh"

namespace NeuralNetwork {
    __global__ void set_input_kernel(Neuron *layer, float const *inputs) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        layer[tid].set_input_value(inputs[tid]);
    }

    __global__ void feed_forward_kernel(Neuron *layer) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        layer[tid].feed_forward();
    }

    __global__ void get_result_kernel(Neuron *layer, float * arr) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        arr[tid] = layer[tid].get_value();
        layer[tid].set_value(0);
    }


    Layer::Layer(): _size(0), neurons(nullptr) {

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

    __host__ void Layer::set_input(float const *inputs) {
        set_input_kernel<<<1, _size>>>(neurons, inputs);
    }

    __host__ void Layer::feed_forward() {
        feed_forward_kernel<<<1, _size>>>(neurons);
    }
    __host__ float * Layer::get_result() {
        auto *output = new float[_size];
        float *dev_output;
        cudaMalloc(&dev_output, _size*sizeof(float));
        get_result_kernel<<<1, _size>>>(neurons, dev_output);
        cudaMemcpy(output, dev_output, _size*sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_output);
        /*
        for(size_t it = 0; it < _size; ++it) {
            printf("%f, ", output[it]);
        }
        printf("\n");*/
        return output;
    }

    __host__ Neuron* Layer::raw() {
        return neurons;
    }
}