//
// Created by alexandre on 16.06.20.
//


#include "Neuron.cuh"
#include "Layer.cuh"

namespace NeuralNetwork {
    __global__ void get_result_kernel(Neuron *layer, float * arr) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        arr[tid] = layer[tid].get_value();
        layer[tid].set_value(0);
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
}