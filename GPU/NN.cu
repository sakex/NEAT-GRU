//
// Created by alexandre on 16.06.20.
//

#include "Neuron.cuh"
#include "NN.cuh"

namespace NeuralNetwork {

    float sigmoid(float const value) {
        return 1 / (1 + std::exp(-value));
    }

    void softmax(float *input, unsigned size) {
        float total = 0;
        for (unsigned i = 0; i < size; ++i) {
            if (input[i] < 0.) input[i] = 0.;
            else total += input[i];
        }
        if (total > 1) {
            for (unsigned i = 0; i < size; ++i) {
                input[i] /= total;
            }
        }
    }

    NN::NN() :
    neurons_count(0),
    layers{nullptr},
    layer_count(0),
    layer_addresses{nullptr} {
    }

    NN::NN(Topology_ptr const &topology) : neurons_count(0), layers(nullptr), layer_count(0), layer_addresses{nullptr} {
        init_topology(topology);
    }

    NN::~NN() {
        delete[] layer_addresses;
        delete_layers();
    }

    __global__ void free_connections_kernel(Neuron *neurons) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        neurons[tid].free_connections();
    }

    void NN::delete_layers() {
        free_connections_kernel<<<1, neurons_count>>>(layers);
        cudaDeviceSynchronize();
        cudaFree(layers);
    }

    __global__ void set_connections_kernel(Neuron *layers, CUDAConnectionCount *connection_count, size_t N) {
        for (size_t it = 0; it < N; ++it) {
            CUDAConnectionCount *count = &connection_count[it];
            Neuron *input_neuron_ptr = &layers[count->pos];
            input_neuron_ptr->set_connections_count(count->count);
        }
    }

    __global__ void connect_neurons_kernel(Neuron *layers, CUDAPhenotype *phenotypes, size_t N) {
        for (size_t it = 0; it < N; ++it) {
            CUDAPhenotype *phen = &phenotypes[it];
            Neuron *input_neuron_ptr = &layers[phen->input_pos];
            Neuron *output_neuron_ptr = &layers[phen->output_pos];

            printf("Input: %i, Output: %i\n", phen->input_pos, phen->output_pos);

            input_neuron_ptr->add_connection(
                    output_neuron_ptr,
                    phen->input_weight,
                    phen->memory_weight,
                    phen->reset_input_weight,
                    phen->reset_memory_weight,
                    phen->update_input_weight,
                    phen->update_memory_weight);
        }
    }

    void NN::init_topology(Topology_ptr const &topology) {
        cudaError err;
        layer_count = topology->get_layers();
        std::vector<int> const &sizes = topology->get_layers_size();
        layer_addresses = new int[layer_count + 1];
        neurons_count = 0;
        int i = 0;
        for (; i < layer_count; ++i) {
            layer_addresses[i] = neurons_count;
            neurons_count += sizes[i];
        }
        layer_addresses[i] = neurons_count;
        err = cudaMalloc(&layers, sizeof(Neuron) * (unsigned long)neurons_count);
        if(err) {
            std::cerr << "Failed at cudaMalloc Neurons with error:\n" << err << ": " << cudaGetErrorString(err) << std::endl;
            throw;
        }
        Topology::relationships_map &relationships = topology->get_relationships();
        std::vector<CUDAPhenotype> phenotype_vec;
        std::vector<CUDAConnectionCount> connection_counts;

        for (auto &it : relationships) {
            connection_counts.push_back({
                                                layer_addresses[it.first[0]] + it.first[1],
                                                it.second.size()}
            );

            for (Phenotype *phenotype : it.second) {
                if (phenotype->is_disabled()) {
                    continue;
                }
                Phenotype::point input = phenotype->get_input();
                Phenotype::point output = phenotype->get_output();
                phenotype_vec.push_back({
                                                phenotype->get_input_weight(),
                                                phenotype->get_memory_weight(),
                                                phenotype->get_reset_input_weight(),
                                                phenotype->get_reset_memory_weight(),
                                                phenotype->get_update_input_weight(),
                                                phenotype->get_update_memory_weight(),
                                                layer_addresses[input[0]] + input[1],
                                                layer_addresses[output[0]] + output[1]
                                        });
            }
        }
        CUDAConnectionCount *device_counts;
        err = cudaMalloc(&device_counts, sizeof(CUDAConnectionCount) * connection_counts.size());
        if(err) {
            std::cerr << "Failed at cudaMalloc CUDAConnectionCount with error:\n" << cudaGetErrorString(err) << std::endl;
            throw;
        }
        err = cudaMemcpy(device_counts, connection_counts.data(), sizeof(CUDAConnectionCount) * connection_counts.size(),
                   cudaMemcpyHostToDevice);
        if(err) {
            std::cerr << "Failed at cudaMemcpy CUDAConnectionCount with error:\n" << cudaGetErrorString(err) << std::endl;
            throw;
        }
        set_connections_kernel<<<1, 1>>>(layers, device_counts, connection_counts.size());
        err = cudaDeviceSynchronize();
        if(err) {
            std::cerr << "Failed at set_connections_kernel " << std::endl;
            throw;
        }
        err = cudaFree(device_counts);
        if(err) {
            std::cerr << "Failed at cudaFree(device_counts) " << std::endl;
            throw;
        }


        CUDAPhenotype *device_phenotypes;
        cudaMalloc(&device_phenotypes, sizeof(CUDAPhenotype) * phenotype_vec.size());
        cudaMemcpy(device_phenotypes, phenotype_vec.data(), sizeof(CUDAPhenotype) * phenotype_vec.size(),
                   cudaMemcpyHostToDevice);
        connect_neurons_kernel<<<1, 1>>>(layers, device_phenotypes, phenotype_vec.size());
        err = cudaDeviceSynchronize();
        if(err) {
            std::cerr << "Failed at connect_neurons_kernel " << cudaGetErrorString(err) << std::endl;
            throw;
        }
        cudaFree(device_phenotypes);
    }

    __global__ void feed_forward_kernel(Neuron *layer, int from) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        layer[from + tid].feed_forward();
        __syncthreads();
    }

    __global__ void get_result_kernel(Neuron *layer, float *arr, int from) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        arr[tid] = layer[from + tid].get_value();
        layer[from + tid].set_value(0);
        __syncthreads();
    }


    float *NN::compute(const float *inputs_array) {
        cudaError err;
        set_inputs(inputs_array);
        int from, to, distance;
        for (size_t pos = 0; pos < layer_count - 1; ++pos) {
            from = layer_addresses[pos];
            to = layer_addresses[pos + 1];
            distance = to - from;
            feed_forward_kernel<<<1, distance>>>(layers, from);
            err = cudaDeviceSynchronize();
            if(err) {
                std::cerr << "Failed at feedforward index: " << pos << std::endl;
                throw;
            }
        }

        auto *output = new float[distance];
        float *dev_output;
        cudaMalloc(&dev_output, distance * sizeof(float));
        get_result_kernel<<<1, distance>>>(layers, dev_output, from);
        cudaMemcpy(output, dev_output, distance * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(dev_output);
        err = cudaDeviceSynchronize();
        if(err) {
            std::cerr << "Failed at copy output " << std::endl;
            throw;
        }

        softmax(output, distance);
        return output;
    }

    void NN::reset_state() {
        /*for (int it = 0; it < layer_count; ++it) {
            for (size_t j = 0; j < layers[it].size(); ++j) {
                layers[it][j]->reset_state();
            }
        }*/
    }

    __global__ void set_input_kernel(Neuron *layer, float const *inputs) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        layer[tid].set_input_value(inputs[tid]);
    }

    void NN::set_inputs(const float *inputs_array) {
        int const from = layer_addresses[0];
        int const to = layer_addresses[1];
        int const distance = from - to;

        float *dev_inputs;
        cudaMalloc(&dev_inputs, distance * sizeof(float));
        cudaMemcpy(dev_inputs, inputs_array, distance * sizeof(float), cudaMemcpyHostToDevice);

        set_input_kernel<<<1, distance>>>(layers, dev_inputs);
        cudaFree(dev_inputs);
        cudaDeviceSynchronize();
    }

} /* namespace NeuralNetwork */
