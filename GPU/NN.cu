//
// Created by alexandre on 16.06.20.
//

#include "Neuron.cuh"
#include "NN.cuh"

namespace NeuralNetwork {

    size_t NN::current_id = 0;

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
        id = current_id++;
    }

    NN::NN(Topology_ptr const &topology) : neurons_count(0), layers(nullptr), layer_count(0), layer_addresses{nullptr} {
        cudaStreamCreate(&stream);
        init_topology(topology);
    }

    NN::~NN() {
        delete[] layer_addresses;
        delete_layers();
        cudaStreamDestroy(stream);
    }

    __global__ void free_connections_kernel(Neuron *neurons) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        neurons[tid].free_connections();
    }

    void NN::delete_layers() {
        free_connections_kernel<<<1, neurons_count, id, stream>>>(layers);
        cudaFree(layers);
    }

    __global__ void init_kernel(Neuron *neurons) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        neurons[tid].init();
    }

    __global__ void set_connections_kernel(Neuron *layers, CUDAConnectionCount *connection_count) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        CUDAConnectionCount *count = &connection_count[tid];
        Neuron *input_neuron_ptr = &layers[count->pos];
        input_neuron_ptr->set_connections_count(count->count);
    }

    __global__ void connect_neurons_kernel(Neuron *layers, CUDAPhenotype *phenotypes, size_t N) {
        for (size_t it = 0; it < N; ++it) {
            CUDAPhenotype *phen = &phenotypes[it];
            Neuron *input_neuron_ptr = &layers[phen->input_pos];
            Neuron *output_neuron_ptr = &layers[phen->output_pos];

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
        cudaMalloc(&layers, sizeof(Neuron) * neurons_count);
        init_kernel<<<1, neurons_count, id, stream>>>(layers);
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
        cudaMalloc(&device_counts, sizeof(CUDAConnectionCount) * connection_counts.size());

        cudaMemcpyAsync(device_counts, connection_counts.data(),
                   sizeof(CUDAConnectionCount) * connection_counts.size(),
                   cudaMemcpyHostToDevice, stream);

        set_connections_kernel<<<1, connection_counts.size(), id, stream>>>(layers, device_counts);

        cudaFree(device_counts);


        CUDAPhenotype *device_phenotypes;
        cudaMalloc(&device_phenotypes, sizeof(CUDAPhenotype) * phenotype_vec.size());
        cudaMemcpyAsync(device_phenotypes, phenotype_vec.data(), sizeof(CUDAPhenotype) * phenotype_vec.size(),
                   cudaMemcpyHostToDevice, stream);
        connect_neurons_kernel<<<1, 1, id, stream>>>(layers, device_phenotypes, phenotype_vec.size());
        cudaFree(device_phenotypes);
    }

    __global__ void ff_connections(Connection *connections, float const value) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        connections[tid].activate(value);
    }

    __global__ void feed_forward_kernel(Neuron *layer, int from) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        Neuron *n = &layer[from + tid];
        float const value = n->compute_value();
        ff_connections<<<1, n->last_connection_added>>>(n->connections, value);
    }

    __global__ void get_result_kernel(Neuron *layer, float *arr, int from) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        arr[tid] = layer[from + tid].get_value();
        layer[from + tid].set_value(0);
    }


    float *NN::compute(const float *inputs_array) {
        set_inputs(inputs_array);
        size_t pos = 0;

        for (; pos < layer_count - 1; ++pos) {
            int from = layer_addresses[pos];
            int to = layer_addresses[pos + 1];
            int distance = to - from;
            feed_forward_kernel<<<1, distance, id, stream>>>(layers, from);
        }

        int from = layer_addresses[pos];
        int to = layer_addresses[pos + 1];
        int distance = to - from;

        auto *output = new float[distance];
        float *dev_output;
        cudaMalloc(&dev_output, distance * sizeof(float));
        get_result_kernel<<<1, distance, id, stream>>>(layers, dev_output, from);
        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(output, dev_output, distance * sizeof(float), cudaMemcpyDeviceToHost, stream);
        cudaFree(dev_output);
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
        int const distance = to - from;

        float *dev_inputs;
        cudaMalloc(&dev_inputs, distance * sizeof(float));
        cudaMemcpyAsync(dev_inputs, inputs_array, distance * sizeof(float), cudaMemcpyHostToDevice, stream);

        set_input_kernel<<<1, distance, id, stream>>>(layers, dev_inputs);
        cudaFree(dev_inputs);
    }

} /* namespace NeuralNetwork */
