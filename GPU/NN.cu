//
// Created by alexandre on 16.06.20.
//

#include "Layer.cuh"
#include "NN.cuh"
#include "Connection.cuh"
#include "Neuron.cuh"

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

    NN::NN() : layers(nullptr), layer_count(0) {
    }

    NN::NN(Topology_ptr const &topology) : layers(nullptr), layer_count(0) {
        init_topology(topology);
    }

    NN::~NN() {
        delete_layers();
    }

    void NN::delete_layers() {
        cudaFree(layers);
    }

    __global__ void connect_neurons_kernel(Neuron **layers, CUDAPhenotype *phenotypes) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        CUDAPhenotype *phen = &phenotypes[tid];
        Neuron *input_neuron_ptr = &layers[phen->input_pos[0]][phen->input_pos[1]];
        Neuron *output_neuron_ptr = &layers[phen->input_pos[0]][phen->input_pos[1]];

        printf("%f\n", phen->reset_input_weight);
        input_neuron_ptr->add_connection(
                output_neuron_ptr,
                phen->input_weight,
                phen->memory_weight,
                phen->reset_input_weight,
                phen->reset_memory_weight,
                phen->update_input_weight,
                phen->update_memory_weight);
    }

    void NN::init_topology(Topology_ptr const &topology) {
        layer_count = topology->get_layers();
        delete_layers();
        layers = new Layer[layer_count];
        std::vector<int> const &sizes = topology->get_layers_size();
        for (int i = 0; i < layer_count; ++i) {
            layers[i].set_size(sizes[i]);
        }
        auto **raw_layers = new Neuron *[layer_count];
        Neuron **device_raw_layers;
        for (int it = 0; it < layer_count; ++it) {
            raw_layers[it] = layers[it].raw();
        }
        cudaMalloc(&device_raw_layers, sizeof(Neuron *) * layer_count);
        cudaMemcpy(device_raw_layers, raw_layers, sizeof(Neuron *) * layer_count, cudaMemcpyHostToDevice);
        Topology::relationships_map &relationships = topology->get_relationships();
        std::vector<CUDAPhenotype> phenotype_vec;

        for (auto &it : relationships) {
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
                                                {input[0], input[1]},
                                                {output[0], output[1]}
                                        });
            }
        }
        CUDAPhenotype *device_phenotypes;
        cudaMalloc(&device_phenotypes, sizeof(CUDAPhenotype) * phenotype_vec.size());
        cudaMemcpy(device_phenotypes, phenotype_vec.data(), sizeof(CUDAPhenotype) * phenotype_vec.size(),
                   cudaMemcpyHostToDevice);
        connect_neurons_kernel<<<phenotype_vec.size(), 1>>>(device_raw_layers, device_phenotypes);
        cudaDeviceSynchronize();
        cudaFree(device_phenotypes);
        cudaFree(device_raw_layers);
    }

    float *NN::compute(const float *inputs_array) {
        set_inputs(inputs_array);
        for (int it = 0; it < layer_count - 1; ++it) {
            layers[it].feed_forward();
        }
        Layer &last_layer = layers[layer_count - 1];
        float *output = last_layer.get_result();
        softmax(output, last_layer.size());
        return output;
    }

    void NN::reset_state() {
        /*for (int it = 0; it < layer_count; ++it) {
            for (size_t j = 0; j < layers[it].size(); ++j) {
                layers[it][j]->reset_state();
            }
        }*/
    }

    void NN::set_inputs(const float *inputs_array) {
        Layer &first_layer = layers[0];
        first_layer.set_input(inputs_array);
    }

} /* namespace NeuralNetwork */
