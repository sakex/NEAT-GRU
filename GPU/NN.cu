//
// Created by alexandre on 16.06.20.
//

#include "Neuron.cuh"
#include "NN.cuh"

#include "Connection.cuh"

namespace NeuralNetwork {
    __device__ Connection::Connection(double const _input_weight, double const _memory_weight,
                                      double const riw,
                                      double const rmw,
                                      double const uiw, double const umw, Neuron *output) :
            input_weight(_input_weight),
            memory_weight(_memory_weight),
            reset_input_weight(riw),
            reset_memory_weight(rmw),
            update_input_weight(uiw),
            update_memory_weight(umw),
            output{output} {
    }

    __device__ void Connection::init(double const _input_weight, double const _memory_weight, double const riw,
                                     double const rmw,
                                     double const uiw, double const umw, Neuron *_output) {
        memory = 0;
        prev_input = 0.;
        input_weight = _input_weight;
        memory_weight = _memory_weight;
        reset_input_weight = riw;
        reset_memory_weight = rmw;
        update_input_weight = uiw;
        update_memory_weight = umw;
        output = _output;
    }

    __device__ void Connection::activate(double const value) {
        double const prev_reset = output->get_prev_reset();
        memory = prev_input * input_weight + memory_weight * prev_reset * memory;
        prev_input = value;
        output->increment_memory(memory * memory_weight);
        output->increment_input(value * input_weight);
        output->increment_reset(value * reset_input_weight + memory * reset_memory_weight);
        output->increment_update(value * update_memory_weight + memory * update_memory_weight);
    }
}

#include "Neuron.cuh"

namespace NeuralNetwork {
    __device__ inline double fast_sigmoid(double const value) {
        return value / (1.f + std::abs(value));
    }


    __device__ inline double fast_tanh(double const x) {
        if (std::abs(x) >= 4.97) {
            double const values[2] = {-1., 1.};
            return values[x > 0.];
        }
        double const x2 = x * x;
        double const a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));
        double const b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));
        return a / b;
    }

    __device__ void
    Neuron::add_connection(Neuron *neuron, double const input_weight, double const memory_weight, double const riw,
                           double const rmw,
                           double const uiw, double const umw) {
        Connection *co = &connections[last_connection_added++];
        co->init(input_weight, memory_weight, riw, rmw, uiw, umw, neuron);
    }

    __device__ void Neuron::init() {
        connections = nullptr;
        input = 0.;
        memory = 0.;
        update = 0.;
        reset = 0.;
        prev_reset = 0.;
        last_connection_added = 0;
    }

    __device__ void Neuron::set_connections_count(size_t const value) {
        connections = new Connection[value];
    }

    __device__ void Neuron::increment_input(const double inc_value) {
        atomicAdd(&input, inc_value);
    }

    __device__ void Neuron::increment_update(const double inc_value) {
        atomicAdd(&update, inc_value);
    }

    __device__ void Neuron::increment_memory(const double inc_value) {
        atomicAdd(&memory, inc_value);
    }

    __device__ void Neuron::increment_reset(const double inc_value) {
        atomicAdd(&reset, inc_value);
    }

    __device__  void Neuron::set_value(double new_value) {
        input = new_value;
    }

    __device__ double Neuron::get_prev_reset() const {
        return prev_reset;
    }

    __device__ double Neuron::compute_value() {
        const double update_gate = fast_sigmoid(update);
        const double reset_gate = fast_sigmoid(reset);
        const double current_memory = fast_tanh(input + memory * reset_gate);
        const double value = update_gate * memory + (1.f - update_gate) * current_memory;
        prev_reset = reset_gate;
        reset_value();
        return value;
    }

    __device__ void Neuron::reset_value() {
        input = 0.;
        update = 0.;
        memory = 0.;
    }

    __device__ void Neuron::set_input_value(double new_value) {
        input = new_value;
    }

    __device__ double Neuron::get_value() {
        const double update_gate = fast_sigmoid(update);
        const double reset_gate = fast_sigmoid(reset);
        const double current_memory = fast_tanh(input + memory * reset_gate);
        const double value = update_gate * memory + (1.f - update_gate) * current_memory;
        prev_reset = reset_gate;
        reset_value();
        return fast_tanh(value);
    }

    __device__ void Neuron::free_connections() {
        delete[]connections;
        connections = nullptr;
    }
}

namespace NeuralNetwork {

    size_t NN::current_id = 0;

    NN::NN() :
            stream{nullptr},
            neurons_count(0),
            layers{nullptr},
            layer_count(0),
            layer_addresses{nullptr} {
        id = current_id++;
        cudaStreamCreate(&stream);
    }

    NN::NN(Topology_ptr const &topology) :
            stream{nullptr},
            neurons_count(0),
            layers(nullptr),
            layer_count(0),
            layer_addresses{nullptr} {
        id = current_id++;
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
            connection_counts.push_back({layer_addresses[it.first[0]] + it.first[1],
                                         it.second.phenotypes.size()}
            );

            for (Phenotype *phenotype : it.second.phenotypes) {
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

    __global__ void ff_connections(Connection *connections, double const value) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        connections[tid].activate(value);
    }

    __global__ void feed_forward_kernel(Neuron *layer, int from) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        Neuron *n = &layer[from + tid];
        double const value = n->compute_value();
        ff_connections<<<1, n->last_connection_added>>>(n->connections, value);
    }

    __global__ void get_result_kernel(Neuron *layer, double *arr, int from) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        arr[tid] = layer[from + tid].get_value();
        layer[from + tid].set_value(0);
    }


    double *NN::compute(const double *inputs_array) {
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

        auto *output = new double[distance];
        double *dev_output;
        cudaMalloc(&dev_output, distance * sizeof(double));
        get_result_kernel<<<1, distance, id, stream>>>(layers, dev_output, from);
        cudaStreamSynchronize(stream);
        cudaMemcpyAsync(output, dev_output, distance * sizeof(double), cudaMemcpyDeviceToHost, stream);
        cudaFree(dev_output);
        return output;
    }

    void NN::reset_state() {
        /*for (int it = 0; it < layer_count; ++it) {
            for (size_t j = 0; j < layers[it].size(); ++j) {
                layers[it][j]->reset_state();
            }
        }*/
    }

    __global__ void set_input_kernel(Neuron *layer, double const *inputs) {
        unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
        layer[tid].set_input_value(inputs[tid]);
    }

    void NN::set_inputs(const double *inputs_array) {
        int const from = layer_addresses[0];
        int const to = layer_addresses[1];
        int const distance = to - from;

        double *dev_inputs;
        cudaMalloc(&dev_inputs, distance * sizeof(double));
        cudaMemcpyAsync(dev_inputs, inputs_array, distance * sizeof(double), cudaMemcpyHostToDevice, stream);

        set_input_kernel<<<1, distance, id, stream>>>(layers, dev_inputs);
        cudaFree(dev_inputs);
    }

} /* namespace NeuralNetwork */
