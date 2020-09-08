//
// Created by alexandre on 16.06.20.
//

#include "Neuron.cuh"
#include "NN.cuh"

#include "Connection.cuh"

namespace NeuralNetworkCuda {
    __device__ void Connection::init(double const _input_weight, double const _memory_weight, double const riw,
                                     double const rmw,
                                     double const uiw, double const umw, Neuron *_output) {
        memory = 0.f;
        prev_input = 0.f;
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

        double const update_mem = memory * memory_weight;
        output->increment_state(update_mem,
                                value * input_weight,
                                value * reset_input_weight + memory * reset_memory_weight,
                                value * update_input_weight + memory * update_memory_weight);

    }

    __device__ inline void Connection::reset_state() {
        memory = 0.f;
        prev_input = 0.f;
    }
}

namespace NeuralNetworkCuda {
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

    __device__ inline void
    Neuron::increment_state(double const mem, double const inp, double const res, double const upd) {
        memory += mem;
        input += inp;
        reset += res;
        update += upd;
    }

    __device__ double Neuron::get_prev_reset() const {
        return prev_reset;
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

    __device__ inline void Neuron::set_connections_count(size_t count) {
        connections = new Connection[count]();
    }

    __device__ inline void Neuron::feed_forward() {
        double const update_gate = fast_sigmoid(update);
        double const reset_gate = fast_sigmoid(reset);

        const double current_memory = fast_tanh(input + memory * reset_gate);
        const double value = update_gate * memory + (1.f - update_gate) * current_memory;
        for (int i = 0; i < last_connection_added; ++i) {
            connections[i].activate(value);
        }
        prev_reset = reset_gate;
        reset_value();
    }

    __device__ inline void Neuron::reset_state() {
        reset_value();
        prev_reset = 0.;
        for (int i = 0; i < last_connection_added; ++i) {
            connections[i].reset_state();
        }
    }
}

namespace NeuralNetworkCuda {

    size_t NN::current_id = 0;

    NN::NN() :
            stream{nullptr},
            neurons_count(0),
            layers{nullptr},
            layer_count(0),
            layer_addresses{nullptr} {
        id = current_id++;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }
    }

    NN::NN(NeuralNetwork::Topology &topology) :
            stream{nullptr},
            neurons_count(0),
            layers(nullptr),
            layer_count(0),
            layer_addresses{nullptr} {
        id = current_id++;
        cudaError_t err = cudaStreamCreate(&stream);
        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }
        init_topology(topology);
    }

    NN::~NN() {
        delete[] layer_addresses;
        delete_layers();
        cudaError_t err = cudaStreamDestroy(stream);
        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
        }
    }

    __global__ void free_connections_kernel(Neuron *neurons) {
        unsigned int tid = threadIdx.x;
        neurons[tid].free_connections();
    }

    void NN::delete_layers() {
        free_connections_kernel<<<1, neurons_count, id, stream>>>(layers);
        cudaError_t err = cudaGetLastError();
        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }
        err = cudaFree(layers);
        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }
    }

    __global__ void init_kernel(Neuron *neurons) {
        unsigned int tid = threadIdx.x;
        neurons[tid].init();
    }

    __global__ void set_connections_kernel(Neuron *layers, CUDAConnectionCount *connection_count) {
        unsigned int tid = threadIdx.x;
        CUDAConnectionCount *count = &connection_count[tid];
        Neuron *input_neuron_ptr = &layers[count->pos];
        input_neuron_ptr->set_connections_count(count->count);
    }

    __global__ void connect_neurons_kernel(Neuron *layers, CUDAGene *genes, size_t N) {
        for (size_t it = 0; it < N; ++it) {
            CUDAGene *phen = &genes[it];
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

    void NN::init_topology(NeuralNetwork::Topology &topology) {
        layer_count = topology.get_layers();
        std::vector<int> const &sizes = topology.get_layers_size();
        layer_addresses = new int[layer_count + 1];
        neurons_count = 0;
        int i = 0;
        for (; i < layer_count; ++i) {
            layer_addresses[i] = neurons_count;
            neurons_count += sizes[i];
        }
        layer_addresses[i] = neurons_count;
        cudaError_t err = cudaMalloc(&layers, sizeof(Neuron) * neurons_count);
        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }
        init_kernel<<<1, neurons_count, id, stream>>>(layers);
        err = cudaGetLastError();
        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }
        NeuralNetwork::Topology::relationships_map &relationships = topology.get_relationships();
        std::vector<CUDAGene> gene_vec;
        std::vector<CUDAConnectionCount> connection_counts;

        for (auto &it : relationships) {
            connection_counts.push_back({layer_addresses[it.first[0]] + it.first[1],
                                         it.second.genes.size()}
            );

            for (NeuralNetwork::Gene *gene : it.second.genes) {
                if (gene->is_disabled()) {
                    continue;
                }
                NeuralNetwork::Gene::point input = gene->get_input();
                NeuralNetwork::Gene::point output = gene->get_output();
                gene_vec.push_back({
                                           gene->get_input_weight(),
                                           gene->get_memory_weight(),
                                           gene->get_reset_input_weight(),
                                           gene->get_reset_memory_weight(),
                                           gene->get_update_input_weight(),
                                           gene->get_update_memory_weight(),
                                           layer_addresses[input[0]] + input[1],
                                           layer_addresses[output[0]] + output[1]
                                   });
            }
        }
        CUDAConnectionCount *device_counts;
        err = cudaMalloc(&device_counts, sizeof(CUDAConnectionCount) * connection_counts.size());

        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }

        err = cudaMemcpyAsync(device_counts, connection_counts.data(),
                              sizeof(CUDAConnectionCount) * connection_counts.size(),
                              cudaMemcpyHostToDevice, stream);

        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }

        set_connections_kernel<<<1, connection_counts.size(), id, stream>>>(layers, device_counts);

        err = cudaGetLastError();
        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }

        err = cudaFree(device_counts);

        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }

        CUDAGene *device_genes;
        err = cudaMalloc(&device_genes, sizeof(CUDAGene) * gene_vec.size());

        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }

        err = cudaMemcpyAsync(device_genes, gene_vec.data(), sizeof(CUDAGene) * gene_vec.size(),
                              cudaMemcpyHostToDevice, stream);

        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }

        connect_neurons_kernel<<<1, 1, id, stream>>>(layers, device_genes, gene_vec.size());
        err = cudaGetLastError();
        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }
        err = cudaFree(device_genes);

        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }
    }

    __device__ void NN::compute(const double *inputs_array,
                                int from,
                                int to,
                                int output_size,
                                double *out,
                                int write_from) {
        set_inputs(inputs_array, from, to);
        for (int it = 0; it < neurons_count - output_size; ++it) {
            layers[it].feed_forward();
        }
        for (int it = neurons_count - output_size; it < neurons_count; ++it) {
            out[it - neurons_count + output_size + write_from] = layers[it].get_value();
        }
    }

    __device__ void NN::reset_state() {
        for (int it = 0; it < neurons_count; ++it) {
            layers[it].reset_state();
        }
    }

    __device__ void NN::set_inputs(const double *inputs_array, int const from, int const to) {
        for (int i = from; i < to; ++i) {
            layers[i - from].set_input_value(inputs_array[i]);
        }
    }

} /* namespace NeuralNetwork */

//
// Created by alexandre on 03.09.20.
//

#include "ComputeInstance.cuh"

void set_networks_gpu_instance(ComputeInstance *instance, NeuralNetworkCuda::NN *nets, unsigned long int count) {
    instance->networks = nets;
    instance->networks_count = count;
}

void update_dataset_gpu_instance(ComputeInstance *instance, double const *host_data) {
    const unsigned int size = instance->dim.x * instance->dim.y * instance->dim.z;
    const unsigned int bytes = size * sizeof(double);

    cudaError_t err;
    if (instance->data) {
        err = cudaFree(instance->data);
        if (err) {
            std::cout << cudaGetErrorString(err) << std::endl;
            throw err;
        }
    }

    err = cudaMalloc((double **) &(instance->data), bytes);
    if (err) {
        std::cout << cudaGetErrorString(err) << std::endl;
        throw err;
    }
    err = cudaMemcpy(instance->data, host_data, bytes, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    if (err) {
        std::cout << cudaGetErrorString(err) << std::endl;
        throw err;
    }
}

__global__ void
compute_kernel(const int x,
               const int y,
               const int z,
               NeuralNetworkCuda::NN *networks,
               double *data,
               const int networks_count,
               const int output_size,
               double *d_output) {
    unsigned int id = threadIdx.x;
    if (id < networks_count) {
        NeuralNetworkCuda::NN *net = &networks[id];
        // Number of datasets
        for (int i = 0; i < z; ++i) {
            // Size of each dataset
            for (int j = 0; j < y; ++j) {
                int const from = j * x + i * y;
                int const write_from = id * y * z * output_size + j * output_size + i * y * output_size;
                net->compute(data, from, from + x, output_size, d_output, write_from);
            }
        }
        net->reset_state();
    }
}

void compute_gpu_instance(ComputeInstance *instance, const unsigned int output_size) {
    const unsigned int size = instance->dim.y * instance->dim.z * output_size * instance->networks_count;
    std::cout << "REAL SIZE: " << size << std::endl;
    std::cout << "Dim y: " << instance->dim.y << " Dim z: " << instance->dim.z << " os: " << output_size << " count " << instance->networks_count << std::endl;
    const unsigned int bytes = size * sizeof(double);
    instance->h_output = (double *) malloc(bytes);
    cudaError_t err = cudaMalloc((double **) &instance->d_output, bytes);
    if (err) {
        std::cout << cudaGetErrorString(err) << std::endl;
        throw err;
    }

    NeuralNetworkCuda::NN * networks;
    err = cudaMalloc((NeuralNetworkCuda::NN**)&networks, sizeof(NeuralNetworkCuda::NN) * instance->networks_count);
    if (err) {
        std::cout << "Malloc networks " << cudaGetErrorString(err) << std::endl;
        throw err;
    }

    err = cudaMemcpy(networks, instance->networks, sizeof(NeuralNetworkCuda::NN) * instance->networks_count, cudaMemcpyHostToDevice);
    if (err) {
        std::cout << "Networks memcpy " << cudaGetErrorString(err) << std::endl;
        throw err;
    }

    compute_kernel<<<1, instance->networks_count>>>(
            static_cast<int>(instance->dim.x),
            static_cast<int>(instance->dim.y),
            static_cast<int>(instance->dim.z),
            networks,
            instance->data,
            static_cast<int>(instance->networks_count),
            static_cast<int>(output_size),
            instance->d_output);

    cudaDeviceSynchronize();
    err = cudaGetLastError();
    if (err) {
        std::cout << cudaGetErrorString(err) << std::endl;
        throw err;
    }
    err = cudaMemcpy(instance->h_output, instance->d_output, bytes, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    if (err) {
        std::cout << "Output memcpy" << cudaGetErrorString(err) << std::endl;
        throw err;
    }
    err = cudaFree(instance->d_output);
    if (err) {
        std::cout << cudaGetErrorString(err) << std::endl;
        throw err;
    }
    err = cudaFree(networks);
    if (err) {
        std::cout << cudaGetErrorString(err) << std::endl;
        throw err;
    }
}