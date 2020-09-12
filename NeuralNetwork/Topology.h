/*
 * Topology.h
 *
 *  Created on: Jun 9, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_TOPOLOGY_H_
#define NEURALNETWORK_TOPOLOGY_H_

#include <vector>
#include <unordered_map>
#include <queue>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <algorithm>
#include <functional>

#include "../Private/Mutation.h"
#include "../Private/Generation.h" // Global evolution number
#include "../Private/Bias.h"

#include "../Serializer/Serializer.hpp"
#include "../Private/Random.h"
#include "../Train/static.h"
#include "../Private/ConnectionType.h"


/// Namespace containing the different classes relevant for the neural network
namespace NeuralNetwork {

    struct GeneAndBias {
        Bias bias;
        std::vector<Gene *> genes;
    };

    class Topology : public Serializer::Serializable {
    public:
        using relationships_map = std::unordered_map<Gene::point, GeneAndBias>;

    public:
        /**
         * Function based on the paper http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
         * (Chapter 4.1)
         * To find distance between two genomes
         *
         * The function has been adapted to work with GRU gates
         *
         * @param top1 First topology
         * @param top2 Second topology to find distance with first one
         * @return Distance between top1 and top2
         */
        static double delta_compatibility(Topology const &top1, Topology const &top2);

        static std::shared_ptr<Topology> crossover(Topology &top1, Topology &top2);

    public:
        // Rule of 3
        Topology();

        Topology(Topology const &);

        ~Topology() override;

    public:
        // Operators
        bool operator==(Topology const &) const;

        bool operator<(Topology const &) const;

    public:
        // getters/setters

        /// Set layers
        void set_layers(int);

        /// Get layers
        int get_layers() const;

        /// Set last score
        void set_last_result(double);

        /// Get last score
        double get_last_result() const;

        /// Get all relationships
        relationships_map &get_relationships();

        /// Adds a relationship
        void add_relationship(Gene *, bool init = false);

        /// Sets a topology as belonging to a species
        void set_assigned(bool);

        /// Getter to see if a topology belongs to a species
        bool is_assigned() const;

        /// Getter to see if the last mutation had a positive impact on the score
        bool mutation_positive() const;

        /// Get the layers sizes
        std::vector<int> const &get_layers_size() const;

        void set_bias(std::array<int, 2>, Bias);

        void generate_output_bias();

        std::vector<Bias> const & get_output_bias() const;

    public:
        // Species evolution methods

        /**
         * Optimizes the weights of the last mutation
         * @return True if last mutation is considered optimized
         */
        bool optimize();

        /// Sets the topology to optimized
        [[maybe_unused]] void set_optimized();

        /**
         * Creates new indivuduals that are mutations of the original one
         * @param children_count The number of new topologies to create
         * @param output The vector to which we push the new topologies
         */
        void new_generation(size_t children_count,
                            std::vector<std::shared_ptr<Topology>> &output);


        /**
         * Serializes the topology to string
         *
         * @return - A string containing the serialized topology
         */
        std::string parse_to_string() const override;

    private:
        // Data
        int layers = 0;
        double last_result = 0;
        double best_historical_result = 0;
        double result_before_mutation = 0;
        bool assigned = false;
        std::vector<int> layers_size;
        std::vector<Bias> output_bias;
        relationships_map relationships;
        std::unordered_map<long, Gene *> ev_number_index;
        std::queue<Mutation> mutations;
        std::array<int, 6> fields_order;
        int current_field;

    private:
        // Sub types / static
        using point_pair = std::array<Gene::point, 2>;
        using gene_cb = std::function<void(Gene *)>;

    private:
        // Init/add relationship

        /**
         * Adds a gene to the gene map
         * @param gene Pointer to the gene to add
         */
        void add_to_relationships_map(Gene *gene);

        /**
         * Recursive algorithm that disables genes that don't get any input
         * @param input Input Point
         * @param output Output Point
         */
        void disable_genes(Gene::point const &input, Gene::point const &output);


        /**
         * Recursive algorithm that checks that no two paths overlap
         * @param input Input Point
         * @param output Output Point
         * @return Boolean if a path overrides
         */
        bool path_overrides(Gene::point const &input, Gene::point const &output);

        /**
         * If we have a new layer, we need to reassign genes
         * @param new_size
         */
        void resize(int new_size);

    private:
        // mutations

        /**
         * Copies the initial topology and adds a mutation to it
         * @return shared pointer to the new mutated topology
         */
        std::shared_ptr<Topology> evolve();

        /**
         * Mutates a topology and creates new genes
         * @return a vector of pointers to the newly created genes
         */
        std::vector<Gene *> mutate();

        /**
         * Creates a new mutation based on the newly created genes
         * @param gene The new gene to create the mutation from
         * @param score The original score
         */
        void new_mutation(Gene *gene, double score);

        /**
         * Creates a new gene
         * @param input Input position of the connection representation
         * @param output Output position of the connection representation
         * @return New gene pointer
         */
        Gene *new_gene(Gene::point const &input,
                            Gene::point const &output);

    private:
        /**
         * Util to run over the genes
         * @param cb The callback to run over all the genes
         */
        void iterate_genes(gene_cb &cb) const;
    };

    using Topology_ptr = std::shared_ptr<Topology>;

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_TOPOLOGY_H_ */
