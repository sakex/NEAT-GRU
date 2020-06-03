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
#include "../Private/routines.h" // no layer exception
#include "../Private/Generation.h" // Global evolution number

#include "../Serializer/Serializer.hpp"
#include "../Private/Random.h"
#include "../Train/static.h"


/// Namespace containing the different classes relevant for the neural network
namespace NeuralNetwork {

    class Topology : public Serializer::Serializable {
    public:
        using relationships_map = std::unordered_map<Phenotype::point, std::vector<Phenotype *>>;

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
        static double delta_compatibility(Topology &top1, Topology &top2);

        static std::shared_ptr<Topology> crossover(Topology &top1, Topology &top2);

    public:
        // Rule of 3
        Topology() = default;

        Topology(Topology const &);

        ~Topology() override;

        Topology &operator=(Topology const &);

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
        void set_last_result(long double);

        /// Get last score
        long double get_last_result() const;

        /// Get all relationships
        relationships_map &get_relationships();

        /// Adds a relationship
        void add_relationship(Phenotype *, bool init = false);

        /// Sets a topology as belonging to a species
        void set_assigned(bool);

        /// Getter to see if a topology belongs to a species
        bool is_assigned() const;

        /// Getter to see if the last mutation had a positive impact on the score
        bool mutation_positive() const;

        /// Get the layers sizes
        std::vector<int> const & get_layers_size() const;

    public:
        // Species evolution methods

        /**
         * Optimizes the weights of the last mutation
         * @return True if last mutation is considered optimized
         */
        bool optimize();

        /// Sets the topology to optimized
        void set_optimized();

        /**
         * Creates new indivuduals that are mutations of the original one
         * @param children_count The number of new topologies to create
         * @param output The vector to which we push the new topologies
         */
        void new_generation(size_t children_count,
                            std::vector<std::shared_ptr<Topology>> & output);

    private:
        // Data
        int layers = 0;
        long double last_result = 0;
        long double best_historical_result = 0;
        long double result_before_mutation = 0;
        bool assigned = false;
        std::vector<int> layers_size;
        relationships_map relationships;
        std::unordered_map<long, Phenotype *> ev_number_index;
        std::queue<Mutation> mutations;

    private:
        // Sub types / static
        using point_pair = std::array<Phenotype::point, 2>;
        using phenotype_cb = std::function<void(Phenotype *)>;

    private:
        // Init/add relationship

        /**
         * Adds a phenotype to the phenotype map
         * @param phenotype Pointer to the phenotype to add
         */
        void add_to_relationships_map(Phenotype *phenotype);

        /**
         * Recursive algorithm that disables phenotypes that don't get any input
         * @param input Input Point
         * @param output Output Point
         */
        void disable_phenotypes(Phenotype::point const & input, Phenotype::point const & output);


        /**
         * Recursive algorithm that checks that no two paths overlap
         * @param input Input Point
         * @param output Output Point
         * @return Boolean if a path overrides
         */
        bool path_overrides(Phenotype::point const & input, Phenotype::point const & output);

        /**
         * If we have a new layer, we need to reassign phenotypes
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
         * Mutates a topology and creates new phenotypes
         * @return a vector of pointers to the newly created phenotypes
         */
        std::vector<Phenotype *> mutate();

        /**
         * Creates a new mutation based on the newly created phenotypes
         * @param phenotype The new phenotype to create the mutation from
         * @param score The original score
         */
        void new_mutation(Phenotype * phenotype, long double score);

        /**
         * Creates a new phenotype
         * @param input Input position of the connection representation
         * @param output Output position of the connection representation
         * @return New phenotype pointer
         */
        Phenotype *new_phenotype(Phenotype::point const &input,
                                 Phenotype::point const &output);

    private:
        // Utils
        std::string parse_to_string() const override;

        /**
         * Util to run over the phenotypes
         * @param cb The callback to run over all the phenotypes
         */
        void iterate_phenotypes(phenotype_cb & cb) const;
    };

    using Topology_ptr = std::shared_ptr<Topology>;

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_TOPOLOGY_H_ */
