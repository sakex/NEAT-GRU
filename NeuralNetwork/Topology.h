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
#include "Random.h"

namespace NeuralNetwork {

    class Topology : public Serializer::Serializable {
    public:
        using relationships_map = std::unordered_map<Phenotype::point, std::vector<Phenotype *>>;

    public:
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
        void set_layers(int);

        int get_layers() const;

        void set_last_result(long double);

        long double get_last_result() const;

        relationships_map &get_relationships();

        void add_relationship(Phenotype *, bool init = false);

        void set_assigned(bool);

        bool is_assigned() const;

        bool mutation_positive() const;

    public:
        // Species evolution methods
        bool optimize();

        void set_optimized();

        void new_generation(size_t,
                            std::vector<std::shared_ptr<Topology>> &);

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
        void add_to_relationships_map(Phenotype *phenotype);

        void disable_phenotypes(Phenotype::point const &, Phenotype::point const &);

        bool path_overrides(Phenotype::point const &,
                            Phenotype::point const &);

        void resize(int new_size);

    private:
        // mutations
        std::shared_ptr<Topology> evolve();

        std::vector<Phenotype *> mutate();

        void new_mutation(Phenotype *, long double);

        Phenotype *new_phenotype(Phenotype::point const &input,
                                 Phenotype::point const &output);

    private:
        // Utils
        std::string parse_to_string() const override;

        void iterate_phenotypes(phenotype_cb &) const;
    };

    using Topology_ptr = std::shared_ptr<Topology>;

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_TOPOLOGY_H_ */
