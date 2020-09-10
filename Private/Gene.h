/*
 * Phenotype.h
 *
 *  Created on: Jul 28, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_GENE_H_
#define NEURALNETWORK_GENE_H_

#include <array>
#include <functional>
#include "../Serializer/Serializer.hpp"
#include "ConnectionType.h"

namespace NeuralNetwork {

    class Gene : public Serializer::Serializable {
    public:
        using point = std::array<int, 2>;
        using coordinate = std::array<point, 2>;

    public:
        //constructors
        explicit Gene(point const &, ConnectionType, long);

        explicit Gene(point const &, double, double, double, double, double, double, ConnectionType, long);

        explicit Gene(point const &, point const &, double, double, double, double, double, double, bool, ConnectionType, long);

        explicit Gene(point const &, point const &, double, double, double, double, double, double, ConnectionType, long);

        Gene(Gene const &);

    public:
        // setters
        void set_input_weight(double);

        void set_memory_weight(double);

        void set_reset_input_weight(double);

        void set_reset_memory_weight(double);

        void set_update_input_weight(double);

        void set_update_memory_weight(double);

        void set_disabled(bool);

        void set_output(int first, int second);

        void resize(int former_size, int new_size);

        void decrement_output();

        void disable();

    public:
        // getters

        point const &get_input();

        point const &get_output();

        long get_ev_number() const;

        double get_input_weight() const;

        double get_memory_weight() const;

        double get_reset_input_weight() const;

        double get_reset_memory_weight() const;

        double get_update_input_weight() const;

        double get_update_memory_weight() const;

        ConnectionType get_type() const;

        bool is_disabled() const;

    public:
        // operators

        bool operator<(Gene const &) const;

        bool operator==(Gene const &) const;

    private:
        point input;
        point output;
        double input_weight;
        double memory_weight;
        double reset_input_weight;
        double update_input_weight;
        double reset_memory_weight;
        double update_memory_weight;
        ConnectionType connection_type;
        long int evolution_number;
        bool disabled;

        std::string parse_to_string() const override;
    };

}

namespace std {
    template<>
    struct hash<NeuralNetwork::Gene::point> {
        size_t operator()(const NeuralNetwork::Gene::point &p) const noexcept{
            std::hash<int> hasher;
            std::size_t result = 111;
            result = (result << 1u) ^ hasher(p[0]);
            result = (result << 1u) ^ hasher(p[1]);
            return result;
        }
    };

    template<>
    struct hash<NeuralNetwork::Gene::coordinate> {
        size_t operator()(const NeuralNetwork::Gene::coordinate &arr) const noexcept{
            std::hash<NeuralNetwork::Gene::point> hasher;
            std::size_t result = 144451;
            result = (result << 1u) ^ hasher(arr[0]);
            result = (result << 1u) ^ hasher(arr[1]);
            return result;
        }
    };



}

#endif /* NEURALNETWORK_GENE_H_ */
