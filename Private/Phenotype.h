/*
 * Phenotype.h
 *
 *  Created on: Jul 28, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_PHENOTYPE_H_
#define NEURALNETWORK_PHENOTYPE_H_

#include <array>
#include <functional>
#include "../Serializer/Serializer.hpp"

namespace NeuralNetwork {

    class Phenotype : public Serializer::Serializable {
    public:
        using point = std::array<int, 2>;
        using coordinate = std::array<point, 2>;

    public:
        //constructors
        explicit Phenotype(point const &, long);

        explicit Phenotype(point const &, double, double, double, double, double, double, long);

        explicit Phenotype(point const &, point const &, double, double, double, double, double, double, bool, long);

        explicit Phenotype(point const &, point const &, double, double, double, double, double, double, long);

        Phenotype(Phenotype const &);

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

        bool is_disabled() const;

    public:
        // operators

        bool operator<(Phenotype const &) const;

        bool operator==(Phenotype const &) const;

    private:
        point input;
        point output;
        double input_weight;
        double memory_weight;
        double reset_input_weight;
        double reset_memory_weight;
        double update_input_weight;
        double update_memory_weight;
        long int evolution_number;
        bool disabled;

        std::string parse_to_string() const override;
    };

}

namespace std {
    template<>
    struct hash<NeuralNetwork::Phenotype::point> {
        size_t operator()(const NeuralNetwork::Phenotype::point &p) const noexcept{
            std::size_t h1 = std::hash<int>{}(p[0]);
            std::size_t h2 = std::hash<int>{}(p[1]);
            return h1 ^ (h2 << 1);
        }
    };

    template<>
    struct hash<NeuralNetwork::Phenotype::coordinate> {
        size_t operator()(const NeuralNetwork::Phenotype::coordinate &arr) const noexcept{
            std::size_t h1 = std::hash<NeuralNetwork::Phenotype::point>{}(arr[0]);
            std::size_t h2 = std::hash<NeuralNetwork::Phenotype::point>{}(arr[1]);
            return h1 ^ (h2 << 1);
        }
    };
}

#endif /* NEURALNETWORK_PHENOTYPE_H_ */
