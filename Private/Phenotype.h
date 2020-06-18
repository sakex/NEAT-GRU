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

        explicit Phenotype(point const &, float, float, float, float, float, float, long);

        explicit Phenotype(point const &, point const &, float, float, float, float, float, float, bool, long);

        explicit Phenotype(point const &, point const &, float, float, float, float, float, float, long);

        Phenotype(Phenotype const &);

    public:
        // setters
        void set_input_weight(float);

        void set_memory_weight(float);

        void set_reset_input_weight(float);

        void set_reset_memory_weight(float);

        void set_update_input_weight(float);

        void set_update_memory_weight(float);

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

        float get_input_weight() const;

        float get_memory_weight() const;

        float get_reset_input_weight() const;

        float get_reset_memory_weight() const;

        float get_update_input_weight() const;

        float get_update_memory_weight() const;

        bool is_disabled() const;

    public:
        // operators

        bool operator<(Phenotype const &) const;

        bool operator==(Phenotype const &) const;

    private:
        point input;
        point output;
        float input_weight;
        float memory_weight;
        float reset_input_weight;
        float reset_memory_weight;
        float update_input_weight;
        float update_memory_weight;
        long int evolution_number;
        bool disabled;

        std::string parse_to_string() const override;
    };

}

namespace std {
    template<>
    struct hash<NeuralNetwork::Phenotype::point> {
        size_t operator()(const NeuralNetwork::Phenotype::point &p) const noexcept{
            std::hash<int> hasher;
            std::size_t result = 111;
            result = (result << 1) ^ hasher(p[0]);
            result = (result << 1) ^ hasher(p[1]);
            return result;
        }
    };

    template<>
    struct hash<NeuralNetwork::Phenotype::coordinate> {
        size_t operator()(const NeuralNetwork::Phenotype::coordinate &arr) const noexcept{
            std::hash<NeuralNetwork::Phenotype::point> hasher;
            std::size_t result = 144451;
            result = (result << 1) ^ hasher(arr[0]);
            result = (result << 1) ^ hasher(arr[1]);
            return result;
        }
    };



}

#endif /* NEURALNETWORK_PHENOTYPE_H_ */
