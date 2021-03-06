/*
 * Mutation.h
 *
 *  Created on: Jul 27, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_MUTATION_H_
#define NEURALNETWORK_MUTATION_H_

#include <iostream>
#include <memory>
#include <cmath>
#include <exception>

#include "Gene.h"
#include "MutationField.h"

namespace NeuralNetwork {

    struct InvalidDirectionException : public std::exception {
        const char *what() const noexcept override {
            return "Invalid direction";
        }
    };

    class Mutation {
        /*
         * Gradient descent first searches for a range between two doubles
         * Once it found it, it divides the range in two until finding the right value
         * */
    public:
        Mutation();

        ~Mutation() = default;

        explicit Mutation(Gene *, double);

        Mutation &operator=(Mutation const &);

    public:
        void set_back_to_max();

        unsigned get_iterations() const;

        unsigned get_unfruitful() const;

        void mutate(double);

        bool operator!() const;

        void set_field(int);

        ConnectionType gene_type() const;

    private:
        Gene *gene;
        int field;
        double interval[2] = {static_cast<double>(-INFINITY), static_cast<double>(INFINITY)};
        bool interval_found = false;
        int direction = 0;
        unsigned iterations = 0;
        unsigned unfruitful = 0;
        int gradient = 10;
        double last_result;
        double best_historical_wealth = 0;
        double best_historical_weight = 0;
    };

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_MUTATION_H_ */
