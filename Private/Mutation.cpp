/*
 * Mutation.cpp
 *
 *  Created on: Jul 27, 2019
 *      Author: sakex
 */

#include "Mutation.h"

namespace NeuralNetwork {

    Mutation::Mutation() :
            gene{nullptr},
            field(0),
            last_result(0) {
    }

    Mutation::Mutation(Gene *_gene, double const _last_result) :
            gene{_gene},
            field(0),
            last_result{_last_result} {
    }

    void Mutation::set_back_to_max() {
        MutationField::set(field, gene, best_historical_weight);
        last_result = best_historical_wealth;
    }

    unsigned Mutation::get_iterations() const {
        return iterations;
    }

    unsigned Mutation::get_unfruitful() const {
        return unfruitful;
    }

    void Mutation::set_field(int value) {
        field = value;
        iterations = 0;
        unfruitful = 0;
        best_historical_weight = 0;
        best_historical_wealth = 0;
    }

    void Mutation::mutate(double const wealth) {
        const double weight = MutationField::get(field, gene);
        if (wealth > best_historical_wealth) {
            best_historical_wealth = wealth;
            best_historical_weight = weight;
        }
        const double delta = wealth - last_result;
        const int prev_direction = direction;
        if (direction == 0 || direction == 1)
            direction = delta > 0 ? 1 : -1;
        else if (direction == -1)
            direction = delta > 0 ? -1 : 1;
        else
            throw InvalidDirectionException();
        const int index = direction == -1 ? 1 : 0;
        // If direction is -1, then we update the max value of the inverval, else we update the min value
        interval[index] = weight;
        if (prev_direction != direction && prev_direction != 0) {
            interval_found = true;
        }
        if (!interval_found) {
            ++unfruitful;
            MutationField::set(field, gene, weight + static_cast<double>(gradient * direction));
            gradient--;
            if (!gradient)
                unfruitful = 1000;
        } else {
            iterations++;
            const double new_weight = (interval[1] + interval[0]) / 2;
            MutationField::set(field, gene, new_weight);
        }
        if (std::abs(interval[0] - interval[1]) < .5)
            iterations = 1000;
        last_result = wealth;
    }

    bool Mutation::operator!() const {
        return !gene;
    }

    Mutation &Mutation::operator=(Mutation const &base) {
        if (&base == this) return *this;
        gene = base.gene;
        direction = base.direction;
        iterations = base.iterations;
        unfruitful = base.unfruitful;
        gradient = base.gradient;
        last_result = base.last_result;
        return *this;
    }

} /* namespace NeuralNetwork */
