/*
 * Mutation.cpp
 *
 *  Created on: Jul 27, 2019
 *      Author: sakex
 */

#include "Mutation.h"

namespace NeuralNetwork {

    Mutation::Mutation() :
            phenotype{nullptr},
            field(0),
            last_result(0) {
    }

    Mutation::Mutation(Phenotype *_phenotype, long double const _last_result) :
            phenotype{_phenotype},
            field(0),
            last_result{_last_result} {
    }

    void Mutation::set_back_to_max() {
        MutationField::set(field, phenotype, best_historical_weight);
        last_result = best_historical_wealth;
    }

    unsigned Mutation::get_iterations() const {
        return iterations;
    }

    unsigned Mutation::get_unfruitful() const {
        return unfruitful;
    }

    int Mutation::get_field() const {
        return field;
    }

    void Mutation::set_field(int value) {
        field = value;
        interval[0] = static_cast<double>(-INFINITY);
        interval[1] = static_cast<double>(INFINITY);
        interval_found = false;
        direction = 0;
        iterations = 0;
        unfruitful = 0;
        gradient = 10;
        best_historical_weight = 0;
        best_historical_wealth = 0;
    }

    void Mutation::mutate(long double const wealth) {
        const double weight = MutationField::get(field, phenotype);
        if (wealth > best_historical_wealth) {
            best_historical_wealth = wealth;
            best_historical_weight = weight;
        }
        const long double delta = wealth - last_result;
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
            MutationField::set(field, phenotype, weight + gradient * direction);
            gradient--;
            if (!gradient)
                unfruitful = 1000;
        } else {
            iterations++;
            const double new_weight = (interval[1] + interval[0]) / 2;
            MutationField::set(field, phenotype, new_weight);
        }
        if (std::abs(interval[0] - interval[1]) < .5)
            iterations = 1000;
        last_result = wealth;
    }

    bool Mutation::operator!() const {
        return !phenotype;
    }

    Mutation &Mutation::operator=(Mutation const &base) {
        if(&base == this) return *this;
        phenotype = base.phenotype;
        interval_found = base.interval_found;
        direction = base.direction;
        iterations = base.iterations;
        unfruitful = base.unfruitful;
        gradient = base.gradient;
        last_result = base.last_result;
        return *this;
    }

} /* namespace NeuralNetwork */
