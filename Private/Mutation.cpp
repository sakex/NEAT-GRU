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

    Mutation::Mutation(Phenotype *_phenotype, float const _last_result) :
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
        iterations = 0;
        unfruitful = 0;
        step = 10;
        best_historical_weight = 0;
        best_historical_wealth = 0;
    }

    void Mutation::mutate(float const wealth) {
        const float weight = MutationField::get(field, phenotype);
        if (wealth > best_historical_wealth) {
            best_historical_wealth = wealth;
            best_historical_weight = weight;
        }
        const float delta = wealth - last_result;
        if (direction == 0 || direction == 1)
            direction = delta > 0 ? 1 : -1;
        else if (direction == -1)
            direction = delta > 0 ? -1 : 1;
        else
            throw InvalidDirectionException();
        if (prev_weight != weight)
            MutationField::set(field, phenotype,
                               weight +
                               static_cast<float>(direction) * step * std::abs(delta / (prev_weight - weight)));
        else
            MutationField::set(field, phenotype,
                               weight + static_cast<float>(direction) * step);
        iterations++;
        step += wealth - last_result;
        last_result = wealth;
        prev_weight = weight;
    }

    bool Mutation::operator!() const {
        return !phenotype;
    }

    Mutation &Mutation::operator=(Mutation const &base) {
        if (&base == this) return *this;
        phenotype = base.phenotype;
        direction = base.direction;
        iterations = base.iterations;
        unfruitful = base.unfruitful;
        step = base.step;
        last_result = base.last_result;
        return *this;
    }

} /* namespace NeuralNetwork */
