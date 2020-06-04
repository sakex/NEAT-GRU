//
// Created by sakex on 05/09/2019.
//

#include "Random.h"

namespace utils {
    std::default_random_engine Random::generator;

    int Random::random_number(const int max) {
        return random_between(0, max);
    }

}