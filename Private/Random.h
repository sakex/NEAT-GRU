//
// Created by sakex on 05/09/2019.
//

#ifndef TRADING_RANDOM_H
#define TRADING_RANDOM_H

#include <random>
#include <iostream>

namespace utils {
    class Random {
    public:
        static std::default_random_engine generator;

        template<typename T>
        static T random_between(T min, T max) {
            std::uniform_int_distribution<T> distribution(min, max);
            return distribution(generator);
        }

        template<typename A, typename T>
        static T random_between(A min, T max) {
            std::uniform_int_distribution<T> distribution(min, max);
            return distribution(generator);
        }

        template<typename T>
        static T random_normal(T mu, T sigma) {
            std::normal_distribution<T> distribution(mu, sigma);
            return distribution(generator);
        }

        static int random_number(int);
    };
}

#endif //TRADING_RANDOM_H
