//
// Created by sakex on 03/11/2019.
//

#ifndef TRADING_MUTATIONFIELD_H
#define TRADING_MUTATIONFIELD_H

#include "Phenotype.h"
#include <exception>

namespace NeuralNetwork {
    struct InvalidField : std::exception {
        char const * what() const noexcept override {
            return "Invalid field";
        }
    };

    template<int n>
    struct __MutationField {
    };

    template<>
    struct __MutationField<0> {
        static double get(Phenotype *);

        static void set(Phenotype *, double);
    };

    template<>
    struct __MutationField<1> {
        static double get(Phenotype *);

        static void set(Phenotype *, double);
    };

    template<>
    struct __MutationField<2> {
        static double get(Phenotype *);

        static void set(Phenotype *, double);
    };

    template<>
    struct __MutationField<3> {
        static double get(Phenotype *);

        static void set(Phenotype *, double);
    };

    template<>
    struct __MutationField<4> {
        static double get(Phenotype *);

        static void set(Phenotype *, double);
    };

    template<>
    struct __MutationField<5> {
        static double get(Phenotype *);

        static void set(Phenotype *, double);
    };

    struct MutationField {
        static double get(int, Phenotype *);

        static void set(int, Phenotype *, double);
    };
}


#endif //TRADING_MUTATIONFIELD_H
