//
// Created by sakex on 03/11/2019.
//

#ifndef TRADING_MUTATIONFIELD_H
#define TRADING_MUTATIONFIELD_H

#include "Phenotype.h"
#include <exception>

namespace NeuralNetwork {
    struct InvalidField : std::exception {
        char const *what() const noexcept override {
            return "Invalid field";
        }
    };

    template<int n>
    struct __MutationField {
    };

    template<>
    struct __MutationField<0> {
        static float get(Phenotype *);

        static void set(Phenotype *, float);
    };

    template<>
    struct __MutationField<1> {
        static float get(Phenotype *);

        static void set(Phenotype *, float);
    };

    template<>
    struct __MutationField<2> {
        static float get(Phenotype *);

        static void set(Phenotype *, float);
    };

    template<>
    struct __MutationField<3> {
        static float get(Phenotype *);

        static void set(Phenotype *, float);
    };

    template<>
    struct __MutationField<4> {
        static float get(Phenotype *);

        static void set(Phenotype *, float);
    };

    template<>
    struct __MutationField<5> {
        static float get(Phenotype *);

        static void set(Phenotype *, float);
    };

    struct MutationField {
        static float get(int, Phenotype *);

        static void set(int, Phenotype *, float);
    };
}


#endif //TRADING_MUTATIONFIELD_H
