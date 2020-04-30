//
// Created by sakex on 03/11/2019.
//

#include "MutationField.h"

namespace NeuralNetwork {

    double __MutationField<0>::get(NeuralNetwork::Phenotype *phenotype) {
        return phenotype->get_input_weight();
    }

    void __MutationField<0>::set(NeuralNetwork::Phenotype *phenotype, double const value) {
        phenotype->set_input_weight(value);
    }

    double __MutationField<1>::get(NeuralNetwork::Phenotype *phenotype) {
        return phenotype->get_memory_weight();
    }

    void __MutationField<1>::set(NeuralNetwork::Phenotype *phenotype, double const value) {
        phenotype->set_memory_weight(value);
    }

    double __MutationField<2>::get(NeuralNetwork::Phenotype *phenotype) {
        return phenotype->get_reset_input_weight();
    }

    void __MutationField<2>::set(NeuralNetwork::Phenotype *phenotype, double const value) {
        phenotype->set_reset_input_weight(value);
    }

    double __MutationField<3>::get(NeuralNetwork::Phenotype *phenotype) {
        return phenotype->get_reset_memory_weight();
    }

    void __MutationField<3>::set(NeuralNetwork::Phenotype *phenotype, double const value) {
        phenotype->set_reset_memory_weight(value);
    }

    double __MutationField<4>::get(NeuralNetwork::Phenotype *phenotype) {
        return phenotype->get_update_input_weight();
    }

    void __MutationField<4>::set(NeuralNetwork::Phenotype *phenotype, double const value) {
        phenotype->set_update_input_weight(value);
    }

    double __MutationField<5>::get(NeuralNetwork::Phenotype *phenotype) {
        return phenotype->get_update_memory_weight();
    }

    void __MutationField<5>::set(NeuralNetwork::Phenotype *phenotype, double const value) {
        phenotype->set_update_memory_weight(value);
    }

    double MutationField::get(int field, NeuralNetwork::Phenotype *phenotype) {
        switch (field) {
            case 0:
                return __MutationField<0>::get(phenotype);
            case 1:
                return __MutationField<1>::get(phenotype);
            case 2:
                return __MutationField<2>::get(phenotype);
            case 3:
                return __MutationField<3>::get(phenotype);
            case 4:
                return __MutationField<4>::get(phenotype);
            case 5:
                return __MutationField<5>::get(phenotype);
            default:
                throw InvalidField();
        }
    }

    void MutationField::set(int field, NeuralNetwork::Phenotype *phenotype, double value) {
        switch (field) {
            case 0:
                __MutationField<0>::set(phenotype, value);
                break;
            case 1:
                __MutationField<1>::set(phenotype, value);
                break;
            case 2:
                __MutationField<2>::set(phenotype, value);
                break;
            case 3:
                __MutationField<3>::set(phenotype, value);
                break;
            case 4:
                __MutationField<4>::set(phenotype, value);
                break;
            case 5:
                __MutationField<5>::set(phenotype, value);
                break;
            default:
                throw InvalidField();
        }
    }
}