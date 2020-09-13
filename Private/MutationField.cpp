//
// Created by sakex on 03/11/2019.
//

#include "MutationField.h"

namespace NeuralNetwork {

    double __MutationField<0>::get(NeuralNetwork::Gene *gene) {
        return gene->get_input_weight();
    }

    void __MutationField<0>::set(NeuralNetwork::Gene *gene, double const value) {
        gene->set_input_weight(value);
    }

    double __MutationField<1>::get(NeuralNetwork::Gene *gene) {
        return gene->get_memory_weight();
    }

    void __MutationField<1>::set(NeuralNetwork::Gene *gene, double const value) {
        gene->set_memory_weight(value);
    }

    double __MutationField<2>::get(NeuralNetwork::Gene *gene) {
        return gene->get_reset_input_weight();
    }

    void __MutationField<2>::set(NeuralNetwork::Gene *gene, double const value) {
        gene->set_reset_input_weight(value);
    }

    double __MutationField<3>::get(NeuralNetwork::Gene *gene) {
        return gene->get_reset_memory_weight();
    }

    void __MutationField<3>::set(NeuralNetwork::Gene *gene, double const value) {
        gene->set_reset_memory_weight(value);
    }

    double __MutationField<4>::get(NeuralNetwork::Gene *gene) {
        return gene->get_update_memory_weight();
    }

    void __MutationField<4>::set(NeuralNetwork::Gene *gene, double const value) {
        gene->set_update_memory_weight(value);
    }

    double __MutationField<5>::get(NeuralNetwork::Gene *gene) {
        return gene->get_update_input_weight();
    }

    void __MutationField<5>::set(NeuralNetwork::Gene *gene, double const value) {
        gene->set_update_input_weight(value);
    }

    double MutationField::get(int field, NeuralNetwork::Gene *gene) {
        switch (field) {
            case 0:
                return __MutationField<0>::get(gene);
            case 1:
                return __MutationField<1>::get(gene);
            case 2:
                return __MutationField<2>::get(gene);
            case 3:
                return __MutationField<3>::get(gene);
            case 4:
                return __MutationField<4>::get(gene);
            case 5:
                return __MutationField<5>::get(gene);
            default:
                throw InvalidField(field);
        }
    }

    void MutationField::set(int field, NeuralNetwork::Gene *gene, double value) {
        switch (field) {
            case 0:
                __MutationField<0>::set(gene, value);
                break;
            case 1:
                __MutationField<1>::set(gene, value);
                break;
            case 2:
                __MutationField<2>::set(gene, value);
                break;
            case 3:
                __MutationField<3>::set(gene, value);
                break;
            case 4:
                __MutationField<4>::set(gene, value);
                break;
            case 5:
                __MutationField<5>::set(gene, value);
                break;
            default:
                throw InvalidField(field);
        }
    }
}