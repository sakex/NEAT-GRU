/*
 * Phenotype.cpp
 *
 *  Created on: Jul 28, 2019
 *      Author: sakex
 */

#include "Phenotype.h"

namespace NeuralNetwork {

    Phenotype::Phenotype(point const &_input, long const ev_number) :
            Phenotype(_input, .1, .1, .1, .1, .1, .1, ev_number) {
    }

    Phenotype::Phenotype(point const &_input, double const _input_weight, double const _memory_weight,
                         double const riw, double const rmw, double const uiw, double const umw,
                         long const ev_number) :
            input{_input[0], _input[1]},
            output{0, 0},
            input_weight(_input_weight),
            memory_weight(_memory_weight),
            reset_input_weight(riw),
            reset_memory_weight(rmw),
            update_input_weight(uiw),
            update_memory_weight(umw),
            disabled(false),
            evolution_number(ev_number) {
    }

    Phenotype::Phenotype(point const &_input, point const &_output,
                         double const _input_weight, double const _memory_weight,
                         double const riw, double const rmw, double const uiw, double const umw, const bool _disabled,
                         long const ev_number) :
            input{_input[0], _input[1]},
            output{_output[0], _output[1]},
            input_weight(_input_weight),
            memory_weight(_memory_weight),
            reset_input_weight(riw),
            reset_memory_weight(rmw),
            update_input_weight(uiw),
            update_memory_weight(umw),
            disabled(_disabled),
            evolution_number(ev_number) {
    }

    Phenotype::Phenotype(point const &input, point const &output,
                         double const _input_weight, double const _memory_weight,
                         double const riw, double const rmw, double const uiw, double const umw,
                         long const ev_number) :
            Phenotype(input, output, _input_weight, _memory_weight, riw, rmw, uiw, umw, false, ev_number) {
    }

    Phenotype::Phenotype(Phenotype const &base) :
            input(base.input),
            output(base.output) {
        input_weight = base.input_weight;
        memory_weight = base.memory_weight;
        reset_memory_weight = base.reset_memory_weight;
        reset_input_weight = base.reset_input_weight;
        update_input_weight = base.update_input_weight;
        update_memory_weight = base.update_memory_weight;
        evolution_number = base.evolution_number;
        disabled = base.disabled;
    }

    void Phenotype::set_input_weight(double const new_weight) {
        input_weight = new_weight;
    }

    void Phenotype::set_memory_weight(double const new_weight) {
        memory_weight = new_weight;
    }

    void Phenotype::set_reset_input_weight(double const new_weight) {
        reset_input_weight = new_weight;
    }

    void Phenotype::set_reset_memory_weight(double const new_weight) {
        reset_memory_weight = new_weight;
    }

    void Phenotype::set_update_input_weight(double const new_weight) {
        update_input_weight = new_weight;
    }

    void Phenotype::set_update_memory_weight(double const new_weight) {
        update_memory_weight = new_weight;
    }

    void Phenotype::set_disabled(bool const value) {
        disabled = value;
    }

    void Phenotype::set_output(int const first, int const second) {
        output[0] = first;
        output[1] = second;
    }

    Phenotype::point const &Phenotype::get_input() {
        return input;
    }

    Phenotype::point const &Phenotype::get_output() {
        return output;
    }

    double Phenotype::get_input_weight() const {
        return input_weight;
    }

    double Phenotype::get_memory_weight() const {
        return memory_weight;
    }

    double Phenotype::get_reset_input_weight() const {
        return reset_input_weight;
    }

    double Phenotype::get_reset_memory_weight() const {
        return reset_memory_weight;
    }

    double Phenotype::get_update_input_weight() const {
        return update_input_weight;
    }

    double Phenotype::get_update_memory_weight() const {
        return update_memory_weight;
    }

    long Phenotype::get_ev_number() const {
        return evolution_number;
    }

    void Phenotype::decrement_output() {
        output[0]--;
    }

    void Phenotype::disable() {
        disabled = true;
    }

    bool Phenotype::is_disabled() const {
        return disabled;
    }

    void Phenotype::resize(int const former_size, int const new_size) {
        if (output[0] == former_size) {
            output[0] = new_size;
        }
    }

    std::string Phenotype::parse_to_string() const {
        std::string str = "{\"input\":[" + std::to_string(input[0]) + ","
                          + std::to_string(input[1]) + "], \"output\":["
                          + std::to_string(output[0]) + "," + std::to_string(output[1])
                          + "],\"input_weight\":" + std::to_string(input_weight) +
                          ",\"memory_weight\":" + std::to_string(memory_weight) +
                          ",\"reset_input_weight\":" + std::to_string(reset_input_weight) +
                          ",\"reset_memory_weight\":" + std::to_string(reset_memory_weight) +
                          ",\"update_input_weight\":" + std::to_string(update_input_weight) +
                          ",\"update_memory_weight\":" + std::to_string(update_memory_weight) +
                          ",\"disabled\":"
                          + (disabled ? "true" : "false") + "}";
        return str;
    }

    bool Phenotype::operator<(Phenotype const &that) const {
        return output[0] < that.output[0];
    }

    bool Phenotype::operator==(Phenotype const &that) const {
        return input == that.input && output == that.output;
    }
}
