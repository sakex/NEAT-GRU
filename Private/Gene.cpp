/*
 * Phenotype.cpp
 *
 *  Created on: Jul 28, 2019
 *      Author: sakex
 */

#include "Gene.h"

namespace NeuralNetwork {

    Gene::Gene(point const &_input, ConnectionType type, long const ev_number) :
            Gene(_input, .1f, .1f, .1f, .1f, .1f, .1f, type, ev_number) {
    }

    Gene::Gene(point const &_input, double const _input_weight, double const _memory_weight,
               double const riw, double const uiw, double const rmw, double const umw, ConnectionType type,
               long const ev_number) :
            input{_input[0], _input[1]},
            output{0, 0},
            input_weight(_input_weight),
            memory_weight(_memory_weight),
            reset_input_weight(riw),
            update_input_weight(uiw),
            reset_memory_weight(rmw),
            update_memory_weight(umw),
            connection_type(type),
            evolution_number(ev_number),
            disabled(false) {
    }

    Gene::Gene(point const &_input, point const &_output,
               double const _input_weight, double const _memory_weight,
               double const riw, double const uiw, double const rmw, double const umw, const bool _disabled,
               ConnectionType type, long const ev_number) :
            input{_input[0], _input[1]},
            output{_output[0], _output[1]},
            input_weight(_input_weight),
            memory_weight(_memory_weight),
            reset_input_weight(riw),
            update_input_weight(uiw),
            reset_memory_weight(rmw),
            update_memory_weight(umw),
            connection_type(type),
            evolution_number(ev_number),
            disabled(_disabled) {
    }

    Gene::Gene(point const &input, point const &output,
               double const _input_weight, double const _memory_weight,
               double const riw, double const uiw, double const rmw, double const umw, ConnectionType type,
               long const ev_number) :
            Gene(input, output, _input_weight, _memory_weight, riw, uiw, rmw, umw, false, type, ev_number) {
    }

    Gene::Gene(Gene const &base) :
            input(base.input),
            output(base.output) {
        input_weight = base.input_weight;
        memory_weight = base.memory_weight;
        reset_memory_weight = base.reset_memory_weight;
        reset_input_weight = base.reset_input_weight;
        update_input_weight = base.update_input_weight;
        update_memory_weight = base.update_memory_weight;
        evolution_number = base.evolution_number;
        connection_type = base.connection_type;
        disabled = base.disabled;
    }

    void Gene::set_input_weight(double const new_weight) {
        input_weight = new_weight;
    }

    void Gene::set_memory_weight(double const new_weight) {
        memory_weight = new_weight;
    }

    void Gene::set_reset_input_weight(double const new_weight) {
        reset_input_weight = new_weight;
    }

    void Gene::set_reset_memory_weight(double const new_weight) {
        reset_memory_weight = new_weight;
    }

    void Gene::set_update_input_weight(double const new_weight) {
        update_input_weight = new_weight;
    }

    void Gene::set_update_memory_weight(double const new_weight) {
        update_memory_weight = new_weight;
    }

    void Gene::set_disabled(bool const value) {
        disabled = value;
    }

    void Gene::set_output(int const first, int const second) {
        output[0] = first;
        output[1] = second;
    }

    Gene::point const &Gene::get_input() {
        return input;
    }

    Gene::point const &Gene::get_output() {
        return output;
    }

    double Gene::get_input_weight() const {
        return input_weight;
    }

    double Gene::get_memory_weight() const {
        return memory_weight;
    }

    double Gene::get_reset_input_weight() const {
        return reset_input_weight;
    }

    double Gene::get_update_input_weight() const {
        return update_input_weight;
    }

    double Gene::get_reset_memory_weight() const {
        return reset_memory_weight;
    }

    double Gene::get_update_memory_weight() const {
        return update_memory_weight;
    }

    long Gene::get_ev_number() const {
        return evolution_number;
    }

    ConnectionType Gene::get_type() const {
        return connection_type;
    }

    void Gene::decrement_output() {
        output[0]--;
    }

    void Gene::disable() {
        disabled = true;
    }

    bool Gene::is_disabled() const {
        return disabled;
    }

    void Gene::resize(int const former_size, int const new_size) {
        if (output[0] == former_size) {
            output[0] = new_size;
        }
    }

    std::string Gene::parse_to_string() const {
        std::string str = "{\"input\":[" + std::to_string(input[0]) + ","
                          + std::to_string(input[1]) + "], \"output\":["
                          + std::to_string(output[0]) + "," + std::to_string(output[1])
                          + "],\"input_weight\":" + std::to_string(input_weight) +
                          ",\"memory_weight\":" + std::to_string(memory_weight) +
                          ",\"reset_input_weight\":" + std::to_string(reset_input_weight) +
                          ",\"reset_memory_weight\":" + std::to_string(reset_memory_weight) +
                          ",\"update_input_weight\":" + std::to_string(update_input_weight) +
                          ",\"update_memory_weight\":" + std::to_string(update_memory_weight) +
                          ",\"connection_type\":" + std::to_string(connection_type) +
                          ",\"disabled\":" + (disabled ? "true" : "false") + "}";
        return str;
    }

    bool Gene::operator<(Gene const &that) const {
        return output[0] < that.output[0];
    }

    bool Gene::operator==(Gene const &that) const {
        return input == that.input && output == that.output;
    }
}
