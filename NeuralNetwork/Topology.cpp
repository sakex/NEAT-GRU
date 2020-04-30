/*
 * Topology.cpp
 *
 *  Created on: July 22, 2019
 *      Author: sakex
 */

#include "Topology.h"

constexpr unsigned MAX_ITERATIONS = 10;
constexpr unsigned MAX_UNFRUITFUL = 10;

namespace NeuralNetwork {

    double Topology::delta_compatibility(Topology &top1, Topology &top2) {
        // see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
        // chapter 4.1
        double disjoints = 0, common = 0;
        double W = 0;
        for (std::pair<long, Phenotype *> pair : top1.ev_number_index) {
            std::unordered_map<long, Phenotype *>::const_iterator search_second =
                    top2.ev_number_index.find(pair.first);
            if (search_second != top2.ev_number_index.end()) {
                common++;
                W += std::abs(pair.second->get_input_weight()
                              - search_second->second->get_input_weight()) +
                     std::abs(pair.second->get_memory_weight()
                              - search_second->second->get_memory_weight())
                     + std::abs(pair.second->get_reset_input_weight()
                                - search_second->second->get_reset_input_weight())
                     + std::abs(pair.second->get_reset_memory_weight()
                                - search_second->second->get_reset_memory_weight())
                     + std::abs(pair.second->get_update_input_weight()
                                - search_second->second->get_update_input_weight())
                     + std::abs(pair.second->get_update_memory_weight()
                                - search_second->second->get_update_memory_weight());
            } else {
                disjoints++;
            }
        }
        size_t const size1 = top1.ev_number_index.size();
        size_t const size2 = top2.ev_number_index.size();
        disjoints += (size1 - common);
        double const N = size1 + size2 <= 20 ? 1 : double(size1 + size2) / 20;
        double const output = 2 * disjoints / N + W / common;
        // printf("COMMON: %f, DISJOINTS: %f, W: %f, O: %f\n", common, disjoints, W, output);
        return output;
    }

    Topology_ptr Topology::crossover(Topology &top1, Topology &top2) {
        Topology *best, *worst;
        if (top1.get_last_result() > top2.get_last_result()) {
            best = &top1;
            worst = &top2;
        } else {
            best = &top2;
            worst = &top1;
        }
        Topology_ptr best_copy = std::make_shared<Topology>(*best);
        // Preloading
        std::unordered_map<long, Phenotype *> &copy_ev_number_index = best_copy->ev_number_index;
        for (std::pair<long, Phenotype *> pair : worst->ev_number_index) {
            std::unordered_map<long, Phenotype *>::const_iterator search_best =
                    copy_ev_number_index.find(pair.first);
            if (search_best != copy_ev_number_index.end()) {
                continue;
            }
            auto *copied_phenotype = new Phenotype(*pair.second);
            copied_phenotype->set_disabled(false);
            best_copy->add_relationship(copied_phenotype);
            best_copy->disable_phenotypes(copied_phenotype->get_input(), copied_phenotype->get_output());
            best_copy->new_mutation(copied_phenotype, best_copy->last_result);

        }
        return best_copy;
    }

    Topology::Topology(Topology const &base) :
            relationships(), mutations(), ev_number_index() {
        layers = base.layers;
        layers_size = base.layers_size;
        last_result = base.last_result;
        result_before_mutation = base.last_result;
        best_historical_result = 0;
        phenotype_cb cb = [this](Phenotype *phenotype) {
            if (phenotype->is_disabled()) return;
            auto *copy = new Phenotype(*phenotype);
            add_to_relationships_map(copy);
        };
        base.iterate_phenotypes(cb);
    }

    Topology::~Topology() {
        phenotype_cb cb = [](Phenotype *phenotype) {
            delete phenotype;
        };
        iterate_phenotypes(cb);
        relationships.clear();
        ev_number_index.clear();
    }

    Topology &Topology::operator=(Topology const &base) {
        if (this != &base) {
            layers = base.layers;
            layers_size = base.layers_size;
            last_result = base.last_result;
            result_before_mutation = base.last_result;
            best_historical_result = 0;
            phenotype_cb cb = [](Phenotype *phenotype) {
                delete phenotype;
            };
            iterate_phenotypes(cb);
            relationships.clear();
            ev_number_index.clear();
            for (auto &it : base.relationships) {
                for (Phenotype *phenotype : it.second) {
                    if (phenotype->is_disabled()) continue;
                    auto *copy = new Phenotype(*phenotype);
                    add_to_relationships_map(copy);
                }
            }
        }
        return *this;
    }

    bool Topology::operator==(NeuralNetwork::Topology const &comparison) const {
        for (auto const &pair: ev_number_index) {
            auto const search = comparison.ev_number_index.find(pair.first);
            bool const not_in = search == comparison.ev_number_index.end();
            Phenotype *phen1 = pair.second;
            Phenotype *phen2 = search->second;
            bool const equal = *phen1 == *phen2;
            if (not_in || !equal) return false;
        }
        return true;
    }

    void Topology::set_layers(int const _layers) {
        layers = _layers;
        layers_size.resize(layers, 1);
        layers_size[layers - 1] = layers_size[layers - 2];
        layers_size[layers - 2] = 1;
    }

    int Topology::get_layers() const {
        return layers;
    }

    void Topology::set_last_result(const long double result) {
        last_result = result;
        if (last_result > best_historical_result) best_historical_result = last_result;
    }

    long double Topology::get_last_result() const {
        return last_result;
    }

    Topology::relationships_map &Topology::get_relationships() {
        return relationships;
    }

    void Topology::add_relationship(Phenotype *phenotype, const bool init) {
        if (phenotype->is_disabled()) return;
        Phenotype::point input = phenotype->get_input();
        Phenotype::point output = phenotype->get_output();
        if (!layers)
            throw NoLayer();
        if (input[1] + 1 > layers_size[input[0]]) {
            layers_size[input[0]] = input[1] + 1;
        }
        if (!init && output[0] == layers) {
            resize(output[0]);
            phenotype->decrement_output();
        } else if (output[1] + 1 > layers_size[output[0]]) {
            layers_size[output[0]] = output[1] + 1;
        }
        add_to_relationships_map(phenotype);
    }

    void Topology::set_assigned(bool _assigned) {
        assigned = _assigned;
    }

    bool Topology::is_assigned() const {
        return assigned;
    }

    bool Topology::optimize() {
        size_t mutations_queued = mutations.size();
        if (!mutations_queued) {
            return false;
        }
        Mutation &mutation = mutations.front();
        if (mutation.get_iterations() >= MAX_ITERATIONS || mutation.get_unfruitful() >= MAX_UNFRUITFUL) {
            mutation.set_back_to_max();
            int field = mutation.get_field();
            if (field != 5) {
                mutation.set_field(field + 1);
                return true;
            }
            mutations.pop();
            last_result = best_historical_result;
            if (mutations_queued > 1) {
                result_before_mutation = best_historical_result;
                Mutation &current_mutation = mutations.front();
                current_mutation.mutate(last_result);
                return true;
            }
            return false;
        }
        mutation.mutate(last_result);
        return true;
    }

    bool Topology::mutation_positive() const {
        return best_historical_result > result_before_mutation;
    }

    void Topology::set_optimized() {
        std::queue<Mutation> empty;
        std::swap(mutations, empty);
    }

    void Topology::new_generation(size_t const count,
                                  std::vector<Topology_ptr> &topologies) {
        for (unsigned it = 0; it < count; ++it) {
            topologies.push_back(evolve());
        }
    }

    void Topology::add_to_relationships_map(Phenotype *phenotype) {
        Phenotype::point input = phenotype->get_input();
        auto iterator = relationships.find(input);
        if (iterator != relationships.end()) {
            iterator->second.push_back(phenotype);
        } else {
            relationships[input] = std::vector<Phenotype *>{phenotype};
        }
        ev_number_index[phenotype->get_ev_number()] = phenotype;
    }

    void Topology::disable_phenotypes(Phenotype::point const &input,
                                      Phenotype::point const &output) {
        auto iterator = relationships.find(input);
        if (iterator == relationships.end())
            return;
        std::vector<Phenotype *> base_vector = iterator->second;
        Phenotype *&back = base_vector.back();
        for (Phenotype *&it : base_vector) {
            if (it == back || it->is_disabled()) {
                continue;
            }
            Phenotype::point compared_output = it->get_output();
            if (output == compared_output || path_overrides(compared_output, output)
                || path_overrides(output, compared_output)) {
                it->disable();
            }
        }
    }

    bool Topology::path_overrides(Phenotype::point const &input,
                                  Phenotype::point const &output) {
        relationships_map::const_iterator iterator = relationships.find(input);
        if (iterator == relationships.end())
            return false;
        std::vector<Phenotype *> base_vector = iterator->second;
        for (Phenotype *it : base_vector) {
            if (it->is_disabled()) {
                continue;
            }
            Phenotype::point compared_output = it->get_output();
            if (compared_output == output) {
                return true;
            } else if (compared_output[0] <= output[0]) {
                if (path_overrides(compared_output, output))
                    return true;
            }
        }
        return false;
    }

    void Topology::resize(int const new_max) {
        for (auto &it : relationships) {
            for (Phenotype *phenotype : it.second) {
                phenotype->resize(new_max - 1, new_max);
            }
        }
        set_layers(new_max + 1);
    }

    Topology_ptr Topology::evolve() {
        using utils::Random;
        Topology_ptr new_topology = std::make_shared<Topology>(*this);
        std::vector<Phenotype *> new_phenotypes = new_topology->mutate();
        for (auto const last_phenotype: new_phenotypes) {
            new_topology->new_mutation(last_phenotype, last_result);
        }
        return new_topology;
    }


    std::vector<Phenotype *> Topology::mutate() {
        // Input must already exist and output may or may not exist
        using utils::Random;
        constexpr int MAX_LAYERS = 10;
        constexpr int MAX_POS = 50;
        bool new_output = false;
        int max_layer = std::min(layers, MAX_LAYERS);
        int input_index = layers >= 2 ? Random::random_number(max_layer - 2) : 0;
        int input_position = Random::random_number(layers_size[input_index] - 1);
        /*int new_layer = 0;
        if (input_index == layers - 2 && layers <= MAX_LAYERS) {
            new_layer = Random::random_number(1);
        }*/
        int output_index = Random::random_between(input_index + 1, max_layer);
        //int output_index = input_index + 1 + new_layer;
        int output_position = 0;
        if (output_index < layers - 1) {
            output_position = Random::random_number(std::min(layers_size[output_index], MAX_POS));
            if (output_position >= layers_size[output_index]) {
                new_output = true;
            }
        } else if (output_index == layers - 1) {
            output_position = Random::random_number(layers_size[output_index] - 1);
        } else { // if output_index == layers
            new_output = true;
        }
        Phenotype::point input = {input_index, input_position};
        Phenotype::point output = {output_index, output_position};
        Phenotype *last_phenotype = new_phenotype(input, output);
        std::vector<Phenotype *> added_phenotypes{last_phenotype};
        if (new_output) {
            int _back = layers_size.back();
            output = last_phenotype->get_output();
            int index = Random::random_number(_back - 1);
            Phenotype::point output_output = {layers - 1, index};
            Phenotype *connection = new_phenotype(output, output_output);
            added_phenotypes.push_back(connection);
        }
        disable_phenotypes(input, output);
        return added_phenotypes;
    }

    void Topology::new_mutation(Phenotype *last_phenotype, long double const wealth) {
        mutations.emplace(last_phenotype, wealth);
    }

    Phenotype *Topology::new_phenotype(Phenotype::point const &input,
                                       Phenotype::point const &output) {
        using utils::Random;
        Phenotype::coordinate coordinate{input, output};
        long const ev_number = Generation::number(coordinate);
        double input_weight = Random::random_between(-100, 100) / 100.0;
        double const memory_weight = Random::random_between(-100, 100) / 100.0;
        double const reset_input_weight = Random::random_between(-100, 100) / 100.0;
        double const reset_memory_weight = Random::random_between(-100, 100) / 100.0;
        double const update_input_weight = Random::random_between(-100, 100) / 100.0;
        double const update_memory_weight = Random::random_between(-100, 100) / 100.0;
        auto *phenotype = new Phenotype(input, output, input_weight, memory_weight, reset_input_weight,
                                        reset_memory_weight, update_input_weight, update_memory_weight, ev_number);
        add_relationship(phenotype);
        return phenotype;
    }

    std::string Topology::parse_to_string() const {
        std::string output = "[";
        phenotype_cb cb = [&output](Phenotype *phenotype) -> void {
            output += phenotype->to_string() + ",";
        };
        iterate_phenotypes(cb);
        output.replace(output.end() - 1, output.end(), "]");
        return output;
    }


    bool Topology::operator<(Topology const &comparison) const {
        return last_result < comparison.last_result;
    }

    void Topology::iterate_phenotypes(phenotype_cb &cb) const {
        for (auto const &it : relationships) {
            for (Phenotype *phenotype : it.second) {
                cb(phenotype);
            }
        }
    }

}
