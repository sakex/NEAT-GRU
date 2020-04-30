/*
 * Species.cpp
 *
 *  Created on: Jul 27, 2019
 *      Author: sakex
 */

#include "Species.h"

namespace NeuralNetwork {

    Species::Species() : best_topology{new Topology}, max_individuals(1) {
    }

    Species::Species(Topology_ptr const &topology, const int max) :
            best_topology{new Topology(*topology)} {
        topologies.push_back(topology);
        max_individuals = max;
    }

    bool Species::operator<(Species const &other) {
        return topologies.size() < other.topologies.size();
    }

    void Species::operator>>(Topology_ptr &topology) {
        topologies.push_back(topology);
        update_best(topology);
        ++max_individuals;
    }

    void Species::set_max_individuals(int const _max) {
        max_individuals = _max;
    }

    std::vector<Topology_ptr> &Species::get_topologies() {
        return topologies;
    }

    void Species::natural_selection() {
        std::sort(topologies.begin(), topologies.end(), [](Topology_ptr &t1, Topology_ptr &t2) -> bool {
            return t1->get_last_result() < t2->get_last_result();
        });
        Topology_ptr most_successful = topologies.back();
        update_best(most_successful);
        do_selection();
    }

    void Species::do_selection() {
        int topologies_size = topologies.size();
        if (!topologies_size) return;
        std::vector<Topology_ptr> surviving_topologies;
        std::vector<Topology_ptr> contenders;
        surviving_topologies.reserve(max_individuals);
        int quarter = max_individuals * 3 / 4;
        for (int it = topologies_size / 2; it < topologies_size; ++it) {
            Topology_ptr &topology = topologies[it];
            if (it >= quarter) {
                contenders.push_back(topology);
            } else if (topology->optimize()) {
                surviving_topologies.push_back(topology);
            }
        }
        mate(surviving_topologies, contenders);
        evolve(surviving_topologies, contenders);
        topologies = std::move(surviving_topologies);
        duplicate_best();
    }

    void Species::mate(std::vector<Topology_ptr> &surviving_topologies,
                       std::vector<Topology_ptr> &positive) {
        size_t positive_size = positive.size();
        if (positive_size < 2 || max_individuals <= surviving_topologies.size())
            return;
        for (size_t it = 0; it < positive_size - 2; ++it) {
            Topology_ptr &topology = positive[it];
            if (!topology->mutation_positive()) continue;
            size_t random_index = utils::Random::random_between(it + 1, positive_size - 1);
            Topology_ptr &other_topology = topologies[random_index];
            if (topology->get_layers() != other_topology->get_layers()) continue;
            Topology_ptr child = Topology::crossover(*topology, *other_topology);
            surviving_topologies.push_back(child);
        }
    }

    void Species::evolve(std::vector<Topology_ptr> &surviving_topologies,
                         std::vector<Topology_ptr> &contenders) {
        size_t positive_size = contenders.size();
        if (positive_size == 0 || max_individuals <= surviving_topologies.size())
            return;
        for (size_t it = positive_size - 1; it >= 0; --it) {
            Topology_ptr &topology = contenders[it];
            size_t new_individuals;
            if (max_individuals > surviving_topologies.size()) {
                new_individuals = (max_individuals - surviving_topologies.size()) / (positive_size - it);
            } else {
                break;
            }
            topology->new_generation(new_individuals, surviving_topologies);
        }
    }

    void Species::update_best(Topology_ptr const &most_successful) {
        if (most_successful->get_last_result() >= best_topology->get_last_result()) {
            best_topology = std::make_shared<Topology>(*most_successful);
            best_topology->set_optimized();
        }
    }

    void Species::duplicate_best() {
        int topologies_size = topologies.size();
        int best_quantity = max_individuals - topologies_size;
        if (best_quantity > 0)
            best_topology->new_generation(best_quantity, topologies);
    }

    Topology_ptr Species::get_best() {
        return best_topology;
    }

} /* namespace NeuralNetwork */
