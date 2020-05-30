/*
 * Train.cpp
 *
 *  Created on: May 26, 2019
 *      Author: sakex
 */

#include "Train.h"

namespace Train {

    Train::Train(Game::Game *_game, int const _iterations, int const _max_individuals, int const inputs,
                 int const outputs) :
            best_historical_topology{Topology_ptr{nullptr}}, brains{nullptr} {
        game = _game;
        iterations = _iterations;
        inputs_count = inputs;
        outputs_count = outputs;
        max_individuals = _max_individuals;
        random_new_species();
    }

    Train::Train(Game::Game *_game, int const _iterations, int const _max_individuals, int const inputs,
                 int const outputs, Topology_ptr top) :
            game(_game), best_historical_topology{std::move(top)}, brains{nullptr} {
        iterations = _iterations;
        inputs_count = inputs;
        outputs_count = outputs;
        max_individuals = _max_individuals;
        Species_ptr new_species = std::make_unique<Species>();
        *new_species >> best_historical_topology;
        species.emplace_back(std::move(new_species));
    }

    Train::~Train() {
        delete[] brains;
    }

    void Train::random_new_species() {
        using utils::Random;
        Species_ptr new_species = std::make_unique<Species>();
        int const connections_per_input = std::ceil((double) outputs_count / (double) inputs_count);
        std::vector<int> not_added;
        int output_index = 0;
        for (int i = 0; i < inputs_count; ++i) {
            for (int o = 0; o < connections_per_input; ++o) {
                not_added.push_back(output_index);
                if (output_index < outputs_count - 1) output_index++;
                else output_index = 0;
            }
        }
        std::random_device rd;
        std::mt19937 g(rd());
        for (int count = 0; count < max_individuals; ++count) {
            std::shuffle(not_added.begin(), not_added.end(), g);
            int not_added_it = 0;
            Topology_ptr initial_topology = std::make_shared<Topology>();
            initial_topology->set_layers(2);
            for (int i = 0; i < inputs_count; ++i) {
                Phenotype::point input = {0, i};
                double input_weight = Random::random_between(-100, 100) / 100.0;
                double const memory_weight = Random::random_between(-100, 100) / 100.0;
                double const reset_input_weight = Random::random_between(-100, 100) / 100.0;
                double const reset_memory_weight = Random::random_between(-100, 100) / 100.0;
                double const update_input_weight = Random::random_between(-100, 100) / 100.0;
                double const update_memory_weight = Random::random_between(-100, 100) / 100.0;

                for (int j = 0; j < connections_per_input; ++j) {
                    int index = not_added[not_added_it];
                    ++not_added_it;
                    Phenotype::point output = {1, index};
                    Phenotype::coordinate coordinate = {input, output};
                    auto *phenotype = new Phenotype(input, input_weight, memory_weight, reset_input_weight,
                                                    reset_memory_weight, update_input_weight, update_memory_weight,
                                                    Generation::number(coordinate));
                    phenotype->set_output(1, index);
                    initial_topology->add_relationship(phenotype, true);
                }
            }
            *new_species >> initial_topology;
        }
        species.emplace_back(std::move(new_species));
    }

    inline void Train::assign_results(std::vector<double> const &results) {
        size_t const size = results.size();
#if DEBUG
        assert(size == last_topologies.size());
#endif
        for (size_t it = 0; it < size; ++it) {
            last_topologies[it]->set_last_result(results[it]);
        }
    }

    void Train::start() {
        std::cout << "before reset" << std::endl;
        reset_players();
        std::cout << "after reset" << std::endl;
        int no_progress = 0;
        for (int it = 0; it != iterations; ++it) { // iterations < 0 -> run forever = other end conditions
            std::cout << it << std::endl;
            utils::Timer run_timer("RUN GENERATION");
            std::vector<double> results = run_dataset();
            delete[] brains;
            run_timer.stop();
            assign_results(results);
            update_best();
            if (new_best) {
                no_progress = 0;
            } else {
                no_progress++;
                if (no_progress == 500) {
                    std::cout << "500 generations without progress, exciting" << std::endl;
                    break;
                }
            }
            reset_species();
            utils::Timer selection_timer("NATURAL SELECTION");
            natural_selection();
            selection_timer.stop();
            utils::Timer reset_timer("RESET");
            reset_players();
            reset_timer.stop();
        }
#if DEBUG
        run_dataset();
        assign_results();
        update_best();
        std::vector<Topology_ptr> topologies = topologies_vector();
        std::sort(topologies.begin(), topologies.end(), [](Topology_ptr &top1, Topology_ptr &top2) {
            return top1->get_last_result() > top2->get_last_result();
        });
        for (auto &top: topologies) {
            std::cout << top->get_last_result() << std::endl;
        }
#endif
        plot_best();
        std::cout << "OUT" << std::endl;
    }

    std::vector<Topology_ptr> Train::topologies_vector() {
        std::vector<Topology_ptr> topologies;
        std::mutex mutex;
        auto lambda = [&topologies, &mutex](Species_ptr &spec) {
            for (Topology_ptr &topology : spec->get_topologies()) {
                mutex.lock();
                topologies.push_back(topology);
                mutex.unlock();
            }
        };
        Threading::for_each(species.begin(), species.end(), lambda);
        return topologies;
    }

    void Train::reset_species() {
        /*if ((new_best && turns_without_reassignments >= next_reassignment) ||
            turns_without_reassignments >= 5 * next_reassignment) {*/
        new_best = false;
        /*turns_without_reassignments = 0;
        if (next_reassignment < 20)
            ++next_reassignment;*/
        std::vector<Topology_ptr> topologies = topologies_vector();
        reassign_species(topologies);
        /*} else {
            turns_without_reassignments++;
        }*/
    }

    void Train::reset_players() {
        last_topologies = topologies_vector();
        size_t const size = last_topologies.size();
        brains = new NN[size];
        for (size_t it = 0; it < size; ++it) {
            brains[it].init_topology(last_topologies[it]);
        }
        game->reset_players(brains, size);
        std::cout << "SPECIES: " << species.size() <<
                  " TOPOLOGIES: " << last_topologies.size() << std::endl;
    }

    void Train::reassign_species(std::vector<Topology_ptr> &topologies) {
        topologies[0] = std::make_shared<Topology>(*best_historical_topology);
        species.clear();
        size_t topologies_size = topologies.size();
        std::mutex mutex;
        for (size_t it = 0; it < topologies_size; ++it) {
            Topology_ptr &topology = topologies[it];
            if (!topology->is_assigned()) {
                Species_ptr new_species = std::make_unique<Species>(topology, 1);
                auto lambda = [&topology, &new_species, &mutex](Topology_ptr &other) {
                    double const delta = Topology::delta_compatibility(*topology, *other);
                    if (!other->is_assigned() && delta <= 1) {
                        other->set_assigned(true);
                        mutex.lock();
                        *new_species >> other;
                        mutex.unlock();
                    }
                };
                Threading::for_each(topologies.begin() + it + 1, topologies.end(), lambda);
                species.push_back(std::move(new_species));
            }
            topology->set_assigned(false);
        }
        extinct_species();
    }

    void Train::extinct_species() {
        int species_size = species.size();
        int const new_count = 20;
        if (species_size > new_count) {
            std::sort(species.begin(), species.end(), [](Species_ptr &spec1, Species_ptr &spec2) -> bool {
                return spec1->get_best()->get_last_result() < spec2->get_best()->get_last_result();
            });
            int const cut_at = species_size - new_count;
            int const new_max = max_individuals / new_count;
            species.erase(species.begin(), species.begin() + cut_at);
            for (Species_ptr &spec: species) {
                spec->set_max_individuals(new_max);
            }
        }
    }

    inline std::vector<double> Train::run_dataset() {
        return game->run_generation();
    }

    void Train::natural_selection() {
        NeuralNetwork::Generation::reset();
        std::function<void(Species_ptr &)> lambda =
                [](Species_ptr &spec) -> void {
                    spec->natural_selection();
                };
#if __MULTITHREADED__
        Threading::for_each(species.begin(), species.end(), lambda);
#else
        std::for_each(species.begin(), species.end(), lambda);
#endif
    }

    void Train::update_best() {
        Topology_ptr best = nullptr;
        Topology_ptr worst = nullptr;
        long double max = -10000000;
        long double min = 100000000;
        std::vector<Topology_ptr> topologies = topologies_vector();
        for (Topology_ptr &topology : topologies) {
            long double result = topology->get_last_result();
            if (result > max) {
                max = result;
                best = topology;
            }
            if (result < min) {
                min = result;
                worst = topology;
            }
        }
        std::sort(species.begin(), species.end(), [](Species_ptr &spec1, Species_ptr &spec2) -> bool {
            return spec1->get_best()->get_last_result() < spec2->get_best()->get_last_result();
        });
        if (best_historical_topology == nullptr
            || (max > best_historical_topology->get_last_result() && best != best_historical_topology)) {
            new_best = true;
            best_historical_topology = std::make_shared<Topology>(*best);
        }
        std::cout << worst->get_last_result() << " " << species[0]->get_best()->get_last_result() << " "
                  << best->get_last_result() << " "
                  << best_historical_topology->get_last_result() << std::endl;
    }

    void Train::plot_best() const {
        Game::Player *best = game->post_training(best_historical_topology);
        if (!best) return;
        std::cout << "RESULT: " << (best->get_result() - 3000) / 3000 << std::endl;
        // Serializer::to_file(&*(best->get_topology()), "topology.json");
    }


    Topology_ptr Train::get_best() const {
        return best_historical_topology;
    }
}