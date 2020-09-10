/*
 * Train.cpp
 *
 *  Created on: May 26, 2019
 *      Author: sakex
 */

#include "Train.h"

int Train::Constants::MAX_LAYERS = 0;
int Train::Constants::MAX_PER_LAYER = 0;

namespace Train {
    Train::Train(Game::Game *_game, int _iterations, int _max_individuals, int _max_species,
                 int _max_layers, int _max_per_layer, int inputs, int outputs) :
            best_historical_topology{NeuralNetwork::Topology_ptr{nullptr}}, brains{nullptr} {
        game = _game;
        iterations = _iterations;
        inputs_count = inputs;
        outputs_count = outputs;
        max_individuals = _max_individuals;
        max_species = _max_species;
        history.reserve(_iterations);
        Constants::MAX_LAYERS = _max_layers;
        Constants::MAX_PER_LAYER = _max_per_layer;
        random_new_species();
    }

    Train::Train(Game::Game *_game, int _iterations, int _max_individuals, int _max_species, int _max_layers, int _max_per_layer,
                 int inputs, int outputs, NeuralNetwork::Topology_ptr top) :
            game(_game), best_historical_topology{std::move(top)}, brains{nullptr} {
        iterations = _iterations;
        inputs_count = inputs;
        outputs_count = outputs;
        max_individuals = _max_individuals;
        max_species = _max_species;
        history.reserve(_iterations);
        Constants::MAX_LAYERS = _max_layers;
        Constants::MAX_PER_LAYER = _max_per_layer;
        NeuralNetwork::Species_ptr new_species = std::make_unique<NeuralNetwork::Species>();
        *new_species >> best_historical_topology;
        species.emplace_back(std::move(new_species));
    }

    Train::~Train() {
        delete[] brains;
    }

    void Train::random_new_species() {
        using utils::Random;
        NeuralNetwork::Species_ptr new_species = std::make_unique<NeuralNetwork::Species>();
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
            NeuralNetwork::Topology_ptr initial_topology = std::make_shared<NeuralNetwork::Topology>();
            initial_topology->set_layers(2);
            for (int i = 0; i < inputs_count; ++i) {
                NeuralNetwork::Gene::point input = {0, i};
                for (int j = 0; j < connections_per_input; ++j) {
                    int index = not_added[not_added_it];
                    ++not_added_it;
                    NeuralNetwork::Gene::point output = {1, index};
                    NeuralNetwork::Gene::coordinate coordinate = {input, output};

                    double input_weight = Random::random_between(-100, 100) / 100.0f;
                    double const memory_weight = Random::random_between(-100, 100) / 100.0f;
                    double const reset_input_weight = Random::random_between(-100, 100) / 100.0f;
                    double const update_input_weight = Random::random_between(-100, 100) / 100.0f;
                    double const reset_memory_weight = Random::random_between(-100, 100) / 100.0f;
                    double const update_memory_weight = Random::random_between(-100, 100) / 100.0f;
                    NeuralNetwork::ConnectionType type =
                            (NeuralNetwork::ConnectionType)Random::random_between(
                                    (int)NeuralNetwork::ConnectionType::Sigmoid, (int)NeuralNetwork::ConnectionType::GRU);

                    auto *gene = new NeuralNetwork::Gene(input, input_weight, memory_weight, reset_input_weight,
                                               update_input_weight, reset_memory_weight, update_memory_weight, type,
                                               NeuralNetwork::Generation::number(coordinate));
                    gene->set_output(1, index);
                    initial_topology->add_relationship(gene, true);
                }
                initial_topology->generate_output_bias();
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
        reset_players();
        int no_progress = 0;
        for (int it = 0; it != iterations; ++it) { // iterations < 0 -> run forever = other end conditions
            std::cout << it << std::endl;
            utils::Timer run_timer("RUN GENERATION");
            std::vector<double> results = run_generation();
            run_timer.stop();
            assign_results(results);
            update_best();
            if (new_best) {
                no_progress = 0;
            } else {
                no_progress++;
                if (no_progress == 50) {
                    std::cout << "50 generations without progress, ending" << std::endl;
                    break;
                }
            }
            if(it % 20 == 0)
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
        std::vector<NeuralNetwork::Topology_ptr> topologies = topologies_vector();
        std::sort(topologies.begin(), topologies.end(), [](NeuralNetwork::Topology_ptr &top1, NeuralNetwork::Topology_ptr &top2) {
            return top1->get_last_result() > top2->get_last_result();
        });
        for (auto &top: topologies) {
            std::cout << top->get_last_result() << std::endl;
        }
#endif
        post_training();
    }

    std::vector<NeuralNetwork::Topology_ptr> Train::topologies_vector() {
        std::vector<NeuralNetwork::Topology_ptr> topologies;
        std::mutex mutex;
        auto lambda = [&topologies, &mutex](NeuralNetwork::Species_ptr &spec) {
            for (NeuralNetwork::Topology_ptr &topology : spec->get_topologies()) {
                mutex.lock();
                topologies.push_back(topology);
                mutex.unlock();
            }
        };
        Threading::for_each(species.begin(), species.end(), lambda);
        return topologies;
    }

    void Train::reset_species() {
        new_best = false;
        std::vector<NeuralNetwork::Topology_ptr> topologies = topologies_vector();
        reassign_species(topologies);
    }

    void Train::reset_players() {
        last_topologies.clear();
        last_topologies = topologies_vector();

        size_t const size = last_topologies.size();
        delete[] brains;
        brains = new NN[size]();
        for (size_t it = 0; it < size; ++it) {
            brains[it].init_topology(*last_topologies[it]);
        }
        game->reset_players(brains, size);
        std::cout << "SPECIES: " << species.size() <<
                  " TOPOLOGIES: " << last_topologies.size() << std::endl;
    }

    void Train::reassign_species(std::vector<NeuralNetwork::Topology_ptr> &topologies) {
        topologies[0] = best_historical_topology;
        species.clear();
        size_t topologies_size = topologies.size();
        std::mutex mutex;
        for (size_t it = 0; it < topologies_size; ++it) {
            NeuralNetwork::Topology_ptr &topology = topologies[it];
            if (!topology->is_assigned()) {
                NeuralNetwork::Species_ptr new_species = std::make_unique<NeuralNetwork::Species>(topology, 1);
                auto lambda = [&topology, &new_species, &mutex](NeuralNetwork::Topology_ptr &other) {
                    if(other->is_assigned()) return;
                    double const delta = NeuralNetwork::Topology::delta_compatibility(*topology, *other);
                    if (delta <= 2.) {
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
        std::cout << "SPECIES SIZE BEFORE CUT: " << species_size << std::endl;
        int const new_count = std::min(max_species, species_size);
        if (species_size > new_count) {
            std::sort(species.begin(), species.end(), [](NeuralNetwork::Species_ptr &spec1, NeuralNetwork::Species_ptr &spec2) -> bool {
                return spec1->get_best()->get_last_result() < spec2->get_best()->get_last_result();
            });
            int const cut_at = species_size - new_count;
            species.erase(species.begin(), species.begin() + cut_at);
        }
        int const new_max = max_individuals / new_count;
        for (NeuralNetwork::Species_ptr &spec: species) {
            spec->set_max_individuals(new_max);
        }
    }

    inline std::vector<double> Train::run_generation() {
        return game->run_generation();
    }

    void Train::natural_selection() {
        NeuralNetwork::Generation::reset();
        std::function<void(NeuralNetwork::Species_ptr &)> lambda =
                [](NeuralNetwork::Species_ptr &spec) -> void {
                    spec->natural_selection();
                };
#if __MULTITHREADED__
        Threading::for_each(species.begin(), species.end(), lambda);
#else
        std::for_each(species.begin(), species.end(), lambda);
#endif
    }

    void Train::update_best() {
        NeuralNetwork::Topology_ptr best = nullptr;
        NeuralNetwork::Topology_ptr worst = nullptr;
        double max = -std::numeric_limits<double>::max();
        double min = std::numeric_limits<double>::max();

        std::vector<NeuralNetwork::Topology_ptr> topologies = topologies_vector();

        for (NeuralNetwork::Topology_ptr &topology : topologies) {
            double result = topology->get_last_result();
            if (result > max) {
                max = result;
                best = topology;
            }
            if (result < min) {
                min = result;
                worst = topology;
            }
        }

        std::sort(species.begin(), species.end(), [](NeuralNetwork::Species_ptr &spec1, NeuralNetwork::Species_ptr &spec2) -> bool {
            return spec1->get_best()->get_last_result() < spec2->get_best()->get_last_result();
        });

        if (best_historical_topology == nullptr
            || (max >= best_historical_topology->get_last_result() && best != best_historical_topology)) {
            new_best = true;
            best_historical_topology = best;
            NeuralNetwork::Topology best_copy(*best);
            history.push_back(std::move(best_copy));
        }

        std::cout << worst->get_last_result() << " " << species[0]->get_best()->get_last_result() << " "
                  << best->get_last_result() << " "
                  << best_historical_topology->get_last_result() << std::endl;
    }

    void Train::post_training() const {
        /*Serializer::to_file(best_historical_topology->to_string(), "topology.json");
        auto *net = new NN(*best_historical_topology);*/
        game->post_training(history.data(), history.size());
    }

}
