//
// Created by sakex on 02.06.20.
//


#include "TrendGame.h"


TrendGame::TrendGame() = default;

void get_current_max_index(std::vector<int> const &vec, std::vector<int> & output) {
    int current_count[DIFFERENT_NUMBERS];
    for(auto & v: current_count) v = 0;
    output.reserve(vec.size());
    for (int current : vec) {
        current_count[current]++;
        int * const max_elem = std::max_element(current_count, current_count + DIFFERENT_NUMBERS);
        int const max_index = std::distance(current_count, max_elem);
        output.push_back(max_index);
    }
}

std::vector<Dataset> TrendGame::generate_dataset() {
    std::vector<Dataset> datasets;
    datasets.reserve(100);
    for (size_t it = 0; it < 100; ++it) {
        Dataset ds {std::vector<int>(), std::vector<int>()};
        ds.data.reserve(100);
        for (size_t j = 0; j < 100; ++j) {
            int number = utils::Random::random_number(DIFFERENT_NUMBERS);
            ds.data.push_back(number);
        }
        get_current_max_index(ds.data, ds.most_frequent);
        datasets.push_back(ds);
    }
    return datasets;
}

std::vector<double> TrendGame::do_run_generation() {
    std::vector<Dataset> datasets = generate_dataset();
    auto &&cb = [&datasets](Player &player) {
        double input[DIFFERENT_NUMBERS];
        for (Dataset &ds: datasets) {
            for (size_t it = 0; it < ds.data.size(); ++ it) {
                int const index = ds.data[it];
                for (double &v: input) v = -1.;
                input[index] = 1.;
                std::vector<double> result = player.network->compute(input);
                auto max_elem = std::max_element(result.begin(), result.end());
                int const max_index = std::distance(result.begin(), max_elem);
                player.score -= (max_index != ds.most_frequent[it]);
            }
            player.network->reset_state();
        }
    };
    Threading::for_each(players.begin(), players.end(), cb);
    std::vector<double> outputs;
    outputs.reserve(players.size());
    std::transform(players.begin(), players.end(),
                   std::back_inserter(outputs),
                   [](const Player &player) { return player.score; });
    return outputs;
}

void TrendGame::do_reset_players(NN *nets, size_t count) {
    players.clear();
    players.reserve(count);
    for (size_t it = 0; it < count; ++it) players.push_back({&nets[it], 0});
}

void TrendGame::do_post_training(Topology_ptr top) {
    auto * net = new NeuralNetwork::NN(top);
    Player player {net, 0};
    std::vector<Dataset> datasets = generate_dataset();
    double input[DIFFERENT_NUMBERS];
    for (Dataset &ds: datasets) {
        std::vector<int> sequence;
        sequence.reserve(datasets[0].data.size());
        for (size_t it = 0; it < ds.data.size(); ++ it) {
            int const index = ds.data[it];
            for (double &v: input) v = -1.;
            input[index] = 1.;
            std::vector<double> result = player.network->compute(input);
            auto max_elem = std::max_element(result.begin(), result.end());
            int const max_index = std::distance(result.begin(), max_elem);
            player.score -= (max_index != ds.most_frequent[it]);
            sequence.push_back(max_index);
        }
        std::cout << "Real Data" << std::endl;
        for(auto i: ds.data) std::cout << i << " ";
        std::cout << std::endl;
        std::cout << "Real sequence" << std::endl;
        for(auto i: ds.most_frequent) std::cout << i << " ";
        std::cout << std::endl << "AI sequence" << std::endl;
        for(auto i: sequence) std::cout << i << " ";
        std::cout << "\n==========================================\n" << std::endl;
        player.network->reset_state();
    }
    std::cout << "Final score: " << player.score << std::endl;
}