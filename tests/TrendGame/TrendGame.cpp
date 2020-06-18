//
// Created by sakex on 02.06.20.
//


#include "TrendGame.h"


TrendGame::TrendGame() : players(), datasets() {
    generate_dataset();
}

void get_current_max_index(std::vector<int> const &vec, std::vector<int> &output) {
    int current_count[DIFFERENT_NUMBERS];
    for (auto &v: current_count) v = 0;
    output.reserve(vec.size());
    for (int current : vec) {
        current_count[current]++;
        int *const max_elem = std::max_element(current_count, current_count + DIFFERENT_NUMBERS);
        int const max_index = std::distance(current_count, max_elem);
        output.push_back(max_index);
    }
}

void TrendGame::generate_dataset() {
    datasets.clear();
    datasets.reserve(150);
    for (size_t it = 0; it < 150; ++it) {
        Dataset ds{std::vector<int>(), std::vector<int>()};
        ds.data.reserve(50);
        for (size_t j = 0; j < 50; ++j) {
            int number = utils::Random::random_number(DIFFERENT_NUMBERS - 1);
            ds.data.push_back(number);
        }
        get_current_max_index(ds.data, ds.most_frequent);
        datasets.push_back(ds);
    }
}

std::vector<float> TrendGame::do_run_generation() {
    auto const &_datasets = datasets;
    auto cb = [&_datasets](Player &player) {
        std::array<float, DIFFERENT_NUMBERS> input{};
        for (Dataset const &ds: _datasets) {
            for (size_t it = 0; it < ds.data.size(); ++it) {
                float *result;
                int const index = ds.data[it];
                for (auto &i: input) i = -1.;
                input[index] = 1.;
                result = player.network->compute(input.data());
                auto max_elem = std::max_element(result, result + DIFFERENT_NUMBERS);
                int const max_index = std::distance(result, max_elem);
                player.score -= (max_index != ds.most_frequent[it]);
                delete[] result;
            }
            player.network->reset_state();
        }
    };
    std::for_each(players.begin(), players.end(), cb);
    std::vector<float> outputs;
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

void TrendGame::do_post_training(NN *net) {
    Player player{net, 0};
    generate_dataset();
    float input[DIFFERENT_NUMBERS];
    for (Dataset &ds: datasets) {
        std::vector<int> sequence;
        sequence.reserve(datasets[0].data.size());
        for (size_t it = 0; it < ds.data.size(); ++it) {
            int const index = ds.data[it];
            for (float &v: input) v = -1.;
            input[index] = 1.;
            float *result = player.network->compute(input);
            auto max_elem = std::max_element(result, result + DIFFERENT_NUMBERS);
            int const max_index = std::distance(result, max_elem);
            player.score -= (max_index != ds.most_frequent[it]);
            sequence.push_back(max_index);
            delete[] result;
        }
        std::cout << "Real Data" << std::endl;
        for (auto i: ds.data) std::cout << i << " ";
        std::cout << std::endl;
        std::cout << "Real sequence" << std::endl;
        for (auto i: ds.most_frequent) std::cout << i << " ";
        std::cout << std::endl << "AI sequence" << std::endl;
        for (auto i: sequence) std::cout << i << " ";
        std::cout << "\n==========================================\n" << std::endl;
        player.network->reset_state();
    }
    std::cout << "Final score: " << player.score << std::endl;
}