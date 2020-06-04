//
// Created by sakex on 02.06.20.
//


#include "Memory.h"

Memory::Memory() : players() {
    generate_random_grids();
}

void Memory::generate_random_grids() {
    datasets.clear();
    datasets.reserve(100);
    for(int _ = 0; _ < 100; ++_){
        numbers_list numbers;
        for (int i = 0; i < NUMBERS / 2; ++i) {
            constexpr int NUMBERS_BY_2 = (NUMBERS - 2) / 2;
            double const value = static_cast<double>(i * 2  - NUMBERS_BY_2) / static_cast<double>(NUMBERS_BY_2);
            numbers[i*2] = value;
            numbers[i*2 + 1] = value;
        }
        std::shuffle(std::begin(numbers), std::end(numbers), std::mt19937(std::random_device()()));
        datasets.push_back(numbers);
    }

}

std::vector<double> Memory::do_run_generation() {
    auto & _datasets = datasets;
    auto &&cb = [_datasets](MemoryPlayer &player) {
        player.play_rounds(_datasets);
    };
    Threading::for_each(players.begin(), players.end(), cb);
    std::vector<double> outputs;
    outputs.reserve(players.size());
    std::transform(players.begin(), players.end(),
                   std::back_inserter(outputs),
                   [](const MemoryPlayer &player) { return player.score(); });
    return outputs;
}

void Memory::do_reset_players(NN *nets, size_t count) {
    players.clear();
    players.reserve(count);
    for (size_t it = 0; it < count; ++it) {
        players.emplace_back(&nets[it]);
    }
}

void Memory::do_post_training(Topology_ptr top) {
    auto * net = new NeuralNetwork::NN(top);
    MemoryPlayer player(net);
    generate_random_grids();
    player.play_rounds(datasets, true);
    std::cout << "Final score: " << player.score() << std::endl;
    delete net;
}