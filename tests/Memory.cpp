//
// Created by sakex on 02.06.20.
//


#include "Memory.h"

Memory::Memory() : players() {

}

std::vector<double> Memory::do_run_generation() {
    auto &&cb = [](MemoryPlayer &player) {
        player.play_rounds(50);
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
    player.play_rounds(50, true);
}