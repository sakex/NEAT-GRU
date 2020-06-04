//
// Created by sakex on 02.06.20.
//

#include "MemoryPlayer.h"

MemoryPlayer::MemoryPlayer(NeuralNetwork::NN *net) : network(net), grid() {
}

std::array<int, 2> max_two_values(std::vector<double> const &input) {
    int first_max = -1;
    double first_max_value = -100000;
    int second_max = -1;
    double second_max_value = -100000;
    for (size_t i = 0; i < input.size(); ++i) {
        double const v = input[i];
        if (v > first_max_value) {
            second_max_value = first_max_value;
            first_max_value = v;
            second_max = first_max;
            first_max = i;
        } else if (v > second_max_value) {
            second_max_value = v;
            second_max = i;
        }
    }
    return {first_max, second_max};
}

void MemoryPlayer::play_rounds(std::vector<numbers_list> const &datasets, bool showing) {
    for (numbers_list const &list: datasets) {
        grid.reset(list);
        _score -= play(showing);
        network->reset_state();
        if (showing)
            std::cout << "====================================" << std::endl;
    }
}

long MemoryPlayer::score() const {
    return _score;
}

long MemoryPlayer::play(bool showing) {
    long tries = 0;
    double game_info[NUMBERS];
    numbers_list first_arr = grid.pick_two(0, 1);
    std::copy(first_arr.begin(), first_arr.end(), game_info);
    std::vector<double> first_computed = network->compute(game_info);
    std::array<int, 2> plays = max_two_values(first_computed);
    while (!grid.has_won() && tries < 100) {
        numbers_list current_game = grid.pick_two(plays[0], plays[1]);
        std::copy(current_game.begin(), current_game.end(), game_info);
        if (showing) {
            for (double it : game_info) std::cout << it << " ";
            std::cout << std::endl;
        }
        std::vector<double> computed = network->compute(game_info);
        plays = max_two_values(computed);
        tries++;
    }
    if (grid.has_won())
        return tries;
    return tries - grid.get_found();
}