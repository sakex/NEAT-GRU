//
// Created by sakex on 02.06.20.
//

#ifndef NEAT_GRU_MEMORYGRID_H
#define NEAT_GRU_MEMORYGRID_H

#include <algorithm>
#include <array>
#include "constants.h"

typedef std::array<double, NUMBERS> numbers_list;

class MemoryGrid {
public:
    MemoryGrid();

    bool has_won() const;

    numbers_list pick_two(int pos1, int pos2);

    void reset();

    long get_found();

private:
    numbers_list numbers;
    std::array<bool, NUMBERS> found;
    bool won;
};


#endif //NEAT_GRU_MEMORYGRID_H
