//
// Created by sakex on 02.06.20.
//

#include "MemoryGrid.h"

#include <random>
#include <iostream>

MemoryGrid::MemoryGrid() : numbers(), found(), won(false) {
    for (bool &v: found) v = false;
}

void MemoryGrid::reset(numbers_list const & list) {
    for (bool &v: found) v = false;
    won = false;
    numbers = list;
}

bool MemoryGrid::has_won() const {
    return won;
}

numbers_list MemoryGrid::pick_two(int const pos1, int const pos2) {
    numbers_list arr_cp = numbers;
    for (int i = 0; i < NUMBERS; ++i) {
        if (i != pos1 && i != pos2 && !found[i]) {
            arr_cp[i] = -10;
        }
    }
    if (arr_cp[pos1] == arr_cp[pos2]) {
        found[pos1] = true;
        found[pos2] = true;
        for (bool f: found) if (!f) return arr_cp;
        won = true;
    }
    return arr_cp;
}

long MemoryGrid::get_found() {
    long count = 0;
    for (bool f: found) count += f;
    return count;
}