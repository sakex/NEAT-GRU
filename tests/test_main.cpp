//
// Created by sakex on 02.06.20.
//

#include "../Train/Train.h"
#include "Memory.h"


int main() {
    auto *mem = new Memory;
    Train::Train train(mem, 1000, 1000, 5, 10, NUMBERS, NUMBERS);
    train.start();
    return 0;
}
