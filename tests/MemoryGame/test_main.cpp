//
// Created by sakex on 02.06.20.
//

#include "Train.h"
#include "Memory.h"


int main() {
    auto *mem = new Memory;
    Train::Train train(mem, 500, 300, 4, 40, NUMBERS, NUMBERS);
    train.start();
    return 0;
}
