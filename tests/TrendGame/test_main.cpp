//
// Created by sakex on 02.06.20.
//

#include "Train.h"
#include "TrendGame.h"


int main() {
    auto *mem = new TrendGame;
    Train::Train train(mem, 1000, 1000, 10, 60,
                       DIFFERENT_NUMBERS, DIFFERENT_NUMBERS);
    train.start();
    return 0;
}
