//
// Created by alexandre on 03.09.20.
//

#include "Train.h"
#include "GpuGame.h"

int main() {
    Dim dim {
        5,
        10,
        3
    };
    auto * game = new GpuGame(dim);
    Train::Train train(game, 500, 500, 20, 4, 60, 5, 1);
    train.start();
    return 0;
}
