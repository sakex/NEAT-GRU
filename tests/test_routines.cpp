//
// Created by alexandre on 07.06.20.
//

#include "../Private/routines.h"

int main() {
    double arr[] = {-1., -.1, 0, .5, 1., 10.};
    NeuralNetwork::softmax(arr, 6);
    for(auto i: arr) std::cout << i << " ";
    std::cout << std::endl;
    return 0;
}