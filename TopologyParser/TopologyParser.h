//
// Created by sakex on 13.11.19.
//

#ifndef TRADING_TOPOLOGYPARSER_H
#define TRADING_TOPOLOGYPARSER_H

#include <nlohmann/json.hpp>
#include "../NeuralNetwork/Topology.h"


struct TopologyParser {
    static NeuralNetwork::Topology parse(nlohmann::json & j);
};


#endif //TRADING_TOPOLOGYPARSER_H
