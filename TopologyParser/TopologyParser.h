//
// Created by sakex on 13.11.19.
//

#ifndef TRADING_TOPOLOGYPARSER_H
#define TRADING_TOPOLOGYPARSER_H

#include <nlohmann/json.hpp>
#include "../NeuralNetwork/Topology.h"

/// Namespace for parsing topologies
struct TopologyParser {
    /**
     * Parse a JSON and returns a topology
     * @param j Json to be converted to a Topology
     * @return Topology from the json
     */
    static NeuralNetwork::Topology parse(nlohmann::json & j);
};


#endif //TRADING_TOPOLOGYPARSER_H
