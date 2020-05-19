/*
 * Species.h
 *
 *  Created on: Jul 27, 2019
 *      Author: sakex
 */

#ifndef NEURALNETWORK_SPECIES_H_
#define NEURALNETWORK_SPECIES_H_

#include <vector>
#include <memory>
#include <algorithm>

#include "../NeuralNetwork/Topology.h"

namespace NeuralNetwork {

    class Species {

    public:
        Species();

        Species(Topology_ptr const &, int);

        virtual ~Species() = default;

    public:
        bool operator<(Species const &);

        void operator>>(Topology_ptr &);

    public:
        void set_max_individuals(int);

    public:
        std::vector<Topology_ptr> &get_topologies();

        Topology_ptr get_best();

    public:
        void natural_selection();

    private:
        int max_individuals;
        std::vector<Topology_ptr> topologies;
        Topology_ptr best_topology;

    private:
        void do_selection();

        void evolve(std::vector<Topology_ptr> &,
                    std::vector<Topology_ptr> &);

        void mate(std::vector<Topology_ptr> &,
                  std::vector<Topology_ptr> &);

        void update_best(Topology_ptr const &);

        void duplicate_best();
    };

    using Species_ptr = std::unique_ptr<Species>;

} /* namespace NeuralNetwork */

#endif /* NEURALNETWORK_SPECIES_H_ */
