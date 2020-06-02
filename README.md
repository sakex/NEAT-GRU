# NEAT-GRU

## Install Nlohmann/json
    sudo apt-get install -y nlohmann-json-dev
    
## Generate doc with Doxygen
    doxygen doc.conf
    
## Build
### Using a CMake subdirectory
 Add to your projects CMakeLists.txt:
    
    add_subdirectory(NEAT-GRU NEAT)
    target_link_libraries(trading NEAT_GRU)
    
### Shared library
    mkdir Release
    cd Release
    cmake -DCMAKE_BUILD_TYPE=Release -D__MULTITHREADED__=1 .. .
    sudo make install
    
## How to use

- Implement the abstract class `Game::Game`:
    
      #include <neat/Game/Game.h>
      class Simulation: public Game {
        std::vector<double> do_run_generation() override {...}
        void do_reset_players(NN * nets, size_t count) override {...};
        void do_post_training(Topology_ptr topology) override {...};
      }
      
- Create a `Train` instance and give it a pointer to a `Simulation` instance:

      auto * sim = new Simulation();
      const int iterations = 1000;
      const int max_individuals = 300; // Individuals per generation
      const int inputs = 10; // Input neurons
      const int outputs = 5; // Output neurons
      Train train(sim, max_individuals, inputs, outputs);
      train.start(); // Runs the training, will output the resulting network to "topologies.json"