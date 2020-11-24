# NEAT-GRU ![Tests](https://github.com/sakex/NEAT-GRU/workflows/Rust/badge.svg?branch=master)

# As a C++ library

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
      
      
 # As a Rust Library
 
 In `Cargo.toml`:
 
     [dependencies]
     neat-gru = "0.1.10"
     
 Create a struct that implements the `Game` trait
 
     use neat_gru::game::Game;
     use neat_gru::neural_network::NeuralNetwork;
     
     struct Player {
         net: NeuralNetwork,
         score: f64
     }
     
     impl Player {
         pub fn new(net: NeuralNetwork) -> Player {
             Player {
                 net = net,
                 score: 0f64
             }
         }
     }

     struct Simulation {
         players: Vec<Player>
     }
     
     impl Simulation {
         pub fn new() -> Simulation {
             Simulation {
                 players: Vec::new()   
             }
         }
     }
     
     impl Game for TradingSimulation {
        // Loss function
        fn run_generation(&mut self) -> Vec<f64> {
            self.players.iter().map(... Your logic here ).collect()
        }
     
        // Reset networks
        fn reset_players(&mut self, nets: &[NeuralNetwork]) {
            self.players.clear();
            self.players.reserve(nets.len());
            self.players = nets
                .into_iter()
                .map(|net| Player::new(net.clone()))
                .collect();
         }
         
        // Called at the end of training
        fn post_training(&mut self, history: &[Topology]) {
            // Iter on best topologies and upload the best one
        }

    }

 Launch a training
 
         let sim = Simulation::new();
         
         let mut runner = Train::new();
         runner
            .simulation(sim)
            .inputs(input_count)
            .outputs(output_count as i32)
            .iterations(nb_generations as i32)
            .max_layers((hidden_layers + 2) as i32)
            .max_per_layers(hidden_layers as i32)
            .max_species(max_species as i32)
            .max_individuals(max_individuals as i32)
            .start();

