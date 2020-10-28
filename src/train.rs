use crate::game::Game;
use crate::ffi::{Simulation, fit};
use std::ffi::c_void;

/// The train struct is used to train a Neural Network on a simulation with the NEAT algorithm
pub struct Train<T>
    where T: Game {
    c_sim: Option<Box<Simulation<T>>>,
    iterations_: i32,
    max_individuals_: i32,
    max_species_: i32,
    max_layers_: i32,
    max_per_layers_: i32,
    inputs_: Option<i32>,
    outputs_: Option<i32>,
}

impl<T> Train<T>
    where T: Game {
    /// Creates a Train<T: Game> instance
    ///
    /// Default values are:
    /// - iterations -> the number of generations to be run: 1000
    /// - max_individuals -> number of networks per generation: 100
    /// - max_layers -> maximum number of layers
    /// - max_per_layers -> maximum number of neurons per layer
    ///
    /// Mandatory fields (use setters):
    /// - inputs -> the number of neurons on the first layer
    /// - outputs -> the number of neurons on the last layer
    pub fn new() -> Train<T> {
        let iterations_: i32 = 1000;
        let max_individuals_: i32 = 100;
        let max_species_: i32 = 100;
        let inputs_ = None;
        let outputs_ = None;

        Train {
            c_sim: None,
            iterations_,
            max_individuals_,
            max_species_,
            max_layers_: 4,
            max_per_layers_: 20,
            inputs_,
            outputs_,
        }
    }

    /// Returns a smart pointer to the simulation
    pub fn get_simulation(&mut self) -> Box<T> {
        unsafe {
            Box::from_raw(std::mem::transmute(self.c_sim.as_ref().unwrap().get()))
        }
    }


    /// Sets the simulation we want to train on
    ///
    /// # Arguments
    ///
    /// `sim` - The simulation we want to train on, it has to implement the trait Game
    pub fn simulation(&mut self, sim: T) -> &mut Self {
        let c_sim = Simulation::new(sim);
        self.c_sim = Some(c_sim);
        self
    }

    /// Sets the number of iterations
    ///
    /// Iterations is the maximum number of generations to be run, optional and defaults to 1000
    ///
    /// # Arguments
    ///
    /// `it` - The number of generations to be run
    pub fn iterations(&mut self, it: i32) -> &mut Self {
        self.iterations_ = it;
        self
    }

    /// Sets the number of networks per generation
    ///
    /// This function is optional as the number of max individuals defaults to 100
    ///
    /// # Arguments
    ///
    /// `v` - The number of networks per generation
    pub fn max_individuals(&mut self, v: i32) -> &mut Self {
        self.max_individuals_ = v;
        self
    }

    /// Sets the number of maximum species per generation
    ///
    /// This function is optional as the number of max species defaults to 100
    ///
    /// # Arguments
    ///
    /// `v` - The number of maximum species per generation
    pub fn max_species(&mut self, v: i32) -> &mut Self {
        self.max_species_ = v;
        self
    }

    /// Sets the number of neurons on the first layer
    ///
    /// This function has to be called in order to start training
    ///
    /// # Arguments
    ///
    /// `v` - The number of neurons on the first layer
    pub fn inputs(&mut self, v: i32) -> &mut Self {
        self.inputs_ = Some(v);
        self
    }

    /// Sets the number of neurons on the last layer
    ///
    /// This function has to be called in order to start training
    ///
    /// # Arguments
    ///
    /// `v` - The number of neurons on the last layer
    pub fn outputs(&mut self, v: i32) -> &mut Self {
        self.outputs_ = Some(v);
        self
    }

    /// Sets the maximum number of layers for the networks
    ///
    /// This function is optional as the max number of layers defaults to 10
    ///
    /// # Arguments
    ///
    /// `v` - The maximum number of layers
    pub fn max_layers(&mut self, v: i32) -> &mut Self {
        self.max_layers_ = v;
        self
    }

    /// Sets the maximum number of neurons per layers for the networks
    ///
    /// This function is optional as the max neurons per layer defaults to 50
    ///
    /// # Arguments
    ///
    /// `v` - The maximum number of neurons per layers
    pub fn max_per_layers(&mut self, v: i32) -> &mut Self {
        self.max_per_layers_ = v;
        self
    }

    /// Starts the training with the given parameters
    ///
    /// # Example
    ///
    /// ```
    /// let sim = Simulation::new(); // Has to implement trait Game
    /// let mut runner: Train<TradingSimulation> = Train::new();
    /// runner.simulation(sim).inputs(5).outputs(1);
    /// runner.start();
    /// ```
    pub fn start(&mut self) {
        let inputs = match self.inputs_ {
            Some(v) => v,
            None => { panic!("Didn't provide a number of inputs") }
        };

        let outputs = match self.outputs_ {
            Some(v) => v,
            None => { panic!("Didn't provide a number of inputs") }
        };

        unsafe {
            let c_sim = self.c_sim.take().unwrap();
            let ptr = Box::into_raw(c_sim) as *mut c_void;
            fit(ptr,
                self.iterations_,
                self.max_individuals_,
                self.max_species_,
                self.max_layers_,
                self.max_per_layers_,
                inputs, outputs);
            let ptr: *mut Simulation<T> = std::mem::transmute(ptr);
            self.c_sim = Some(Box::from_raw(ptr));
        }
    }
}