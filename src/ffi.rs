use std::os::raw::{c_double, c_ulong, c_int, c_char};
use std::ffi::c_void;
use crate::game::Game;
use std::mem;
use crate::neural_network::NeuralNetwork;
use crate::topology::Topology;
#[cfg(feature = "gpu")]
use crate::gpu_compute_instance::ComputeInstance;
#[cfg(feature = "gpu")]
use crate::dim::Dim;


#[repr(C)]
#[derive(Clone)]
pub struct Simulation<T>
    where T: Game {
    pub run_generation: unsafe extern fn(*mut T) -> *mut c_double,
    pub reset_players: unsafe extern fn(*mut T, *mut c_void, ptr_size: c_ulong, c_ulong),
    pub post_training: unsafe extern fn(*mut T, *const c_void, ptr_size: c_ulong, size: c_ulong),
    pub game: *mut c_void,
}

impl<T> Simulation<T>
    where T: Game {
    pub fn new(game: T) -> Box<Simulation<T>> {
        unsafe extern "C" fn run_generation<T>(game_ptr: *mut T) -> *mut c_double
            where T: Game {
            let game_ref = &mut *game_ptr;
            let mut vec = game_ref.run_generation();
            vec.shrink_to_fit();
            let c_ptr = vec.as_mut_ptr();
            mem::forget(vec);
            c_ptr
        }

        unsafe extern "C" fn reset_players<T>(game_ptr: *mut T, wrappers: *mut c_void, ptr_size: c_ulong, size: c_ulong)
            where T: Game {
            let mut nets: Vec<NeuralNetwork> = Vec::new();
            nets.reserve(size as usize);
            let ptr_size: isize = ptr_size as isize;
            for i in 0..(size as isize) {
                let net = wrappers.offset(i * ptr_size);
                nets.push(NeuralNetwork::new(net));
            };
            let game_ref = &mut *game_ptr;
            game_ref.reset_players(nets.as_slice());
        }

        unsafe extern "C" fn post_training<T>(game_ptr: *mut T, history: *const c_void, ptr_size: c_ulong, size: c_ulong)
            where T: Game {
            let mut nets: Vec<Topology> = Vec::new();
            nets.reserve(size as usize);
            let ptr_size: isize = ptr_size as isize;
            for i in 0..(size as isize) {
                let net = history.offset(i * ptr_size);
                nets.push(Topology::new(net));
            };
            let game_ref = &mut *game_ptr;
            game_ref.post_training(nets.as_slice());
        }

        let ptr = Box::into_raw(Box::new(game)) as *mut c_void;

        Box::new(Simulation {
            run_generation,
            reset_players,
            post_training,
            game: ptr,
        })
    }

    pub fn get(&self) -> *mut c_void {
        self.game
    }
}

#[cfg(feature = "gpu")]
#[link(name = "GPU", kind = "static")]
extern "C" {
    pub fn compute_gpu_instance(instance: *mut ComputeInstance, output_size: c_ulong);
    pub fn update_dataset_gpu_instance(instance: *mut ComputeInstance, data: *const c_double);
    pub fn set_networks_gpu_instance(instance: *mut ComputeInstance,nets: *mut c_void, count: c_ulong);
}

#[link(name = "neat", kind = "static")]
extern "C" {
    pub fn reset_network_state(net: *mut c_void);
    pub fn compute_network(net: *mut c_void, input: *mut c_double) -> *mut c_double;
    pub fn network_from_string(json: *const c_char) -> *mut c_void;
    pub fn network_from_topology(topology: *const c_void) -> *mut c_void;
    pub fn topology_to_string(topology: *const c_void) -> *const c_char;
    pub fn fit(sim: *mut c_void, iterations: c_int, max_individuals: c_int, max_species: c_int,
               max_layers: c_int, max_per_layers: c_int, inputs: c_int, outputs: c_int);
    pub fn topology_delta_compatibility(topology1: *const c_void, topology2: *const c_void) -> c_double;
}