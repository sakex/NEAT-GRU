use std::ffi::{c_void, CStr};
use std::fmt;
use crate::ffi::{topology_to_string, network_from_topology, topology_delta_compatibility};
use crate::neural_network::NeuralNetwork;

/// Wrapper to the C++ class Topology
#[derive(Clone)]
pub struct Topology {
    ptr: *const c_void
}

impl Topology {
    pub fn new(ptr: *const c_void) -> Topology {
        Topology {
            ptr
        }
    }

    pub fn to_string(&self) -> String {
        unsafe {
            let c_buf = topology_to_string(self.ptr);
            CStr::from_ptr(c_buf).to_str().unwrap().to_string()
        }
    }

    pub fn delta_compatibility(topology1: &Topology, topology2: &Topology) -> f64 {
        unsafe {
            topology_delta_compatibility(topology1.ptr, topology2.ptr)
        }
    }
}

impl fmt::Display for Topology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

impl Into<NeuralNetwork> for &Topology {
    fn into(self) -> NeuralNetwork {
        let topology_ptr: *const c_void = self.ptr;
        let network_ptr: *mut c_void = unsafe { network_from_topology(topology_ptr) };
        NeuralNetwork::new(network_ptr)
    }
}