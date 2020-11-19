use std::ffi::{c_void, CStr, CString};
use std::fmt;
use crate::ffi::{topology_to_string, network_from_topology, topology_delta_compatibility, topologies_equal, topology_from_string};
use crate::neural_network::NeuralNetwork;
use std::os::raw::c_char;

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

    pub fn from_string(serialized: &str) -> Topology {
        unsafe {
            let c_string = CString::new(serialized).unwrap();
            let char_ptr = c_string.as_ptr() as *const c_char;
            let topology_ptr = topology_from_string(char_ptr);
            Topology::new(topology_ptr)
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

impl PartialEq for Topology {
    fn eq(&self, other: &Topology) -> bool {
        unsafe { topologies_equal(self.ptr, other.ptr) }
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