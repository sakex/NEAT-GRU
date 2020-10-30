use crate::ffi::{compute_network, reset_network_state, network_from_string};
use std::os::raw::{c_double, c_char};
use std::ffi::{c_void, CString};
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

/// ## FFI wrapper for the NeuralNetwork C++ class
///
/// ### /!\ It contains a c_void* pointer. Accessing it from different threads is unsafe
///
/// # Example
///
/// ```
/// let serialized: String = fs::read_to_string("topology_test.json")
///         .expect("Something went wrong reading the file");
/// let mut net = NeuralNetwork::from_string(&serialized);
/// let input_1: Vec<f64> = vec![0.5, 0.5];
/// let input_2: Vec<f64> = vec![-0.5, -0.5];
///
/// let output_1 = net.compute(&input_1, 1);
/// let output_2 = net.compute(&input_2, 1);
/// let output_3 = net.compute(&input_1, 1);
///
/// // 1 and 2 should by definition be different
/// assert_ne!(output_1, output_2);
/// assert_ne!(output_1, output_3);
///
/// //Because of GRU gates, giving the same input twice won't yield the same output
/// assert_ne!(output_2, output_3);
///
/// // Reset
/// net.reset_state();
/// let output_4 = net.compute(&input_1, 1);
///
/// // After resetting, giving the same input sequence should yield the same results
/// assert_eq!(output_1, output_4);
/// ```
#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
#[derive(Clone)]
pub struct NeuralNetwork {
    ptr: *mut c_void
}

unsafe impl Send for NeuralNetwork {}

unsafe impl Sync for NeuralNetwork {}

#[cfg_attr(target_arch = "wasm32", wasm_bindgen)]
impl NeuralNetwork {
    /// Creates a new NeuralNetwork wrapper from a c void pointer
    ///
    /// # Arguments
    /// `ptr` - The C pointer to the C++ NeuralNetwork
    #[inline]
    pub fn new(ptr: *mut c_void) -> NeuralNetwork {
        NeuralNetwork { ptr }
    }

    /// Computes the Neural Network and returns the result
    ///
    /// # Arguments
    ///
    /// `input` - The input values on the first layer
    /// `output_length` - The number of outputs expected
    #[inline]
    pub fn compute(&self, input: &[f64], output_length: usize) -> Vec<f64> {
        unsafe {
            let ptr: *mut c_double = compute_network(self.ptr, input.as_ptr() as *mut f64);
            let vec: Vec<f64> = Vec::from_raw_parts(ptr, output_length, output_length);
            vec
        }
    }

    /// Resets the hidden state of a neural network
    ///
    /// Use it to make the network `forget` previous dataset during a generation
    #[inline]
    pub fn reset_state(&mut self) {
        unsafe {
            reset_network_state(self.ptr);
        }
    }

    /// Parses a neural network to string
    ///
    /// # Arguments
    ///
    /// `serialized` - A string represented a JSON serialisation of a neural network's topology
    ///
    /// # Returns
    ///
    /// A neural network instance /!\ The pointer is allocated on the heap
    ///
    /// # Example
    ///
    /// ```
    /// let serialized: String = fs::read_to_string("topology_test.json").expect("Something went wrong reading the file");
    /// let mut net = NeuralNetwork::from_string(&serialized);
    ///  net.compute(&vec![0.5, 0.5], 1);
    /// ```
    #[allow(dead_code)]
    #[inline]
    pub fn from_string(serialized: &str) -> NeuralNetwork {
        unsafe {
            let c_string = CString::new(serialized).unwrap();
            let char_ptr = c_string.as_ptr() as *const c_char;
            let network_ptr = network_from_string(char_ptr);
            NeuralNetwork::new(network_ptr)
        }
    }

    #[allow(dead_code)]
    #[inline]
    pub fn get_ptr(&self) -> *mut c_void {
        self.ptr
    }
}