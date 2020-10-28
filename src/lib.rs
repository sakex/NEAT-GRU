#[cfg(feature = "gpu")]
pub mod dim;
pub mod topology;
mod ffi;
pub mod game;
pub mod neural_network;
pub mod train;
#[cfg(feature = "gpu")]
pub mod gpu_compute_instance;
#[cfg(test)]
mod tests;