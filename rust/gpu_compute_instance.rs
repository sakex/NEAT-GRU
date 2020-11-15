use crate::ffi::{compute_gpu_instance, update_dataset_gpu_instance, set_networks_gpu_instance};
use crate::dim::Dim;
use std::os::raw::{c_double, c_ulong};
use futures_util::core_reexport::ffi::c_void;
use std::ptr;

#[repr(C)]
pub struct ComputeInstance {
    dim: Dim,
    h_output: *mut c_double,
    d_output: *mut c_double,
    networks_count: c_ulong,
    networks: *mut c_void,
}

impl ComputeInstance {
    pub fn new(dim: Dim) -> ComputeInstance {
        ComputeInstance {
            dim,
            h_output: ptr::null_mut(),
            d_output: ptr::null_mut(),
            networks_count: 0,
            networks: ptr::null_mut(),
        }
    }

    #[inline]
    unsafe fn structure_output(ptr: *mut c_double, player_count: usize, folds: usize, sub_periods: usize, sub_period_size: usize, output_count: usize) -> Vec<Vec<Vec<Vec<Vec<f64>>>>> {
        let mut output= Vec::new();
        output.reserve(player_count * folds * sub_periods * output_count);
        let player_dataset_size = folds * sub_periods * output_count * sub_period_size;
        let fold_size = sub_periods * output_count * sub_period_size;
        let sub_period_size = output_count * sub_period_size;
        let dword = std::mem::size_of::<c_double>();
        for player_index in 0..player_count {
            let mut player_vec = Vec::new();
            for fold_index in 0..folds {
                let mut fold_vec = Vec::new();
                for sub_period_index in 0..sub_periods {
                    let mut sub_period_vec = Vec::new();
                    for output_index in 0..sub_period_size {
                        let from = player_index * player_dataset_size + fold_index * fold_size + sub_period_index * sub_period_size + output_index * output_count;
                        let outputs = Vec::from_raw_parts(ptr.offset((from * dword) as isize), output_count, output_count);
                        sub_period_vec.push(outputs);
                    }
                    fold_vec.push(sub_period_vec);
                }
                player_vec.push(fold_vec);
            }
            output.push(player_vec);
        }
        output
    }

    pub fn compute(&mut self,
                   output_size: c_ulong,
                   player_count: usize,
                   folds: usize,
                   sub_periods: usize) -> Vec<Vec<Vec<Vec<Vec<f64>>>>> {
        unsafe {
            let self_ptr: *mut ComputeInstance = self;
            compute_gpu_instance(self_ptr, output_size);
            ComputeInstance::structure_output(self.h_output, player_count, folds, sub_periods, self.dim.y as usize, output_size as usize)
        }
    }

    pub fn update_dataset(&mut self, data: &[c_double]) {
        unsafe {
            let self_ptr: *mut ComputeInstance = self;
            let data_ptr = data.as_ptr();
            update_dataset_gpu_instance(self_ptr, data_ptr);
        };
    }

    pub fn set_networks(&mut self, networks: *mut c_void, count: u64) {
        unsafe {
            let self_ptr: *mut ComputeInstance = self;
            set_networks_gpu_instance(self_ptr, networks, count);
        };
    }
}