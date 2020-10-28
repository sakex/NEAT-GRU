use std::os::raw::c_ulong;

#[repr(C)]
#[derive(Clone)]
pub struct Dim {
    pub x: c_ulong,
    pub y: c_ulong,
    pub z: c_ulong
}

impl Dim {
    pub fn new(x: c_ulong, y: c_ulong, z: c_ulong) -> Dim {
        Dim {
            x, y, z
        }
    }
}