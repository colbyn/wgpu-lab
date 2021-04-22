//! The parts of this example enabling MSAA are:
//! *    The render pipeline is created with a sample_count > 1.
//! *    A new texture with a sample_count > 1 is created and set as the color_attachment instead of the swapchain.
//! *    The swapchain is now specified as a resolve_target.
//!
//! The parts of this example enabling LineList are:
//! *   Set the primitive_topology to PrimitiveTopology::LineList.
//! *   Vertices and Indices describe the two points that make up a line.
#![allow(unused)]
mod framework;
mod render;
mod utils;

use std::{borrow::Cow, iter};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use either::Either;



fn main() {
    framework::run("Colbyn's App");
}
