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

use std::{borrow::Cow, iter};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use either::Either;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct Vertex {
    pos: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    pub const fn new(x: f32, y: f32) -> Self {
        Vertex{pos: [x, y, 0.0], color: [0.5, 0.0, 0.5]}
    }
}


#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct GlobalUniforms {
    viewport: [f32; 2],
}

impl GlobalUniforms {
    fn new(sc_desc: &wgpu::SwapChainDescriptor) -> Self {
        Self {
            viewport: [sc_desc.width as f32, sc_desc.height as f32]
        }
    }

    fn update(&mut self, sc_desc: &wgpu::SwapChainDescriptor) {
        self.viewport = [sc_desc.width as f32, sc_desc.height as f32];
    }
}


struct Example {
    bundle: wgpu::RenderBundle,
    shader: Either<wgpu::ShaderModule, (wgpu::ShaderModule, wgpu::ShaderModule)>,
    pipeline_layout: wgpu::PipelineLayout,
    multisampled_framebuffer: wgpu::TextureView,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    sample_count: u32,
    rebuild_bundle: bool,
    sc_desc: wgpu::SwapChainDescriptor,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl Example {
    fn create_bundle(
        device: &wgpu::Device,
        sc_desc: &wgpu::SwapChainDescriptor,
        shader: &Either<wgpu::ShaderModule, (wgpu::ShaderModule, wgpu::ShaderModule)>,
        pipeline_layout: &wgpu::PipelineLayout,
        sample_count: u32,
        vertex_buffer: &wgpu::Buffer,
        vertex_count: u32,
        // uniform_buf: &wgpu::Buffer,
        // bind_group: &wgpu::BindGroup,
        uniform_buffer: &wgpu::Buffer,
        uniform_bind_group: &wgpu::BindGroup,
    ) -> wgpu::RenderBundle {
        log::info!("sample_count: {}", sample_count);
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: match shader {
                    Either::Left(shader) => shader,
                    Either::Right((shader, _)) => {
                        shader
                    },
                },
                entry_point: match shader {
                    Either::Left(_) => "vs_main",
                    Either::Right((_, _)) => "main",
                },
                buffers: &[
                    wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                        step_mode: wgpu::InputStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float3, 1 => Float3],
                    }
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: match shader {
                    Either::Left(shader) => shader,
                    Either::Right((_, shader)) => shader,
                },
                entry_point: match shader {
                    Either::Left(_) => "fs_main",
                    Either::Right((_, _)) => "main",
                },
                // targets: &[sc_desc.format.into()],
                targets: &[wgpu::ColorTargetState {
                    format: sc_desc.format,
                    color_blend: wgpu::BlendState {
                        src_factor: wgpu::BlendFactor::SrcAlpha,
                        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                        operation: wgpu::BlendOperation::Add,
                    },
                    alpha_blend: wgpu::BlendState {
                        src_factor: wgpu::BlendFactor::One,
                        dst_factor: wgpu::BlendFactor::One,
                        operation: wgpu::BlendOperation::Max,
                    },
                    write_mask: wgpu::ColorWrite::ALL,
                }]
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: sample_count,
                ..Default::default()
            },
        });
        let mut encoder = device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            label: None,
            color_formats: &[sc_desc.format],
            depth_stencil_format: None,
            sample_count,
        });
        encoder.set_pipeline(&pipeline);
        encoder.set_vertex_buffer(0, vertex_buffer.slice(..));
        encoder.set_bind_group(0, &uniform_bind_group, &[]);
        encoder.draw(0..vertex_count, 0..1);
        encoder.finish(&wgpu::RenderBundleDescriptor {
            label: Some("main"),
        })
    }

    fn create_multisampled_framebuffer(
        device: &wgpu::Device,
        sc_desc: &wgpu::SwapChainDescriptor,
        sample_count: u32,
    ) -> wgpu::TextureView {
        let multisampled_texture_extent = wgpu::Extent3d {
            width: sc_desc.width,
            height: sc_desc.height,
            depth: 1,
        };
        let multisampled_frame_descriptor = &wgpu::TextureDescriptor {
            size: multisampled_texture_extent,
            mip_level_count: 1,
            sample_count: sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: sc_desc.format,
            usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
            label: None,
        };

        device
            .create_texture(multisampled_frame_descriptor)
            .create_view(&wgpu::TextureViewDescriptor::default())
    }
}

impl framework::Example for Example {
    fn init(
        sc_desc: &wgpu::SwapChainDescriptor,
        adapter: &wgpu::Adapter,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
    ) -> Self {
        let sample_count = 1;
        let mut flags = wgpu::ShaderFlags::VALIDATION;
        match adapter.get_info().backend {
            wgpu::Backend::Metal | wgpu::Backend::Vulkan => {
                flags |= wgpu::ShaderFlags::EXPERIMENTAL_TRANSLATION
            }
            _ => (), //TODO
        }
        let vs_module = device.create_shader_module(&wgpu::include_spirv!("../shaders/shader.vert.spv"));
        let fs_module = device.create_shader_module(&wgpu::include_spirv!("../shaders/shader.frag.spv"));
        let global_uniforms: GlobalUniforms = GlobalUniforms::new(&sc_desc);
        let uniform_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[global_uniforms]),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            }
        );
        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("uniform_bind_group_layout"),
        });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                }
            ],
            label: Some("uniform_bind_group"),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let multisampled_framebuffer = Example::create_multisampled_framebuffer(
            device,
            sc_desc,
            sample_count
        );
        
        const VERTICES: &[Vertex] = &[
            Vertex::new(-1.0, 1.0),
            Vertex::new(-1.0, -1.0),
            Vertex::new(1.0, 1.0),

            Vertex::new(-1.0 * 0.6 + 0.25, 1.0 * 0.6),
            Vertex::new(-1.0 * 0.6 + 0.25, -1.0 * 0.6),
            Vertex::new(1.0 * 0.6 + 0.25, 1.0 * 0.6),
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(VERTICES),
            usage: wgpu::BufferUsage::VERTEX,
        });
        let vertex_count = VERTICES.len() as u32;
        let shader = Either::Right((vs_module, fs_module));

        let bundle = Example::create_bundle(
            device,
            &sc_desc,
            &shader,
            &pipeline_layout,
            sample_count,
            &vertex_buffer,
            vertex_count,
            &uniform_buffer,
            &uniform_bind_group,
        );

        Example {
            bundle,
            shader,
            pipeline_layout,
            multisampled_framebuffer,
            vertex_buffer,
            vertex_count,
            sample_count,
            rebuild_bundle: true,
            sc_desc: sc_desc.clone(),
            uniform_buffer,
            uniform_bind_group,
        }
    }

    fn update(&mut self, event: winit::event::WindowEvent) {
        match event {
            winit::event::WindowEvent::KeyboardInput { input, .. } => {
                if let winit::event::ElementState::Pressed = input.state {
                    match input.virtual_keycode {
                        // TODO: Switch back to full scans of possible options when we expose
                        //       supported sample counts to the user.
                        // Some(winit::event::VirtualKeyCode::Left) => {
                        //     if self.sample_count == 4 {
                        //         self.sample_count = 1;
                        //         self.rebuild_bundle = true;
                        //     }
                        // }
                        // Some(winit::event::VirtualKeyCode::Right) => {
                        //     if self.sample_count == 1 {
                        //         self.sample_count = 4;
                        //         self.rebuild_bundle = true;
                        //     }
                        // }
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }

    fn resize(
        &mut self,
        sc_desc: &wgpu::SwapChainDescriptor,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) {
        let global_uniforms: GlobalUniforms = GlobalUniforms {
            viewport: [sc_desc.width as f32, sc_desc.height as f32],
        };
        self.uniform_buffer = device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: Some("Uniform Buffer"),
                contents: bytemuck::cast_slice(&[global_uniforms]),
                usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
            }
        );
        let uniform_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            ],
            label: Some("uniform_bind_group_layout"),
        });
        self.uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                }
            ],
            label: Some("uniform_bind_group"),
        });
        self.sc_desc = sc_desc.clone();
        self.multisampled_framebuffer = Example::create_multisampled_framebuffer(
            device,
            sc_desc,
            self.sample_count
        );
    }

    fn render(
        &mut self,
        frame: &wgpu::SwapChainTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &framework::Spawner,
    ) {
        if self.rebuild_bundle {
            self.bundle = Example::create_bundle(
                device,
                &self.sc_desc,
                &self.shader,
                &self.pipeline_layout,
                self.sample_count,
                &self.vertex_buffer,
                self.vertex_count,
                &self.uniform_buffer,
                &self.uniform_bind_group,
            );
            self.multisampled_framebuffer = Example::create_multisampled_framebuffer(
                device,
                &self.sc_desc,
                self.sample_count
            );
            self.rebuild_bundle = false;
        }
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {label: None}
        );
        let ops = wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color{
                r: 0.5,
                g: 0.5,
                b: 0.5,
                a: 1.0,
            }),
            store: true,
        };
        let rpass_color_attachment = if self.sample_count == 1 {
            wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &frame.view,
                resolve_target: None,
                ops,
            }
        } else {
            wgpu::RenderPassColorAttachmentDescriptor {
                attachment: &self.multisampled_framebuffer,
                resolve_target: Some(&frame.view),
                ops,
            }
        };
        encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[rpass_color_attachment],
                depth_stencil_attachment: None,
            })
            .execute_bundles(iter::once(&self.bundle));

        queue.submit(iter::once(encoder.finish()));
    }
}

fn main() {
    framework::run::<Example>("msaa-line");
}
