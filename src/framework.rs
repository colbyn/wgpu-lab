#![allow(unused)]

use std::future::Future;
use std::time::{Duration, Instant};
use winit::{
    event::{self, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};
use std::{borrow::Cow, iter};
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;
use either::Either;


///////////////////////////////////////////////////////////////////////////////
// BASICS
///////////////////////////////////////////////////////////////////////////////

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

pub enum ShaderStage {
    Vertex,
    Fragment,
    Compute,
}

pub struct Shader {
    module: wgpu::ShaderModule,
    entrypoint: String,
}

pub struct Shaders {
    vertex: Shader,
    fragment: Shader,
}

impl Shaders {
    pub fn to_vertex_state<'a>(&'a self) -> wgpu::VertexState<'a> {
        wgpu::VertexState {
            module: &self.vertex.module,
            entry_point: &self.vertex.entrypoint,
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &[
                        wgpu::VertexAttribute {
                            offset: 0,
                            shader_location: 0,
                            format: wgpu::VertexFormat::Float3,
                        },
                        wgpu::VertexAttribute {
                            offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                            shader_location: 1,
                            format: wgpu::VertexFormat::Float3,
                        }
                    ],
                }
            ],
        }
    }
    pub fn to_fragment_state<'a>(&'a self, targets: &'a [wgpu::ColorTargetState]) -> wgpu::FragmentState<'a> {
        let color_blend = wgpu::BlendState {
            src_factor: wgpu::BlendFactor::SrcAlpha,
            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
            operation: wgpu::BlendOperation::Add,
        };
        let alpha_blend = wgpu::BlendState {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Max,
        };
        wgpu::FragmentState {
            module: &self.fragment.module,
            entry_point: &self.fragment.entrypoint,
            targets
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// GEOMETRY
///////////////////////////////////////////////////////////////////////////////

pub struct Geometry {
    vertex_buffer: wgpu::Buffer,
    vertices: Vec<Vertex>,
}

impl Geometry {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }
    pub fn new(device: &wgpu::Device) -> Self {
        let vertices: Vec<Vertex> = vec![
            Vertex::new(-1.0, 1.0),
            Vertex::new(-1.0, -1.0),
            Vertex::new(1.0, 1.0),

            Vertex::new(-1.0 * 0.6 + 0.25, 1.0 * 0.6),
            Vertex::new(-1.0 * 0.6 + 0.25, -1.0 * 0.6),
            Vertex::new(1.0 * 0.6 + 0.25, 1.0 * 0.6),
        ];

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsage::VERTEX,
        });

        Geometry {
            vertex_buffer,
            vertices,
        }
    }

}



///////////////////////////////////////////////////////////////////////////////
// APPLICATION
///////////////////////////////////////////////////////////////////////////////

pub struct Application {
    shaders: Shaders,
    pipeline_layout: wgpu::PipelineLayout,
    multisampled_framebuffer: wgpu::TextureView,
    geometry: Geometry,
    sample_count: u32,
    rebuild_bundle: bool,
    sc_desc: wgpu::SwapChainDescriptor,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
}

impl Application {
    fn optional_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_features() -> wgpu::Features {
        wgpu::Features::empty()
    }
    fn required_limits() -> wgpu::Limits {
        wgpu::Limits::default()
    }

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

        let multisampled_framebuffer = Application::create_multisampled_framebuffer(
            device,
            sc_desc,
            sample_count
        );

        let geometry = Geometry::new(&device);

        let shaders = Shaders{
            vertex: Shader{
                module: vs_module,
                entrypoint: String::from("main"),
            },
            fragment: Shader{
                module: fs_module,
                entrypoint: String::from("main"),
            },
        };

        let app = Application {
            shaders: shaders,
            pipeline_layout,
            multisampled_framebuffer,
            geometry,
            sample_count,
            rebuild_bundle: true,
            sc_desc: sc_desc.clone(),
            uniform_buffer,
            uniform_bind_group,
        };
        app
    }

    fn render(
        &mut self,
        bundle: &mut wgpu::RenderBundle,
        frame: &wgpu::SwapChainTexture,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _spawner: &Spawner,
    ) {
        ///////////////////////////////////////////////////////////////////////
        // INIT
        ///////////////////////////////////////////////////////////////////////
        if self.rebuild_bundle {
            *bundle = self.create_bundle(device);
            self.multisampled_framebuffer = Application::create_multisampled_framebuffer(
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

        ///////////////////////////////////////////////////////////////////////
        // FINALIZE
        ///////////////////////////////////////////////////////////////////////
        let bundle: &wgpu::RenderBundle = &bundle;
        encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
            color_attachments: &[rpass_color_attachment],
            depth_stencil_attachment: None,
        }).execute_bundles(iter::once(bundle));
        queue.submit(iter::once(encoder.finish()));
    }
    fn create_bundle(&self, device: &wgpu::Device) -> wgpu::RenderBundle {
        let color_target_state = &[wgpu::ColorTargetState {
            format: self.sc_desc.format,
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
        }];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&self.pipeline_layout),
            vertex: self.shaders.to_vertex_state(),
            fragment: Some(self.shaders.to_fragment_state(color_target_state)),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: self.sample_count,
                ..Default::default()
            },
        });
        let mut encoder = device.create_render_bundle_encoder(&wgpu::RenderBundleEncoderDescriptor {
            label: None,
            color_formats: &[self.sc_desc.format],
            depth_stencil_format: None,
            sample_count: self.sample_count,
        });
        encoder.set_pipeline(&pipeline);
        encoder.set_vertex_buffer(0, self.geometry.vertex_buffer.slice(..));
        encoder.set_bind_group(0, &self.uniform_bind_group, &[]);
        encoder.draw(0 .. self.geometry.vertex_count() as u32, 0..1);
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

///////////////////////////////////////////////////////////////////////////////
// SETUP
///////////////////////////////////////////////////////////////////////////////

struct Setup {
    window: winit::window::Window,
    event_loop: EventLoop<()>,
    instance: wgpu::Instance,
    size: winit::dpi::PhysicalSize<u32>,
    surface: wgpu::Surface,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}

async fn setup(title: &str) -> Setup {
    let event_loop = EventLoop::new();
    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title(title);
    let window = builder.build(&event_loop).unwrap();
    let backend = wgpu::BackendBit::METAL;
    let power_preference = wgpu::PowerPreference::LowPower;
    let instance = wgpu::Instance::new(backend);
    let (size, surface) = unsafe {
        let size = window.inner_size();
        let surface = instance.create_surface(&window);
        (size, surface)
    };
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference,
            compatible_surface: Some(&surface),
        })
        .await
        .expect("No suitable GPU adapters found on the system!");

    let adapter_info = adapter.get_info();
    println!("Using {} ({:?})", adapter_info.name, adapter_info.backend);

    let optional_features = Application::optional_features();
    let required_features = Application::required_features();
    let adapter_features = adapter.features();
    assert!(
        adapter_features.contains(required_features),
        "Adapter does not support required features for this app: {:?}",
        required_features - adapter_features
    );

    let needed_limits = Application::required_limits();

    let trace_dir = std::env::var("WGPU_TRACE");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: (optional_features & adapter_features) | required_features,
                limits: needed_limits,
            },
            trace_dir.ok().as_ref().map(std::path::Path::new),
        )
        .await
        .expect("Unable to find a suitable GPU adapter!");

    Setup {
        window,
        event_loop,
        instance,
        size,
        surface,
        adapter,
        device,
        queue,
    }
}

///////////////////////////////////////////////////////////////////////////////
// INGRESS - START
///////////////////////////////////////////////////////////////////////////////

fn start(setup: Setup) {
    let Setup {window, event_loop, instance, size, surface, adapter, device, queue} = setup;
    let spawner = Spawner::new();
    let mut sc_desc = wgpu::SwapChainDescriptor {
        usage: wgpu::TextureUsage::RENDER_ATTACHMENT,
        format: adapter.get_swap_chain_preferred_format(&surface),
        width: size.width,
        height: size.height,
        present_mode: wgpu::PresentMode::Mailbox,
    };
    let mut swap_chain = device.create_swap_chain(&surface, &sc_desc);
    let mut app = Application::init(&sc_desc, &adapter, &device, &queue);
    let mut bundle = app.create_bundle(&device);
    let mut last_update_inst = Instant::now();

    ///////////////////////////////////////////////////////////////////////////
    // EVENT LOOP
    ///////////////////////////////////////////////////////////////////////////
    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter); // force ownership by the closure
        *control_flow = {
            if cfg!(feature = "metal-auto-capture") {
                ControlFlow::Exit
            } else {
                ControlFlow::Poll
            }
        };
        ///////////////////////////////////////////////////////////////////////
        // EVENT LOOP HELPERS
        ///////////////////////////////////////////////////////////////////////
        let is_exit_window_event = |event: &WindowEvent| -> bool {
            match event {
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                } => {
                    true
                }
                WindowEvent::CloseRequested => {
                    true
                }
                _ => {
                    false
                }
            }
        };
        ///////////////////////////////////////////////////////////////////////
        // EVENTS
        ///////////////////////////////////////////////////////////////////////
        match event {
            ///////////////////////////////////////////////////////////////////
            // PRE-DRAW - REDRAW PROCESSING IS ABOUT TO BEGIN
            ///////////////////////////////////////////////////////////////////
            event::Event::MainEventsCleared => {
                // Clamp to some max framerate to avoid busy-looping too much
                // (we might be in wgpu::PresentMode::Mailbox, thus discarding superfluous frames)
                //
                // winit has window.current_monitor().video_modes() but that is a list of all full screen video modes.
                // So without extra dependencies it's a bit tricky to get the max refresh rate we can run the window on.
                // Therefore we just go with 60fps - sorry 120hz+ folks!
                let target_frametime = Duration::from_secs_f64(1.0 / 60.0);
                let time_since_last_frame = last_update_inst.elapsed();
                if time_since_last_frame >= target_frametime {
                    window.request_redraw();
                    last_update_inst = Instant::now();
                } else {
                    *control_flow = ControlFlow::WaitUntil(
                        Instant::now() + target_frametime - time_since_last_frame,
                    );
                }
                spawner.run_until_stalled();
            }
            ///////////////////////////////////////////////////////////////////
            // RESIZE WINDOW EVENT
            ///////////////////////////////////////////////////////////////////
            event::Event::WindowEvent {event: WindowEvent::Resized(size), ..} => {
                sc_desc.width = if size.width == 0 { 1 } else { size.width };
                sc_desc.height = if size.height == 0 { 1 } else { size.height };
                let frame = match swap_chain.get_current_frame() {
                    Ok(frame) => frame,
                    Err(_) => {
                        swap_chain = device.create_swap_chain(&surface, &sc_desc);
                        swap_chain
                            .get_current_frame()
                            .expect("Failed to acquire next swap chain texture!")
                    }
                };
                app.sc_desc = sc_desc.clone();
                // app.render(&frame.output, &device, &queue, &spawner);
                app.render(&mut bundle, &frame.output, &device, &queue, &spawner);
            }
            ///////////////////////////////////////////////////////////////////
            // GENERAL WINDOW EVENT
            ///////////////////////////////////////////////////////////////////
            event::Event::WindowEvent { event, .. } => match event {
                e if is_exit_window_event(&e) => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            ///////////////////////////////////////////////////////////////////
            // REDRAW
            ///////////////////////////////////////////////////////////////////
            event::Event::RedrawRequested(_) => {
                let frame = match swap_chain.get_current_frame() {
                    Ok(frame) => frame,
                    Err(_) => {
                        swap_chain = device.create_swap_chain(&surface, &sc_desc);
                        swap_chain
                            .get_current_frame()
                            .expect("Failed to acquire next swap chain texture!")
                    }
                };
                app.render(&mut bundle, &frame.output, &device, &queue, &spawner);
            }
            ///////////////////////////////////////////////////////////////////
            // NOTHING TO DO
            ///////////////////////////////////////////////////////////////////
            _ => {}
        }
    });
}


///////////////////////////////////////////////////////////////////////////////
// ASYNC RUNTIME
///////////////////////////////////////////////////////////////////////////////

pub struct Spawner<'a> {
    executor: async_executor::LocalExecutor<'a>,
}


impl<'a> Spawner<'a> {
    fn new() -> Self {
        Self {
            executor: async_executor::LocalExecutor::new(),
        }
    }

    #[allow(dead_code)]
    pub fn spawn_local(&self, future: impl Future<Output = ()> + 'a) {
        self.executor.spawn(future).detach();
    }

    fn run_until_stalled(&self) {
        while self.executor.try_tick() {}
    }
}

///////////////////////////////////////////////////////////////////////////////
// ENTRYPOINT
///////////////////////////////////////////////////////////////////////////////

pub fn run(title: &str) {
    start(pollster::block_on(setup(title)));
}
