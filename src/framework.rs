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

pub fn cast_slice<T>(data: &[T]) -> &[u8] {
    use std::{mem::size_of, slice::from_raw_parts};
    unsafe { from_raw_parts(data.as_ptr() as *const u8, data.len() * size_of::<T>()) }
}

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

pub struct Application {
    bundle: wgpu::RenderBundle,
    shaders: Shaders,
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

        let bundle = Application::create_bundle(
            device,
            &sc_desc,
            &shaders,
            &pipeline_layout,
            sample_count,
            &vertex_buffer,
            vertex_count,
            &uniform_buffer,
            &uniform_bind_group,
        );

        Application {
            bundle,
            shaders: shaders,
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
        self.multisampled_framebuffer = Application::create_multisampled_framebuffer(
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
        _spawner: &Spawner,
    ) {
        if self.rebuild_bundle {
            self.bundle = Application::create_bundle(
                device,
                &self.sc_desc,
                &self.shaders,
                &self.pipeline_layout,
                self.sample_count,
                &self.vertex_buffer,
                self.vertex_count,
                &self.uniform_buffer,
                &self.uniform_bind_group,
            );
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
        encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[rpass_color_attachment],
                depth_stencil_attachment: None,
            })
            .execute_bundles(iter::once(&self.bundle));

        queue.submit(iter::once(encoder.finish()));
    }
    fn create_bundle(
        device: &wgpu::Device,
        sc_desc: &wgpu::SwapChainDescriptor,
        shaders: &Shaders,
        pipeline_layout: &wgpu::PipelineLayout,
        sample_count: u32,
        vertex_buffer: &wgpu::Buffer,
        vertex_count: u32,
        uniform_buffer: &wgpu::Buffer,
        uniform_bind_group: &wgpu::BindGroup,
    ) -> wgpu::RenderBundle {
        let color_target_state = &[wgpu::ColorTargetState {
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
        }];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: shaders.to_vertex_state(),
            fragment: Some(shaders.to_fragment_state(color_target_state)),
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
    let chrome_tracing_dir = std::env::var("WGPU_CHROME_TRACE");
    wgpu_subscriber::initialize_default_subscriber(
        chrome_tracing_dir.as_ref().map(std::path::Path::new).ok(),
    );

    let event_loop = EventLoop::new();
    let mut builder = winit::window::WindowBuilder::new();
    builder = builder.with_title(title);
    #[cfg(windows_OFF)] // TODO
    {
        use winit::platform::windows::WindowBuilderExtWindows;
        builder = builder.with_no_redirection_bitmap(true);
    }
    let window = builder.build(&event_loop).unwrap();
    
    let backend = if let Ok(backend) = std::env::var("WGPU_BACKEND") {
        match backend.to_lowercase().as_str() {
            "vulkan" => wgpu::BackendBit::VULKAN,
            "metal" => wgpu::BackendBit::METAL,
            "dx12" => wgpu::BackendBit::DX12,
            "dx11" => wgpu::BackendBit::DX11,
            "gl" => wgpu::BackendBit::GL,
            "webgpu" => wgpu::BackendBit::BROWSER_WEBGPU,
            other => panic!("Unknown backend: {}", other),
        }
    } else {
        wgpu::BackendBit::PRIMARY
    };
    // TODO
    let backend = wgpu::BackendBit::METAL;
    //  ^^^^^^
    let power_preference = if let Ok(power_preference) = std::env::var("WGPU_POWER_PREF") {
        match power_preference.to_lowercase().as_str() {
            "low" => wgpu::PowerPreference::LowPower,
            "high" => wgpu::PowerPreference::HighPerformance,
            other => panic!("Unknown power preference: {}", other),
        }
    } else {
        wgpu::PowerPreference::default()
    };
    // TODO
    let power_preference = wgpu::PowerPreference::LowPower;
    //  ^^^^^^^^^^^^^^^^
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
    let mut last_update_inst = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        let _ = (&instance, &adapter); // force ownership by the closure
        *control_flow = if cfg!(feature = "metal-auto-capture") {
            ControlFlow::Exit
        } else {
            ControlFlow::Poll
        };
        match event {
            event::Event::MainEventsCleared => {
                {
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
            }
            event::Event::WindowEvent {event: WindowEvent::Resized(size), ..} => {
                log::info!("Resizing to {:?}", size);
                sc_desc.width = if size.width == 0 { 1 } else { size.width };
                sc_desc.height = if size.height == 0 { 1 } else { size.height };
                app.resize(&sc_desc, &device, &queue);
                swap_chain = device.create_swap_chain(&surface, &sc_desc);
            }
            event::Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput {
                    input:
                        event::KeyboardInput {
                            virtual_keycode: Some(event::VirtualKeyCode::Escape),
                            state: event::ElementState::Pressed,
                            ..
                        },
                    ..
                }
                | WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {
                    app.update(event);
                }
            },
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

                app.render(&frame.output, &device, &queue, &spawner);
            }
            _ => {}
        }
    });
}


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

pub fn run(title: &str) {
    start(pollster::block_on(setup(title)));
}
