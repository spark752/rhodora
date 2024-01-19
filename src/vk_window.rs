use crate::rh_error::*;
use log::info;
use std::sync::Arc;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceError, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::{Format, FormatProperties},
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType,
            DebugUtilsMessenger, DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateInfo, InstanceExtensions,
    },
    swapchain::{Surface, SurfaceCapabilities},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild; // build_vk_surface impl
use winit::window::Window as WinitWindow;
use winit::window::WindowBuilder as WinitWindowBuilder;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event_loop::EventLoop,
};

pub struct VkWindowProperties {
    pub dimensions: [u32; 2],
    pub title: String,
}

pub struct VkWindow {
    instance: Arc<Instance>,
    surface: Arc<Surface>,
    physical: Arc<PhysicalDevice>,
    device: Arc<Device>,
    graphics_queue: Arc<Queue>,
}

impl VkWindow {
    pub fn new(
        properties: Option<VkWindowProperties>,
        event_loop: &EventLoop<()>,
        debug_layers: bool,
    ) -> Result<Self, RhError> {
        let properties = if let Some(wp) = properties {
            wp
        } else {
            VkWindowProperties {
                dimensions: [640, 480],
                title: "VK Greenbell".to_string(),
            }
        };
        let library = VulkanLibrary::new()?;
        let required_extensions = InstanceExtensions {
            ext_debug_utils: debug_layers,
            ..vulkano_win::required_extensions(&library)
        };
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )?;
        let _debug_callback =
            unsafe {
                DebugUtilsMessenger::new(
                    instance.clone(),
                    DebugUtilsMessengerCreateInfo {
                        message_severity: DebugUtilsMessageSeverity::ERROR
                            | DebugUtilsMessageSeverity::WARNING
                            | DebugUtilsMessageSeverity::INFO
                            | DebugUtilsMessageSeverity::VERBOSE,
                        message_type: DebugUtilsMessageType::GENERAL
                            | DebugUtilsMessageType::VALIDATION
                            | DebugUtilsMessageType::PERFORMANCE,
                        ..DebugUtilsMessengerCreateInfo::user_callback(
                            Arc::new(|msg| {
                                let severity = if msg.severity.intersects(
                                    DebugUtilsMessageSeverity::ERROR,
                                ) {
                                    "error"
                                } else if msg.severity.intersects(
                                    DebugUtilsMessageSeverity::WARNING,
                                ) {
                                    "warning"
                                } else if msg
                                    .severity
                                    .intersects(DebugUtilsMessageSeverity::INFO)
                                {
                                    "information"
                                } else if msg.severity.intersects(
                                    DebugUtilsMessageSeverity::VERBOSE,
                                ) {
                                    "verbose"
                                } else {
                                    panic!("no-impl");
                                };

                                let ty = if msg
                                    .ty
                                    .intersects(DebugUtilsMessageType::GENERAL)
                                {
                                    "general"
                                } else if msg.ty.intersects(
                                    DebugUtilsMessageType::VALIDATION,
                                ) {
                                    "validation"
                                } else if msg.ty.intersects(
                                    DebugUtilsMessageType::PERFORMANCE,
                                ) {
                                    "performance"
                                } else {
                                    panic!("no-impl");
                                };

                                println!(
                                    "{} {} {}: {}",
                                    msg.layer_prefix.unwrap_or("unknown"),
                                    ty,
                                    severity,
                                    msg.description
                                );
                            }),
                        )
                    },
                )
                .ok()
            };
        let size = LogicalSize::new(
            f64::from(properties.dimensions[0]),
            f64::from(properties.dimensions[1]),
        );
        let surface = WinitWindowBuilder::new()
            .with_title(properties.title)
            .with_inner_size(size)
            .build_vk_surface(event_loop, instance.clone())?;
        let device_extensions = DeviceExtensions {
            khr_swapchain: true,
            ..DeviceExtensions::empty()
        };
        let (physical, queue_family_index) = instance
            .enumerate_physical_devices()?
            .filter(|p| p.api_version() >= vulkano::Version::V1_3)
            .filter(|p| p.supported_extensions().contains(&device_extensions))
            .filter_map(|p| {
                p.queue_family_properties()
                    .iter()
                    .enumerate()
                    .position(|(i, q)| {
                        q.queue_flags.intersects(QueueFlags::GRAPHICS)
                            && p.surface_support(i as u32, &surface)
                                .unwrap_or(false)
                    })
                    .map(|i| (p, i as u32))
            })
            .min_by_key(|(p, _)| match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            })
            .ok_or(RhError::DeviceNotFound)?;
        let (device, mut queues) = Device::new(
            physical.clone(),
            DeviceCreateInfo {
                enabled_extensions: device_extensions,
                enabled_features: Features {
                    // Vulkan 1.3 things:
                    dynamic_rendering: true,
                    synchronization2: true,
                    ..Features::empty()
                },
                queue_create_infos: vec![QueueCreateInfo {
                    queue_family_index,
                    ..Default::default()
                }],
                ..Default::default()
            },
        )?;
        info!("Using Vulkan version : {}", device.api_version());
        let graphics_queue = queues.next().ok_or(RhError::QueueNotFound)?;

        Ok(Self {
            instance,
            surface,
            physical,
            device,
            graphics_queue,
        })
    }

    pub fn physical(&self) -> &Arc<PhysicalDevice> {
        &self.physical
    }

    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub fn graphics_queue(&self) -> &Arc<Queue> {
        &self.graphics_queue
    }

    pub fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    pub fn surface(&self) -> &Arc<Surface> {
        &self.surface
    }

    pub fn surface_caps(
        &self,
    ) -> Result<SurfaceCapabilities, PhysicalDeviceError> {
        self.physical
            .surface_capabilities(&self.surface, Default::default())
    }

    pub fn format_properties(
        &self,
        format: Format,
    ) -> Result<FormatProperties, PhysicalDeviceError> {
        self.physical.format_properties(format)
    }

    pub fn winit_window(&self) -> Result<&WinitWindow, RhError> {
        self.surface
            .object()
            .ok_or(RhError::WindowNotFound)?
            .downcast_ref::<WinitWindow>()
            .ok_or(RhError::WindowNotFound)
    }

    pub fn dimensions(&self) -> Result<PhysicalSize<u32>, RhError> {
        Ok(self.winit_window()?.inner_size())
    }
}
