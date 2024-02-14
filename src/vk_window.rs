use crate::rh_error::RhError;
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
    swapchain::{Surface, SurfaceCapabilities, SurfaceInfo},
    VulkanLibrary,
};
use vulkano_win::VkSurfaceBuild; // build_vk_surface impl
use winit::window::Window as WinitWindow;
use winit::window::WindowBuilder as WinitWindowBuilder;
use winit::{
    dpi::{LogicalSize, PhysicalSize},
    event_loop::EventLoop,
};

pub struct Properties {
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
    /// # Errors
    /// May return `RhError`
    ///
    /// # Panics
    /// Panics if some `usize` value does not fit into `u32` parameters of
    /// vulkano functions. These are usually very small values so this is
    /// not expected to happen.
    #[allow(clippy::too_many_lines)]
    pub fn new(
        properties: Option<Properties>,
        event_loop: &EventLoop<()>,
        debug_layers: bool,
    ) -> Result<Self, RhError> {
        let properties = properties.map_or_else(
            || Properties {
                dimensions: [640, 480],
                title: "Rhodora".to_string(),
            },
            |wp| wp,
        );
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
            // Interface for Vulkan validation layers.
            // DebugUtilsMessenger is unsafe so this block is required to be
            // unsafe.
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
                                } else  {
                                    "verbose"
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
                                } else {
                                    "performance"
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
                            && p.surface_support(
                                u32::try_from(i)
                                    .expect("queue_family_index out of range"),
                                &surface,
                            )
                            .unwrap_or(false)
                    })
                    .map(|i| (p, u32::try_from(i).expect("value out of range")))
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

    #[must_use]
    pub const fn physical(&self) -> &Arc<PhysicalDevice> {
        &self.physical
    }

    #[must_use]
    pub const fn device(&self) -> &Arc<Device> {
        &self.device
    }

    #[must_use]
    pub const fn graphics_queue(&self) -> &Arc<Queue> {
        &self.graphics_queue
    }

    #[must_use]
    pub const fn instance(&self) -> &Arc<Instance> {
        &self.instance
    }

    #[must_use]
    pub const fn surface(&self) -> &Arc<Surface> {
        &self.surface
    }

    /// # Errors
    /// May return `RhError`
    pub fn surface_caps(
        &self,
    ) -> Result<SurfaceCapabilities, PhysicalDeviceError> {
        self.physical
            .surface_capabilities(&self.surface, SurfaceInfo::default())
    }

    /// # Errors
    /// May return `RhError`
    pub fn format_properties(
        &self,
        format: Format,
    ) -> Result<FormatProperties, PhysicalDeviceError> {
        self.physical.format_properties(format)
    }

    /// # Errors
    /// May return `RhError`
    pub fn winit_window(&self) -> Result<&WinitWindow, RhError> {
        self.surface
            .object()
            .ok_or(RhError::WindowNotFound)?
            .downcast_ref::<WinitWindow>()
            .ok_or(RhError::WindowNotFound)
    }

    /// # Errors
    /// May return `RhError`
    pub fn dimensions(&self) -> Result<PhysicalSize<u32>, RhError> {
        Ok(self.winit_window()?.inner_size())
    }
}
