use crate::rh_error::RhError;
use log::info;
use std::sync::Arc;
use vulkano::{
    device::{
        physical::{PhysicalDevice, PhysicalDeviceType},
        Device, DeviceCreateInfo, DeviceExtensions, Features, Queue,
        QueueCreateInfo, QueueFlags,
    },
    format::{Format, FormatProperties},
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessageType,
            DebugUtilsMessenger, DebugUtilsMessengerCallback,
            DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateInfo, InstanceExtensions,
    },
    swapchain::{Surface, SurfaceCapabilities, SurfaceInfo},
    Validated, VulkanError, VulkanLibrary,
};
use winit::window::Window as WinitWindow;
use winit::window::WindowBuilder;
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

        // vulkano 0.34 deprecates `vulkano_win` which was getting the required
        // extensions and gets them from `Surface` instead.
        let required_extensions = InstanceExtensions {
            ext_debug_utils: debug_layers,
            ..Surface::required_extensions(event_loop)
        };
        let instance = Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: required_extensions,
                ..Default::default()
            },
        )
        .map_err(Validated::unwrap)?;

        // Interface for Vulkan validation layers.
        // DebugUtilsMessenger is unsafe so this block is required to be
        // unsafe.
        let _debug_callback = unsafe {
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
                        DebugUtilsMessengerCallback::new(
                            |msg_severity, msg_type, callback_data| {
                                let severity = if msg_severity.intersects(
                                    DebugUtilsMessageSeverity::ERROR,
                                ) {
                                    "error"
                                } else if msg_severity.intersects(
                                    DebugUtilsMessageSeverity::WARNING,
                                ) {
                                    "warning"
                                } else if msg_severity
                                    .intersects(DebugUtilsMessageSeverity::INFO)
                                {
                                    "information"
                                } else {
                                    "verbose"
                                };

                                let ty = if msg_type
                                    .intersects(DebugUtilsMessageType::GENERAL)
                                {
                                    "general"
                                } else if msg_type.intersects(
                                    DebugUtilsMessageType::VALIDATION,
                                ) {
                                    "validation"
                                } else {
                                    "performance"
                                };

                                // The layers will mostly report "info" type
                                // messages so the `info!` macro seems like a
                                // reasonable choice
                                info!(
                                    "{} {} {}: {}",
                                    callback_data
                                        .message_id_name
                                        .unwrap_or("unknown"),
                                    ty,
                                    severity,
                                    callback_data.message
                                );
                            },
                        ),
                    )
                },
            )
            .ok()
        }; // unsafe block

        let size = LogicalSize::new(
            f64::from(properties.dimensions[0]),
            f64::from(properties.dimensions[1]),
        );

        // Previously there was no actual thing called `window` just a
        // surface created with `build_vk_surface`. But that was deprecated
        // in vulkano 0.34 so now there is this. The `window` gets passed to
        // create the `surface` but is not currently stored elsewhere.
        let window = Arc::new(
            WindowBuilder::new()
                .with_title(properties.title)
                .with_inner_size(size)
                .build(event_loop)?,
        );
        let surface = Surface::from_window(instance.clone(), window)
            .map_err(Validated::unwrap)?;

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
        )
        .map_err(Validated::unwrap)?;
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
    ) -> Result<SurfaceCapabilities, Validated<VulkanError>> {
        self.physical
            .surface_capabilities(&self.surface, SurfaceInfo::default())
    }

    /// Returns the format properties of this window's physical device
    ///
    /// # Panics
    /// Will panic if a `vulkano::ValidationError` is returned by Vulkan
    #[must_use]
    pub fn format_properties(&self, format: Format) -> FormatProperties {
        self.physical.format_properties(format).unwrap() // Box<ValidationError>
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
