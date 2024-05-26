use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use log::{debug, error, info, warn};
use std::ffi::CStr;
use vulkano::{
    instance::{
        debug::{DebugUtilsMessageSeverity, DebugUtilsMessageType},
        Instance,
    },
    Validated, VulkanError, VulkanObject,
};

/// Called during initialization to create the validation callback
pub(crate) fn validation_callback(
    instance: Arc<Instance>,
) -> Option<RhDebugUtilsMessenger> {
    {
        RhDebugUtilsMessenger::new(instance).ok()
    }
}

#[must_use]
pub struct RhDebugUtilsMessenger {
    handle: ash::vk::DebugUtilsMessengerEXT,
    instance: Arc<Instance>,
}

impl RhDebugUtilsMessenger {
    /// Initializes a debug callback. Based on vulkano 0.34.1
    /// `DebugUtilsMessenger` from *debug.rs*. Renamed to lessen confusion
    /// since similarly named vulkano types are used unchanged.
    ///
    /// # Errors
    /// Returns a `VulkanError` if creation fails
    #[inline]
    pub fn new(
        instance: Arc<Instance>,
    ) -> Result<Self, Validated<VulkanError>> {
        unsafe { Ok(Self::new_unchecked(instance)?) }
    }

    unsafe fn new_unchecked(
        instance: Arc<Instance>,
    ) -> Result<Self, VulkanError> {
        let message_severity = DebugUtilsMessageSeverity::ERROR
            | DebugUtilsMessageSeverity::WARNING
            | DebugUtilsMessageSeverity::INFO
            | DebugUtilsMessageSeverity::VERBOSE;
        let message_type = DebugUtilsMessageType::GENERAL
            | DebugUtilsMessageType::VALIDATION
            | DebugUtilsMessageType::PERFORMANCE;

        let create_info_vk = ash::vk::DebugUtilsMessengerCreateInfoEXT {
            flags: ash::vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: message_severity.into(),
            message_type: message_type.into(),
            pfn_user_callback: Some(rh_callback),
            p_user_data: std::ptr::null::<std::ffi::c_void>().cast_mut(),
            ..Default::default()
        };

        let handle = {
            let fns = instance.fns();
            let mut output = std::mem::MaybeUninit::uninit();
            (fns.ext_debug_utils.create_debug_utils_messenger_ext)(
                instance.handle(),
                &create_info_vk,
                std::ptr::null(),
                output.as_mut_ptr(),
            )
            .result()
            .map_err(VulkanError::from)?;
            output.assume_init()
        };

        Ok(Self { handle, instance })
    }
}

impl Drop for RhDebugUtilsMessenger {
    #[inline]
    fn drop(&mut self) {
        unsafe {
            let fns = self.instance.fns();
            (fns.ext_debug_utils.destroy_debug_utils_messenger_ext)(
                self.instance.handle(),
                self.handle,
                std::ptr::null(),
            );
        }
    }
}

/// The callback itself. This code is based on the vulkano 0.34.1 "trampoline"
/// function in *debug.rs* with the changes noted below.
///
/// The vulkano version uses the user data pointer to jump to a user defined
/// callback, but in this project that was just going to a simple debug output
/// function, so the trampoline aspect was removed and the output done here
/// directly.
///
/// The main issue was that some of the fields from Vulkan (as returned by ash)
/// may contain null pointers, and vulkano 0.34.1 (and earlier) handle these
/// incorrectly. The Rust compiler was unable to detect this until 1.78 but
/// once it did it caused  an immediate panic at start up. The issue has been
/// patched in vulkano but (as of this writing) has not been released. This
/// project was not using any of those fields anyway so the pointers are not
/// used at all here.
pub(super) unsafe extern "system" fn rh_callback(
    message_severity_vk: ash::vk::DebugUtilsMessageSeverityFlagsEXT,
    message_types_vk: ash::vk::DebugUtilsMessageTypeFlagsEXT,
    callback_data_vk: *const ash::vk::DebugUtilsMessengerCallbackDataEXT,
    _user_data_vk: *mut std::ffi::c_void,
) -> ash::vk::Bool32 {
    // Since we box the closure, the type system doesn't detect that the `UnwindSafe`
    // bound is enforced. Therefore we enforce it manually.
    let _ = catch_unwind(AssertUnwindSafe(move || {
        let ash::vk::DebugUtilsMessengerCallbackDataEXT {
            s_type: _,
            p_next: _,
            flags: _,
            p_message_id_name,
            message_id_number: _,
            p_message,
            queue_label_count: _,
            p_queue_labels: _,
            cmd_buf_label_count: _,
            p_cmd_buf_labels: _,
            object_count: _,
            p_objects: _,
        } = *callback_data_vk;

        let name =
            p_message_id_name
                .as_ref()
                .map_or("unknown", |p_message_id_name| {
                    {
                        CStr::from_ptr(p_message_id_name)
                            .to_str()
                            .unwrap_or("unknown")
                    }
                });
        let msg_type = {
            let ty: DebugUtilsMessageType = message_types_vk.into();
            if ty.intersects(DebugUtilsMessageType::GENERAL) {
                "general"
            } else if ty.intersects(DebugUtilsMessageType::VALIDATION) {
                "validation"
            } else {
                "performance"
            }
        };
        let message =
            CStr::from_ptr(p_message).to_str().unwrap_or("(no message)");
        let msg_severity: DebugUtilsMessageSeverity =
            message_severity_vk.into();
        if msg_severity.intersects(DebugUtilsMessageSeverity::ERROR) {
            error!("{} {}: {}", name, msg_type, message);
        } else if msg_severity.intersects(DebugUtilsMessageSeverity::WARNING) {
            warn!("{} {}: {}", name, msg_type, message);
        } else if msg_severity.intersects(DebugUtilsMessageSeverity::INFO) {
            info!("{} {}: {}", name, msg_type, message);
        } else {
            debug!("{} {}: {}", name, msg_type, message);
        };
    }));

    ash::vk::FALSE
}
