use std::cell::Cell;
use std::ffi::CStr;
use std::os::raw::{c_char, c_void};
use std::ptr;
use std::sync::Arc;

use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{extensions::ext::DebugReport, vk, Entry};
use renderdoc::{RenderDoc, V100};

#[test]
#[ignore]
fn smoke() {
    let had_error = Cell::new(false);
    let mut rd = RenderDoc::<V100>::new().ok();

    unsafe {
        let entry = Entry::new().unwrap();
        let app_name = CStr::from_bytes_with_nul(b"fuzzyblue smoke test\0").unwrap();
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::builder()
                    .application_info(
                        &vk::ApplicationInfo::builder()
                            .application_name(&app_name)
                            .application_version(0)
                            .engine_name(&app_name)
                            .engine_version(0)
                            .api_version(vk::make_version(1, 0, 36)),
                    )
                    .enabled_extension_names(&[DebugReport::name().as_ptr()]),
                None,
            )
            .unwrap();

        let debug_report_loader = DebugReport::new(&entry, &instance);
        let debug_call_back = debug_report_loader
            .create_debug_report_callback(
                &vk::DebugReportCallbackCreateInfoEXT::builder()
                    .flags(
                        vk::DebugReportFlagsEXT::ERROR
                            | vk::DebugReportFlagsEXT::WARNING
                            | vk::DebugReportFlagsEXT::PERFORMANCE_WARNING
                            | vk::DebugReportFlagsEXT::INFORMATION,
                    )
                    .pfn_callback(Some(vulkan_debug_callback))
                    .user_data(&had_error as *const _ as *mut _),
                None,
            )
            .unwrap();

        let (pdevice, queue_family_index) = instance
            .enumerate_physical_devices()
            .unwrap()
            .iter()
            .map(|pdevice| {
                instance
                    .get_physical_device_queue_family_properties(*pdevice)
                    .iter()
                    .enumerate()
                    .filter_map(|(index, ref info)| {
                        if info.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                            Some((*pdevice, index as u32))
                        } else {
                            None
                        }
                    })
                    .next()
            })
            .filter_map(|v| v)
            .next()
            .expect("no graphics device available");

        let device = Arc::new(
            instance
                .create_device(
                    pdevice,
                    &vk::DeviceCreateInfo::builder()
                        .queue_create_infos(&[vk::DeviceQueueCreateInfo::builder()
                            .queue_family_index(queue_family_index)
                            .queue_priorities(&[1.0])
                            .build()])
                        .enabled_features(&vk::PhysicalDeviceFeatures {
                            robust_buffer_access: vk::TRUE,
                            ..Default::default()
                        }),
                    None,
                )
                .unwrap(),
        );
        let queue = device.get_device_queue(queue_family_index as u32, 0);

        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::builder()
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                    .queue_family_index(queue_family_index),
                None,
            )
            .unwrap();

        if let Some(ref mut rd) = rd {
            rd.start_frame_capture(renderdoc::DevicePointer::from(ptr::null()), ptr::null());
        }

        let cmd = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::builder()
                    .command_buffer_count(1)
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )
            .unwrap()[0];

        let builder = Arc::new(fuzzyblue::Builder::new(
            &instance,
            device.clone(),
            vk::PipelineCache::null(),
            pdevice,
            queue_family_index,
            None,
        ));

        device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();

        let pending = fuzzyblue::Atmosphere::build(
            builder,
            cmd,
            // Simplified for speed
            &fuzzyblue::Parameters {
                scattering_r_size: 8,
                scattering_mu_size: 32,
                scattering_mu_s_size: 8,
                scattering_nu_size: 2,
                ..Default::default()
            },
        );

        device.end_command_buffer(cmd).unwrap();

        device
            .queue_submit(
                queue,
                &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()],
                vk::Fence::null(),
            )
            .unwrap();

        device.device_wait_idle().unwrap();

        drop(pending);

        if let Some(ref mut rd) = rd {
            rd.end_frame_capture(renderdoc::DevicePointer::from(ptr::null()), ptr::null());
        }

        device.destroy_command_pool(pool, None);
        device.destroy_device(None);
        debug_report_loader.destroy_debug_report_callback(debug_call_back, None);
        instance.destroy_instance(None);
    }

    if had_error.get() {
        panic!("vulkan reported an error");
    }
}

unsafe extern "system" fn vulkan_debug_callback(
    flags: vk::DebugReportFlagsEXT,
    _: vk::DebugReportObjectTypeEXT,
    _: u64,
    _: usize,
    _: i32,
    _: *const c_char,
    p_message: *const c_char,
    user_data: *mut c_void,
) -> u32 {
    eprintln!(
        "{:?} {}",
        flags,
        CStr::from_ptr(p_message).to_string_lossy()
    );
    if flags.contains(vk::DebugReportFlagsEXT::ERROR) {
        let had_error = &*(user_data as *const Cell<bool>);
        had_error.set(true);
    }
    vk::FALSE
}
