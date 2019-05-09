use std::ffi::CStr;
use std::fs::File;
use std::sync::Arc;
use std::{mem, ptr, slice};

use ash::version::{DeviceV1_0, EntryV1_0, InstanceV1_0};
use ash::{vk, Device, Entry};
use half::f16;
use openexr::frame_buffer::PixelStruct;
use renderdoc::{RenderDoc, V100};

fn main() {
    let mut rd = RenderDoc::<V100>::new().ok();
    unsafe {
        let entry = Entry::new().unwrap();
        let app_name = CStr::from_bytes_with_nul(b"fuzzyblue smoke test\0").unwrap();
        let instance = entry
            .create_instance(
                &vk::InstanceCreateInfo::builder().application_info(
                    &vk::ApplicationInfo::builder()
                        .application_name(&app_name)
                        .application_version(0)
                        .engine_name(&app_name)
                        .engine_version(0)
                        .api_version(vk::make_version(1, 0, 36)),
                ),
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

        let memory_props = instance.get_physical_device_memory_properties(pdevice);

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

        let params = fuzzyblue::Parameters {
            usage: vk::ImageUsageFlags::TRANSFER_SRC,
            dst_stage_mask: vk::PipelineStageFlags::TRANSFER,
            dst_access_mask: vk::AccessFlags::TRANSFER_READ,
            layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            // Simplified for speed
            order: 4,
            scattering_r_size: 16,
            scattering_mu_size: 64,
            scattering_mu_s_size: 16,
            scattering_nu_size: 4,
            ..Default::default()
        };

        let transmittance_buf = Buffer::<[f32; 4]>::new(
            &device,
            &memory_props,
            params.transmittance_extent().width * params.transmittance_extent().height,
        );
        let irradiance_buf = Buffer::<[f32; 4]>::new(
            &device,
            &memory_props,
            params.irradiance_extent().width * params.irradiance_extent().height,
        );
        let scattering_buf = Buffer::<[f16; 4]>::new(
            &device,
            &memory_props,
            params.scattering_extent().width
                * params.scattering_extent().height
                * params.scattering_extent().depth,
        );

        let builder = Arc::new(fuzzyblue::Builder::new(
            &instance,
            device.clone(),
            vk::PipelineCache::null(),
            pdevice,
            queue_family_index,
            None,
        ));

        // Precompute look-up tables
        device
            .begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::builder()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .unwrap();

        let pending = fuzzyblue::Atmosphere::build(builder, cmd, &params);

        // Pipeline barriers of build ensure this is blocked until the images are fully written
        let atmosphere = pending.atmosphere();
        for &(image, buf, extent) in &[
            (
                atmosphere.transmittance(),
                transmittance_buf.handle,
                vk::Extent3D {
                    width: atmosphere.transmittance_extent().width,
                    height: atmosphere.transmittance_extent().height,
                    depth: 1,
                },
            ),
            (
                atmosphere.irradiance(),
                irradiance_buf.handle,
                vk::Extent3D {
                    width: atmosphere.irradiance_extent().width,
                    height: atmosphere.irradiance_extent().height,
                    depth: 1,
                },
            ),
            (
                atmosphere.scattering(),
                scattering_buf.handle,
                atmosphere.scattering_extent(),
            ),
        ] {
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buf,
                &[vk::BufferImageCopy {
                    buffer_offset: 0,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                    image_extent: extent,
                }],
            );
        }

        device.end_command_buffer(cmd).unwrap();

        device
            .queue_submit(
                queue,
                &[vk::SubmitInfo::builder().command_buffers(&[cmd]).build()],
                vk::Fence::null(),
            )
            .unwrap();

        device.device_wait_idle().unwrap();

        write_image(
            "transmittance",
            &*transmittance_buf.ptr,
            atmosphere.transmittance_extent().width,
            atmosphere.transmittance_extent().height,
            1,
        );
        write_image(
            "irradiance",
            &*irradiance_buf.ptr,
            atmosphere.irradiance_extent().width,
            atmosphere.irradiance_extent().height,
            1,
        );
        write_image(
            "scattering",
            &*scattering_buf.ptr,
            atmosphere.scattering_extent().width,
            atmosphere.scattering_extent().height,
            atmosphere.scattering_extent().depth,
        );

        drop(pending);

        transmittance_buf.destroy(&device);
        irradiance_buf.destroy(&device);
        scattering_buf.destroy(&device);

        if let Some(ref mut rd) = rd {
            rd.end_frame_capture(renderdoc::DevicePointer::from(ptr::null()), ptr::null());
        }

        device.destroy_command_pool(pool, None);
        device.destroy_device(None);
        instance.destroy_instance(None);
    }
}

fn find_memory_type(
    device_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..device_props.memory_type_count {
        if type_bits & (1 << i) != 0
            && device_props.memory_types[i as usize]
                .property_flags
                .contains(flags)
        {
            return Some(i);
        }
    }
    None
}

unsafe fn allocate(
    device: &Device,
    device_props: &vk::PhysicalDeviceMemoryProperties,
    reqs: vk::MemoryRequirements,
    flags: vk::MemoryPropertyFlags,
) -> Option<vk::DeviceMemory> {
    let ty = find_memory_type(device_props, reqs.memory_type_bits, flags)?;
    Some(
        device
            .allocate_memory(
                &vk::MemoryAllocateInfo {
                    allocation_size: reqs.size,
                    memory_type_index: ty,
                    ..Default::default()
                },
                None,
            )
            .unwrap(),
    )
}

struct Buffer<T: Copy> {
    handle: vk::Buffer,
    mem: vk::DeviceMemory,
    ptr: *mut [T],
}

impl<T: Copy> Buffer<T> {
    unsafe fn new(
        device: &Device,
        memory_props: &vk::PhysicalDeviceMemoryProperties,
        pixels: u32,
    ) -> Self {
        let bytes = u64::from(pixels) * mem::size_of::<T>() as u64;
        let handle = device
            .create_buffer(
                &vk::BufferCreateInfo {
                    size: bytes,
                    usage: vk::BufferUsageFlags::TRANSFER_DST,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        let mem = {
            let reqs = device.get_buffer_memory_requirements(handle);
            allocate(
                device,
                memory_props,
                reqs,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            )
            .unwrap()
        };
        device.bind_buffer_memory(handle, mem, 0).unwrap();
        let ptr = device
            .map_memory(mem, 0, bytes, Default::default())
            .unwrap() as _;
        let ptr = slice::from_raw_parts_mut(ptr, pixels as usize);
        Self { handle, mem, ptr }
    }

    unsafe fn destroy(&self, device: &Device) {
        device.destroy_buffer(self.handle, None);
        device.free_memory(self.mem, None);
    }
}

fn write_image<T: PixelStruct>(name: &str, data: &[T], width: u32, height: u32, depth: u32) {
    use openexr::{FrameBuffer, Header, ScanlineOutputFile};
    let mut file = File::create(format!("{}.exr", name)).unwrap();

    let mut header = Header::new();
    header.set_resolution(width, height);
    for layer in 0..depth {
        for (&channel, ty) in ['R', 'G', 'B', 'A']
            .iter()
            .zip((0..).map(|i| T::channel(i).0))
        {
            header.add_channel(&channel_name(channel, layer, depth), ty);
        }
    }
    let mut exr_file = ScanlineOutputFile::new(&mut file, &header).unwrap();

    let mut fb = FrameBuffer::new(width, height);
    for (layer, data) in data.chunks(width as usize * height as usize).enumerate() {
        let channels = ['R', 'G', 'B', 'A']
            .iter()
            .map(|&c| channel_name(c, layer as u32, depth))
            .collect::<Vec<_>>();
        fb.insert_channels(
            &[&channels[0], &channels[1], &channels[2], &channels[3]],
            &data,
        );
    }
    exr_file.write_pixels(&fb).unwrap();
}

fn channel_name(color: char, layer: u32, layer_count: u32) -> String {
    if layer_count == 1 {
        color.to_string()
    } else {
        format!("{}.{}", layer, color)
    }
}
