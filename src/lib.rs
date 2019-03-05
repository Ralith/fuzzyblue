use std::{mem, ptr, sync::Arc};

use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, Device, Instance};
use vk_shader_macros::include_glsl;

const TRANSMITTANCE: &[u32] = include_glsl!("shaders/transmittance.comp", debug);
const SCATTERING: &[u32] = include_glsl!("shaders/scattering.comp", debug);
const GATHERING: &[u32] = include_glsl!("shaders/gathering.comp", debug);
const MULTISCATTERING: &[u32] = include_glsl!("shaders/multiscattering.comp", debug);

const SCATTERING_EXTENT: vk::Extent3D = vk::Extent3D {
    width: 32,
    height: 64,
    depth: 32,
};
const TRANSMITTANCE_EXTENT: vk::Extent2D = vk::Extent2D {
    width: 32,
    height: 128,
};
const GATHERING_EXTENT: vk::Extent2D = vk::Extent2D {
    width: 32,
    height: 32,
};

/// Order of scattering to compute
const ORDER: usize = 4;

/// Constructs `Atmosphere`s
///
/// Must not be dropped before all created `Atmosphers` and `PendingAtmospheres` are dropped.
pub struct Builder {
    device: Arc<Device>,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    gfx_queue_family: u32,
    compute_queue_family: Option<u32>,
    sampler: vk::Sampler,
    params_layout: vk::DescriptorSetLayout,
    transmittance: Pass,
    scattering: Pass,
    // These reuse scattering's (ds_)layout
    gathering: vk::Pipeline,
    gathering_shader: vk::ShaderModule,
    multiscattering: vk::Pipeline,
    multiscattering_shader: vk::ShaderModule,
}

impl Drop for Builder {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline(self.transmittance.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.transmittance.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.transmittance.ds_layout, None);
            self.device
                .destroy_shader_module(self.transmittance.shader, None);
            self.device.destroy_pipeline(self.scattering.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.scattering.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.scattering.ds_layout, None);
            self.device
                .destroy_shader_module(self.scattering.shader, None);
            self.device.destroy_pipeline(self.gathering, None);
            self.device
                .destroy_shader_module(self.gathering_shader, None);
            self.device.destroy_pipeline(self.multiscattering, None);
            self.device
                .destroy_shader_module(self.multiscattering_shader, None);
            self.device.destroy_sampler(self.sampler, None);
            self.device
                .destroy_descriptor_set_layout(self.params_layout, None);
        }
    }
}

impl Builder {
    pub fn new(
        instance: &Instance,
        device: Arc<Device>,
        cache: vk::PipelineCache,
        physical: vk::PhysicalDevice,
        gfx_queue_family: u32,
        compute_queue_family: Option<u32>,
    ) -> Self {
        unsafe {
            let params_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                    ]),
                    None,
                )
                .unwrap();

            let sampler = device
                .create_sampler(
                    &vk::SamplerCreateInfo {
                        min_filter: vk::Filter::LINEAR,
                        mag_filter: vk::Filter::LINEAR,
                        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();

            let transmittance_ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                    ]),
                    None,
                )
                .unwrap();
            let transmittance_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[params_layout, transmittance_ds_layout]),
                    None,
                )
                .unwrap();
            let transmittance_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&TRANSMITTANCE),
                    None,
                )
                .unwrap();

            let scattering_ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                    ]),
                    None,
                )
                .unwrap();
            let scattering_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[params_layout, scattering_ds_layout]),
                    None,
                )
                .unwrap();
            let scattering_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&SCATTERING),
                    None,
                )
                .unwrap();

            let gathering_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&GATHERING),
                    None,
                )
                .unwrap();

            let multiscattering_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&MULTISCATTERING),
                    None,
                )
                .unwrap();

            let p_name = b"main\0".as_ptr() as *const i8;

            let mut pipelines = device
                .create_compute_pipelines(
                    cache,
                    &[
                        vk::ComputePipelineCreateInfo {
                            stage: vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::COMPUTE,
                                module: transmittance_shader,
                                p_name,
                                ..Default::default()
                            },
                            layout: transmittance_layout,
                            ..Default::default()
                        },
                        vk::ComputePipelineCreateInfo {
                            stage: vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::COMPUTE,
                                module: scattering_shader,
                                p_name,
                                ..Default::default()
                            },
                            layout: scattering_layout,
                            ..Default::default()
                        },
                        vk::ComputePipelineCreateInfo {
                            stage: vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::COMPUTE,
                                module: gathering_shader,
                                p_name,
                                ..Default::default()
                            },
                            layout: scattering_layout,
                            ..Default::default()
                        },
                        vk::ComputePipelineCreateInfo {
                            stage: vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::COMPUTE,
                                module: multiscattering_shader,
                                p_name,
                                ..Default::default()
                            },
                            layout: scattering_layout,
                            ..Default::default()
                        },
                    ],
                    None,
                )
                .unwrap()
                .into_iter();

            let transmittance = Pass {
                shader: transmittance_shader,
                pipeline: pipelines.next().unwrap(),
                layout: transmittance_layout,
                ds_layout: transmittance_ds_layout,
            };
            let scattering = Pass {
                shader: scattering_shader,
                pipeline: pipelines.next().unwrap(),
                layout: scattering_layout,
                ds_layout: scattering_ds_layout,
            };
            let gathering = pipelines.next().unwrap();
            let multiscattering = pipelines.next().unwrap();

            Self {
                device,
                memory_props: instance.get_physical_device_memory_properties(physical),
                gfx_queue_family,
                compute_queue_family,
                sampler,
                params_layout,
                transmittance,
                scattering,
                gathering,
                gathering_shader,
                multiscattering,
                multiscattering_shader,
            }
        }
    }

    /// Build an `Atmosphere` that will be usable when `cmd` is fully executed.
    pub fn build(&self, cmd: vk::CommandBuffer, atmosphere_params: &Params) -> PendingAtmosphere {
        unsafe {
            // common: 1 uniform
            // transmittance: 1 storage image
            // scattering: 1 sampled image, 2 storage image
            // gathering: 1 sampled image, 2 storage images
            // multiscattering: 1 sampled image, 2 storage image
            // ====
            // 1 uniform, 4 sampled images, 8 storage images
            let descriptor_pool = self
                .device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::builder()
                        .max_sets(6)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: 1,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                descriptor_count: 3,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_IMAGE,
                                descriptor_count: 7,
                            },
                        ]),
                    None,
                )
                .unwrap();

            let mut descriptor_sets = self
                .device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[
                            self.params_layout,
                            self.transmittance.ds_layout,
                            self.scattering.ds_layout,
                            self.scattering.ds_layout,
                            self.scattering.ds_layout,
                        ]),
                )
                .unwrap()
                .into_iter();
            let params_ds = descriptor_sets.next().unwrap();
            let transmittance_ds = descriptor_sets.next().unwrap();
            let scattering_ds = descriptor_sets.next().unwrap();
            let gathering_ds = descriptor_sets.next().unwrap();
            let multiscattering_ds = descriptor_sets.next().unwrap();

            let scattering_image_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_3D,
                format: vk::Format::R16G16B16A16_SFLOAT,
                extent: SCATTERING_EXTENT,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let single_order = self.alloc_image(&scattering_image_info);

            let gathering_image_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R16G16B16A16_SFLOAT,
                extent: vk::Extent3D {
                    width: GATHERING_EXTENT.width,
                    height: GATHERING_EXTENT.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let gathered = self.alloc_image(&gathering_image_info);
            let gathered_sum = self.alloc_image(&gathering_image_info);

            let transmittance_lut = self.alloc_image(&vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R16G16B16A16_SFLOAT,
                extent: vk::Extent3D {
                    width: TRANSMITTANCE_EXTENT.width,
                    height: TRANSMITTANCE_EXTENT.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            });

            let scattering = self.alloc_image(&scattering_image_info);

            let params = self
                .device
                .create_buffer(
                    &vk::BufferCreateInfo {
                        size: mem::size_of::<Params>() as vk::DeviceSize,
                        usage: vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let params_mem = {
                let reqs = self.device.get_buffer_memory_requirements(params);
                allocate(
                    &self.device,
                    &self.memory_props,
                    reqs,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .unwrap()
            };
            self.device
                .bind_buffer_memory(params, params_mem, 0)
                .unwrap();

            self.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet {
                        dst_set: params_ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: params,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: transmittance_ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: transmittance_lut.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: scattering_ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: transmittance_lut.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: scattering_ds,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: single_order.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: scattering_ds,
                        dst_binding: 2,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: scattering.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: gathering_ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: single_order.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: gathering_ds,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: gathered.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: gathering_ds,
                        dst_binding: 2,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: gathered_sum.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: multiscattering_ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: gathered.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: multiscattering_ds,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: single_order.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: multiscattering_ds,
                        dst_binding: 2,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: scattering.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                ],
                &[],
            );

            let init_barrier = vk::ImageMemoryBarrier {
                dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            let write_read_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            let read_write_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_READ,
                dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };
            let write_barrier = vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::SHADER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::GENERAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            //
            // Write commands
            //

            self.device.cmd_update_buffer(
                cmd,
                params,
                0,
                &mem::transmute::<_, [u8; 32]>(*atmosphere_params),
            );
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                Default::default(),
                &[],
                &[vk::BufferMemoryBarrier {
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask: vk::AccessFlags::UNIFORM_READ,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    buffer: params,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                    ..Default::default()
                }],
                &[
                    vk::ImageMemoryBarrier {
                        image: transmittance_lut.handle,
                        ..init_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: single_order.handle,
                        ..init_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: gathered.handle,
                        ..init_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: gathered_sum.handle,
                        ..init_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: scattering.handle,
                        ..init_barrier
                    },
                ],
            );

            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.transmittance.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.transmittance.layout,
                0,
                &[params_ds, transmittance_ds],
                &[],
            );
            self.device.cmd_dispatch(
                cmd,
                TRANSMITTANCE_EXTENT.width,
                TRANSMITTANCE_EXTENT.height,
                1,
            );

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                Default::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    image: transmittance_lut.handle,
                    ..write_read_barrier
                }],
            );

            self.device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.scattering.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.scattering.layout,
                1,
                &[scattering_ds],
                &[],
            );
            self.device.cmd_dispatch(
                cmd,
                SCATTERING_EXTENT.width,
                SCATTERING_EXTENT.height,
                SCATTERING_EXTENT.depth,
            );

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                Default::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    image: single_order.handle,
                    ..write_read_barrier
                }],
            );

            for _ in 1..ORDER {
                self.device
                    .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.gathering);
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.scattering.layout,
                    1,
                    &[gathering_ds],
                    &[],
                );
                self.device
                    .cmd_dispatch(cmd, GATHERING_EXTENT.width, GATHERING_EXTENT.height, 1);

                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    Default::default(),
                    &[],
                    &[],
                    &[
                        vk::ImageMemoryBarrier {
                            image: gathered.handle,
                            ..write_read_barrier
                        },
                        vk::ImageMemoryBarrier {
                            image: single_order.handle,
                            ..read_write_barrier
                        },
                        vk::ImageMemoryBarrier {
                            image: scattering.handle,
                            ..write_barrier
                        },
                    ],
                );

                self.device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.multiscattering,
                );
                self.device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    self.scattering.layout,
                    1,
                    &[multiscattering_ds],
                    &[],
                );
                self.device.cmd_dispatch(
                    cmd,
                    SCATTERING_EXTENT.width,
                    SCATTERING_EXTENT.height,
                    SCATTERING_EXTENT.depth,
                );

                self.device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    Default::default(),
                    &[],
                    &[],
                    &[
                        vk::ImageMemoryBarrier {
                            image: gathered.handle,
                            ..read_write_barrier
                        },
                        vk::ImageMemoryBarrier {
                            image: single_order.handle,
                            ..write_read_barrier
                        },
                    ],
                );
            }

            let params_queue_transfer = [vk::BufferMemoryBarrier {
                src_access_mask: vk::AccessFlags::UNIFORM_READ,
                src_queue_family_index: self.compute_queue_family.unwrap_or(self.gfx_queue_family),
                dst_queue_family_index: self.gfx_queue_family,
                buffer: params,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            }];
            let buffer_barriers = if self
                .compute_queue_family
                .map_or(false, |x| x != self.gfx_queue_family)
            {
                &params_queue_transfer[..]
            } else {
                &[][..]
            };

            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                Default::default(),
                &[],
                buffer_barriers,
                &[vk::ImageMemoryBarrier {
                    image: scattering.handle,
                    src_queue_family_index: self
                        .compute_queue_family
                        .unwrap_or(self.gfx_queue_family),
                    dst_queue_family_index: self.gfx_queue_family,
                    ..write_read_barrier
                }],
            );

            PendingAtmosphere {
                device: self.device.clone(),
                inner: Some(Atmosphere {
                    device: self.device.clone(),
                    scattering,
                    params,
                    params_mem,
                }),
                descriptor_pool,
                single_order,
                gathered,
                gathered_sum,
                transmittance_lut,
            }
        }
    }

    unsafe fn alloc_image(&self, info: &vk::ImageCreateInfo) -> Image {
        let handle = self.device.create_image(info, None).unwrap();
        let reqs = self.device.get_image_memory_requirements(handle);
        let memory = allocate(
            &self.device,
            &self.memory_props,
            reqs,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .unwrap();
        self.device.bind_image_memory(handle, memory, 0).unwrap();
        let view = self
            .device
            .create_image_view(
                &vk::ImageViewCreateInfo {
                    image: handle,
                    view_type: match info.image_type {
                        vk::ImageType::TYPE_1D => vk::ImageViewType::TYPE_1D,
                        vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
                        vk::ImageType::TYPE_3D => vk::ImageViewType::TYPE_3D,
                        _ => unreachable!("unknown image type"),
                    },
                    format: info.format,
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        Image {
            handle,
            view,
            memory,
        }
    }
}

struct Image {
    handle: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Params {
    /// Maximum height of atmosphere (m)
    pub h_atm: f32,
    /// Radius of planet (m)
    pub r_planet: f32,
    /// Scale height of rayleigh particles (m)
    pub h_r: f32,
    /// Scale height of mie particles (m)
    pub h_m: f32,
    /// Rayleigh scattering coefficients for each color's wavelength (m^-1)
    pub beta_r: [f32; 3],
    /// Mie scattering coefficient (m^-1)
    pub beta_m: f32,
}

pub const IOR_AIR: f64 = 1.0003;
/// molecules/m^3
pub const DENSITY_AIR: f64 = 2.545e25;
/// red wavelength
pub const LAMBDA_R: f64 = 6.5e-7;
/// green wavelength
pub const LAMBDA_G: f64 = 5.1e-7;
/// blue wavelength
pub const LAMBDA_B: f64 = 4.75e-7;

pub fn compute_beta_rayleigh(ior: f64, molecular_density: f64, wavelength: f64) -> f64 {
    8.0 * std::f64::consts::PI.powi(3) * (ior.powi(2) - 1.0).powi(2)
        / (3.0 * molecular_density * wavelength.powi(4))
}

impl Default for Params {
    fn default() -> Self {
        let r = compute_beta_rayleigh(IOR_AIR, DENSITY_AIR, LAMBDA_R) as f32;
        let g = compute_beta_rayleigh(IOR_AIR, DENSITY_AIR, LAMBDA_G) as f32;
        let b = compute_beta_rayleigh(IOR_AIR, DENSITY_AIR, LAMBDA_B) as f32;
        Self {
            h_atm: 80_000.0,
            r_planet: 6371e3,
            h_r: 8_000.0,
            h_m: 1_200.0,
            beta_r: [r, g, b],
            beta_m: 2e-5, // Fudged, per Bruneton
        }
    }
}

struct Pass {
    shader: vk::ShaderModule,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    ds_layout: vk::DescriptorSetLayout,
}

/// An atmosphere being prepared by the GPU
///
/// Must not be dropped before the `vk::CommandBuffer` passed to `Builder::build` has completed execution
pub struct PendingAtmosphere {
    device: Arc<Device>,
    inner: Option<Atmosphere>,
    descriptor_pool: vk::DescriptorPool,
    single_order: Image,
    gathered: Image,
    gathered_sum: Image,
    transmittance_lut: Image,
}

impl Drop for PendingAtmosphere {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for &image in &[
                &self.single_order,
                &self.gathered,
                &self.gathered_sum,
                &self.transmittance_lut,
            ] {
                self.device.destroy_image_view(image.view, None);
                self.device.destroy_image(image.handle, None);
                self.device.free_memory(image.memory, None);
            }
        }
    }
}

impl PendingAtmosphere {
    /// Call if precompute took place on a different queue family than that of the gfx queue that
    /// will be used for drawing
    ///
    /// `cmd` is the command buffer that this `Atmosphere` will be drawn with.
    pub unsafe fn acquire_ownership(
        &self,
        cmd: vk::CommandBuffer,
        compute_queue_family: u32,
        gfx_queue_family: u32,
    ) {
        debug_assert!(compute_queue_family != gfx_queue_family);
        self.device.cmd_pipeline_barrier(
            cmd,
            Default::default(),
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            Default::default(),
            &[],
            &[vk::BufferMemoryBarrier {
                dst_access_mask: vk::AccessFlags::UNIFORM_READ,
                src_queue_family_index: compute_queue_family,
                dst_queue_family_index: gfx_queue_family,
                buffer: self.inner.as_ref().unwrap().params,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            }],
            &[vk::ImageMemoryBarrier {
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                old_layout: vk::ImageLayout::GENERAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: compute_queue_family,
                dst_queue_family_index: gfx_queue_family,
                image: self.inner.as_ref().unwrap().scattering.handle,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            }],
        );
    }

    /// Call when the `vk::CommandBuffer` passed to `Builder::build` has completed execution
    pub unsafe fn assert_ready(mut self) -> Atmosphere {
        self.inner.take().unwrap()
    }
}

/// An atmosphere that's ready for rendering
///
/// As with any Vulkan object, this must not be dropped while in use by a rendering operation.
pub struct Atmosphere {
    device: Arc<Device>,
    scattering: Image,
    params: vk::Buffer,
    params_mem: vk::DeviceMemory,
}

impl Drop for Atmosphere {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image_view(self.scattering.view, None);
            self.device.destroy_image(self.scattering.handle, None);
            self.device.free_memory(self.scattering.memory, None);
            self.device.destroy_buffer(self.params, None);
            self.device.free_memory(self.params_mem, None);
        }
    }
}

pub struct Renderer {
    device: Arc<Device>,
    sampler: vk::Sampler,
    vert_shader: vk::ShaderModule,
    frag_shader: vk::ShaderModule,
    ds_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    descriptor_pool: vk::DescriptorPool,
    ds: vk::DescriptorSet,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.ds_layout, None);
            self.device.destroy_shader_module(self.vert_shader, None);
            self.device.destroy_shader_module(self.frag_shader, None);
            self.device.destroy_sampler(self.sampler, None);
        }
    }
}

const FULLSCREEN_VERT: &[u32] = include_glsl!("shaders/fullscreen.vert");
const INSIDE_FRAG: &[u32] = include_glsl!("shaders/inside.frag", debug);

impl Renderer {
    /// Construct an atmosphere renderer
    ///
    /// `inverse_z` indicates whether points distant from the camera have depth value 1.0
    pub fn new(
        device: Arc<Device>,
        cache: vk::PipelineCache,
        inverse_z: bool,
        render_pass: vk::RenderPass,
    ) -> Self {
        unsafe {
            let vert_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&FULLSCREEN_VERT),
                    None,
                )
                .unwrap();

            let frag_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&INSIDE_FRAG),
                    None,
                )
                .unwrap();

            let sampler = device
                .create_sampler(
                    &vk::SamplerCreateInfo {
                        min_filter: vk::Filter::LINEAR,
                        mag_filter: vk::Filter::LINEAR,
                        mipmap_mode: vk::SamplerMipmapMode::NEAREST,
                        address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                        border_color: vk::BorderColor::FLOAT_TRANSPARENT_BLACK,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();

            let ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            p_immutable_samplers: ptr::null(),
                        },
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            p_immutable_samplers: &sampler,
                        },
                    ]),
                    None,
                )
                .unwrap();

            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[ds_layout])
                        .push_constant_ranges(&[vk::PushConstantRange {
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            offset: 0,
                            size: mem::size_of::<DrawParams>() as u32,
                        }]),
                    None,
                )
                .unwrap();

            let entry_point = b"main\0".as_ptr() as *const i8;
            let noop_stencil_state = vk::StencilOpState {
                fail_op: vk::StencilOp::KEEP,
                pass_op: vk::StencilOp::KEEP,
                depth_fail_op: vk::StencilOp::KEEP,
                compare_op: vk::CompareOp::ALWAYS,
                compare_mask: 0,
                write_mask: 0,
                reference: 0,
            };
            let pipeline = device
                .create_graphics_pipelines(
                    cache,
                    &[vk::GraphicsPipelineCreateInfo::builder()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::VERTEX,
                                module: vert_shader,
                                p_name: entry_point,
                                p_specialization_info: &*vk::SpecializationInfo::builder()
                                    .data(&(inverse_z as u32 as f32).to_bits().to_ne_bytes())
                                    .map_entries(&[vk::SpecializationMapEntry {
                                        constant_id: 0,
                                        offset: 0,
                                        size: 4,
                                    }]),
                                ..Default::default()
                            },
                            vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::FRAGMENT,
                                module: frag_shader,
                                p_name: entry_point,
                                ..Default::default()
                            },
                        ])
                        .vertex_input_state(&Default::default())
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::builder()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::builder()
                                .scissor_count(1)
                                .viewport_count(1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::builder()
                                .cull_mode(vk::CullModeFlags::NONE)
                                .polygon_mode(vk::PolygonMode::FILL)
                                .line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::builder()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .depth_stencil_state(
                            &vk::PipelineDepthStencilStateCreateInfo::builder()
                                .depth_test_enable(true)
                                .depth_compare_op(if inverse_z {
                                    vk::CompareOp::GREATER_OR_EQUAL
                                } else {
                                    vk::CompareOp::LESS_OR_EQUAL
                                })
                                .front(noop_stencil_state)
                                .back(noop_stencil_state)
                                .max_depth_bounds(inverse_z as u32 as f32)
                                .min_depth_bounds((1 - inverse_z as u32) as f32),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::builder().attachments(&[
                                vk::PipelineColorBlendAttachmentState {
                                    blend_enable: vk::TRUE,
                                    src_color_blend_factor: vk::BlendFactor::ONE,
                                    dst_color_blend_factor: vk::BlendFactor::ONE,
                                    color_blend_op: vk::BlendOp::ADD,
                                    src_alpha_blend_factor: vk::BlendFactor::ONE,
                                    dst_alpha_blend_factor: vk::BlendFactor::ZERO,
                                    alpha_blend_op: vk::BlendOp::ADD,
                                    color_write_mask: vk::ColorComponentFlags::all(),
                                },
                            ]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(pipeline_layout)
                        .render_pass(render_pass)
                        .build()],
                    None,
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::builder()
                        .max_sets(1)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: 1,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                descriptor_count: 1,
                            },
                        ]),
                    None,
                )
                .unwrap();

            let ds = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[ds_layout]),
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            Self {
                device,
                sampler,
                vert_shader,
                frag_shader,
                ds_layout,
                pipeline_layout,
                pipeline,
                descriptor_pool,
                ds,
            }
        }
    }

    pub fn draw(
        &self,
        cmd: vk::CommandBuffer,
        atmosphere: &Atmosphere,
        params: &DrawParams,
        viewport: vk::Extent2D,
    ) {
        unsafe {
            self.device.update_descriptor_sets(
                &[
                    vk::WriteDescriptorSet {
                        dst_set: self.ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &vk::DescriptorBufferInfo {
                            buffer: atmosphere.params,
                            offset: 0,
                            range: vk::WHOLE_SIZE,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: self.ds,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: atmosphere.scattering.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                ],
                &[],
            );

            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.ds],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                &mem::transmute::<_, [u8; 92]>(*params),
            );
            // TODO: Double-check necessity
            self.device.cmd_set_viewport(
                cmd,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: viewport.width as f32,
                    height: viewport.height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            self.device.cmd_set_scissor(
                cmd,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: viewport,
                }],
            );
            self.device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct DrawParams {
    /// (projection * view)^-1
    pub inverse_viewproj: [[f32; 4]; 4],
    /// Unit vector from the center of the planet to the camera's origin
    pub zenith: [f32; 3],
    /// Altitude of the camera's origin above sea level
    pub height: f32,
    /// Unit vector towards the sun
    pub sun_direction: [f32; 3],
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
