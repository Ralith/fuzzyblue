use std::{mem, ptr, sync::Arc};

use ash::version::{DeviceV1_0, InstanceV1_0};
use ash::{vk, Device, Instance};
use vk_shader_macros::include_glsl;

const TRANSMITTANCE: &[u32] = include_glsl!("shaders/transmittance.comp");
const SINGLE_SCATTERING: &[u32] = include_glsl!("shaders/single_scattering.comp");
const SCATTERING_DENSITY: &[u32] = include_glsl!("shaders/scattering_density.comp");
const MULTIPLE_SCATTERING: &[u32] = include_glsl!("shaders/multiple_scattering.comp");
const DIRECT_IRRADIANCE: &[u32] = include_glsl!("shaders/direct_irradiance.comp");
const INDIRECT_IRRADIANCE: &[u32] = include_glsl!("shaders/indirect_irradiance.comp");

/// Constructs `Atmosphere`s
pub struct Builder {
    device: Arc<Device>,
    memory_props: vk::PhysicalDeviceMemoryProperties,
    gfx_queue_family: u32,
    compute_queue_family: Option<u32>,
    sampler: vk::Sampler,
    params_ds_layout: vk::DescriptorSetLayout,
    render_ds_layout: vk::DescriptorSetLayout,
    frame_ds_layout: vk::DescriptorSetLayout,
    transmittance: Pass,
    single_scattering: Pass,
    direct_irradiance: Pass,
    indirect_irradiance: Pass,
    scattering_density: Pass,
    multiple_scattering: Pass,
}

impl Drop for Builder {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_sampler(self.sampler, None);
            self.device
                .destroy_descriptor_set_layout(self.params_ds_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.render_ds_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.frame_ds_layout, None);
            for &pass in &[
                &self.transmittance,
                &self.single_scattering,
                &self.direct_irradiance,
                &self.indirect_irradiance,
                &self.scattering_density,
                &self.multiple_scattering,
            ] {
                self.device.destroy_pipeline(pass.pipeline, None);
                self.device.destroy_pipeline_layout(pass.layout, None);
                self.device
                    .destroy_descriptor_set_layout(pass.ds_layout, None);
                self.device.destroy_shader_module(pass.shader, None);
            }
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
            let params_ds_layout = device
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
                        .set_layouts(&[params_ds_layout, transmittance_ds_layout]),
                    None,
                )
                .unwrap();
            let transmittance_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&TRANSMITTANCE),
                    None,
                )
                .unwrap();

            let direct_irradiance_ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        // transmittance
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // delta_irradiance
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                    ]),
                    None,
                )
                .unwrap();
            let direct_irradiance_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[params_ds_layout, direct_irradiance_ds_layout]),
                    None,
                )
                .unwrap();
            let direct_irradiance_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&DIRECT_IRRADIANCE),
                    None,
                )
                .unwrap();

            let indirect_irradiance_ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        // single_rayleigh
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // single_mie
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // multiple
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // delta_irradiance
                        vk::DescriptorSetLayoutBinding {
                            binding: 3,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                        // irradiance
                        vk::DescriptorSetLayoutBinding {
                            binding: 4,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                    ]),
                    None,
                )
                .unwrap();
            let indirect_irradiance_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[params_ds_layout, indirect_irradiance_ds_layout])
                        .push_constant_ranges(&[vk::PushConstantRange {
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            offset: 0,
                            size: 4,
                        }]),
                    None,
                )
                .unwrap();
            let indirect_irradiance_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&INDIRECT_IRRADIANCE),
                    None,
                )
                .unwrap();

            let scattering_ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        // transmittance
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // delta_rayleigh
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                        // delta_mie
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                        // scattering
                        vk::DescriptorSetLayoutBinding {
                            binding: 3,
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
                        .set_layouts(&[params_ds_layout, scattering_ds_layout]),
                    None,
                )
                .unwrap();
            let scattering_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&SINGLE_SCATTERING),
                    None,
                )
                .unwrap();

            let scattering_density_ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        // transmittance
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // single_rayleigh
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // single_mie
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // multiple_scattering
                        vk::DescriptorSetLayoutBinding {
                            binding: 3,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // irradiance
                        vk::DescriptorSetLayoutBinding {
                            binding: 4,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // scattering_density
                        vk::DescriptorSetLayoutBinding {
                            binding: 5,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                    ]),
                    None,
                )
                .unwrap();
            let scattering_density_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[params_ds_layout, scattering_density_ds_layout])
                        .push_constant_ranges(&[vk::PushConstantRange {
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            offset: 0,
                            size: 4,
                        }]),
                    None,
                )
                .unwrap();
            let scattering_density_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&SCATTERING_DENSITY),
                    None,
                )
                .unwrap();

            let multiple_scattering_ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        // transmittance
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // scattering_density
                        vk::DescriptorSetLayoutBinding {
                            binding: 1,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: &sampler,
                        },
                        // delta_multiple_scattering
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                        // scattering
                        vk::DescriptorSetLayoutBinding {
                            binding: 3,
                            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            p_immutable_samplers: ptr::null(),
                        },
                    ]),
                    None,
                )
                .unwrap();
            let multiple_scattering_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[params_ds_layout, multiple_scattering_ds_layout])
                        .push_constant_ranges(&[vk::PushConstantRange {
                            stage_flags: vk::ShaderStageFlags::COMPUTE,
                            offset: 0,
                            size: 4,
                        }]),
                    None,
                )
                .unwrap();
            let multiple_scattering_shader = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&MULTIPLE_SCATTERING),
                    None,
                )
                .unwrap();

            let render_ds_layout = device
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
                        vk::DescriptorSetLayoutBinding {
                            binding: 2,
                            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            p_immutable_samplers: &sampler,
                        },
                    ]),
                    None,
                )
                .unwrap();

            let frame_ds_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::builder().bindings(&[
                        vk::DescriptorSetLayoutBinding {
                            binding: 0,
                            descriptor_type: vk::DescriptorType::INPUT_ATTACHMENT,
                            descriptor_count: 1,
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            p_immutable_samplers: ptr::null(),
                        },
                    ]),
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
                                module: direct_irradiance_shader,
                                p_name,
                                ..Default::default()
                            },
                            layout: direct_irradiance_layout,
                            ..Default::default()
                        },
                        vk::ComputePipelineCreateInfo {
                            stage: vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::COMPUTE,
                                module: indirect_irradiance_shader,
                                p_name,
                                ..Default::default()
                            },
                            layout: indirect_irradiance_layout,
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
                                module: scattering_density_shader,
                                p_name,
                                ..Default::default()
                            },
                            layout: scattering_density_layout,
                            ..Default::default()
                        },
                        vk::ComputePipelineCreateInfo {
                            stage: vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::COMPUTE,
                                module: multiple_scattering_shader,
                                p_name,
                                ..Default::default()
                            },
                            layout: multiple_scattering_layout,
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
            let direct_irradiance = Pass {
                shader: direct_irradiance_shader,
                pipeline: pipelines.next().unwrap(),
                layout: direct_irradiance_layout,
                ds_layout: direct_irradiance_ds_layout,
            };
            let indirect_irradiance = Pass {
                shader: indirect_irradiance_shader,
                pipeline: pipelines.next().unwrap(),
                layout: indirect_irradiance_layout,
                ds_layout: indirect_irradiance_ds_layout,
            };
            let single_scattering = Pass {
                shader: scattering_shader,
                pipeline: pipelines.next().unwrap(),
                layout: scattering_layout,
                ds_layout: scattering_ds_layout,
            };
            let scattering_density = Pass {
                shader: scattering_density_shader,
                pipeline: pipelines.next().unwrap(),
                layout: scattering_density_layout,
                ds_layout: scattering_density_ds_layout,
            };
            let multiple_scattering = Pass {
                shader: multiple_scattering_shader,
                pipeline: pipelines.next().unwrap(),
                layout: multiple_scattering_layout,
                ds_layout: multiple_scattering_ds_layout,
            };
            debug_assert!(pipelines.next().is_none());

            Self {
                device,
                memory_props: instance.get_physical_device_memory_properties(physical),
                gfx_queue_family,
                compute_queue_family,
                sampler,
                params_ds_layout,
                render_ds_layout,
                frame_ds_layout,
                transmittance,
                direct_irradiance,
                indirect_irradiance,
                single_scattering,
                scattering_density,
                multiple_scattering,
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

    pub(crate) fn device(&self) -> &Arc<Device> {
        &self.device
    }

    pub(crate) fn render_ds_layout(&self) -> vk::DescriptorSetLayout {
        self.render_ds_layout
    }
    pub(crate) fn frame_ds_layout(&self) -> vk::DescriptorSetLayout {
        self.frame_ds_layout
    }
}

struct Image {
    handle: vk::Image,
    view: vk::ImageView,
    memory: vk::DeviceMemory,
}

/// A single layer of a `DensityProfile`
///
/// An atmosphere layer of width 'width', and whose density is defined as
///   'exp_term' * exp('exp_scale' * h) + 'linear_term' * h + 'constant_term',
/// clamped to [0,1], and where h is the altitude.
pub struct DensityProfileLayer {
    pub width: f32,
    pub exp_term: f32,
    pub exp_scale: f32,
    pub linear_term: f32,
    pub constant_term: f32,
}

/// A collection of `DensityProfileLayer`s
///
/// An atmosphere density profile made of several layers on top of each other
/// (from bottom to top). The width of the last layer is ignored, i.e. it always
/// extend to the top atmosphere boundary. The profile values vary between 0
/// (null density) to 1 (maximum density).
pub struct DensityProfile {
    pub layers: [DensityProfileLayer; 2],
}

/// Parameters governing generated skies
///
/// Distances in km.
///
/// # LUT dimensions
///
/// All values are encoded in various ways to achieve a useful distribution of precision.
///
/// - μ (mu): view angle from vertical
/// - μ_s (mu_s): sun angle from vertical
/// - r: distance from planet origin
/// - ν (nu): view angle from sun
pub struct Parameters {
    /// Extra usage flags for the generated look-up tables
    pub usage: vk::ImageUsageFlags,
    /// Stage mask for synchronizing precompute
    pub dst_stage_mask: vk::PipelineStageFlags,
    /// Access mask for synchronizing precompute
    pub dst_access_mask: vk::AccessFlags,
    /// Layout the look-up tables should end in
    pub layout: vk::ImageLayout,

    /// Number of light bounces to simulate
    pub order: u32,

    /// View angle precision for the transmittance look-up table
    pub transmittance_mu_size: u32,
    /// Height precision for the transmittance look-up table
    pub transmittance_r_size: u32,
    /// Height precision for the scattering look-up table
    pub scattering_r_size: u32,
    /// View angle precision for the scattering look-up table
    pub scattering_mu_size: u32,
    /// Sun angle precision for the scattering look-up table
    pub scattering_mu_s_size: u32,
    /// Sun azimuth precision for the scattering look-up table
    pub scattering_nu_size: u32,
    /// Sun angle precision for the lighting look-up table
    pub irradiance_mu_s_size: u32,
    /// Height precision for the lighting look-up table
    pub irradiance_r_size: u32,

    /// The solar irradiance at the top of the atmosphere.
    pub solar_irradiance: [f32; 3],
    /// The sun's angular radius. Warning: the implementation uses approximations
    /// that are valid only if this angle is smaller than 0.1 radians.
    pub sun_angular_radius: f32,
    /// The distance between the planet center and the bottom of the atmosphere.
    pub bottom_radius: f32,
    /// The distance between the planet center and the top of the atmosphere.
    pub top_radius: f32,
    /// The density profile of air molecules, i.e. a function from altitude to
    /// dimensionless values between 0 (null density) and 1 (maximum density).
    pub rayleigh_density: DensityProfile,
    /// The scattering coefficient of air molecules at the altitude where their
    /// density is maximum (usually the bottom of the atmosphere), as a function of
    /// wavelength. The scattering coefficient at altitude h is equal to
    /// 'rayleigh_scattering' times 'rayleigh_density' at this altitude.
    pub rayleigh_scattering: [f32; 3],
    /// The density profile of aerosols, i.e. a function from altitude to
    /// dimensionless values between 0 (null density) and 1 (maximum density).
    pub mie_density: DensityProfile,
    /// The scattering coefficient of aerosols at the altitude where their density
    /// is maximum (usually the bottom of the atmosphere), as a function of
    /// wavelength. The scattering coefficient at altitude h is equal to
    /// 'mie_scattering' times 'mie_density' at this altitude.
    pub mie_scattering: [f32; 3],
    /// The extinction coefficient of aerosols at the altitude where their density
    /// is maximum (usually the bottom of the atmosphere), as a function of
    /// wavelength. The extinction coefficient at altitude h is equal to
    /// 'mie_extinction' times 'mie_density' at this altitude.
    pub mie_extinction: [f32; 3],
    /// The asymetry parameter for the Cornette-Shanks phase function for the
    /// aerosols.
    pub mie_phase_function_g: f32,
    /// The density profile of air molecules that absorb light (e.g. ozone), i.e.
    /// a function from altitude to dimensionless values between 0 (null density)
    /// and 1 (maximum density).
    pub absorbtion_density: DensityProfile,
    /// The extinction coefficient of molecules that absorb light (e.g. ozone) at
    /// the altitude where their density is maximum, as a function of wavelength.
    /// The extinction coefficient at altitude h is equal to
    /// 'absorption_extinction' times 'absorption_density' at this altitude.
    pub absorbtion_extinction: [f32; 3],
    /// The average albedo of the ground.
    pub ground_albedo: [f32; 3],
    /// The cosine of the maximum Sun zenith angle for which atmospheric scattering
    /// must be precomputed (for maximum precision, use the smallest Sun zenith
    /// angle yielding negligible sky light radiance values. For instance, for the
    /// Earth case, 102 degrees is a good choice - yielding mu_s_min = -0.2).
    pub mu_s_min: f32,
}

impl Parameters {
    pub fn transmittance_extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.transmittance_mu_size,
            height: self.transmittance_r_size,
        }
    }

    pub fn irradiance_extent(&self) -> vk::Extent2D {
        vk::Extent2D {
            width: self.irradiance_mu_s_size,
            height: self.irradiance_r_size,
        }
    }

    pub fn scattering_extent(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.scattering_nu_size * self.scattering_mu_s_size,
            height: self.scattering_mu_size,
            depth: self.scattering_r_size,
        }
    }
}

// Taken from Bruneton's paper
// /// Wavelength of red light
// pub const LAMBDA_R: f32 = 680e-9;
// /// Wavelength of green light
// pub const LAMBDA_G: f32 = 550e-9;
// /// Wavelength of blue light
// pub const LAMBDA_B: f32 = 440e-9;

// /// Average index of refraction Earth's atmosphere, used to compute `Params::default().beta_r`
// pub const IOR_AIR: f32 = 1.0003;

// /// Number density of Earth's atmosphere at sea level (molecules/m^3)
// pub const DENSITY_AIR: f32 = 2.545e25;

// /// Extinction coefficients for ozone on Earth
// pub const OZONE_ABSORBTION_COEFFICIENT: [f32; 3] = [0.000650, 0.001881, 0.000085];

// /// Compute the Rayleigh scattering factor at a certain wavelength
// ///
// /// `ior` - index of refraction
// /// `molecular_density` - number of Rayleigh particles (i.e. molecules) per cubic m at sea level
// /// `wavelength` - wavelength to compute β_R for
// pub fn beta_rayleigh(ior: f32, molecular_density: f32, wavelength: f32) -> f32 {
//     8.0 * std::f32::consts::PI.powi(3) * (ior.powi(2) - 1.0).powi(2)
//         / (3.0 * molecular_density * wavelength.powi(4))
// }

// /// Compute the wavelength-independent Mie scattering factor
// ///
// /// `ior` - index of refraction of the aerosol particle
// /// `molecular_density` - number of Mie particles (i.e. aerosols) per cubic meter at sea level
// /// `wavelength` - wavelength to compute β_R for
// pub fn beta_mie(ior: f32, particle_density: f32) -> f32 {
//     8.0 * std::f32::consts::PI.powi(3) * (ior.powi(2) - 1.0).powi(2) / (3.0 * particle_density)
// }

// impl Default for Params {
//     fn default() -> Self {
//         // from Bruneton
//         let beta_m = 2.2e-5;
//         let beta_e_m = beta_m / 0.9;
//         Self {
//             h_atm: 80_000.0,
//             r_planet: 6371e3,
//             h_r: 8_000.0,
//             h_m: 1_200.0,
//             beta_r: [r, g, b],
//             beta_m,
//             beta_e_o: OZONE_EXTINCTION_COEFFICIENT,
//             beta_e_m,
//         }
//     }
// }

impl Default for Parameters {
    fn default() -> Self {
        Self {
            usage: vk::ImageUsageFlags::default(),
            dst_stage_mask: vk::PipelineStageFlags::FRAGMENT_SHADER,
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,

            order: 4,

            transmittance_mu_size: 256,
            transmittance_r_size: 64,
            scattering_r_size: 32,
            scattering_mu_size: 128,
            scattering_mu_s_size: 32,
            scattering_nu_size: 8,
            irradiance_mu_s_size: 64,
            irradiance_r_size: 16,

            solar_irradiance: [1.474, 1.850, 1.91198],
            sun_angular_radius: 0.004675,
            bottom_radius: 6360.0,
            top_radius: 6420.0,
            rayleigh_density: DensityProfile {
                layers: [
                    DensityProfileLayer {
                        width: 0.0,
                        exp_term: 0.0,
                        exp_scale: 0.0,
                        linear_term: 0.0,
                        constant_term: 0.0,
                    },
                    DensityProfileLayer {
                        width: 0.0,
                        exp_term: 1.0,
                        exp_scale: -0.125,
                        linear_term: 0.0,
                        constant_term: 0.0,
                    },
                ],
            },
            rayleigh_scattering: [0.005802, 0.013558, 0.033100],
            mie_density: DensityProfile {
                layers: [
                    DensityProfileLayer {
                        width: 0.0,
                        exp_term: 0.0,
                        exp_scale: 0.0,
                        linear_term: 0.0,
                        constant_term: 0.0,
                    },
                    DensityProfileLayer {
                        width: 0.0,
                        exp_term: 1.0,
                        exp_scale: -0.833333,
                        linear_term: 0.0,
                        constant_term: 0.0,
                    },
                ],
            },
            mie_scattering: [0.003996, 0.003996, 0.003996],
            mie_extinction: [0.004440, 0.004440, 0.004440],
            mie_phase_function_g: 0.8,
            absorbtion_density: DensityProfile {
                layers: [
                    DensityProfileLayer {
                        width: 25.0,
                        exp_term: 0.0,
                        exp_scale: 0.0,
                        linear_term: 0.066667,
                        constant_term: -0.666667,
                    },
                    DensityProfileLayer {
                        width: 0.0,
                        exp_term: 0.0,
                        exp_scale: 0.0,
                        linear_term: -0.066667,
                        constant_term: 2.666667,
                    },
                ],
            },
            absorbtion_extinction: [6.5e-4, 1.881e-3, 8.5e-5],
            ground_albedo: [0.1, 0.1, 0.1],
            mu_s_min: -0.207912,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct ParamsRaw {
    solar_irradiance: [f32; 3],
    sun_angular_radius: f32,
    rayleigh_scattering: [f32; 3],
    bottom_radius: f32,
    mie_scattering: [f32; 3],
    top_radius: f32,
    mie_extinction: [f32; 3],
    mie_phase_function_g: f32,
    ground_albedo: [f32; 3],
    mu_s_min: f32,
    absorbtion_extinction: [f32; 3],

    transmittance_mu_size: u32,
    transmittance_r_size: u32,
    scattering_r_size: u32,
    scattering_mu_size: u32,
    scattering_mu_s_size: u32,
    scattering_nu_size: u32,
    irradiance_mu_s_size: u32,
    irradiance_r_size: u32,

    rayleigh_density: DensityProfileRaw,
    mie_density: DensityProfileRaw,
    absorbtion_density: DensityProfileRaw,
}

impl ParamsRaw {
    fn new(x: &Parameters) -> Self {
        Self {
            solar_irradiance: x.solar_irradiance,
            sun_angular_radius: x.sun_angular_radius,
            rayleigh_scattering: x.rayleigh_scattering,
            bottom_radius: x.bottom_radius,
            mie_scattering: x.mie_scattering,
            top_radius: x.top_radius,
            mie_extinction: x.mie_extinction,
            mie_phase_function_g: x.mie_phase_function_g,
            ground_albedo: x.ground_albedo,
            mu_s_min: x.mu_s_min,
            absorbtion_extinction: x.absorbtion_extinction,
            transmittance_mu_size: x.transmittance_mu_size,
            transmittance_r_size: x.transmittance_r_size,
            scattering_r_size: x.scattering_r_size,
            scattering_mu_size: x.scattering_mu_size,
            scattering_mu_s_size: x.scattering_mu_s_size,
            scattering_nu_size: x.scattering_nu_size,
            irradiance_mu_s_size: x.irradiance_mu_s_size,
            irradiance_r_size: x.irradiance_r_size,
            rayleigh_density: DensityProfileRaw::new(&x.rayleigh_density),
            mie_density: DensityProfileRaw::new(&x.mie_density),
            absorbtion_density: DensityProfileRaw::new(&x.absorbtion_density),
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct DensityProfileRaw {
    layers: [DensityProfileLayerRaw; 2],
}

impl DensityProfileRaw {
    fn new(x: &DensityProfile) -> Self {
        Self {
            layers: [
                DensityProfileLayerRaw::new(&x.layers[0]),
                DensityProfileLayerRaw::new(&x.layers[1]),
            ],
        }
    }
}

#[repr(C)]
#[repr(align(16))]
#[derive(Copy, Clone)]
struct DensityProfileLayerRaw {
    width: f32,
    exp_term: f32,
    exp_scale: f32,
    linear_term: f32,
    constant_term: f32,
}

impl DensityProfileLayerRaw {
    fn new(x: &DensityProfileLayer) -> Self {
        Self {
            width: x.width,
            exp_term: x.exp_term,
            exp_scale: x.exp_scale,
            linear_term: x.linear_term,
            constant_term: x.constant_term,
        }
    }
}

struct Pass {
    shader: vk::ShaderModule,
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
    ds_layout: vk::DescriptorSetLayout,
}

/// An atmosphere that's ready for rendering
///
/// As with any Vulkan object, this must not be dropped while in use by a rendering operation.
pub struct Atmosphere {
    builder: Arc<Builder>,
    descriptor_pool: vk::DescriptorPool,
    ds: vk::DescriptorSet,
    transmittance: Image,
    transmittance_extent: vk::Extent2D,
    scattering: Image,
    scattering_extent: vk::Extent3D,
    irradiance: Image,
    irradiance_extent: vk::Extent2D,
    params: vk::Buffer,
    params_mem: vk::DeviceMemory,
}

impl Drop for Atmosphere {
    fn drop(&mut self) {
        let device = &*self.builder.device;
        unsafe {
            for &image in &[&self.transmittance, &self.scattering, &self.irradiance] {
                device.destroy_image_view(image.view, None);
                device.destroy_image(image.handle, None);
                device.free_memory(image.memory, None);
            }
            device.destroy_buffer(self.params, None);
            device.free_memory(self.params_mem, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

impl Atmosphere {
    /// Build an `Atmosphere` that will be usable when `cmd` is fully executed.
    pub fn build(
        builder: Arc<Builder>,
        cmd: vk::CommandBuffer,
        atmosphere_params: &Parameters,
    ) -> PendingAtmosphere {
        let device = &*builder.device;
        unsafe {
            // common: 1 uniform
            // transmittance: 1 storage image
            // direct irradiance: 1 image-sampler, 1 storage image
            // indirect irradiance: 3 image-samplers, 2 storage images
            // single scattering: 1 image-sampler, 3 storage images
            // scattering density: 5 image-samplers, 1 storage image
            // multiple scattering: 2 image-samplers, 2 storage images
            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::builder()
                        .max_sets(7)
                        .pool_sizes(&[
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::UNIFORM_BUFFER,
                                descriptor_count: 1,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                                descriptor_count: 12,
                            },
                            vk::DescriptorPoolSize {
                                ty: vk::DescriptorType::STORAGE_IMAGE,
                                descriptor_count: 10,
                            },
                        ]),
                    None,
                )
                .unwrap();

            let mut descriptor_sets = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[
                            builder.params_ds_layout,
                            builder.transmittance.ds_layout,
                            builder.direct_irradiance.ds_layout,
                            builder.indirect_irradiance.ds_layout,
                            builder.single_scattering.ds_layout,
                            builder.scattering_density.ds_layout,
                            builder.multiple_scattering.ds_layout,
                        ]),
                )
                .unwrap()
                .into_iter();
            let params_ds = descriptor_sets.next().unwrap();
            let transmittance_ds = descriptor_sets.next().unwrap();
            let direct_irradiance_ds = descriptor_sets.next().unwrap();
            let indirect_irradiance_ds = descriptor_sets.next().unwrap();
            let single_scattering_ds = descriptor_sets.next().unwrap();
            let scattering_density_ds = descriptor_sets.next().unwrap();
            let multiple_scattering_ds = descriptor_sets.next().unwrap();
            debug_assert!(descriptor_sets.next().is_none());

            let persistent_pool = device
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
                                descriptor_count: 2,
                            },
                        ]),
                    None,
                )
                .unwrap();

            let render_ds = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::builder()
                        .descriptor_pool(persistent_pool)
                        .set_layouts(&[builder.render_ds_layout]),
                )
                .unwrap()
                .into_iter()
                .next()
                .unwrap();

            let transmittance_extent = atmosphere_params.transmittance_extent();
            let transmittance = builder.alloc_image(&vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R32G32B32A32_SFLOAT,
                extent: vk::Extent3D {
                    width: transmittance_extent.width,
                    height: transmittance_extent.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | atmosphere_params.usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            });

            let irradiance_extent = atmosphere_params.irradiance_extent();
            let irradiance_image_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R32G32B32A32_SFLOAT,
                extent: vk::Extent3D {
                    width: irradiance_extent.width,
                    height: irradiance_extent.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | atmosphere_params.usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            let delta_irradiance = builder.alloc_image(&irradiance_image_info);
            let irradiance = builder.alloc_image(&irradiance_image_info);

            let scattering_extent = atmosphere_params.scattering_extent();
            let scattering_image_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_3D,
                format: vk::Format::R16G16B16A16_SFLOAT,
                extent: scattering_extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::SAMPLED
                    | atmosphere_params.usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };
            // TODO: These could be merged
            let delta_rayleigh = builder.alloc_image(&scattering_image_info);
            let delta_mie = builder.alloc_image(&scattering_image_info);
            let scattering = builder.alloc_image(&scattering_image_info);
            // TODO: This could overlap with delta_rayleigh/mie, since they are not used simultaneously
            let delta_multiple_scattering = builder.alloc_image(&scattering_image_info);
            let scattering_density = builder.alloc_image(&scattering_image_info);

            let params = device
                .create_buffer(
                    &vk::BufferCreateInfo {
                        size: mem::size_of::<ParamsRaw>() as vk::DeviceSize,
                        usage: vk::BufferUsageFlags::UNIFORM_BUFFER
                            | vk::BufferUsageFlags::TRANSFER_DST,
                        ..Default::default()
                    },
                    None,
                )
                .unwrap();
            let params_mem = {
                let reqs = device.get_buffer_memory_requirements(params);
                allocate(
                    device,
                    &builder.memory_props,
                    reqs,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )
                .unwrap()
            };
            device.bind_buffer_memory(params, params_mem, 0).unwrap();

            device.update_descriptor_sets(
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
                            image_view: transmittance.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: direct_irradiance_ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: transmittance.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: direct_irradiance_ds,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_irradiance.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: single_scattering_ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: transmittance.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: single_scattering_ds,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_rayleigh.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: single_scattering_ds,
                        dst_binding: 2,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_mie.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: single_scattering_ds,
                        dst_binding: 3,
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
                        dst_set: scattering_density_ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: transmittance.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: scattering_density_ds,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_rayleigh.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: scattering_density_ds,
                        dst_binding: 2,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_mie.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: scattering_density_ds,
                        dst_binding: 3,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_multiple_scattering.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: scattering_density_ds,
                        dst_binding: 4,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_irradiance.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: scattering_density_ds,
                        dst_binding: 5,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: scattering_density.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: indirect_irradiance_ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_rayleigh.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: indirect_irradiance_ds,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_mie.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: indirect_irradiance_ds,
                        dst_binding: 2,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_multiple_scattering.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: indirect_irradiance_ds,
                        dst_binding: 3,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_irradiance.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: indirect_irradiance_ds,
                        dst_binding: 4,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: irradiance.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: multiple_scattering_ds,
                        dst_binding: 0,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: transmittance.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: multiple_scattering_ds,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: scattering_density.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: multiple_scattering_ds,
                        dst_binding: 2,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: delta_multiple_scattering.view,
                            image_layout: vk::ImageLayout::GENERAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: multiple_scattering_ds,
                        dst_binding: 3,
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
                        dst_set: render_ds,
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
                        dst_set: render_ds,
                        dst_binding: 1,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: transmittance.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        },
                        ..Default::default()
                    },
                    vk::WriteDescriptorSet {
                        dst_set: render_ds,
                        dst_binding: 2,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                        p_image_info: &vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: scattering.view,
                            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
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

            device.cmd_update_buffer(
                cmd,
                params,
                0,
                &mem::transmute::<_, [u8; 320]>(ParamsRaw::new(atmosphere_params)),
            );
            device.cmd_pipeline_barrier(
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
                        image: transmittance.handle,
                        ..init_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: delta_rayleigh.handle,
                        ..init_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: delta_mie.handle,
                        ..init_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: scattering.handle,
                        ..init_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: irradiance.handle,
                        new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        ..init_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: delta_irradiance.handle,
                        ..init_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: delta_multiple_scattering.handle,
                        ..init_barrier
                    },
                ],
            );

            // Transmittance
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                builder.transmittance.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                builder.transmittance.layout,
                0,
                &[params_ds, transmittance_ds],
                &[],
            );
            device.cmd_dispatch(
                cmd,
                transmittance_extent.width,
                transmittance_extent.height,
                1,
            );

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                Default::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    image: transmittance.handle,
                    ..write_read_barrier
                }],
            );

            // Direct irradiance
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                builder.direct_irradiance.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                builder.direct_irradiance.layout,
                1,
                &[direct_irradiance_ds],
                &[],
            );
            device.cmd_dispatch(cmd, irradiance_extent.width, irradiance_extent.height, 1);

            // Single scattering
            device.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                builder.single_scattering.pipeline,
            );
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                builder.single_scattering.layout,
                1,
                &[single_scattering_ds],
                &[],
            );
            device.cmd_dispatch(
                cmd,
                scattering_extent.width,
                scattering_extent.height,
                scattering_extent.depth,
            );

            device.cmd_clear_color_image(
                cmd,
                irradiance.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            );

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                Default::default(),
                &[],
                &[],
                &[vk::ImageMemoryBarrier {
                    image: irradiance.handle,
                    src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                    old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    ..write_barrier
                }],
            );

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                Default::default(),
                &[],
                &[],
                &[
                    vk::ImageMemoryBarrier {
                        image: delta_rayleigh.handle,
                        ..write_read_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: delta_mie.handle,
                        ..write_read_barrier
                    },
                ],
            );

            // Compute higher-order effects
            for order in 2..=atmosphere_params.order {
                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    Default::default(),
                    &[],
                    &[],
                    &[
                        vk::ImageMemoryBarrier {
                            image: scattering_density.handle,
                            src_access_mask: vk::AccessFlags::SHADER_READ,
                            ..init_barrier
                        },
                        vk::ImageMemoryBarrier {
                            image: delta_irradiance.handle,
                            ..write_read_barrier
                        },
                        vk::ImageMemoryBarrier {
                            image: delta_multiple_scattering.handle,
                            ..write_read_barrier
                        },
                    ],
                );

                // Scattering density
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    builder.scattering_density.pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    builder.scattering_density.layout,
                    0,
                    &[params_ds, scattering_density_ds],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    builder.scattering_density.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &order.to_ne_bytes(),
                );
                device.cmd_dispatch(
                    cmd,
                    scattering_extent.width,
                    scattering_extent.height,
                    scattering_extent.depth,
                );

                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    Default::default(),
                    &[],
                    &[],
                    &[
                        // Scattering density reads this
                        vk::ImageMemoryBarrier {
                            image: delta_irradiance.handle,
                            ..read_write_barrier
                        },
                        // Previous irradiance pass output must be written
                        vk::ImageMemoryBarrier {
                            image: irradiance.handle,
                            ..write_barrier
                        },
                    ],
                );

                // Indirect irradiance
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    builder.indirect_irradiance.pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    builder.indirect_irradiance.layout,
                    0,
                    &[params_ds, indirect_irradiance_ds],
                    &[],
                );
                device.cmd_push_constants(
                    cmd,
                    builder.indirect_irradiance.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &(order - 1).to_ne_bytes(),
                );
                device.cmd_dispatch(cmd, irradiance_extent.width, irradiance_extent.height, 1);

                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    Default::default(),
                    &[],
                    &[],
                    &[
                        vk::ImageMemoryBarrier {
                            image: scattering_density.handle,
                            ..write_read_barrier
                        },
                        vk::ImageMemoryBarrier {
                            image: scattering.handle,
                            ..write_barrier
                        },
                        vk::ImageMemoryBarrier {
                            image: delta_multiple_scattering.handle,
                            src_access_mask: vk::AccessFlags::SHADER_READ,
                            ..init_barrier
                        },
                    ],
                );

                // Multiscattering
                device.cmd_bind_pipeline(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    builder.multiple_scattering.pipeline,
                );
                device.cmd_bind_descriptor_sets(
                    cmd,
                    vk::PipelineBindPoint::COMPUTE,
                    builder.multiple_scattering.layout,
                    0,
                    &[params_ds, multiple_scattering_ds],
                    &[],
                );
                device.cmd_dispatch(
                    cmd,
                    scattering_extent.width,
                    scattering_extent.height,
                    scattering_extent.depth,
                );
            }

            // Finalize layouts and transfer to graphics queue
            let src_queue_family_index = builder
                .compute_queue_family
                .unwrap_or(builder.gfx_queue_family);
            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                atmosphere_params.dst_stage_mask,
                Default::default(),
                &[],
                &[vk::BufferMemoryBarrier {
                    src_access_mask: vk::AccessFlags::UNIFORM_READ,
                    src_queue_family_index,
                    dst_queue_family_index: builder.gfx_queue_family,
                    buffer: params,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                    ..Default::default()
                }],
                &[
                    vk::ImageMemoryBarrier {
                        image: scattering.handle,
                        dst_access_mask: atmosphere_params.dst_access_mask,
                        new_layout: atmosphere_params.layout,
                        src_queue_family_index,
                        dst_queue_family_index: builder.gfx_queue_family,
                        ..write_read_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: irradiance.handle,
                        dst_access_mask: atmosphere_params.dst_access_mask,
                        new_layout: atmosphere_params.layout,
                        src_queue_family_index,
                        dst_queue_family_index: builder.gfx_queue_family,
                        ..write_read_barrier
                    },
                    vk::ImageMemoryBarrier {
                        image: transmittance.handle,
                        src_access_mask: vk::AccessFlags::default(),
                        dst_access_mask: atmosphere_params.dst_access_mask,
                        old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                        new_layout: atmosphere_params.layout,
                        src_queue_family_index,
                        dst_queue_family_index: builder.gfx_queue_family,
                        ..write_read_barrier
                    },
                ],
            );

            PendingAtmosphere {
                device: builder.device.clone(),
                descriptor_pool,
                inner: Some(Self {
                    builder,
                    descriptor_pool: persistent_pool,
                    ds: render_ds,
                    transmittance,
                    transmittance_extent,
                    scattering,
                    scattering_extent,
                    irradiance,
                    irradiance_extent,
                    params,
                    params_mem,
                }),
                delta_irradiance,
                delta_mie,
                delta_rayleigh,
                scattering_density,
                delta_multiple_scattering,
            }
        }
    }

    pub fn transmittance(&self) -> vk::Image {
        self.transmittance.handle
    }
    pub fn transmittance_view(&self) -> vk::ImageView {
        self.transmittance.view
    }
    pub fn transmittance_extent(&self) -> vk::Extent2D {
        self.transmittance_extent
    }
    pub fn scattering(&self) -> vk::Image {
        self.scattering.handle
    }
    pub fn scattering_view(&self) -> vk::ImageView {
        self.scattering.view
    }
    pub fn scattering_extent(&self) -> vk::Extent3D {
        self.scattering_extent
    }
    pub fn irradiance(&self) -> vk::Image {
        self.irradiance.handle
    }
    pub fn irradiance_view(&self) -> vk::ImageView {
        self.irradiance.view
    }
    pub fn irradiance_extent(&self) -> vk::Extent2D {
        self.irradiance_extent
    }

    pub(crate) fn descriptor_set(&self) -> vk::DescriptorSet {
        self.ds
    }
}

/// An atmosphere being prepared by the GPU
///
/// Must not be dropped before the `vk::CommandBuffer` passed to `Builder::build` has completed execution
pub struct PendingAtmosphere {
    device: Arc<Device>,
    descriptor_pool: vk::DescriptorPool,
    inner: Option<Atmosphere>,
    delta_irradiance: Image,
    delta_rayleigh: Image,
    delta_mie: Image,
    scattering_density: Image,
    delta_multiple_scattering: Image,
}

impl Drop for PendingAtmosphere {
    fn drop(&mut self) {
        unsafe {
            for &image in &[
                &self.delta_irradiance,
                &self.delta_rayleigh,
                &self.delta_mie,
                &self.scattering_density,
                &self.delta_multiple_scattering,
            ] {
                self.device.destroy_image_view(image.view, None);
                self.device.destroy_image(image.handle, None);
                self.device.free_memory(image.memory, None);
            }
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}

impl PendingAtmosphere {
    /// Call if precompute completed on a different queue family than that of the gfx queue that
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
        let inner = self.inner.as_ref().unwrap();
        let barrier = vk::ImageMemoryBarrier {
            dst_access_mask: vk::AccessFlags::SHADER_READ,
            old_layout: vk::ImageLayout::GENERAL,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            src_queue_family_index: compute_queue_family,
            dst_queue_family_index: gfx_queue_family,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };
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
                buffer: inner.params,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            }],
            &[
                vk::ImageMemoryBarrier {
                    image: inner.transmittance.handle,
                    old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    ..barrier
                },
                vk::ImageMemoryBarrier {
                    image: inner.scattering.handle,
                    ..barrier
                },
                vk::ImageMemoryBarrier {
                    image: inner.irradiance.handle,
                    ..barrier
                },
            ],
        );
    }

    /// Access the `Atmosphere` while it may not yet be ready
    pub unsafe fn atmosphere(&self) -> &Atmosphere {
        self.inner.as_ref().unwrap()
    }

    /// Call when the `vk::CommandBuffer` passed to `Builder::build` has completed execution
    pub unsafe fn assert_ready(mut self) -> Atmosphere {
        self.inner.take().unwrap()
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
