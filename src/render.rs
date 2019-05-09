use std::{mem, sync::Arc};

use ash::version::DeviceV1_0;
use ash::{vk, Device};
use vk_shader_macros::include_glsl;

use crate::{Atmosphere, Builder};

const FULLSCREEN: &[u32] = include_glsl!("shaders/fullscreen.vert");
const RENDER_SKY: &[u32] = include_glsl!("shaders/render_sky.frag");

// TODO: Rasterize icospheres rather than raytracing
pub struct Renderer {
    device: Arc<Device>,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl Drop for Renderer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}

impl Renderer {
    /// Construct an atmosphere renderer
    pub fn new(
        builder: &Builder,
        cache: vk::PipelineCache,
        render_pass: vk::RenderPass,
        subpass: u32,
    ) -> Self {
        let device = builder.device().clone();
        unsafe {
            let vert = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&FULLSCREEN),
                    None,
                )
                .unwrap();

            let frag = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::builder().code(&RENDER_SKY),
                    None,
                )
                .unwrap();

            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::builder()
                        .set_layouts(&[builder.render_ds_layout()])
                        .push_constant_ranges(&[vk::PushConstantRange {
                            stage_flags: vk::ShaderStageFlags::FRAGMENT,
                            offset: 0,
                            size: mem::size_of::<DrawParamsRaw>() as u32,
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
            let mut pipelines = device
                .create_graphics_pipelines(
                    cache,
                    &[vk::GraphicsPipelineCreateInfo::builder()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::VERTEX,
                                module: vert,
                                p_name: entry_point,
                                ..Default::default()
                            },
                            vk::PipelineShaderStageCreateInfo {
                                stage: vk::ShaderStageFlags::FRAGMENT,
                                module: frag,
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
                                .depth_test_enable(false)
                                .front(noop_stencil_state)
                                .back(noop_stencil_state),
                        )
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::builder().attachments(&[
                                vk::PipelineColorBlendAttachmentState {
                                    blend_enable: vk::TRUE,
                                    src_color_blend_factor: vk::BlendFactor::ONE,
                                    dst_color_blend_factor: vk::BlendFactor::SRC1_COLOR,
                                    color_blend_op: vk::BlendOp::ADD,
                                    src_alpha_blend_factor: vk::BlendFactor::ZERO,
                                    dst_alpha_blend_factor: vk::BlendFactor::ONE,
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
                        .subpass(subpass)
                        .build()],
                    None,
                )
                .unwrap()
                .into_iter();

            device.destroy_shader_module(vert, None);
            device.destroy_shader_module(frag, None);

            let pipeline = pipelines.next().unwrap();

            Self {
                device,
                pipeline_layout,
                pipeline,
            }
        }
    }

    pub fn draw(&self, cmd: vk::CommandBuffer, atmosphere: &Atmosphere, params: &DrawParameters) {
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[atmosphere.descriptor_set()],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                &mem::transmute::<_, [u8; 92]>(DrawParamsRaw::new(params)),
            );
            self.device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }
}

/// Rendering parameters for an individual frame
///
/// All coordinates are in the planet's reference frame.
#[derive(Debug, Copy, Clone)]
pub struct DrawParameters {
    /// (projection * view)^-1
    pub inverse_viewproj: [[f32; 4]; 4],
    pub camera_position: [f32; 3],
    pub sun_direction: [f32; 3],
}

#[repr(C)]
struct DrawParamsRaw {
    inverse_viewproj: [[f32; 4]; 4],
    camera_position: [f32; 3],
    _padding: u32,
    sun_direction: [f32; 3],
}

impl DrawParamsRaw {
    fn new(x: &DrawParameters) -> Self {
        Self {
            inverse_viewproj: x.inverse_viewproj,
            camera_position: x.camera_position,
            _padding: 0,
            sun_direction: x.sun_direction,
        }
    }
}
