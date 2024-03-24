//! Demo of egui integration
use egui::{
    epaint::Shadow, style::Margin, vec2, Align, Align2, Color32, Frame,
    Rounding,
};
use log::info;
use nalgebra_glm as glm;
use rhodora::{
    boss::*,
    camera,
    camera::Camera,
    egui_integration::{Gui, GuiConfig},
    keyboard::Keyboard,
    mesh_import::ImportOptions,
    pbr_lights::*,
    vk_window,
};
use std::{path::Path, time::Duration};
use vulkano::image::SampleCount;
use winit::{event::VirtualKeyCode, event_loop::EventLoop};

const BACKGROUND: [f32; 4] = [0.016, 0.016, 0.016, 1.0]; // Linear space
const FRAME_DURATION: Duration = Duration::from_micros(8200);
const FRAMES_IN_FLIGHT: usize = 2;
const SAMPLES: SampleCount = SampleCount::Sample4;
const WIDTH: u32 = 800;
const HEIGHT: u32 = 800;
const VSYNC: bool = true;
const SIM_RATE: f32 = 1.0 / 30.0;

pub fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();

    let mut boss = Boss::new(
        &event_loop,
        Some(vk_window::Properties {
            dimensions: [WIDTH, HEIGHT],
            title: "GUI".to_string(),
        }),
        SAMPLES,
        VSYNC,
        FRAMES_IN_FLIGHT,
    )
    .unwrap();

    let mut cbb = boss.create_primary_cbb().unwrap();

    // Load the model. Assume this example is ran by "cargo run --example"
    // from the project root.
    boss.load_mesh(
        &mut cbb,
        Path::new("./examples/cube.obj"),
        &ImportOptions::default(),
    )
    .unwrap();
    let transfer_future = boss.start_transfer(cbb).unwrap();
    let m = glm::translate(
        &glm::Mat4::identity(),
        &glm::vec3(0.0f32, 1.0f32, 0.0f32),
    );
    let m = glm::rotate_z(&m, 0.78f32);
    let m = glm::rotate_x(&m, 0.45f32);
    boss.model_manager.update(0, Some(m), Some(true));
    transfer_future.wait(None).unwrap();

    let mut gui = Gui::new(
        &event_loop,
        boss.surface().clone(),
        boss.graphics_queue().clone(),
        &GuiConfig {
            preferred_format: Some(vulkano::format::Format::B8G8R8A8_SRGB),
            ..Default::default()
        },
    )
    .unwrap();

    let mut keyboard = Keyboard::new();
    let mut camera = Camera::new(camera::Properties {
        position: glm::vec3(0.0f32, -10.0f32, 0.0f32),
        ..Default::default()
    });
    let lights = PbrLights::default();

    // Main loop
    boss.start_loop(SIM_RATE, Some(FRAME_DURATION));
    event_loop.run(move |event, _, control_flow| {
        // Events go first to egui for consideration
        if gui.update(&event) {
            // egui consumed the event so it was some window event and there
            // is nothing more to do?
            info!("egui consumed event");
            //return;
        }

        let (mut do_tick, do_render) =
            boss.handle_event(&event, &mut keyboard, control_flow);

        while do_tick.is_some() {
            // Process keys
            if keyboard.is_pressed(VirtualKeyCode::Escape) {
                control_flow.set_exit();
                return;
            }
            keyboard.tick();
            do_tick = boss.finish_tick(false);
        }

        if do_render {
            // Start rendering
            let action = boss.render_start().unwrap();
            match action {
                Action::Return => return,
                Action::Resize(dimensions) => {
                    camera.aspect_ratio(
                        dimensions.width as f32 / dimensions.height as f32,
                    );
                }
                _ => (),
            }

            // egui
            gui.immediate_ui(|gui| {
                let ctx = gui.context();
                egui::Window::new("Transparent Window")
                    .anchor(
                        Align2([Align::RIGHT, Align::TOP]),
                        vec2(-300.0, 300.0),
                    ) // anchor makes this widget immovable
                    .resizable(false)
                    .default_width(300.0)
                    .frame(
                        Frame::none()
                            .fill(Color32::from_white_alpha(125))
                            .shadow(Shadow {
                                extrusion: 8.0,
                                color: Color32::from_black_alpha(125),
                            })
                            .rounding(Rounding::same(5.0))
                            .inner_margin(Margin::same(10.0)),
                    )
                    .show(&ctx, |ui| {
                        ui.colored_label(Color32::BLACK, "Content :)");
                    });
                egui::SidePanel::left("left_panel").show(&ctx, |ui| {
                    ui.add(egui::widgets::Label::new("Hello, World?"));
                    if ui.button("Click Me").clicked() {
                        println!("Button clicked");
                    }
                });
            })
            .unwrap();

            // Do the actual rendering
            boss.render_all(BACKGROUND, &camera, &lights, Some(&mut gui))
                .unwrap();
        }
    }); // event_loop.run closure
}
