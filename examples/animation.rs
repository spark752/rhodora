//! Demo of animation using rhodora
use nalgebra_glm as glm;
use rhodora::{
    boss::*,
    camera::{self, Camera},
    egui_integration::Gui,
    keyboard::Keyboard,
    mesh_import::ImportOptions,
    model_manager::JointTransforms,
    pbr_lights::PbrLights,
    util, vk_window,
};
use std::{path::Path, time::Duration};
use vulkano::image::SampleCount;
use winit::{event::VirtualKeyCode, event_loop::EventLoop};

const BACKGROUND: [f32; 4] = [0.016, 0.016, 0.016, 1.0]; // Linear space
const FRAME_DURATION: Duration = Duration::from_micros(5000);
const FRAMES_IN_FLIGHT: usize = 2;
const SAMPLES: SampleCount = SampleCount::Sample4;
const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;
const VSYNC: bool = false;
const FILENAME: &str = "./examples/assets/slenderman.gltf";
const SIM_RATE: f32 = 1.0 / 30.0;
const MIN_FOCAL_LEN: f32 = 35.0;
const MAX_FOCAL_LEN: f32 = 3000.0;
const DEFAULT_FOCAL_LEN: f32 = 50.0;
const MIN_TARGET_Z: f32 = 0.0;
const MAX_TARGET_Z: f32 = 5.0;
const ZOOM_RATE: f32 = 1.2 * SIM_RATE;
const TILT_RATE: f32 = 1.0 * SIM_RATE;

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let file_path = if args.len() < 2 {
        FILENAME.to_string()
    } else {
        args[1].clone()
    };

    let event_loop = EventLoop::new();

    let mut boss = Boss::new(
        &event_loop,
        Some(vk_window::Properties {
            dimensions: [WIDTH, HEIGHT],
            title: "Anim".to_string(),
        }),
        SAMPLES,
        VSYNC,
        FRAMES_IN_FLIGHT,
    )
    .unwrap();

    // Create a default camera
    let mut focal_length: f32 = DEFAULT_FOCAL_LEN;
    let mut target_z: f32 = 1.5;
    let mut update_camera: bool = false;
    let mut camera = Camera::new(camera::Properties {
        aspect_ratio: WIDTH as f32 / HEIGHT as f32,
        fovy: util::focal_length_to_fovy(focal_length),
        position: glm::vec3(0.0, -8.0, target_z),
        target: glm::vec3(0.0, 0.0, target_z),
    });

    // Get a command buffer used for transfering data to the GPU
    let mut cbb = boss.create_primary_cbb().unwrap();

    // Load the model
    boss.load_model(&mut cbb, Path::new(&file_path), &ImportOptions::default())
        .unwrap();
    boss.model_manager
        .update(0, Some(glm::Mat4::identity()), Some(true));

    // Other objects could be created here so that they could all be transfered
    // in one queue execution. Any other one time commands could be recorded to
    // cbb as well.
    // There is nothing else to do in this case, so build and execute the
    // command buffer:
    let transfer_future = boss.start_transfer(cbb).unwrap();

    // Test some animation things
    let (skeletons, animations) =
        rhodora::mesh_import::gltf_file::load_animations(Path::new(&file_path))
            .unwrap();

    // Now wait for the future that shows the GPU finished the earlier transfers
    // and signalled the fence. This means the data needed for rendering should
    // be ready.
    transfer_future.wait(None).unwrap();

    // Keypress things
    let mut keyboard = Keyboard::new();

    // For this app just try the default lights
    let lights = PbrLights {
        ..Default::default()
    };

    // Main loop
    boss.start_loop(SIM_RATE, Some(FRAME_DURATION));
    event_loop.run(move |event, _, control_flow| {
        // Run continually even without receiving OS events. For non game type
        // applications set_wait() would be a better choice.
        //control_flow.set_poll();
        let (mut do_tick, do_render) =
            boss.handle_event(&event, &mut keyboard, control_flow);

        while do_tick.is_some() {
            // Process keys
            if keyboard.is_pressed(VirtualKeyCode::Escape) {
                control_flow.set_exit();
                return;
            }
            if keyboard.is_pressed(VirtualKeyCode::Z) {
                focal_length = (focal_length.log2() + ZOOM_RATE).exp2();
                update_camera = true;
            }
            if keyboard.is_pressed(VirtualKeyCode::X) {
                focal_length = (focal_length.log2() - ZOOM_RATE).exp2();
                update_camera = true;
            }
            if keyboard.is_pressed(VirtualKeyCode::W) {
                target_z += TILT_RATE;
                update_camera = true;
            }
            if keyboard.is_pressed(VirtualKeyCode::S) {
                target_z -= TILT_RATE;
                update_camera = true;
            }
            keyboard.tick();

            if update_camera {
                focal_length = focal_length.clamp(MIN_FOCAL_LEN, MAX_FOCAL_LEN);
                camera.zoom(util::focal_length_to_fovy(focal_length));
                target_z = target_z.clamp(MIN_TARGET_Z, MAX_TARGET_Z);
                camera.target(&glm::vec3(0.0, 0.0, target_z));
                update_camera = false;
            }

            // Finish this tick and do another if necessary
            do_tick = boss.finish_tick(false);
        }

        if do_render {
            // Do this first so on `Action::Return` we won't do anything else
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

            // Animation stuff
            if let Some(elapsed) = boss.elapsed() {
                let time = elapsed.as_secs_f32();
                let modu = time % (animations[0].max_time + 0.2_f32);

                let mut bone_stuff = JointTransforms::default();
                rhodora::animation::animate(
                    &skeletons[0],
                    &animations[0],
                    &mut bone_stuff.0,
                    modu,
                );
                boss.model_manager.update_joints(0, bone_stuff);
            }

            // Do the actual rendering. The `None` parameter indicates that
            // we don't have a GUI to draw, but Rust makes us provide a
            // type that implements the `GuiTrait` that would be used if we
            // did have one, which is rather annoying.
            boss.render_all(BACKGROUND, &camera, &lights, None::<&mut Gui>)
                .unwrap();
        }
    }); // event_loop.run closure
}
