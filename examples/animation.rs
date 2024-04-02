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

const CAMERA_Y: f32 = -8.0;
const MIN_FOCAL_LEN: f32 = 35.0;
const MAX_FOCAL_LEN: f32 = 3000.0;
const DEFAULT_FOCAL_LEN: f32 = 50.0;
const MIN_TARGET_Z: f32 = 0.0;
const MAX_TARGET_Z: f32 = 5.0;
const DEFAULT_TARGET_Z: f32 = 1.5;
const ZOOM_RATE: f32 = 1.2 * SIM_RATE;
const TILT_RATE: f32 = 1.0 * SIM_RATE;
const ORBIT_RATE: f32 = 1.2 * SIM_RATE;

struct CameraManager {
    focal_length: f32,
    orbit_angle: f32,
    target_z: f32,
    update_needed: bool,
}

impl CameraManager {
    fn zoom(&mut self, rate: f32) {
        self.focal_length = (self.focal_length.log2() + rate)
            .exp2()
            .clamp(MIN_FOCAL_LEN, MAX_FOCAL_LEN);
        self.update_needed = true;
    }

    fn tilt(&mut self, rate: f32) {
        self.target_z =
            (self.target_z + rate).clamp(MIN_TARGET_Z, MAX_TARGET_Z);
        self.update_needed = true;
    }

    fn orbit(&mut self, rate: f32) {
        self.orbit_angle =
            (self.orbit_angle - rate) % (2.0_f32 * std::f32::consts::PI);
        self.update_needed = true;
    }

    fn update(&mut self, camera: &mut Camera) {
        if self.update_needed {
            camera.position(&glm::rotate_vec3(
                &glm::vec3(0.0_f32, CAMERA_Y, 0.0_f32),
                self.orbit_angle,
                &glm::vec3(0.0_f32, 0.0_f32, 1.0_f32),
            ));
            camera.zoom(util::focal_length_to_fovy(self.focal_length));
            camera.target(&glm::vec3(0.0, 0.0, self.target_z));
            self.update_needed = false;
        }
    }
}

impl Default for CameraManager {
    fn default() -> Self {
        Self {
            focal_length: DEFAULT_FOCAL_LEN,
            orbit_angle: 0.0_f32,
            target_z: DEFAULT_TARGET_Z,
            update_needed: true,
        }
    }
}

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

    // Create a camera
    let mut camera = Camera::new(camera::Properties {
        aspect_ratio: WIDTH as f32 / HEIGHT as f32,
        ..Default::default()
    });
    let mut camera_manager = CameraManager::default();
    camera_manager.update(&mut camera);

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
                camera_manager.zoom(ZOOM_RATE);
            }
            if keyboard.is_pressed(VirtualKeyCode::X) {
                camera_manager.zoom(-ZOOM_RATE);
            }
            if keyboard.is_pressed(VirtualKeyCode::W) {
                camera_manager.tilt(TILT_RATE);
            }
            if keyboard.is_pressed(VirtualKeyCode::S) {
                camera_manager.tilt(-TILT_RATE);
            }
            if keyboard.is_pressed(VirtualKeyCode::A) {
                camera_manager.orbit(ORBIT_RATE);
            }
            if keyboard.is_pressed(VirtualKeyCode::D) {
                camera_manager.orbit(-ORBIT_RATE);
            }
            keyboard.tick();
            camera_manager.update(&mut camera);

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
            if !skeletons.is_empty() && !animations.is_empty() {
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
