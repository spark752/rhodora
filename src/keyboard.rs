use crate::types::KeyboardHandler;
use winit::event::{ElementState, VirtualKeyCode};

/// One possible implementation for keyboard handling. The event handler only
/// uses the `KeyboardHandler` trait to register events and the rest is used
/// by application code.
///
/// `VirtualKeyCode` is the symbolic name a keyboard key. It should not depend
/// on keyboard layout, so Q should be Q no matter where the Q is.
/// The physical key is represented by a `ScanCode` instead.
/// This implementation is currently a very basic/limited one from an example.
/// The intent is to be able to distinguish a key that has just been pressed
/// from one that is held while also allowing multiple keys pressed at one
/// time.
/// The example used 255 for the array size. This seems to be based on nothing.
/// The winit `VirtualKeyCode` enum currently has 163 entries. There is no
/// macro in stable Rust to count this automatically.
const ARRAY_SIZE: usize = 180;

pub struct Keyboard {
    current_keys: [bool; ARRAY_SIZE],
    previous_keys: [bool; ARRAY_SIZE],
}

impl Default for Keyboard {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyboardHandler for Keyboard {
    fn input(&mut self, keycode: VirtualKeyCode, state: ElementState) {
        match state {
            ElementState::Pressed => self.current_keys[keycode as usize] = true,
            ElementState::Released => {
                self.current_keys[keycode as usize] = false;
            }
        }
    }
}

impl Keyboard {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            current_keys: [false; ARRAY_SIZE],
            previous_keys: [false; ARRAY_SIZE],
        }
    }

    /// Application should call at the end of a tick to track which keys
    /// have changed
    pub fn tick(&mut self) {
        // Rust implements bools as bytes and arrays have a copy trait so
        // this copy shouldn't be that expensive
        self.previous_keys = self.current_keys;
    }

    pub fn pressed(&mut self, keycode: VirtualKeyCode) {
        self.current_keys[keycode as usize] = true;
    }

    pub fn released(&mut self, keycode: VirtualKeyCode) {
        self.current_keys[keycode as usize] = false;
    }

    /// Application can call to see if a particular key is pressed
    #[must_use]
    pub const fn is_pressed(&self, keycode: VirtualKeyCode) -> bool {
        self.current_keys[keycode as usize]
    }

    /// Application can call to see if a particular key is pressed now
    /// but wasn't last tick
    #[must_use]
    pub const fn is_just_pressed(&self, keycode: VirtualKeyCode) -> bool {
        self.current_keys[keycode as usize]
            && !self.previous_keys[keycode as usize]
    }
}
