//! Recommend using with
//! `RUSTFLAGS="-C target-cpu=x86-86-v2" cargo bench`
//! and that end users compile their applications in this way. That enables
//! SSE4.2 support (released late in 2008) which should be a safe default and
//! gives 5 to 10% performance improvement in these benchmarks.
//!
//! The current benchmarks are just to explore how benchmarking and SIMD
//! code generation works. The functions tested are not critical for speed
//! and many of them aren't currently used at all in the library itself.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use nalgebra_glm as glm;
use rhodora::dualquat::{self, DualQuat};

const COUNT: usize = 100;
const MUL: f32 = 1.0_f32 / (COUNT as f32);

fn dq_to_mat4(c: &mut Criterion) {
    let q = glm::quat(0.36516f32, 0.54772f32, 0.73030f32, 0.18257f32);
    let v = glm::vec3(-0.7, 5.1, -21.0);
    let dq = black_box(DualQuat::new(&q, &v));

    c.bench_function(
        "dq_to_mat4", //
        |b| b.iter(|| dualquat::to_mat4(&dq)),
    );
}

fn use_this_arr() -> [[f32; 4]; 4] {
    // This should be a valid rotation and translation matrix
    {
        [
            [1.0f32, 0.0f32, 0.0f32, 0.0f32], // column 0
            [0.0f32, 0.3584f32, -0.9336f32, 0.0f32], // column 1
            [0.0f32, 0.9336f32, 0.3584f32, 0.0f32], // column 2
            [5.0f32, 7.0f32, 9.0f32, 1.0f32], // column 3
        ]
    }
}

fn mat4_to_dq(c: &mut Criterion) {
    let m: glm::Mat4 = use_this_arr().into();
    let m = black_box(m);

    c.bench_function(
        "mat4_to_dq", //
        |b| b.iter(|| dualquat::from_mat4(&m)),
    );
}

fn arr_to_dq(c: &mut Criterion) {
    let arr = black_box(use_this_arr());
    c.bench_function(
        "arr_to_dq", //
        |b| b.iter(|| <[[f32; 4]; 4] as Into<DualQuat>>::into(arr)),
    );
}

fn use_these_dqs() -> (DualQuat, DualQuat) {
    let dq1 = DualQuat::new(
        &glm::quat_angle_axis(
            0.376_f32, //
            &glm::vec3(0.0_f32, 0.0_f32, 1.0_f32),
        ),
        &glm::vec3(3.0_f32, 1.4_f32, 0.0_f32),
    );
    let dq2 = DualQuat::new(
        &glm::quat_angle_axis(
            0.512_f32, //
            &glm::vec3(0.0_f32, 1.0_f32, 0.0_f32),
        ),
        &glm::vec3(1.2_f32, 0.0_f32, -4.0_f32),
    );
    (dq1, dq2)
}

fn sep_interpolate(c: &mut Criterion) {
    let (dq1, dq2) = use_these_dqs();
    let dq1 = black_box(dq1);
    let dq2 = black_box(dq2);
    c.bench_function(
        "sep interpolate", //
        |b| {
            b.iter(|| {
                for i in 0..=COUNT {
                    let _ = dualquat::sep(&dq1, &dq2, (i as f32) * MUL);
                }
            })
        },
    );
}

fn dlb_interpolate(c: &mut Criterion) {
    let (dq1, dq2) = use_these_dqs();
    let dq1 = black_box(dq1);
    let dq2 = black_box(dq2);
    c.bench_function(
        "dlb interpolate", //
        |b| {
            b.iter(|| {
                for i in 0..=COUNT {
                    let _ = dualquat::dlb(&dq1, &dq2, (i as f32) * MUL);
                }
            })
        },
    );
}
criterion_group!(
    benches,
    dq_to_mat4,
    mat4_to_dq,
    arr_to_dq,
    sep_interpolate,
    dlb_interpolate
);
criterion_main!(benches);
