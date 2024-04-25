// Some code inspired by
// https://github.com/KhronosGroup/glTF-Tutorials/

use super::types::{
    ImportMaterial, ImportOptions, ImportVertex, MeshLoaded, Submesh,
};
use crate::{
    animation::{
        Animation, AnimationChannel, Interpolation, JointInfo, Keyframe,
        Skeleton,
    },
    dualquat::{self, DualQuat},
    mesh_import::ImportError,
    rh_error::RhError,
    vertex::{IndexBuffer, InterBuffer},
};
use ahash::{HashMap, HashMapExt};
use gltf::{
    animation::util::ReadOutputs,
    buffer::{self, Data},
    image::Source,
    mesh::util::{ReadIndices, ReadJoints, ReadNormals, ReadPositions},
    mesh::Mode,
    Document, Gltf, Node, Primitive, Semantic,
};
use log::{debug, error, info, trace, warn};
use nalgebra_glm as glm;
use std::{fs, io, path::Path};

#[cfg(feature = "rayon")]
use rayon::prelude::*;

/// Comparison value for approximate equality of two scale vectors
const SCALE_EPSILON: f32 = 0.005;

#[derive(Clone, Debug)]
struct NodeInfo {
    name: String,
    parent: usize,
    root: usize,
    children: Vec<usize>,
    translation: glm::Vec3,
    rotation: glm::Quat,
    scale: glm::Vec3,
}

#[derive(Clone, Debug, Default)]
struct Rotation {
    time: f32,
    data: glm::Quat,
}

#[derive(Clone, Debug)]
struct RotationChannel {
    interpolation: Interpolation,
    channel: Vec<Rotation>,
}

#[derive(Clone, Debug, Default)]
struct Translation {
    time: f32,
    data: glm::Vec3,
}

#[derive(Clone, Debug)]
struct TranslationChannel {
    interpolation: Interpolation,
    channel: Vec<Translation>,
}

#[derive(Clone, Debug)]
struct RawAnimation {
    name: String,
    r_channels: HashMap<usize, RotationChannel>,
    t_channels: HashMap<usize, TranslationChannel>,
}

// Validate a glTF for compatibility. Returns index and vertex count.
fn validate(p: &Primitive) -> Result<(usize, usize), RhError> {
    // Mesh must be made of indexed triangles
    if p.mode() != Mode::Triangles {
        Err(ImportError::NoTriangles)?;
    }
    let indices = (p.indices().ok_or(ImportError::NoIndices))?;
    let idx_count = indices.count();

    // Positions are required
    let positions =
        (p.get(&Semantic::Positions).ok_or(ImportError::NoPositions))?;
    let vert_count = positions.count();

    // Normals are highly recommended. If present there must be the same number
    // of normals as there are positions.
    let normals_option = p.get(&Semantic::Normals);
    if let Some(ref normals) = normals_option {
        if normals.count() != vert_count {
            Err(ImportError::CountMismatch)?;
        }
    }

    // Check optional features just for info
    let uv_option = p.get(&Semantic::TexCoords(0));
    let joint_option = p.get(&Semantic::Joints(0));

    // A little info
    info!(
        "Submesh={}, vertices={}, triangles={}, has_normals={}, has_uv={}, has_joints={}",
        p.index(),
        vert_count,
        idx_count / 3,
        normals_option.is_some(),
        uv_option.is_some(),
        joint_option.is_some(),
    );

    Ok((idx_count, vert_count))
}

fn load_impl<P>(path: P) -> Result<(Document, Vec<buffer::Data>), RhError>
where
    P: AsRef<Path>,
{
    let path = path.as_ref();
    let base = path.parent().unwrap_or_else(|| Path::new("./"));
    let file = fs::File::open(path).map_err(RhError::StdIoError)?;
    let reader = io::BufReader::new(file);
    let gltf = Gltf::from_reader(reader)
        .map_err(|e| RhError::GltfError(Box::new(e)))?;
    let buffers = gltf::import_buffers(&gltf.document, Some(base), gltf.blob)
        .map_err(|e| RhError::GltfError(Box::new(e)))?;

    // Some info
    let buffer_count = buffers.len();
    info!(
        "{:?}, base path={:?}, buffer count={}, first buffer length={} ",
        path,
        base,
        buffer_count,
        buffers[0].len(),
    );
    if buffer_count != 1 {
        warn!("buffer count={} is not 1", buffer_count);
    }

    Ok((gltf.document, buffers))
}

/// Load a glTF file. Only a very limited subset of glTF functionality is
/// supported. The current focus of the project is on models which share
/// textures, therefore glTF files that embed images are not supported.
/// Tested with files exported from Blender 3.6.8 using the "glTF Separate"
/// option. glTF defines +Y up, +Z forward so `swizzle` is always used and
/// therefore ignored.
///
/// # Errors
/// May return `RhError`
//#[allow(clippy::cognitive_complexity)]
#[allow(clippy::too_many_lines)]
pub fn load(
    path: &Path,
    import_options: &ImportOptions,
    vb_index: &mut IndexBuffer,
    vb_inter: &mut InterBuffer,
) -> Result<MeshLoaded, RhError> {
    let scale = import_options.scale;
    let mut submeshes = Vec::new();
    let mut first_index = 0u32;
    let mut vertex_offset = 0i32;

    // Load the gltf file and mesh data but not the textures
    let (document, buffers) = load_impl(path)?;

    // Can contain multiple meshes which can contain multiple primitives. Each
    // primitive is treated as a submesh to match the .obj format support.
    for m in document.meshes() {
        info!("mesh={}, name={:?}", m.index(), m.name());
        for p in m.primitives() {
            let benchmark = std::time::Instant::now();

            // Validate certain aspects of the glTF. These should include that
            // the number of positions is equal to the number of normals etc.
            // since we don't know how to interpret the data if they aren't
            // equal.
            let (idx_count, vert_count) = validate(&p)?;

            // Create a reader for the data buffer
            let reader = p.reader(|x| Some(&buffers[x.index()]));

            // Read the indices and store them as 16 bits. There is a `into_u32`
            // provided in the glTF library but not one for 16 bits.
            let mut import_indices: Vec<u16> = Vec::new();
            let idx_data =
                reader.read_indices().ok_or(ImportError::NoIndices)?;
            match idx_data {
                ReadIndices::U8(it) => {
                    for i in it {
                        import_indices.push(u16::from(i));
                    }
                }
                ReadIndices::U16(it) => {
                    for i in it {
                        import_indices.push(i);
                    }
                }
                ReadIndices::U32(it) => {
                    info!("Trying to convert 32 bit indices to 16 bit");
                    for i in it {
                        import_indices.push(
                            u16::try_from(i)
                                .map_err(|_| RhError::IndexTooLarge)?,
                        );
                    }
                }
            }

            // Create an import vertex buffer to build up the other data
            let mut import_vertices: Vec<ImportVertex> = Vec::new();

            // Read and store positions into the import vertex buffer, scaling
            // if needed
            let pos_data =
                reader.read_positions().ok_or(ImportError::NoPositions)?;
            let ReadPositions::Standard(it) = pos_data else {
                return Err(ImportError::SparseMesh)?;
            };
            for i in it {
                let v = ImportVertex {
                    position: glm::Vec3::new(
                        i[0] * scale,
                        -i[2] * scale,
                        i[1] * scale,
                    ),
                    ..Default::default()
                };
                import_vertices.push(v);
            }

            // Read and store normals if they exist
            let has_normals = {
                reader.read_normals().map_or(false, |norm_data| {
                    if let ReadNormals::Standard(it) = norm_data {
                        for (i, norm) in it.enumerate() {
                            if i < import_vertices.len() {
                                import_vertices[i].normal =
                                    glm::Vec3::new(norm[0], -norm[2], norm[1]);
                            }
                        }
                        true
                    } else {
                        warn!("Sparse normals not supported");
                        false
                    }
                })
            };

            // Read and store the texture coordinates if they exist
            if let Some(uv_data) = reader.read_tex_coords(0) {
                for (i, uv) in uv_data.into_f32().enumerate() {
                    if i < import_vertices.len() {
                        import_vertices[i].tex_coord = uv;
                    }
                }
            }

            // Read and store the joints if they exist
            if let Some(joint_data) = reader.read_joints(0) {
                let ReadJoints::U8(joint_it) = joint_data else {
                    // Could try to fit these into u8 but if they fit that
                    // would have probably been used by the file
                    return Err(ImportError::BigJointIndices)?;
                };
                let weight_data = reader
                    .read_weights(0)
                    .ok_or(ImportError::NoWeights)?
                    .into_f32();
                for (i, (id_array, weights)) in
                    joint_it.zip(weight_data).enumerate()
                {
                    trace!("Joint ids={:?} weights={:?}", id_array, weights);
                    let sum: f32 = weights.iter().sum();
                    if (sum - 1.0_f32).abs() > 0.02_f32 {
                        warn!("Vertex {i} weights aren't normalized={sum}");
                    }
                    if i < import_vertices.len() {
                        import_vertices[i].joint_ids = id_array;
                        import_vertices[i].weights = weights;
                    }
                }
            }

            // Validate that there is the expected amount of information
            if vert_count != import_vertices.len() {
                error!(
                    "Vertex count mismatch {} != {}",
                    vert_count,
                    import_vertices.len()
                );
                Err(ImportError::CountMismatch)?;
            }

            // Do possible additional processing of the intermediate data
            if !has_normals {
                warn!(
                    "Missing normals are being calculated and might be wrong"
                );
                super::util::calculate_normals(
                    &import_indices,
                    &mut import_vertices,
                );
            }

            // Collect information
            let vertex_count = i32::try_from(vert_count)
                .map_err(|_| RhError::VertexCountTooLarge)?;
            let index_count = u32::try_from(idx_count)
                .map_err(|_| RhError::IndexCountTooLarge)?;
            submeshes.push(Submesh {
                index_count,
                first_index,
                vertex_offset,
                vertex_count,
                material_id: p.material().index().unwrap_or(0),
            });

            // Collect into the output struct
            vb_inter.append(&import_vertices);
            vb_index.indices.append(&mut import_indices); // Consumes input
            debug!("Processing took {:?}", benchmark.elapsed());

            // Prepare for next submesh
            vertex_offset += vertex_count;
            first_index += index_count;
        }
    }

    Ok(MeshLoaded {
        submeshes,
        materials: {
            let base_path = path.parent().unwrap_or_else(|| Path::new("."));
            load_materials(base_path, &document)
        },
        order_option: import_options.order_option.clone(),
    })
}

fn load_materials(
    base_path: &Path,
    document: &Document,
) -> Vec<ImportMaterial> {
    info!("Materials={}", document.materials().count());

    // Data is collected but textures are not loaded here. There is not much
    // computation and probably a small number of textures so it is likely
    // not worth parallelizing this,
    document.materials().map(|m| {
        let pbr = m.pbr_metallic_roughness();
        let diffuse = {
            let base = pbr.base_color_factor();
            [base[0], base[1], base[2]]
        };
        let roughness = pbr.roughness_factor();
        let metalness = pbr.metallic_factor();
        let colour_filename = {
            pbr.base_color_texture().map_or_else(String::new, |tex| {
                let source = tex.texture().source().source();
                if let Source::Uri { uri, mime_type: _ } = source {
                    let ret = base_path.join(uri);
                    ret.display().to_string()
                } else {
                    String::new()
                }
            })
        };
        info!(
            "Material {} name={} texture={} diffuse={:?} roughness={} metalness={}",
            m.index().unwrap_or(0),
            m.name().unwrap_or("N/A"),
            colour_filename,
            diffuse,
            roughness,
            metalness,
        );

        ImportMaterial {
            colour_filename,
            diffuse,
            roughness,
            metalness,
        }
    }).collect()
}

/// Recursive node tree traversal
fn traverse_tree(
    node: &Node,
    tree: &mut HashMap<usize, NodeInfo>,
    parent: usize,
    root: usize,
) {
    // Walk children of this node
    for child in node.children() {
        traverse_tree(&child, tree, node.index(), root);
    }

    // Collect node information
    let name = node
        .name()
        .map_or_else(|| format!("node.{}", node.index()), ToString::to_string);

    let mut children = Vec::new();
    for child in node.children() {
        children.push(child.index());
    }
    let (t, r, s) = node.transform().decomposed();

    // Insert it into a hashmap
    tree.insert(
        node.index(),
        NodeInfo {
            name,
            parent,
            root,
            children,
            translation: t.into(),
            rotation: r.into(),
            scale: s.into(),
        },
    );
}

/// Swizzles a quaternion from Y axis up to Z axis up
fn quat_swizzle(q: &glm::Quat) -> glm::Quat {
    glm::quat(q.i, -q.k, q.j, q.w)
}

/// Swizzles a vector from Y axis up to Z axis up
fn vec_swizzle(v: &glm::Vec3) -> glm::Vec3 {
    glm::vec3(v.x, -v.z, v.y)
}

/// Creates `JointInfo` from `NodeInfo` by adding the inverse binding
fn joint_info_from_node(node_info: &NodeInfo, inv_bind: DualQuat) -> JointInfo {
    JointInfo {
        name: node_info.name.clone(),
        parent: node_info.parent,
        children: node_info.children.clone(),
        inv_bind,
        bind: DualQuat::new(
            &quat_swizzle(&node_info.rotation),
            &vec_swizzle(&node_info.translation),
        ), // Z axis up
    }
}

fn load_skeleton(
    document: &Document,
    buffers: &[Data],
) -> Result<Vec<Skeleton>, RhError> {
    let mut ret = Vec::new();

    // Full node tree
    let mut full_tree = HashMap::<usize, NodeInfo>::new();
    for scene in document.scenes() {
        for node in scene.nodes() {
            let index = node.index();
            traverse_tree(&node, &mut full_tree, index, index);
        }
    }
    debug!("full tree={:?}", full_tree);

    // Skins
    for skin in document.skins() {
        let reader = skin.reader(|x| Some(&buffers[x.index()]));
        let Some(iter) = reader.read_inverse_bind_matrices() else {
            return Err(ImportError::NoInverseBind)?;
        };

        let mut current_root: Option<usize> = None;
        let mut joint_tree = HashMap::<usize, JointInfo>::new();
        let mut joint_to_node = Vec::new();

        for (ibm, node) in iter.zip(skin.joints()) {
            let node_index = node.index();
            joint_to_node.push(node_index);
            let Some(node_info) = full_tree.get(&node_index) else {
                return Err(ImportError::NoNodeInfo(node_index))?;
            };

            // All of the root nodes for this skin should be the same
            if let Some(x) = current_root {
                if x != node_info.root {
                    Err(ImportError::ConflictingRootNodes(skin.index()))?;
                }
            }
            current_root = Some(node_info.root);

            // Scaled joints are not supported
            let compare = glm::not_equal_eps(
                &node_info.scale,
                &glm::vec3(1.0f32, 1.0f32, 1.0f32),
                SCALE_EPSILON,
            );
            if compare.x || compare.y || compare.z {
                Err(ImportError::ScaledJoints(node_index))?;
            }

            // Insert into the joint info hashmap
            joint_tree.insert(
                node_index,
                joint_info_from_node(
                    node_info,
                    dualquat::swizzle(&ibm.into()), // Z axis up
                ),
            );
        }

        // Make sure that a root node was found
        let Some(root_index) = current_root else {
            return Err(ImportError::NoRootNode(skin.index()))?;
        };

        // Make sure that root node matches what gltf crate says.
        // Ideally tree traversal could be replaced by using this, but the else
        // case is always triggered by test files so more research is needed.
        if let Some(alt_root) = skin.skeleton() {
            if root_index != alt_root.index() {
                Err(ImportError::ConflictingRootNodes(skin.index()))?;
            }
        } else {
            warn!(
                "skin {} does not report a root node, using node index {}",
                skin.index(),
                root_index
            );
        }

        // Make sure root node is in joint tree without disturbing it
        if joint_tree.get(&root_index).is_none() {
            if let Some(node_info) = full_tree.get(&root_index) {
                debug!(
                    "Adding root node {} for skin {}",
                    root_index,
                    skin.index()
                );

                // The root node may have children that are not joints
                // (submeshes etc.) that should not be in the skeleton.
                let mut joint_info =
                    joint_info_from_node(node_info, DualQuat::default());
                let mut child_joints = Vec::new();
                for child in joint_info.children {
                    if joint_tree.contains_key(&child) {
                        child_joints.push(child);
                    }
                }
                if child_joints.is_empty() {
                    // Thought it was root node but has no joints as children?
                    Err(ImportError::NoRootNode(skin.index()))?;
                }
                joint_info.children = child_joints;

                // Since this isn't a joint, there isn't an inverse binding to
                // store, but there still could be a binding (node translation &
                // rotation). This should be ok since the binding may effect
                // children, but the inverse binding is only used in the final
                // calculation of the joint transform... and this isn't a joint.
                joint_tree.insert(root_index, joint_info);
            } else {
                // Skeleton says there is a root node but it isn't in full tree
                Err(ImportError::NoRootNode(skin.index()))?;
            }
        }

        // Store
        let name = skin.name().map_or_else(
            || format!("skin.{}", skin.index()),
            ToString::to_string,
        );
        ret.push(Skeleton {
            name,
            root: root_index,
            joint_to_node,
            tree: joint_tree,
        });
    }
    Ok(ret)
}

fn load_raw_animations(
    document: &Document,
    buffers: &[Data],
) -> Result<Vec<RawAnimation>, RhError> {
    use gltf::accessor::Iter;

    let mut ret = Vec::new();
    for animation in document.animations() {
        debug!("animation name={:?}", animation.name());
        let mut r_channels = HashMap::<usize, RotationChannel>::new();
        let mut t_channels = HashMap::<usize, TranslationChannel>::new();

        for channel in animation.channels() {
            let node = channel.target().node();
            let interpolation = match channel.sampler().interpolation() {
                gltf::animation::Interpolation::Step => Interpolation::Step,
                gltf::animation::Interpolation::Linear => Interpolation::Linear,
                gltf::animation::Interpolation::CubicSpline => {
                    Interpolation::CubicSpline
                }
            };
            let reader = channel.reader(|x| Some(&buffers[x.index()]));
            let times: Vec<f32> = if let Some(inputs) = reader.read_inputs() {
                match inputs {
                    Iter::Standard(times) => times.collect(),
                    Iter::Sparse(_) => {
                        return Err(ImportError::SparseAnimation)?;
                    }
                }
            } else {
                error!("Animation does not contain a sampler");
                return Err(ImportError::NoSampler)?;
            };

            if let Some(outputs) = reader.read_outputs() {
                match outputs {
                    ReadOutputs::Rotations(x) => {
                        let mut chan = Vec::new();
                        let q: Vec<glm::Quat> =
                            x.into_f32().map(Into::into).collect();
                        for (time, data) in times.iter().zip(&q) {
                            chan.push(Rotation {
                                time: *time,
                                data: quat_swizzle(data), // Z axis up
                            });
                        }
                        r_channels.insert(
                            node.index(),
                            RotationChannel {
                                interpolation,
                                channel: chan,
                            },
                        );
                    }
                    ReadOutputs::Translations(x) => {
                        let mut chan = Vec::new();
                        let v: Vec<glm::Vec3> = x.map(Into::into).collect();
                        for (time, data) in times.iter().zip(&v) {
                            chan.push(Translation {
                                time: *time,
                                data: vec_swizzle(data), // Z axis up
                            });
                        }
                        t_channels.insert(
                            node.index(),
                            TranslationChannel {
                                interpolation,
                                channel: chan,
                            },
                        );
                    }
                    ReadOutputs::Scales(x) => {
                        for d in x {
                            // Only a scale of 1 is supported. Warn if
                            // another value is present.
                            let scale: glm::Vec3 = d.into();
                            let comp = glm::not_equal_eps(
                                &scale,
                                &glm::vec3(1.0, 1.0, 1.0),
                                SCALE_EPSILON,
                            );
                            if comp.x || comp.y || comp.z {
                                warn!(
                                    "animation {} node {} scale ignored",
                                    animation.index(),
                                    node.index()
                                );
                                break;
                            }
                        }
                    }
                    ReadOutputs::MorphTargetWeights(_) => {
                        Err(ImportError::Morphing)?;
                    }
                }
            } else {
                Err(ImportError::NoSampler)?;
            };
        }

        // Store
        let name = animation.name().map_or_else(
            || format!("animation.{}", animation.index()),
            ToString::to_string,
        );
        ret.push(RawAnimation {
            name,
            r_channels,
            t_channels,
        });
    } // animation
    Ok(ret)
}

fn weight(start: f32, end: f32, current: f32) -> f32 {
    const EPSILON: f32 = 0.0005_f32;
    ((current - start) / (end - start).max(EPSILON)).clamp(0.0f32, 1.0f32)
}

/// Helper function for determining rotation
fn find_rotation(
    r_channel: &RotationChannel,
    initial_frame: &Rotation,
    current_time: f32,
) -> glm::Quat {
    let mut frame = initial_frame;
    for f in &r_channel.channel {
        if f.time < current_time {
            // This frame has a time before the current time, so
            // make it the new candidate frame. (Note that `frame`
            // and `f` are both references.)
            frame = f;
        } else {
            // This frame has a time equal or greater than the
            // desired time, so stop looping.
            if r_channel.interpolation == Interpolation::Step {
                // Step interpolation uses the candidate frame
                return frame.data;
            }
            // Other interpolation options are linear and cubic
            // spline. Cubic spline isn't supported so if it wasn't
            // filtered out already it will be treated as linear.
            return glm::quat_slerp(
                &frame.data,
                &f.data,
                weight(frame.time, f.time, current_time),
            );
        }
    }
    // Fall through past the end of the channel so return data
    // from candidate frame
    frame.data
}

/// Helper function for determining translation
fn find_translation(
    t_channel: &TranslationChannel,
    initial_frame: &Translation,
    current_time: f32,
) -> glm::Vec3 {
    let mut frame = initial_frame;
    for f in &t_channel.channel {
        if f.time < current_time {
            // This frame has a time before the current time, so
            // make it the new candidate frame. (Note that `frame`
            // and `f` are both references.)
            frame = f;
        } else {
            // This frame has a time equal or greater than the
            // desired time, so stop looping.
            if t_channel.interpolation == Interpolation::Step {
                // Step interpolation uses the candidate frame
                return frame.data;
            }
            // Other interpolation options are linear and cubic
            // spline. Cubic spline isn't supported so if it wasn't
            // filtered out already it will be treated as linear.
            return glm::lerp(
                &frame.data,
                &f.data,
                weight(frame.time, f.time, current_time),
            );
        }
    }
    // Fall through past the end of the channel so return data
    // from candidate frame
    frame.data
}

// Returns the animation channel and the maximum timestamp in that channel
fn process_animation_node(
    raw: &RawAnimation,
    skeleton: &Skeleton,
    node_index: usize,
) -> (AnimationChannel, f32) {
    // The animation data has no direct reference to the skeleton but all the
    // data depends on the skeleton's hiearchy. Just holding the skeleton in
    // its binding pose requires that pose being in the data. That means it
    // is unlikely that channels will be missing from the animation data.
    // But if it is, filling it in with binding data seems like the best
    // solution. That information is in the skeleton but it is conveniently
    // accessed by node index.
    let (initial_rot, initial_trans) = {
        skeleton.tree.get(&node_index).map_or_else(
            || {
                // This shouldn't happen. Even a root that is not a joint is
                // added to the skeleton. But if it does, use some defaults.
                warn!("node_index={} not in tree", node_index);
                (glm::Quat::identity(), glm::Vec3::zeros())
            },
            |joint_info| dualquat::decompose(&joint_info.bind),
        )
    };
    let initial_rot_frame = Rotation {
        time: 0.0_f32,
        data: initial_rot,
    };
    let initial_trans_frame = Translation {
        time: 0.0_f32,
        data: initial_trans,
    };

    // Input animation data is keyed by node index
    let r_channel_opt = raw.r_channels.get(&node_index);
    let t_channel_opt = raw.t_channels.get(&node_index);

    // The channels may contain different timestamps. Collect all of
    // those values to make sure each one is processed.
    // This is also a convenient place to check the interpolation mode. The
    // output mode will default to `Interpolation::Step` but will be changed to
    // `Interpolation::Linear` if either channel does not use
    // `Interpolation::Step`.
    let mut times = Vec::new();
    let mut interpolation = Interpolation::Step;
    if let Some(r_channel) = r_channel_opt {
        for data in &r_channel.channel {
            times.push(data.time);
        }
        if r_channel.interpolation != Interpolation::Step {
            interpolation = Interpolation::Linear;
        }
    }
    if let Some(t_channel) = t_channel_opt {
        for data in &t_channel.channel {
            times.push(data.time);
        }
        if t_channel.interpolation != Interpolation::Step {
            interpolation = Interpolation::Linear;
        }
    }
    // Since the timestamps are f32, regular `sort` can't be used. If there are
    // NaNs consider them equal (but expect trouble somewhere else).
    times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    // The regular `dedup` is implemented for f32 but doing an approximate
    // equals is probably a better idea. It seems unlikely that animation will
    // need keyframes that are actually less than 1 mS apart.
    times.dedup_by(|a, b| (*a - *b).abs() < 0.001_f32);
    let max_time = *(times.last().unwrap_or(&0.0_f32));

    // Collect the data into dual quaternions. If the rotations and translations
    // have different timestamps values will be interpolated.
    // Processing may be done in parallel using rayon's threadpool.
    #[cfg(feature = "rayon")]
    let it = times.par_iter();
    #[cfg(not(feature = "rayon"))]
    let it = times.iter();

    let keyframes = it
        .map(|current_time| {
            let rot = {
                r_channel_opt.map_or(initial_rot, |r_channel| {
                    find_rotation(r_channel, &initial_rot_frame, *current_time)
                })
            };
            let trans = {
                t_channel_opt.map_or(initial_trans, |t_channel| {
                    find_translation(
                        t_channel,
                        &initial_trans_frame,
                        *current_time,
                    )
                })
            };
            Keyframe {
                time: *current_time,
                data: DualQuat::new(&rot, &trans),
            }
        })
        .collect();

    (
        AnimationChannel {
            interpolation,
            data: keyframes,
        },
        max_time,
    )
}

/// Loads skeleton and animation from a glTF file
///
/// # Errors
/// May return `RhError`
pub fn load_animations(
    path: &Path,
) -> Result<(Vec<Skeleton>, Vec<Animation>), RhError> {
    let (document, buffers) = load_impl(path)?;
    let skeletons = load_skeleton(&document, &buffers)?;
    let raw_animations = load_raw_animations(&document, &buffers)?;

    debug!("skeletons={:?}", skeletons);
    //debug!("animations={:?}", animations);

    let mut animations = Vec::new();
    for raw in &raw_animations {
        // The raw animation has two hashmaps, one with rotation channels and
        // one with translation channels. They are both keyed by node index
        // (which is not the joint index). They probably both have entries for
        // all the same nodes, but that should not be relied on.

        // Collect all the keys for both maps into a non-duplicated vector.
        let mut nodes: Vec<usize> = raw
            .r_channels
            .keys()
            .chain(raw.t_channels.keys())
            .copied()
            .collect();
        nodes.sort_unstable(); // Fast and safe for sorting `Vec<usize>`
        nodes.dedup();

        // How to tie this animation to the skeleton? FIXME
        let skeleton = &skeletons[0];

        // Process every node present in this animation
        let mut channels: HashMap<usize, AnimationChannel> = HashMap::new();
        let mut max_time = 0.0_f32;

        // This loop is not parallelized across nodes because of the mutability
        // of `max_time` and `channels` but `process_animation_node` does
        // parallelize the node processing.
        for node_index in nodes {
            let (channel, max_channel_time) =
                process_animation_node(raw, skeleton, node_index);
            max_time = max_time.max(max_channel_time);
            channels.insert(node_index, channel);
        }

        animations.push(Animation {
            name: raw.name.clone(),
            max_time,
            channels,
        });
    }
    //debug!("animations={:?}", animations); // Probaby a LOT of data

    Ok((skeletons, animations))
}
