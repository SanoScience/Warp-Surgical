# SPDX-License-Identifier: Apache-2.0
"""Helpers wiring :class:`MarchingCubesBuffers` output into ``newton.viewer``.

Each call to :func:`log_mc_surface` runs the MC pipeline and updates a named
mesh asset on the viewer. MC triangle indices stay GPU-resident and are
flattened with a small device kernel before being handed to the viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton._src.geometry.flags import ParticleFlags
from newton._src.utils.mesh import compute_vertex_normals

from .grid import GridAuxState
from .kernels.marching_cubes import (
    MarchingCubesBuffers,
    MarchingCubesTables,
    compute_mc_topology,
    compute_mc_topology_dirty,
    compute_mc_vertex_positions,
    compute_visible_flags_kernel,
)
from .tool import Tool

_SCOPED_TIMER_DICT: dict[str, list[float]] | None = None
_SCOPED_TIMER_GPU_DICT: dict[str, list[float]] | None = None
_SCOPED_TIMER_GPU_BREAKDOWN: dict[str, dict[str, list[float]]] | None = None
_SCOPED_TIMER_SYNC_NAMES: set[str] = set()
_SCOPED_TIMER_SYNC_ENABLED: bool = False
_SCOPED_TIMER_CUDA_FILTER: int = 0


@dataclass
class CryoSurfaceFrame:
    """Flat MC triangle data prepared for external Slang-style renderers."""

    positions: wp.array
    normals: wp.array
    uv3: wp.array
    material_id: wp.array | None
    state_rgba: wp.array | None
    vertex_count: int
    triangle_count: int
    buffer_generation: int
    procedural_coord: wp.array | None = None


def set_scoped_timer_dict(
    timer_dict: dict[str, list[float]] | None,
    *,
    gpu_dict: dict[str, list[float]] | None = None,
    gpu_breakdown: dict[str, dict[str, list[float]]] | None = None,
    sync_names: set[str] | None = None,
    sync_enabled: bool = False,
    cuda_filter: int = 0,
) -> None:
    """Set the shared timing sinks and options used by local timer scopes."""
    global _SCOPED_TIMER_DICT  # noqa: PLW0603
    global _SCOPED_TIMER_GPU_DICT  # noqa: PLW0603
    global _SCOPED_TIMER_GPU_BREAKDOWN  # noqa: PLW0603
    global _SCOPED_TIMER_SYNC_NAMES  # noqa: PLW0603
    global _SCOPED_TIMER_SYNC_ENABLED  # noqa: PLW0603
    global _SCOPED_TIMER_CUDA_FILTER  # noqa: PLW0603
    _SCOPED_TIMER_DICT = timer_dict
    _SCOPED_TIMER_GPU_DICT = gpu_dict
    _SCOPED_TIMER_GPU_BREAKDOWN = gpu_breakdown
    _SCOPED_TIMER_SYNC_NAMES = set() if sync_names is None else set(sync_names)
    _SCOPED_TIMER_SYNC_ENABLED = bool(sync_enabled)
    _SCOPED_TIMER_CUDA_FILTER = int(cuda_filter)


class _ScopedTimerRecorder:
    """Wrap ``wp.ScopedTimer`` and optionally collect CUDA activity totals."""

    def __init__(self, name: str):
        self.name = name
        self._gpu_scope = name in _SCOPED_TIMER_SYNC_NAMES
        cuda_filter = _SCOPED_TIMER_CUDA_FILTER if self._gpu_scope else 0
        synchronize = _SCOPED_TIMER_SYNC_ENABLED if self._gpu_scope else False
        self._timer = wp.ScopedTimer(
            name,
            use_nvtx=True,
            synchronize=synchronize,
            print=False,
            dict=_SCOPED_TIMER_DICT,
            cuda_filter=cuda_filter,
        )

    def __enter__(self):
        self._timer.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        result = self._timer.__exit__(exc_type, exc_value, traceback)
        if self._gpu_scope and _SCOPED_TIMER_GPU_DICT is not None:
            timing_results = getattr(self._timer, "timing_results", ())
            if timing_results:
                total_ms = float(sum(r.elapsed for r in timing_results))
                _SCOPED_TIMER_GPU_DICT.setdefault(self.name, []).append(total_ms)
                if _SCOPED_TIMER_GPU_BREAKDOWN is not None:
                    scope_breakdown = _SCOPED_TIMER_GPU_BREAKDOWN.setdefault(self.name, {})
                    for timing in timing_results:
                        scope_breakdown.setdefault(timing.name, []).append(float(timing.elapsed))
        return result


def make_scoped_timer(name: str) -> _ScopedTimerRecorder:
    """Create a configured timer for the local MC/render hot path."""
    return _ScopedTimerRecorder(name)


def _scoped_timer(name: str) -> _ScopedTimerRecorder:
    """Backward-compatible local timer factory."""
    return make_scoped_timer(name)


def compose_orientation(
    rot_x: int = 0,
    rot_y: int = 0,
    rot_z: int = 0,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> tuple[int, int, int, bool, bool, bool]:
    """Compose 90-degree rotations + axis flips into kernel-ready args.

    Returns ``(src_x, src_y, src_z, fx, fy, fz)`` where ``src_i`` is the
    world axis (0=X, 1=Y, 2=Z) that feeds texture axis ``i`` and ``f*``
    are post-permutation flip flags. Rotations are applied in the order
    X -> Y -> Z after the flips, so the UI can treat flip and rotation
    controls independently.

    With ``rot_i`` in ``{0,1,2,3}`` and three flips this reaches all 48
    signed-permutation orientations of the cube symmetry group.
    """
    # Signed-permutation matrix: each row has one +/-1 entry. We compose
    # rotation matrices (3x3 with integer entries) into a single one and
    # read back ``(src, sign)`` per row.
    M = np.eye(3, dtype=np.int8)
    if flip_x:
        M = np.diag([-1, 1, 1]).astype(np.int8) @ M
    if flip_y:
        M = np.diag([1, -1, 1]).astype(np.int8) @ M
    if flip_z:
        M = np.diag([1, 1, -1]).astype(np.int8) @ M

    # 90-degree rotations around each axis (right-handed).
    R = {
        0: np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.int8),
        1: np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.int8),
        2: np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int8),
    }
    for axis, raw_rotation_count in ((0, rot_x), (1, rot_y), (2, rot_z)):
        rotation_count = int(raw_rotation_count) % 4
        for _ in range(rotation_count):
            M = R[axis] @ M

    src = [0, 0, 0]
    sign = [1, 1, 1]
    for i in range(3):
        row = M[i]
        nz = np.nonzero(row)[0]
        if nz.size != 1:
            # Should not happen for signed-permutation matrices but fall
            # back to identity in case of caller error.
            src[i] = i
            sign[i] = 1
        else:
            j = int(nz[0])
            src[i] = j
            sign[i] = int(row[j])
    return src[0], src[1], src[2], sign[0] < 0, sign[1] < 0, sign[2] < 0


def _configure_atlas_filter(mesh_gl) -> None:
    """Use level-0 clamp+linear filtering for the baked atlas texture.

    Newton uploads mipmaps by default. For our atlas, higher mip levels blend
    unrelated neighbouring tiles together. Clamp to level 0 avoids that, while
    linear filtering on the base level preserves the texel variation baked
    inside each triangle tile.
    """
    if mesh_gl.texture_id is None:
        return
    texture_handle = int(mesh_gl.texture_id.value) if hasattr(mesh_gl.texture_id, "value") else int(mesh_gl.texture_id)
    if getattr(mesh_gl, "_cutting_filtered_texture_handle", None) == texture_handle:
        return
    from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415
    gl = RendererGL.gl
    gl.glBindTexture(gl.GL_TEXTURE_2D, mesh_gl.texture_id)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BASE_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAX_LEVEL, 0)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    mesh_gl._cutting_filtered_texture_handle = texture_handle


def _enable_textured_render(mesh_gl) -> None:
    """Monkey-patch a MeshGL instance to sample its albedo map during draw.

    Two fixes per draw call:

    * ``Material.w`` controls ``texture_enable`` and defaults to ``0``.
      We flip it to ``1.0`` for this mesh draw so the atlas is sampled.
    * ``ObjectColor`` defaults to Newton's warm brown ``(0.7, 0.5, 0.3)``.
      Newton multiplies texture colour by that albedo, so temporarily draw
      this mesh as white to avoid tinting the baked atlas.

    Current Newton sets those GL attributes inside ``MeshGL.render()`` from
    ``mesh_gl.color`` and ``mesh_gl.material``. Override the object fields,
    not only the GL constants, so the change survives that internal reset.
    """
    if getattr(mesh_gl, "_cutting_patched", False):
        return
    orig_render = mesh_gl.render

    def render_patched():
        if mesh_gl.hidden:
            return
        from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415
        gl = RendererGL.gl
        old_color = getattr(mesh_gl, "color", (0.7, 0.5, 0.3))
        old_material = getattr(mesh_gl, "material", (0.5, 0.0, 0.0, 0.0))
        material = list(old_material)
        while len(material) < 4:
            material.append(0.0)
        material[3] = 1.0
        mesh_gl.color = (1.0, 1.0, 1.0)
        mesh_gl.material = tuple(float(v) for v in material[:4])
        gl.glVertexAttrib4f(8, *mesh_gl.material)
        gl.glVertexAttrib3f(7, *mesh_gl.color)
        try:
            orig_render()
        finally:
            mesh_gl.color = old_color
            mesh_gl.material = old_material
            gl.glVertexAttrib4f(8, *old_material)
            gl.glVertexAttrib3f(7, *old_color)

    mesh_gl.render = render_patched
    mesh_gl._cutting_patched = True


def _enable_texture_upload_cache(mesh_gl) -> None:
    """Skip MeshGL texture re-uploads when the atlas image is unchanged.

    Newton's ``MeshGL.update()`` always calls ``update_texture(texture)``.
    For the cryo atlas that means deleting and recreating the GL texture every
    frame even when the atlas bake did not change. Patch the instance so the
    upload only happens when the atlas revision changes, falling back to object
    identity checks when no explicit revision is provided.
    """
    if getattr(mesh_gl, "_cutting_texture_cache_patched", False):
        return

    orig_update_texture = mesh_gl.update_texture

    def update_texture_patched(texture=None):
        next_revision = getattr(mesh_gl, "_cutting_next_texture_revision", None)
        last_revision = getattr(mesh_gl, "_cutting_texture_revision", None)
        last_source = getattr(mesh_gl, "_cutting_texture_source", None)

        if texture is None:
            orig_update_texture(texture)
            mesh_gl._cutting_texture_revision = None
            mesh_gl._cutting_texture_source = None
            mesh_gl._cutting_filtered_texture_handle = None
            return

        if mesh_gl.texture_id is not None:
            if next_revision is not None and next_revision == last_revision and texture is last_source:
                return
            if next_revision is None and texture is last_source:
                return

        orig_update_texture(texture)
        mesh_gl._cutting_texture_revision = next_revision
        mesh_gl._cutting_texture_source = texture
        mesh_gl._cutting_filtered_texture_handle = None

    mesh_gl.update_texture = update_texture_patched
    mesh_gl._cutting_texture_cache_patched = True


def _install_cryo_shape_shader_defaults(gl_shaders) -> None:
    """Keep patched cryo sampler uniforms valid before any object render."""
    shader_shape = getattr(gl_shaders, "ShaderShape", None)
    if shader_shape is None or getattr(shader_shape, "_cutting_cryo_defaults_patched", False):
        return

    orig_update = shader_shape.update

    def update_patched(self, *args, **kwargs):
        result = orig_update(self, *args, **kwargs)
        loc_volume = self._get_uniform_location("cryo_volume")
        loc_enabled = self._get_uniform_location("cryo_volume_enabled")
        if loc_volume >= 0:
            self._gl.glUniform1i(loc_volume, 3)
        if loc_enabled >= 0:
            self._gl.glUniform1i(loc_enabled, 0)
        return result

    shader_shape.update = update_patched
    shader_shape._cutting_cryo_defaults_patched = True
    shader_shape._cutting_cryo_defaults_orig_update = orig_update


def install_cryo_volume_shader_patch() -> bool:
    """Patch Newton's shape shader to optionally sample a cryo 3D texture.

    The patch must run before ``ViewerGL`` is constructed because Newton
    compiles the module-level shader strings in ``RendererGL.__init__``.
    Returns ``False`` if the expected shader anchors are not present.
    """
    try:
        from newton._src.viewer.gl import shaders as gl_shaders  # noqa: PLC0415
    except Exception:
        return False

    if "cryo_volume_enabled" in gl_shaders.shape_fragment_shader:
        old_cryo_multiply = "        albedo *= pow(cryo_color, vec3(2.2));\n"
        new_cryo_assign = "        albedo = pow(cryo_color, vec3(2.2));\n"
        if old_cryo_multiply in gl_shaders.shape_fragment_shader:
            gl_shaders.shape_fragment_shader = gl_shaders.shape_fragment_shader.replace(
                old_cryo_multiply,
                new_cryo_assign,
                1,
            )
        _install_cryo_shape_shader_defaults(gl_shaders)
        return True

    vertex_shader = gl_shaders.shape_vertex_shader
    fragment_shader = gl_shaders.shape_fragment_shader

    vertex_replacements = (
        (
            "layout (location = 2) in vec2 aTexCoord;\n",
            "layout (location = 2) in vec2 aTexCoord;\nlayout (location = 9) in vec3 aTexCoord3;\n",
        ),
        (
            "out vec2 TexCoord;\n",
            "out vec2 TexCoord;\nout vec3 TexCoord3;\n",
        ),
        (
            "    TexCoord = aTexCoord;\n",
            "    TexCoord = aTexCoord;\n    TexCoord3 = aTexCoord3;\n",
        ),
    )
    fragment_replacements = (
        (
            "in vec2 TexCoord;\n",
            "in vec2 TexCoord;\nin vec3 TexCoord3;\n",
        ),
        (
            "uniform sampler2D albedo_map;\n",
            (
                "uniform sampler2D albedo_map;\n"
                "uniform bool cryo_volume_enabled;\n"
                "uniform sampler3D cryo_volume;\n"
                "uniform vec3 cryo_volume_scale;\n"
                "uniform ivec3 cryo_volume_src_axis;\n"
                "uniform bvec3 cryo_volume_flip;\n"
            ),
        ),
        (
            "const float PI = 3.14159265359;\n",
            (
                "float cutting_select_axis(vec3 uv, int axis)\n"
                "{\n"
                "    if (axis == 0) return uv.x;\n"
                "    if (axis == 1) return uv.y;\n"
                "    return uv.z;\n"
                "}\n\n"
                "float cutting_scale_about_centre(float v, float scale)\n"
                "{\n"
                "    return (v - 0.5) / max(scale, 1.0e-6) + 0.5;\n"
                "}\n\n"
                "vec3 cutting_cryo_uv(vec3 uv3)\n"
                "{\n"
                "    vec3 uv = vec3(\n"
                "        cutting_scale_about_centre(cutting_select_axis(uv3, cryo_volume_src_axis.x), cryo_volume_scale.x),\n"
                "        cutting_scale_about_centre(cutting_select_axis(uv3, cryo_volume_src_axis.y), cryo_volume_scale.y),\n"
                "        cutting_scale_about_centre(cutting_select_axis(uv3, cryo_volume_src_axis.z), cryo_volume_scale.z));\n"
                "    if (cryo_volume_flip.x) uv.x = 1.0 - uv.x;\n"
                "    if (cryo_volume_flip.y) uv.y = 1.0 - uv.y;\n"
                "    if (cryo_volume_flip.z) uv.z = 1.0 - uv.z;\n"
                "    return clamp(uv, vec3(0.0), vec3(1.0));\n"
                "}\n\n"
                "const float PI = 3.14159265359;\n"
            ),
        ),
        (
            (
                "    if (texture_enable > 0.5)\n"
                "    {\n"
                "        vec3 tex_color = texture(albedo_map, TexCoord).rgb;\n"
                "        albedo *= pow(tex_color, vec3(2.2));\n"
                "    }\n"
            ),
            (
                "    if (texture_enable > 0.5)\n"
                "    {\n"
                "        vec3 tex_color = texture(albedo_map, TexCoord).rgb;\n"
                "        albedo *= pow(tex_color, vec3(2.2));\n"
                "    }\n"
                "    if (cryo_volume_enabled)\n"
                "    {\n"
                "        vec3 cryo_color = texture(cryo_volume, cutting_cryo_uv(TexCoord3)).rgb;\n"
                "        albedo = pow(cryo_color, vec3(2.2));\n"
                "    }\n"
            ),
        ),
    )

    for needle, replacement in vertex_replacements:
        if needle not in vertex_shader:
            return False
        vertex_shader = vertex_shader.replace(needle, replacement, 1)
    for needle, replacement in fragment_replacements:
        if needle not in fragment_shader:
            return False
        fragment_shader = fragment_shader.replace(needle, replacement, 1)

    gl_shaders.shape_vertex_shader = vertex_shader
    gl_shaders.shape_fragment_shader = fragment_shader
    _install_cryo_shape_shader_defaults(gl_shaders)
    return True


def _delete_cryo_volume_resources(mesh_gl) -> None:
    from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

    gl = RendererGL.gl
    if gl is None:
        return
    texture_id = getattr(mesh_gl, "_cutting_cryo_volume_texture", None)
    if texture_id is not None:
        try:
            gl.glDeleteTextures(1, texture_id)
        except Exception:
            pass
        mesh_gl._cutting_cryo_volume_texture = None
    uv3_vbo = getattr(mesh_gl, "_cutting_cryo_uv3_vbo", None)
    if uv3_vbo is not None:
        try:
            gl.glDeleteBuffers(1, uv3_vbo)
        except Exception:
            pass
        mesh_gl._cutting_cryo_uv3_vbo = None
    mesh_gl._cutting_cryo_uv3_cuda_buffer = None


def _ensure_cryo_volume_destroy_patch(mesh_gl) -> None:
    if getattr(mesh_gl, "_cutting_cryo_destroy_patched", False):
        return
    orig_destroy = mesh_gl.destroy

    def destroy_patched():
        _delete_cryo_volume_resources(mesh_gl)
        orig_destroy()

    mesh_gl.destroy = destroy_patched
    mesh_gl._cutting_cryo_destroy_patched = True


def _upload_cryo_volume_texture(mesh_gl, host_volume: np.ndarray, *, linear_filter: bool = True) -> None:
    """Upload ``(nx, ny, nz, 3)`` uint8 cryo data as a GL ``sampler3D``."""
    import ctypes  # noqa: PLC0415

    if host_volume is None:
        return
    if host_volume.ndim != 4 or host_volume.shape[-1] != 3 or host_volume.dtype != np.uint8:
        raise ValueError(
            f"expected cryo host volume (nx, ny, nz, 3) uint8, got shape={host_volume.shape} dtype={host_volume.dtype}"
        )

    filter_mode = "linear" if bool(linear_filter) else "nearest"
    source_key = (id(host_volume), tuple(int(v) for v in host_volume.shape), str(host_volume.dtype), filter_mode)
    if (
        getattr(mesh_gl, "_cutting_cryo_volume_texture", None) is not None
        and getattr(mesh_gl, "_cutting_cryo_volume_source_key", None) == source_key
    ):
        return

    from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

    gl = RendererGL.gl
    _delete_cryo_volume_resources(mesh_gl)

    nx, ny, nz = (int(host_volume.shape[i]) for i in range(3))
    max_size = gl.GLint()
    gl.glGetIntegerv(gl.GL_MAX_3D_TEXTURE_SIZE, max_size)
    if max(nx, ny, nz) > int(max_size.value):
        raise ValueError(
            f"cryo volume shape {(nx, ny, nz)} exceeds GL_MAX_3D_TEXTURE_SIZE={int(max_size.value)}"
        )
    upload = np.ascontiguousarray(np.asarray(host_volume).transpose(2, 1, 0, 3))
    texture_id = gl.GLuint()
    gl.glGenTextures(1, texture_id)
    gl.glBindTexture(gl.GL_TEXTURE_3D, texture_id)
    prev_alignment = gl.GLint()
    gl.glGetIntegerv(gl.GL_UNPACK_ALIGNMENT, prev_alignment)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    gl_filter = gl.GL_LINEAR if bool(linear_filter) else gl.GL_NEAREST
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MIN_FILTER, gl_filter)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_MAG_FILTER, gl_filter)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_3D, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
    gl.glTexImage3D(
        gl.GL_TEXTURE_3D,
        0,
        gl.GL_RGB8,
        nx,
        ny,
        nz,
        0,
        gl.GL_RGB,
        gl.GL_UNSIGNED_BYTE,
        upload.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
    )
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, prev_alignment.value)
    gl.glBindTexture(gl.GL_TEXTURE_3D, 0)

    mesh_gl._cutting_cryo_volume_texture = texture_id
    mesh_gl._cutting_cryo_volume_source_key = source_key


def _upload_cryo_uv3_attrib(
    mesh_gl,
    vertex_uv3: wp.array,
    *,
    revision: int | None = None,
    source_id: int | None = None,
) -> None:
    """Attach static MC vertex UV3s as MeshGL vertex attribute location 9."""
    import ctypes  # noqa: PLC0415

    from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

    gl = RendererGL.gl
    count = int(vertex_uv3.shape[0])
    base_id = id(vertex_uv3) if source_id is None else int(source_id)
    source_key = (base_id, count) if revision is None else (base_id, count, int(revision))
    if (
        getattr(mesh_gl, "_cutting_cryo_uv3_vbo", None) is not None
        and getattr(mesh_gl, "_cutting_cryo_uv3_source_key", None) == source_key
        and getattr(mesh_gl, "_cutting_cryo_uv3_count", None) == count
    ):
        return

    uv3_vbo = getattr(mesh_gl, "_cutting_cryo_uv3_vbo", None)
    if uv3_vbo is None:
        uv3_vbo = gl.GLuint()
        gl.glGenBuffers(1, uv3_vbo)
    gl.glBindVertexArray(mesh_gl.vao)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, uv3_vbo)
    nbytes = count * 3 * 4
    uploaded = False
    if _mesh_gl_interop_requested(mesh_gl):
        gl.glBufferData(gl.GL_ARRAY_BUFFER, nbytes, None, gl.GL_STATIC_DRAW)
        try:
            cuda_buffer = wp.RegisteredGLBuffer(
                int(uv3_vbo.value),
                mesh_gl.device,
                flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            )
            if getattr(cuda_buffer, "resource", None) is not None:
                mapped = cuda_buffer.map(dtype=wp.vec3, shape=vertex_uv3.shape)
                wp.copy(mapped, vertex_uv3)
                cuda_buffer.unmap()
                mesh_gl._cutting_cryo_uv3_cuda_buffer = cuda_buffer
                uploaded = True
        except Exception:
            mesh_gl._cutting_cryo_uv3_cuda_buffer = None
    if not uploaded:
        host_uv3 = np.ascontiguousarray(vertex_uv3.numpy().reshape((count, 3)).astype(np.float32, copy=False))
        gl.glBufferData(gl.GL_ARRAY_BUFFER, host_uv3.nbytes, host_uv3.ctypes.data, gl.GL_STATIC_DRAW)
    gl.glVertexAttribPointer(9, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, ctypes.c_void_p(0))
    gl.glEnableVertexAttribArray(9)
    gl.glBindVertexArray(0)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

    mesh_gl._cutting_cryo_uv3_vbo = uv3_vbo
    mesh_gl._cutting_cryo_uv3_source_key = source_key
    mesh_gl._cutting_cryo_uv3_count = count


def _gl_current_program(gl) -> int:
    program = gl.GLint()
    gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM, program)
    return int(program.value)


def _uniform_location(gl, program: int, name: str) -> int:
    import ctypes  # noqa: PLC0415

    return int(gl.glGetUniformLocation(program, ctypes.c_char_p(name.encode("utf-8"))))


def _set_cryo_volume_uniforms(mesh_gl, enabled: bool) -> None:
    from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

    gl = RendererGL.gl
    program = _gl_current_program(gl)
    if program <= 0:
        return
    loc_enabled = _uniform_location(gl, program, "cryo_volume_enabled")
    if loc_enabled < 0:
        return
    loc_texture = _uniform_location(gl, program, "cryo_volume")
    if loc_texture >= 0:
        gl.glUniform1i(loc_texture, 3)
    gl.glUniform1i(loc_enabled, int(bool(enabled)))
    if not enabled:
        return

    texture_id = getattr(mesh_gl, "_cutting_cryo_volume_texture", None)
    if texture_id is None:
        gl.glUniform1i(loc_enabled, 0)
        return

    loc_scale = _uniform_location(gl, program, "cryo_volume_scale")
    loc_axis = _uniform_location(gl, program, "cryo_volume_src_axis")
    loc_flip = _uniform_location(gl, program, "cryo_volume_flip")
    gl.glActiveTexture(gl.GL_TEXTURE3)
    gl.glBindTexture(gl.GL_TEXTURE_3D, texture_id)
    if loc_scale >= 0:
        gl.glUniform3f(loc_scale, *getattr(mesh_gl, "_cutting_cryo_volume_scale", (1.0, 1.0, 1.0)))
    if loc_axis >= 0:
        gl.glUniform3i(loc_axis, *getattr(mesh_gl, "_cutting_cryo_volume_src_axis", (0, 1, 2)))
    if loc_flip >= 0:
        gl.glUniform3i(loc_flip, *getattr(mesh_gl, "_cutting_cryo_volume_flip", (0, 0, 0)))


def _enable_cryo_volume_render(
    mesh_gl,
    host_volume: np.ndarray,
    vertex_uv3: wp.array,
    *,
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    src_axis: tuple[int, int, int] = (0, 1, 2),
    flip: tuple[bool, bool, bool] = (False, False, False),
    linear_filter: bool = True,
    uv3_revision: int | None = None,
    uv3_source_id: int | None = None,
) -> None:
    """Patch a MeshGL instance to sample a static GL 3D texture in shader."""
    if not hasattr(mesh_gl, "vao"):
        return
    _ensure_cryo_volume_destroy_patch(mesh_gl)
    _upload_cryo_volume_texture(mesh_gl, host_volume, linear_filter=linear_filter)
    if uv3_revision is None and uv3_source_id is None:
        _upload_cryo_uv3_attrib(mesh_gl, vertex_uv3)
    elif uv3_revision is None:
        _upload_cryo_uv3_attrib(mesh_gl, vertex_uv3, source_id=uv3_source_id)
    else:
        _upload_cryo_uv3_attrib(mesh_gl, vertex_uv3, revision=uv3_revision, source_id=uv3_source_id)

    mesh_gl._cutting_cryo_volume_scale = tuple(float(v) for v in scale)
    mesh_gl._cutting_cryo_volume_src_axis = tuple(int(v) for v in src_axis)
    mesh_gl._cutting_cryo_volume_flip = tuple(1 if bool(v) else 0 for v in flip)

    if getattr(mesh_gl, "_cutting_cryo_volume_patched", False):
        return
    orig_render = mesh_gl.render

    def render_patched():
        if mesh_gl.hidden:
            return
        from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

        gl = RendererGL.gl
        old_color = getattr(mesh_gl, "color", (0.7, 0.5, 0.3))
        mesh_gl.color = (1.0, 1.0, 1.0)
        gl.glVertexAttrib3f(7, *mesh_gl.color)
        try:
            _set_cryo_volume_uniforms(mesh_gl, True)
            orig_render()
        finally:
            _set_cryo_volume_uniforms(mesh_gl, False)
            mesh_gl.color = old_color
            gl.glVertexAttrib3f(7, *old_color)

    mesh_gl.render = render_patched
    mesh_gl._cutting_cryo_volume_patched = True


def build_segmentation_color_texture(
    labels: np.ndarray,
    material_colors: np.ndarray,
    *,
    device: str | wp.context.Device | None = None,
) -> tuple[np.ndarray, wp.array]:
    """Build a static RGB volume from integer labels and material colours.

    The host output matches the cryo/CT texture convention: ``uint8``
    ``(nx, ny, nz, 3)``. The device output is a ``wp.array3d(dtype=wp.vec3)`` in
    ``[0, 1]`` for the atlas fallback and diagnostics.
    """
    label_volume = np.asarray(labels)
    if label_volume.ndim != 3 or not np.issubdtype(label_volume.dtype, np.integer):
        raise ValueError(f"expected integer labels with shape (nx, ny, nz), got {label_volume.shape} {label_volume.dtype}")
    if label_volume.size and int(label_volume.min()) < 0:
        raise ValueError("segmentation labels must be non-negative")

    colors = np.asarray(material_colors, dtype=np.float32)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError(f"expected material colors with shape (n, 3), got {colors.shape}")
    max_label = int(label_volume.max()) if label_volume.size else 0
    if max_label >= int(colors.shape[0]):
        raise ValueError(f"segmentation label {max_label} has no material color (n={int(colors.shape[0])})")

    colors_u8 = np.rint(np.clip(colors, 0.0, 1.0) * 255.0).astype(np.uint8)
    host = np.ascontiguousarray(colors_u8[label_volume.astype(np.int64, copy=False)])
    as_float = host.astype(np.float32) / 255.0
    return host, wp.array(as_float, dtype=wp.vec3, device=device)


def _uint32_index_view(indices: wp.array) -> wp.array:
    """Return an unsigned view of a non-negative triangle index buffer."""
    return indices if indices.dtype == wp.uint32 else indices.view(dtype=wp.uint32)


def _mesh_gl_interop_requested(mesh_gl) -> bool:
    """Whether Newton's MeshGL instance should use CUDA/GL interop."""
    try:
        from newton._src.viewer.gl import opengl as newton_gl_opengl  # noqa: PLC0415
    except Exception:
        return False
    device = getattr(mesh_gl, "device", None)
    return bool(getattr(newton_gl_opengl, "ENABLE_CUDA_INTEROP", False) and device is not None and device.is_cuda)


def _update_mesh_indices_cuda(mesh_gl, indices: wp.array) -> bool:
    """Refresh a MeshGL EBO through CUDA/GL interop when available."""
    if not _mesh_gl_interop_requested(mesh_gl):
        return False
    if not hasattr(mesh_gl, "ebo"):
        return False

    from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

    gl = RendererGL.gl
    indices_u32 = _uint32_index_view(indices)
    nbytes = int(len(indices_u32)) * 4

    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_gl.ebo)
    if nbytes > int(getattr(mesh_gl, "ebo_size", 0)):
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, nbytes, None, gl.GL_DYNAMIC_DRAW)
        mesh_gl.ebo_size = nbytes
        mesh_gl._cutting_index_cuda_buffer = None

    try:
        cuda_buffer = getattr(mesh_gl, "_cutting_index_cuda_buffer", None)
        if cuda_buffer is None:
            cuda_buffer = wp.RegisteredGLBuffer(
                int(mesh_gl.ebo.value),
                mesh_gl.device,
                flags=wp.RegisteredGLBuffer.WRITE_DISCARD,
            )
            # Warp may fall back to a staging copy if true interop is unavailable.
            # That fallback binds GL_ARRAY_BUFFER internally, so keep the explicit
            # EBO host path below for element buffers.
            if getattr(cuda_buffer, "resource", None) is None:
                mesh_gl._cutting_index_cuda_buffer = None
                return False
            mesh_gl._cutting_index_cuda_buffer = cuda_buffer
        mapped = cuda_buffer.map(dtype=wp.uint32, shape=indices_u32.shape)
        wp.copy(mapped, indices_u32)
        cuda_buffer.unmap()
    except Exception:
        mesh_gl._cutting_index_cuda_buffer = None
        return False

    mesh_gl.indices = indices_u32
    mesh_gl.num_indices = int(len(indices_u32))
    return True


def _precreate_mesh_gl_for_gpu_indices(
    viewer: newton.viewer.ViewerBase,
    name: str,
    num_points: int,
    num_indices: int,
    *,
    hidden: bool,
    backface_culling: bool = True,
) -> bool:
    """Create MeshGL before log_mesh so we can seed its EBO ourselves."""
    if not hasattr(viewer, "objects") or name in viewer.objects:
        return False
    if viewer.__class__.__name__ != "ViewerGL":
        return False
    try:
        from newton._src.viewer.gl import opengl as newton_gl_opengl  # noqa: PLC0415
        from newton._src.viewer.gl.opengl import MeshGL  # noqa: PLC0415
    except Exception:
        return False
    device = getattr(viewer, "device", None)
    if not bool(getattr(newton_gl_opengl, "ENABLE_CUDA_INTEROP", False) and device is not None and device.is_cuda):
        return False
    viewer.objects[name] = MeshGL(
        int(num_points),
        int(num_indices),
        device,
        hidden=hidden,
        backface_culling=backface_culling,
    )
    return True


def _update_mesh_indices(mesh_gl, indices: wp.array) -> bool:
    """Refresh a Newton MeshGL element buffer without recreating the object."""
    if not hasattr(mesh_gl, "ebo"):
        return False
    if _update_mesh_indices_cuda(mesh_gl, indices):
        return True
    from newton._src.viewer.gl.opengl import RendererGL  # noqa: PLC0415

    gl = RendererGL.gl
    mesh_gl.indices = _uint32_index_view(indices)
    host_indices = mesh_gl.indices.numpy()
    nbytes = int(host_indices.nbytes)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, mesh_gl.ebo)
    if nbytes > int(getattr(mesh_gl, "ebo_size", 0)):
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, nbytes, host_indices.ctypes.data, gl.GL_DYNAMIC_DRAW)
        mesh_gl.ebo_size = nbytes
    else:
        gl.glBufferSubData(gl.GL_ELEMENT_ARRAY_BUFFER, 0, nbytes, host_indices.ctypes.data)
    mesh_gl.num_indices = int(len(mesh_gl.indices))
    return True


@wp.func
def _select_axis(uv: wp.vec3, axis: int) -> float:
    if axis == 0:
        return uv[0]
    if axis == 1:
        return uv[1]
    return uv[2]


@wp.func
def _scale_about_centre(v: float, s: float) -> float:
    """Scale a normalised [0,1] coordinate about 0.5 by ``s`` (>1 zooms in)."""
    return (v - 0.5) / s + 0.5


@wp.kernel
def _sample_3d_texture_kernel(
    vertex_uv3: wp.array(dtype=wp.vec3),
    texture: wp.array3d(dtype=wp.vec3),
    tex_nx: int,
    tex_ny: int,
    tex_nz: int,
    src_x: int,
    src_y: int,
    src_z: int,
    flip_x: int,
    flip_y: int,
    flip_z: int,
    scale_x: float,
    scale_y: float,
    scale_z: float,
    out_rgb: wp.array(dtype=wp.vec3),
):
    """Nearest-neighbour sample the 3D texture at each vertex's UV3.

    ``src_{x,y,z}`` = signed-permutation source axis (0/1/2) per texture
    axis. ``flip_{x,y,z}`` mirror the texture index after permutation.
    ``scale_{x,y,z}`` zoom each texture axis about the centre (0.5) - a
    scale of 2 samples the middle half of the volume along that axis,
    useful when the cryosection mouse body only fills a central fraction
    of the image frame.
    """
    i = wp.tid()
    uv = vertex_uv3[i]
    tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)
    tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)
    tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)
    ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)
    iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)
    iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)
    if flip_x != 0:
        ix = (tex_nx - 1) - ix
    if flip_y != 0:
        iy = (tex_ny - 1) - iy
    if flip_z != 0:
        iz = (tex_nz - 1) - iz
    out_rgb[i] = texture[ix, iy, iz]


@wp.kernel
def _expand_triangles_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    vertex_pos: wp.array(dtype=wp.vec3),
    vertex_uv3: wp.array(dtype=wp.vec3),
    num_triangles: int,
    flat_pos: wp.array(dtype=wp.vec3),
    tri_centroid_uv3: wp.array(dtype=wp.vec3),
):
    """Expand an indexed MC mesh into flat per-triangle vertex triplets.

    Produces ``flat_pos[3*num_triangles]`` with no vertex reuse and
    ``tri_centroid_uv3[num_triangles]`` holding the average UV3 of each
    triangle's three source vertices. Flat shading lets us encode a
    per-triangle colour via a texture atlas without hitting the GL
    bilinear interpolation problem.
    """
    t = wp.tid()
    if t >= num_triangles:
        return
    v0 = tri_indices[t, 0]
    v1 = tri_indices[t, 1]
    v2 = tri_indices[t, 2]
    flat_pos[t * 3 + 0] = vertex_pos[v0]
    flat_pos[t * 3 + 1] = vertex_pos[v1]
    flat_pos[t * 3 + 2] = vertex_pos[v2]
    tri_centroid_uv3[t] = (vertex_uv3[v0] + vertex_uv3[v1] + vertex_uv3[v2]) * (1.0 / 3.0)


@wp.kernel
def _expand_triangle_vertices_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    vertex_pos: wp.array(dtype=wp.vec3),
    vertex_uv3: wp.array(dtype=wp.vec3),
    num_triangles: int,
    flat_pos: wp.array(dtype=wp.vec3),
    flat_uv3: wp.array(dtype=wp.vec3),
):
    """Expand indexed triangles into flat per-corner positions and UV3s."""
    t = wp.tid()
    if t >= num_triangles:
        return
    v0 = tri_indices[t, 0]
    v1 = tri_indices[t, 1]
    v2 = tri_indices[t, 2]
    base = t * 3
    flat_pos[base + 0] = vertex_pos[v0]
    flat_pos[base + 1] = vertex_pos[v1]
    flat_pos[base + 2] = vertex_pos[v2]
    flat_uv3[base + 0] = vertex_uv3[v0]
    flat_uv3[base + 1] = vertex_uv3[v1]
    flat_uv3[base + 2] = vertex_uv3[v2]


@wp.kernel
def _expand_triangle_vertices_with_procedural_coord_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    vertex_pos: wp.array(dtype=wp.vec3),
    vertex_uv3: wp.array(dtype=wp.vec3),
    procedural_coord: wp.array(dtype=wp.vec3),
    num_triangles: int,
    flat_pos: wp.array(dtype=wp.vec3),
    flat_uv3: wp.array(dtype=wp.vec3),
    flat_procedural_coord: wp.array(dtype=wp.vec3),
):
    """Expand indexed triangles into flat positions, UV3s, and procedural coords."""
    t = wp.tid()
    if t >= num_triangles:
        return
    v0 = tri_indices[t, 0]
    v1 = tri_indices[t, 1]
    v2 = tri_indices[t, 2]
    base = t * 3
    flat_pos[base + 0] = vertex_pos[v0]
    flat_pos[base + 1] = vertex_pos[v1]
    flat_pos[base + 2] = vertex_pos[v2]
    flat_uv3[base + 0] = vertex_uv3[v0]
    flat_uv3[base + 1] = vertex_uv3[v1]
    flat_uv3[base + 2] = vertex_uv3[v2]
    flat_procedural_coord[base + 0] = procedural_coord[v0]
    flat_procedural_coord[base + 1] = procedural_coord[v1]
    flat_procedural_coord[base + 2] = procedural_coord[v2]


@wp.kernel
def _expand_triangle_vertices_majority_uv3_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    vertex_pos: wp.array(dtype=wp.vec3),
    particle_uv3: wp.array(dtype=wp.vec3),
    particle_material: wp.array(dtype=wp.int32),
    num_triangles: int,
    flat_pos: wp.array(dtype=wp.vec3),
    flat_uv3: wp.array(dtype=wp.vec3),
):
    """Expand triangles and give each triangle one categorical material UV3."""
    t = wp.tid()
    if t >= num_triangles:
        return
    v0 = tri_indices[t, 0]
    v1 = tri_indices[t, 1]
    v2 = tri_indices[t, 2]
    base = t * 3
    flat_pos[base + 0] = vertex_pos[v0]
    flat_pos[base + 1] = vertex_pos[v1]
    flat_pos[base + 2] = vertex_pos[v2]

    p0 = v0 / 6
    p1 = v1 / 6
    p2 = v2 / 6
    m0 = particle_material[p0]
    m1 = particle_material[p1]
    m2 = particle_material[p2]

    chosen = p0
    if m1 == m2:
        chosen = p1
    elif m0 == m1 or m0 == m2:
        chosen = p0
    uv = particle_uv3[chosen]
    flat_uv3[base + 0] = uv
    flat_uv3[base + 1] = uv
    flat_uv3[base + 2] = uv


@wp.kernel
def _flatten_triangle_indices_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    num_triangles: int,
    flat_indices: wp.array(dtype=wp.int32),
):
    """Flatten ``int32[num_triangles, 3]`` MC indices without a host round-trip."""
    t = wp.tid()
    if t >= num_triangles:
        return
    base = t * 3
    flat_indices[base + 0] = tri_indices[t, 0]
    flat_indices[base + 1] = tri_indices[t, 1]
    flat_indices[base + 2] = tri_indices[t, 2]


@wp.kernel
def _expand_triangle_positions_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    vertex_pos: wp.array(dtype=wp.vec3),
    num_triangles: int,
    flat_pos: wp.array(dtype=wp.vec3),
):
    """Positions-only variant of :func:`_expand_triangle_vertices_kernel`.

    Used on topology-cache hits where ``flat_uv3`` is already valid from a
    prior expansion and only the deformed positions need refreshing.
    """
    t = wp.tid()
    if t >= num_triangles:
        return
    v0 = tri_indices[t, 0]
    v1 = tri_indices[t, 1]
    v2 = tri_indices[t, 2]
    base = t * 3
    flat_pos[base + 0] = vertex_pos[v0]
    flat_pos[base + 1] = vertex_pos[v1]
    flat_pos[base + 2] = vertex_pos[v2]


@wp.kernel
def _expand_triangle_normals_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    vertex_normals: wp.array(dtype=wp.vec3),
    num_triangles: int,
    flat_normals: wp.array(dtype=wp.vec3),
):
    """Expand indexed per-vertex normals onto flat triangle corners."""
    t = wp.tid()
    if t >= num_triangles:
        return
    v0 = tri_indices[t, 0]
    v1 = tri_indices[t, 1]
    v2 = tri_indices[t, 2]
    base = t * 3
    flat_normals[base + 0] = vertex_normals[v0]
    flat_normals[base + 1] = vertex_normals[v1]
    flat_normals[base + 2] = vertex_normals[v2]


@wp.kernel
def _compute_flat_triangle_normals_kernel(
    flat_pos: wp.array(dtype=wp.vec3),
    num_triangles: int,
    flat_normals: wp.array(dtype=wp.vec3),
):
    """Write one geometric normal to all three corners of each flat triangle."""
    t = wp.tid()
    if t >= num_triangles:
        return
    base = t * 3
    p0 = flat_pos[base + 0]
    p1 = flat_pos[base + 1]
    p2 = flat_pos[base + 2]
    n = wp.cross(p1 - p0, p2 - p0)
    if wp.dot(n, n) > 1.0e-16:
        n = wp.normalize(n)
    else:
        n = wp.vec3(0.0, 1.0, 0.0)
    flat_normals[base + 0] = n
    flat_normals[base + 1] = n
    flat_normals[base + 2] = n


@wp.kernel
def _expand_triangle_material_state_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    particle_material: wp.array(dtype=wp.int32),
    num_triangles: int,
    flat_material_id: wp.array(dtype=wp.int32),
    flat_state_rgba: wp.array(dtype=wp.vec4),
):
    """Write categorical majority material and neutral state per flat corner."""
    t = wp.tid()
    if t >= num_triangles:
        return
    v0 = tri_indices[t, 0]
    v1 = tri_indices[t, 1]
    v2 = tri_indices[t, 2]
    p0 = v0 / 6
    p1 = v1 / 6
    p2 = v2 / 6
    m0 = particle_material[p0]
    m1 = particle_material[p1]
    m2 = particle_material[p2]

    mat = m0
    if m1 == m2:
        mat = m1
    elif m0 == m1 or m0 == m2:
        mat = m0

    base = t * 3
    zero = wp.vec4(0.0, 0.0, 0.0, 0.0)
    flat_material_id[base + 0] = mat
    flat_material_id[base + 1] = mat
    flat_material_id[base + 2] = mat
    flat_state_rgba[base + 0] = zero
    flat_state_rgba[base + 1] = zero
    flat_state_rgba[base + 2] = zero


@wp.kernel
def _accumulate_vertex_neighbours_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    vertex_pos: wp.array(dtype=wp.vec3),
    num_triangles: int,
    neighbour_sum: wp.array(dtype=wp.vec3),
    neighbour_degree: wp.array(dtype=wp.int32),
):
    """Accumulate one-ring neighbour sums for Laplacian/Taubin smoothing."""
    t = wp.tid()
    if t >= num_triangles:
        return
    v0 = tri_indices[t, 0]
    v1 = tri_indices[t, 1]
    v2 = tri_indices[t, 2]
    p0 = vertex_pos[v0]
    p1 = vertex_pos[v1]
    p2 = vertex_pos[v2]

    wp.atomic_add(neighbour_sum, v0, p1 + p2)
    wp.atomic_add(neighbour_sum, v1, p0 + p2)
    wp.atomic_add(neighbour_sum, v2, p0 + p1)

    wp.atomic_add(neighbour_degree, v0, 2)
    wp.atomic_add(neighbour_degree, v1, 2)
    wp.atomic_add(neighbour_degree, v2, 2)


@wp.kernel
def _apply_laplacian_step_kernel(
    src_pos: wp.array(dtype=wp.vec3),
    neighbour_sum: wp.array(dtype=wp.vec3),
    neighbour_degree: wp.array(dtype=wp.int32),
    coeff: float,
    dst_pos: wp.array(dtype=wp.vec3),
):
    """Apply one uniform-Laplacian smoothing step with coefficient ``coeff``."""
    vid = wp.tid()
    degree = neighbour_degree[vid]
    p = src_pos[vid]
    if degree <= 0:
        dst_pos[vid] = p
        return
    avg = neighbour_sum[vid] * (1.0 / float(degree))
    dst_pos[vid] = p + (avg - p) * coeff


@wp.kernel
def _sample_triangle_texture_kernel(
    tri_centroid_uv3: wp.array(dtype=wp.vec3),
    texture: wp.array3d(dtype=wp.vec3),
    tex_nx: int,
    tex_ny: int,
    tex_nz: int,
    src_x: int,
    src_y: int,
    src_z: int,
    flip_x: int,
    flip_y: int,
    flip_z: int,
    scale_x: float,
    scale_y: float,
    scale_z: float,
    num_triangles: int,
    tri_rgb: wp.array(dtype=wp.vec3),
):
    """Sample the 3D texture at each triangle's centroid UV3."""
    t = wp.tid()
    if t >= num_triangles:
        return
    uv = tri_centroid_uv3[t]
    tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)
    tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)
    tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)
    ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)
    iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)
    iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)
    if flip_x != 0:
        ix = (tex_nx - 1) - ix
    if flip_y != 0:
        iy = (tex_ny - 1) - iy
    if flip_z != 0:
        iz = (tex_nz - 1) - iz
    tri_rgb[t] = texture[ix, iy, iz]


@wp.kernel
def _fill_triangle_atlas_kernel(
    tri_rgb: wp.array(dtype=wp.vec3),
    atlas_width: int,
    atlas_height: int,
    tile_size: int,
    num_triangles: int,
    atlas_rgb: wp.array2d(dtype=wp.vec3),
):
    """Paint each triangle's colour into its ``tile_size x tile_size`` tile."""
    tx, ty, sub = wp.tid()
    tiles_per_row = atlas_width / tile_size
    tid = ty * tiles_per_row + tx
    if tid >= num_triangles:
        return
    px = sub % tile_size
    py = sub / tile_size
    x = tx * tile_size + px
    y = ty * tile_size + py
    if x >= atlas_width or y >= atlas_height:
        return
    atlas_rgb[y, x] = tri_rgb[tid]


@wp.kernel
def _fill_triangle_material_atlas_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    particle_material: wp.array(dtype=wp.int32),
    material_colors: wp.array(dtype=wp.vec3),
    atlas_width: int,
    atlas_height: int,
    tile_size: int,
    num_triangles: int,
    atlas_rgb: wp.array2d(dtype=wp.vec3),
):
    """Paint each MC triangle tile with the majority material of its owners."""
    tx, ty, sub = wp.tid()
    tiles_per_row = atlas_width / tile_size
    tid = ty * tiles_per_row + tx
    if tid >= num_triangles:
        return

    px = sub % tile_size
    py = sub / tile_size
    x = tx * tile_size + px
    y = ty * tile_size + py
    if x >= atlas_width or y >= atlas_height:
        return

    p0 = tri_indices[tid, 0] / 6
    p1 = tri_indices[tid, 1] / 6
    p2 = tri_indices[tid, 2] / 6
    m0 = particle_material[p0]
    m1 = particle_material[p1]
    m2 = particle_material[p2]

    mat = m0
    if m1 == m2:
        mat = m1
    elif m0 == m1 or m0 == m2:
        mat = m0

    atlas_rgb[y, x] = material_colors[mat]


@wp.func
def _lerp_vec3(a: wp.vec3, b: wp.vec3, t: float) -> wp.vec3:
    return a + (b - a) * t


@wp.func
def _cold_warm_stress_color(value: float) -> wp.vec3:
    """Cold-blue to hot-red display ramp for normalized stress values."""
    x = wp.clamp(value, 0.0, 1.0)
    c0 = wp.vec3(0.02, 0.08, 0.70)
    c1 = wp.vec3(0.00, 0.72, 1.00)
    c2 = wp.vec3(0.05, 0.88, 0.18)
    c3 = wp.vec3(1.00, 0.90, 0.05)
    c4 = wp.vec3(1.00, 0.04, 0.00)
    if x < 0.25:
        return _lerp_vec3(c0, c1, x * 4.0)
    if x < 0.50:
        return _lerp_vec3(c1, c2, (x - 0.25) * 4.0)
    if x < 0.75:
        return _lerp_vec3(c2, c3, (x - 0.50) * 4.0)
    return _lerp_vec3(c3, c4, (x - 0.75) * 4.0)


@wp.kernel
def _compute_triangle_tile_uvs_kernel(
    atlas_width: int,
    atlas_height: int,
    tile_size: int,
    num_triangles: int,
    flat_uvs: wp.array(dtype=wp.vec2),
):
    """Assign each flat triangle to a right-triangle footprint inside its tile."""
    t = wp.tid()
    if t >= num_triangles:
        return
    tiles_per_row = atlas_width / tile_size
    tile_x = t % tiles_per_row
    tile_y = t / tiles_per_row
    x0 = float(tile_x * tile_size)
    y0 = float(tile_y * tile_size)
    if tile_size <= 1:
        cx = x0 + 0.5
        cy = y0 + 0.5
        uv = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))
        flat_uvs[t * 3 + 0] = uv
        flat_uvs[t * 3 + 1] = uv
        flat_uvs[t * 3 + 2] = uv
        return
    ax = x0 + 0.5
    ay = y0 + 0.5
    bx = x0 + float(tile_size) - 0.5
    by = y0 + 0.5
    cx = x0 + 0.5
    cy = y0 + float(tile_size) - 0.5
    # Newton flips uploaded numpy textures vertically before glTexImage2D, so
    # atlas rows written in image space (row 0 = top) must be mapped to GL
    # texture space (v = 1 at the top) here.
    flat_uvs[t * 3 + 0] = wp.vec2(ax / float(atlas_width), 1.0 - (ay / float(atlas_height)))
    flat_uvs[t * 3 + 1] = wp.vec2(bx / float(atlas_width), 1.0 - (by / float(atlas_height)))
    flat_uvs[t * 3 + 2] = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))


@wp.kernel
def _compute_vertex_atlas_uvs_kernel(
    atlas_width: int,
    atlas_height: int,
    tile_size: int,
    num_vertices: int,
    out_uvs: wp.array(dtype=wp.vec2),
):
    """Map each vertex id to the centre of its tile in a 2D colour atlas.

    Each vertex owns a ``tile_size x tile_size`` block of pixels so that GL
    bilinear filtering pulls in the same colour from all neighbours and
    doesn't bleed across vertices.
    """
    i = wp.tid()
    if i >= num_vertices:
        out_uvs[i] = wp.vec2(0.0, 0.0)
        return
    tiles_per_row = atlas_width / tile_size
    tile_x = i % tiles_per_row
    tile_y = i / tiles_per_row
    cx = float(tile_x * tile_size + tile_size / 2)
    cy = float(tile_y * tile_size + tile_size / 2)
    out_uvs[i] = wp.vec2(cx / float(atlas_width), 1.0 - (cy / float(atlas_height)))


@wp.kernel
def _fill_atlas_tiles_kernel(
    vertex_rgb: wp.array(dtype=wp.vec3),
    atlas_width: int,
    atlas_height: int,
    tile_size: int,
    num_vertices: int,
    atlas_rgb: wp.array2d(dtype=wp.vec3),
):
    """Paint each vertex's colour into a ``tile_size x tile_size`` block."""
    tx, ty, sub = wp.tid()
    tiles_per_row = atlas_width / tile_size
    vid = ty * tiles_per_row + tx
    if vid >= num_vertices:
        return
    px = sub % tile_size
    py = sub / tile_size
    x = tx * tile_size + px
    y = ty * tile_size + py
    if x >= atlas_width or y >= atlas_height:
        return
    atlas_rgb[y, x] = vertex_rgb[vid]


@wp.func
def _triangle_barycentric(p: wp.vec2, a: wp.vec2, b: wp.vec2, c: wp.vec2) -> wp.vec3:
    v0 = b - a
    v1 = c - a
    v2 = p - a
    den = v0[0] * v1[1] - v1[0] * v0[1]
    if wp.abs(den) < 1.0e-8:
        return wp.vec3(1.0, 0.0, 0.0)
    inv_den = 1.0 / den
    w1 = (v2[0] * v1[1] - v1[0] * v2[1]) * inv_den
    w2 = (v0[0] * v2[1] - v2[0] * v0[1]) * inv_den
    w0 = 1.0 - w1 - w2
    return wp.vec3(w0, w1, w2)


@wp.func
def _clamp_barycentric(w: wp.vec3) -> wp.vec3:
    c0 = wp.max(w[0], 0.0)
    c1 = wp.max(w[1], 0.0)
    c2 = wp.max(w[2], 0.0)
    s = c0 + c1 + c2
    if s <= 1.0e-8:
        return wp.vec3(1.0, 0.0, 0.0)
    inv_s = 1.0 / s
    return wp.vec3(c0 * inv_s, c1 * inv_s, c2 * inv_s)


@wp.kernel
def _fill_triangle_stress_atlas_kernel(
    tri_indices: wp.array2d(dtype=wp.int32),
    cell_stretch: wp.array(dtype=wp.float32),
    color_scale: float,
    atlas_width: int,
    atlas_height: int,
    tile_size: int,
    num_triangles: int,
    atlas_rgb: wp.array2d(dtype=wp.vec3),
):
    """Paint per-triangle tiles with barycentric vertex stress colours."""
    tx, ty, sub = wp.tid()
    tiles_per_row = atlas_width / tile_size
    tid = ty * tiles_per_row + tx
    if tid >= num_triangles:
        return

    px = sub % tile_size
    py = sub / tile_size
    x = tx * tile_size + px
    y = ty * tile_size + py
    if x >= atlas_width or y >= atlas_height:
        return

    s0 = cell_stretch[tri_indices[tid, 0] / 6]
    s1 = cell_stretch[tri_indices[tid, 1] / 6]
    s2 = cell_stretch[tri_indices[tid, 2] / 6]
    if tile_size <= 1:
        stress = (s0 + s1 + s2) * (1.0 / 3.0)
    else:
        a = wp.vec2(0.5, 0.5)
        b = wp.vec2(float(tile_size) - 0.5, 0.5)
        c = wp.vec2(0.5, float(tile_size) - 0.5)
        p = wp.vec2(float(px) + 0.5, float(py) + 0.5)
        bary = _clamp_barycentric(_triangle_barycentric(p, a, b, c))
        stress = s0 * bary[0] + s1 * bary[1] + s2 * bary[2]
    atlas_rgb[y, x] = _cold_warm_stress_color(stress * color_scale)


@wp.kernel
def _bake_triangle_texture_kernel(
    flat_vertex_uv3: wp.array(dtype=wp.vec3),
    texture: wp.array3d(dtype=wp.vec3),
    tex_nx: int,
    tex_ny: int,
    tex_nz: int,
    src_x: int,
    src_y: int,
    src_z: int,
    flip_x: int,
    flip_y: int,
    flip_z: int,
    scale_x: float,
    scale_y: float,
    scale_z: float,
    atlas_width: int,
    atlas_height: int,
    tile_size: int,
    num_triangles: int,
    atlas_rgb: wp.array2d(dtype=wp.vec3),
):
    """Bake a small triangle-local texture into each atlas tile."""
    tx, ty, sub = wp.tid()
    tiles_per_row = atlas_width / tile_size
    tid = ty * tiles_per_row + tx
    if tid >= num_triangles:
        return

    px = sub % tile_size
    py = sub / tile_size
    x = tx * tile_size + px
    y = ty * tile_size + py
    if x >= atlas_width or y >= atlas_height:
        return

    if tile_size <= 1:
        uv = (
            flat_vertex_uv3[tid * 3 + 0]
            + flat_vertex_uv3[tid * 3 + 1]
            + flat_vertex_uv3[tid * 3 + 2]
        ) * (1.0 / 3.0)
    else:
        a = wp.vec2(0.5, 0.5)
        b = wp.vec2(float(tile_size) - 0.5, 0.5)
        c = wp.vec2(0.5, float(tile_size) - 0.5)
        p = wp.vec2(float(px) + 0.5, float(py) + 0.5)
        bary = _clamp_barycentric(_triangle_barycentric(p, a, b, c))
        uv = (
            flat_vertex_uv3[tid * 3 + 0] * bary[0]
            + flat_vertex_uv3[tid * 3 + 1] * bary[1]
            + flat_vertex_uv3[tid * 3 + 2] * bary[2]
        )

    tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)
    tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)
    tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)
    ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)
    iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)
    iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)
    if flip_x != 0:
        ix = (tex_nx - 1) - ix
    if flip_y != 0:
        iy = (tex_ny - 1) - iy
    if flip_z != 0:
        iz = (tex_nz - 1) - iz
    atlas_rgb[y, x] = texture[ix, iy, iz]


@wp.kernel
def _gather_active_particles_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_material: wp.array(dtype=wp.int32),
    material_colors: wp.array(dtype=wp.vec3),
    cut_z: float,
    counter: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
    out_colors: wp.array(dtype=wp.vec3),
):
    """Compact active particles into contiguous (point, color) arrays.

    Particles with world-space Z above ``cut_z`` are skipped so the UI
    cut-plane slider can peel away the dorsal tissue.
    """
    i = wp.tid()
    if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:
        return
    if particle_q[i][2] > cut_z:
        return
    idx = wp.atomic_add(counter, 0, 1)
    out_points[idx] = particle_q[i]
    out_colors[idx] = material_colors[particle_material[i]]


@wp.kernel
def _gather_stress_colored_particles_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    cell_stretch: wp.array(dtype=wp.float32),
    color_scale: float,
    cut_z: float,
    counter: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
    out_colors: wp.array(dtype=wp.vec3),
):
    """Compact active cell-centre particles coloured by normalized stretch."""
    i = wp.tid()
    if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:
        return
    if particle_q[i][2] > cut_z:
        return
    idx = wp.atomic_add(counter, 0, 1)
    out_points[idx] = particle_q[i]
    out_colors[idx] = _cold_warm_stress_color(cell_stretch[i] * color_scale)


@wp.kernel
def _gather_cryo_colored_particles_kernel(
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    particle_grid_xyz: wp.array2d(dtype=wp.int32),
    texture: wp.array3d(dtype=wp.vec3),
    inv_grid_nx: float,
    inv_grid_ny: float,
    inv_grid_nz: float,
    tex_nx: int,
    tex_ny: int,
    tex_nz: int,
    src_x: int,
    src_y: int,
    src_z: int,
    flip_x: int,
    flip_y: int,
    flip_z: int,
    scale_x: float,
    scale_y: float,
    scale_z: float,
    cut_z: float,
    counter: wp.array(dtype=wp.int32),
    out_points: wp.array(dtype=wp.vec3),
    out_colors: wp.array(dtype=wp.vec3),
):
    """Compact active particles into (point, rgb) pairs sampled from a 3D texture.

    ``src_{x,y,z}`` + ``flip_{x,y,z}`` span the 48 signed-permutation
    orientations; ``scale_{x,y,z}`` zoom the sampling window about the
    centre to compensate for aspect-ratio mismatch between the raw cryo
    images and the atlas grid (>1 zooms in).
    """
    i = wp.tid()
    if (particle_flags[i] & wp.int32(ParticleFlags.ACTIVE)) == 0:
        return
    if particle_q[i][2] > cut_z:
        return
    gx = float(particle_grid_xyz[i, 0]) * inv_grid_nx
    gy = float(particle_grid_xyz[i, 1]) * inv_grid_ny
    gz = float(particle_grid_xyz[i, 2]) * inv_grid_nz
    uv = wp.vec3(gx, gy, gz)
    tex_u = _scale_about_centre(_select_axis(uv, src_x), scale_x)
    tex_v = _scale_about_centre(_select_axis(uv, src_y), scale_y)
    tex_w = _scale_about_centre(_select_axis(uv, src_z), scale_z)
    ix = wp.clamp(int(tex_u * float(tex_nx)), 0, tex_nx - 1)
    iy = wp.clamp(int(tex_v * float(tex_ny)), 0, tex_ny - 1)
    iz = wp.clamp(int(tex_w * float(tex_nz)), 0, tex_nz - 1)
    if flip_x != 0:
        ix = (tex_nx - 1) - ix
    if flip_y != 0:
        iy = (tex_ny - 1) - iy
    if flip_z != 0:
        iz = (tex_nz - 1) - iz
    idx = wp.atomic_add(counter, 0, 1)
    out_points[idx] = particle_q[i]
    out_colors[idx] = texture[ix, iy, iz]


@wp.kernel
def _gather_grab_constraint_lines_kernel(
    grab_indices: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    pull_target: wp.vec3,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    particle_idx = grab_indices[tid]
    if particle_idx < 0:
        starts[tid] = pull_target
        ends[tid] = pull_target
        return
    starts[tid] = pull_target
    ends[tid] = particle_q[particle_idx]


@wp.func
def _cluster_edge_slot_a(edge_idx: int) -> int:
    slot = int(0)
    if edge_idx == 1:
        slot = 1
    elif edge_idx == 2:
        slot = 2
    elif edge_idx == 3:
        slot = 3
    elif edge_idx == 4:
        slot = 4
    elif edge_idx == 5:
        slot = 5
    elif edge_idx == 6:
        slot = 6
    elif edge_idx == 7:
        slot = 7
    elif edge_idx == 9:
        slot = 1
    elif edge_idx == 10:
        slot = 2
    elif edge_idx == 11:
        slot = 3
    return slot


@wp.func
def _cluster_edge_slot_b(edge_idx: int) -> int:
    slot = int(1)
    if edge_idx == 1:
        slot = 2
    elif edge_idx == 2:
        slot = 3
    elif edge_idx == 3:
        slot = 0
    elif edge_idx == 4:
        slot = 5
    elif edge_idx == 5:
        slot = 6
    elif edge_idx == 6:
        slot = 7
    elif edge_idx == 7:
        slot = 4
    elif edge_idx == 8:
        slot = 4
    elif edge_idx == 9:
        slot = 5
    elif edge_idx == 10:
        slot = 6
    elif edge_idx == 11:
        slot = 7
    return slot


@wp.kernel
def _gather_cluster_box_edges_kernel(
    indices_by_slot: wp.array(dtype=wp.int32),
    cluster_active: wp.array(dtype=wp.int32),
    particle_q: wp.array(dtype=wp.vec3),
    particle_flags: wp.array(dtype=wp.int32),
    num_clusters: int,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
):
    tid = wp.tid()
    cluster_idx = tid // 12
    edge_idx = tid - cluster_idx * 12

    slot_a = _cluster_edge_slot_a(edge_idx)
    slot_b = _cluster_edge_slot_b(edge_idx)
    particle_a = indices_by_slot[slot_a * num_clusters + cluster_idx]
    particle_b = indices_by_slot[slot_b * num_clusters + cluster_idx]

    active_bit = wp.int32(ParticleFlags.ACTIVE)
    if (
        cluster_active[cluster_idx] == 0
        or particle_a < 0
        or particle_b < 0
        or (particle_flags[particle_a] & active_bit) == 0
        or (particle_flags[particle_b] & active_bit) == 0
    ):
        p = wp.vec3(0.0, 0.0, 0.0)
        if particle_a >= 0:
            p = particle_q[particle_a]
        starts[tid] = p
        ends[tid] = p
    else:
        starts[tid] = particle_q[particle_a]
        ends[tid] = particle_q[particle_b]


# Tool overlay line colours. Active (pedal down) gets a heat-red; idle is a
# muted grey so the cut site pops when the electrode lights up.
_TOOL_COLOR_IDLE = (0.25, 0.25, 0.25)
_TOOL_COLOR_ACTIVE = (1.0, 0.25, 0.05)


class SurfaceRenderer:
    """Wraps the MC output in viewer-ready arrays with per-frame reuse.

    The index buffer is resized only when the triangle count grows; the vertex
    buffer is a constant ``num_particles * 6`` since the (particle, direction)
    vertex slots are fixed.
    """

    def __init__(
        self,
        name: str,
        num_particles: int,
        device: wp.context.Device,
        max_triangles: int | None = None,
    ):
        self.name = name
        self.device = device
        # Reuse these scratch arrays between frames. ``indices_scratch`` holds
        # the compact flat form passed to the viewer; its size tracks the
        # high-water triangle count.
        self._indices_scratch: wp.array | None = None
        self._cached_cap: int = 0
        self._num_particles = num_particles
        # Minimal two-index placeholder buffer used when the mesh is hidden,
        # so we can keep calling log_mesh (which requires a non-empty index
        # array) even while the surface is toggled off.
        self._hidden_indices: wp.array | None = None
        # Scratch flags array used when a material visibility mask is passed.
        # Avoids allocating every frame.
        self._visible_flags: wp.array | None = None
        # All-ones mask used when the caller supplies only a cut plane.
        self._all_visible: wp.array | None = None
        # Flat triangle-soup scratch buffers used when textured rendering
        # is requested. Lazily allocated on first textured update.
        self._flat_pos: wp.array | None = None
        self._flat_uv3: wp.array | None = None
        self._flat_procedural_coord: wp.array | None = None
        self._flat_normals: wp.array | None = None
        self._flat_material_id: wp.array | None = None
        self._flat_state_rgba: wp.array | None = None
        self._tri_centroid_uv3: wp.array | None = None
        self._flat_identity_indices: wp.array | None = None
        self._flat_identity_cap: int = 0
        self._flat_uv3_kind: str | None = None
        self._flat_uv3_revision: int = 0
        self._flat_buffer_generation: int = 0
        self._max_triangles: int = int(max_triangles) if max_triangles is not None else 0
        self._vertex_cap: int = 0
        self._smoothed_pos_a: wp.array | None = None
        self._smoothed_pos_b: wp.array | None = None
        self._smoothed_procedural_coord_a: wp.array | None = None
        self._smoothed_procedural_coord_b: wp.array | None = None
        self._smoothed_normals: wp.array | None = None
        self._neighbour_sum: wp.array | None = None
        self._neighbour_degree: wp.array | None = None
        # Topology cache. When the caller passes a ``topology_revision`` that
        # matches the last full run AND the visibility mode (material mask /
        # cut_z) has not changed, we can skip the cube_cases + emit kernels
        # and the flat-index rebuild: the flat index buffer
        # and flat UV3 buffer are topology-constant so they stay valid.
        # ``_cached_mode`` captures (material_visible is None, id(...), cut_z)
        # so a mode flip invalidates the cache even at the same revision.
        self._cached_topology_rev: int | None = None
        self._cached_count: int = 0
        self._cached_flat_uv3_valid: bool = False
        # ``_cached_flat_indices_valid`` tracks whether ``_indices_scratch``
        # holds the flat (count * 3,) index buffer for the currently cached
        # topology. We update it lazily: a compute_only call may bump the
        # topology revision without rebuilding the scratch, and the next
        # non-compute_only call must detect that and rebuild it.
        self._cached_flat_indices_valid: bool = False
        self._cached_mode: tuple | None = None

    def _ensure_index_capacity(self, n: int) -> wp.array:
        if self._indices_scratch is None or self._cached_cap < n:
            cap = max(16, n * 2)
            self._indices_scratch = wp.zeros(cap, dtype=wp.int32, device=self.device)
            self._cached_cap = cap
        return self._indices_scratch

    def _ensure_vertex_capacity(self, num_vertices: int) -> None:
        if self._vertex_cap >= num_vertices:
            return
        self._vertex_cap = num_vertices
        self._smoothed_pos_a = wp.zeros(num_vertices, dtype=wp.vec3, device=self.device)
        self._smoothed_pos_b = wp.zeros(num_vertices, dtype=wp.vec3, device=self.device)
        self._smoothed_procedural_coord_a = wp.zeros(num_vertices, dtype=wp.vec3, device=self.device)
        self._smoothed_procedural_coord_b = wp.zeros(num_vertices, dtype=wp.vec3, device=self.device)
        self._smoothed_normals = wp.zeros(num_vertices, dtype=wp.vec3, device=self.device)
        self._neighbour_sum = wp.zeros(num_vertices, dtype=wp.vec3, device=self.device)
        self._neighbour_degree = wp.zeros(num_vertices, dtype=wp.int32, device=self.device)

    def _ensure_flat_identity_capacity(self, n: int) -> wp.array:
        if self._flat_identity_indices is None or self._flat_identity_cap < n:
            cap = max(16, n * 2)
            self._flat_identity_indices = wp.array(
                np.arange(cap, dtype=np.int32),
                dtype=wp.int32,
                device=self.device,
            )
            self._flat_identity_cap = cap
        return self._flat_identity_indices

    def _run_taubin_smoothing(
        self,
        tri_indices: wp.array,
        vertex_pos: wp.array,
        num_triangles: int,
        iterations: int,
        lambda_coeff: float,
        mu_coeff: float,
        *,
        scratch_a: wp.array | None = None,
        scratch_b: wp.array | None = None,
    ) -> wp.array:
        self._ensure_vertex_capacity(int(vertex_pos.shape[0]))
        if scratch_a is None:
            scratch_a = self._smoothed_pos_a
        if scratch_b is None:
            scratch_b = self._smoothed_pos_b
        assert scratch_a is not None
        assert scratch_b is not None
        assert self._neighbour_sum is not None
        assert self._neighbour_degree is not None

        src_pos = vertex_pos
        next_dst = scratch_a
        num_vertices = int(vertex_pos.shape[0])
        for _ in range(iterations):
            self._neighbour_sum.zero_()
            self._neighbour_degree.zero_()
            wp.launch(
                _accumulate_vertex_neighbours_kernel,
                dim=num_triangles,
                inputs=[tri_indices, src_pos, int(num_triangles)],
                outputs=[self._neighbour_sum, self._neighbour_degree],
                device=self.device,
            )
            wp.launch(
                _apply_laplacian_step_kernel,
                dim=num_vertices,
                inputs=[
                    src_pos,
                    self._neighbour_sum,
                    self._neighbour_degree,
                    float(lambda_coeff),
                ],
                outputs=[next_dst],
                device=self.device,
            )

            src_pos = next_dst
            next_dst = scratch_b if src_pos is scratch_a else scratch_a

            self._neighbour_sum.zero_()
            self._neighbour_degree.zero_()
            wp.launch(
                _accumulate_vertex_neighbours_kernel,
                dim=num_triangles,
                inputs=[tri_indices, src_pos, int(num_triangles)],
                outputs=[self._neighbour_sum, self._neighbour_degree],
                device=self.device,
            )
            wp.launch(
                _apply_laplacian_step_kernel,
                dim=num_vertices,
                inputs=[
                    src_pos,
                    self._neighbour_sum,
                    self._neighbour_degree,
                    float(mu_coeff),
                ],
                outputs=[next_dst],
                device=self.device,
            )
            src_pos = next_dst
            next_dst = scratch_b if src_pos is scratch_a else scratch_a
        return src_pos

    def _ensure_flat_surface_capacity(self, max_triangles: int, *, include_material_state: bool = False) -> None:
        max_tris = int(max(1, max_triangles))
        needs_surface_alloc = (
            self._flat_pos is None
            or self._flat_uv3 is None
            or self._flat_procedural_coord is None
            or self._flat_normals is None
            or self._tri_centroid_uv3 is None
            or self._max_triangles < max_tris
        )
        needs_material_alloc = bool(
            include_material_state
            and (
                self._flat_material_id is None
                or self._flat_state_rgba is None
                or int(self._flat_material_id.shape[0]) < max_tris * 3
                or int(self._flat_state_rgba.shape[0]) < max_tris * 3
            )
        )
        if not needs_surface_alloc and not needs_material_alloc:
            return
        if needs_surface_alloc:
            self._max_triangles = max_tris
            self._flat_pos = wp.zeros(max_tris * 3, dtype=wp.vec3, device=self.device)
            self._flat_uv3 = wp.zeros(max_tris * 3, dtype=wp.vec3, device=self.device)
            self._flat_procedural_coord = wp.zeros(max_tris * 3, dtype=wp.vec3, device=self.device)
            self._flat_normals = wp.zeros(max_tris * 3, dtype=wp.vec3, device=self.device)
            self._tri_centroid_uv3 = wp.zeros(max_tris, dtype=wp.vec3, device=self.device)
            self._cached_flat_uv3_valid = False
            self._flat_uv3_kind = None
        if needs_material_alloc:
            self._flat_material_id = wp.zeros(max_tris * 3, dtype=wp.int32, device=self.device)
            self._flat_state_rgba = wp.zeros(max_tris * 3, dtype=wp.vec4, device=self.device)
        self._flat_buffer_generation += 1

    def update_cryo_surface_frame(
        self,
        viewer: newton.viewer.ViewerBase,
        aux: GridAuxState,
        particle_q: wp.array,
        particle_flags: wp.array,
        orientation: wp.array,
        tables: MarchingCubesTables,
        buffers: MarchingCubesBuffers,
        mc_factor: float,
        material_visible: wp.array | None = None,
        smooth_normals: bool = True,
        taubin_iterations: int = 2,
        taubin_lambda: float = 0.33,
        taubin_mu: float = -0.34,
        cut_z: float = float("inf"),
        topology_revision: int | None = None,
        dirty_particle_ids: wp.array | None = None,
        dirty_particle_count: int = 0,
        dirty_particle_count_device: wp.array | None = None,
        dirty_particle_capacity: int | None = None,
        dirty_topology_revision: int | None = None,
        include_material_state: bool = False,
    ) -> CryoSurfaceFrame | None:
        """Prepare flat MC buffers for a Slang direct-3D cryo surface draw."""
        count = self.update(
            viewer=viewer,
            aux=aux,
            particle_q=particle_q,
            particle_flags=particle_flags,
            orientation=orientation,
            tables=tables,
            buffers=buffers,
            mc_factor=mc_factor,
            material_visible=material_visible,
            smooth_normals=False,
            taubin_iterations=0,
            cut_z=cut_z,
            topology_revision=topology_revision,
            dirty_particle_ids=dirty_particle_ids,
            dirty_particle_count=dirty_particle_count,
            dirty_particle_count_device=dirty_particle_count_device,
            dirty_particle_capacity=dirty_particle_capacity,
            dirty_topology_revision=dirty_topology_revision,
            compute_only=True,
        )
        if count <= 0:
            return None

        self._ensure_flat_surface_capacity(buffers.max_triangles, include_material_state=include_material_state)
        assert self._flat_pos is not None
        assert self._flat_uv3 is not None
        assert self._flat_procedural_coord is not None
        assert self._flat_normals is not None

        render_vertex_pos = buffers.vertex_pos
        render_procedural_coord = buffers.vertex_uv3
        if taubin_iterations > 0:
            with _scoped_timer("mc.taubin"):
                render_vertex_pos = self._run_taubin_smoothing(
                    tri_indices=buffers.tri_indices,
                    vertex_pos=buffers.vertex_pos,
                    num_triangles=int(count),
                    iterations=int(taubin_iterations),
                    lambda_coeff=float(taubin_lambda),
                    mu_coeff=float(taubin_mu),
                )
                assert self._smoothed_procedural_coord_a is not None
                assert self._smoothed_procedural_coord_b is not None
                render_procedural_coord = self._run_taubin_smoothing(
                    tri_indices=buffers.tri_indices,
                    vertex_pos=buffers.vertex_uv3,
                    num_triangles=int(count),
                    iterations=int(taubin_iterations),
                    lambda_coeff=float(taubin_lambda),
                    mu_coeff=float(taubin_mu),
                    scratch_a=self._smoothed_procedural_coord_a,
                    scratch_b=self._smoothed_procedural_coord_b,
                )

        with _scoped_timer("mc.slang_expand"):
            wp.launch(
                _expand_triangle_vertices_with_procedural_coord_kernel,
                dim=count,
                inputs=[
                    buffers.tri_indices,
                    render_vertex_pos,
                    buffers.vertex_uv3,
                    render_procedural_coord,
                    int(count),
                ],
                outputs=[self._flat_pos, self._flat_uv3, self._flat_procedural_coord],
                device=self.device,
            )
            if include_material_state:
                assert self._flat_material_id is not None
                assert self._flat_state_rgba is not None
                wp.launch(
                    _expand_triangle_material_state_kernel,
                    dim=count,
                    inputs=[buffers.tri_indices, aux.particle_material, int(count)],
                    outputs=[self._flat_material_id, self._flat_state_rgba],
                    device=self.device,
                )

        if smooth_normals:
            self._ensure_vertex_capacity(int(render_vertex_pos.shape[0]))
            assert self._smoothed_normals is not None
            idx = self._ensure_index_capacity(int(count) * 3)
            wp.launch(
                _flatten_triangle_indices_kernel,
                dim=count,
                inputs=[buffers.tri_indices, int(count)],
                outputs=[idx],
                device=self.device,
            )
            with _scoped_timer("mc.normals"):
                render_vertex_normals = compute_vertex_normals(
                    render_vertex_pos,
                    idx[: int(count) * 3],
                    normals=self._smoothed_normals,
                    device=self.device,
                )
            wp.launch(
                _expand_triangle_normals_kernel,
                dim=count,
                inputs=[buffers.tri_indices, render_vertex_normals, int(count)],
                outputs=[self._flat_normals],
                device=self.device,
            )
        else:
            wp.launch(
                _compute_flat_triangle_normals_kernel,
                dim=count,
                inputs=[self._flat_pos, int(count)],
                outputs=[self._flat_normals],
                device=self.device,
            )

        vertex_count = int(count) * 3
        material_id = self._flat_material_id[:vertex_count] if include_material_state and self._flat_material_id is not None else None
        state_rgba = self._flat_state_rgba[:vertex_count] if include_material_state and self._flat_state_rgba is not None else None
        return CryoSurfaceFrame(
            positions=self._flat_pos[:vertex_count],
            normals=self._flat_normals[:vertex_count],
            uv3=self._flat_uv3[:vertex_count],
            material_id=material_id,
            state_rgba=state_rgba,
            vertex_count=vertex_count,
            triangle_count=int(count),
            buffer_generation=self._flat_buffer_generation,
            procedural_coord=self._flat_procedural_coord[:vertex_count],
        )

    def update(
        self,
        viewer: newton.viewer.ViewerBase,
        aux: GridAuxState,
        particle_q: wp.array,
        particle_flags: wp.array,
        orientation: wp.array,
        tables: MarchingCubesTables,
        buffers: MarchingCubesBuffers,
        mc_factor: float,
        hidden: bool = False,
        material_visible: wp.array | None = None,
        segmentation_atlas: CryoTextureAtlas | None = None,
        segmentation_material_colors: wp.array | None = None,
        segmentation_texture_host: np.ndarray | None = None,
        segmentation_volume_direct: bool = False,
        stress_atlas: CryoTextureAtlas | None = None,
        cell_stretch: wp.array | None = None,
        stress_color_scale: float = 1.0,
        cryo_atlas: CryoTextureAtlas | None = None,
        cryo_texture_3d: wp.array | None = None,
        cryo_texture_host: np.ndarray | None = None,
        cryo_volume_direct: bool = False,
        cryo_flip_x: bool = False,
        cryo_flip_y: bool = False,
        cryo_flip_z: bool = False,
        cryo_src_x: int = 0,
        cryo_src_y: int = 1,
        cryo_src_z: int = 2,
        cryo_scale_x: float = 1.0,
        cryo_scale_y: float = 1.0,
        cryo_scale_z: float = 1.0,
        smooth_normals: bool = True,
        taubin_iterations: int = 2,
        taubin_lambda: float = 0.33,
        taubin_mu: float = -0.34,
        cut_z: float = float("inf"),
        topology_revision: int | None = None,
        dirty_particle_ids: wp.array | None = None,
        dirty_particle_count: int = 0,
        dirty_particle_count_device: wp.array | None = None,
        dirty_particle_capacity: int | None = None,
        dirty_topology_revision: int | None = None,
        compute_only: bool = False,
    ) -> int:
        """Run MC and log the result as a mesh asset on ``viewer``.

        Returns the triangle count emitted this frame.

        ``topology_revision`` is a monotonic counter owned by the caller
        (e.g. ``HexDeletionState.topology_revision``) that changes whenever
        the active-flag set affecting MC changes. When supplied, the cube
        cases + emit kernels and the flat-index rebuild are
        skipped on frames where revision and visibility mode both match the
        prior cache, and the cached flat index buffer is reused.

        ``compute_only`` runs the MC core (positions + topology on miss) but
        skips Taubin smoothing, normals, textured expansion, atlas rebake,
        and ``log_mesh``. Useful when an external diagnostic overlay needs
        fresh ``buffers.tri_indices`` / ``buffers.vertex_pos`` without the
        render-side work.

        ``cryo_volume_direct=True`` is a ViewerGL fast path: the mesh remains
        indexed and the shader samples ``cryo_texture_host`` as a 3D texture.
        That avoids per-cut atlas rebuilds and atlas device-to-host copies.

        ``segmentation_texture_host`` + ``segmentation_volume_direct`` use the
        direct GL 3D-volume path for static material-colour segmentation. The
        older ``segmentation_atlas`` + ``segmentation_material_colors`` path is
        retained as a fallback for USD/non-GL output.

        ``stress_atlas`` + ``cell_stretch`` switch the flat-atlas path to a
        dynamic cold-to-warm color ramp keyed by per-cell stretch. This path
        takes precedence over segmentation/cryo coloring when enabled.

        ``dirty_particle_ids`` can be supplied on a topology miss caused only
        by recently changed particle ACTIVE bits. In that case the MC topology
        path recomputes just the incident cubes and preserves the compact mesh
        contract; otherwise it falls back to the full rebuild.
        """
        # If a per-material visibility mask or a world-Z cut plane is in
        # play, fold them into the ACTIVE bit before the MC pipeline sees
        # the flags so hidden tissue drops out of the iso-surface. Note:
        # cut_z is POSITION-dependent (it reads particle_q[i][2]), so we
        # cannot topology-cache across deforming frames when it's finite.
        mc_flags = particle_flags
        needs_mask = material_visible is not None or cut_z != float("inf")
        # Cache mode key: captures everything that invalidates the cached
        # cube-cases/emit result without a topology_revision bump. If the
        # caller reassigns material_visible or changes cut_z, this trips.
        cache_mode = (
            material_visible is None,
            id(material_visible) if material_visible is not None else 0,
            float(cut_z),
        )
        # Topology cache is only safe when cut_z is infinite (position-
        # independent flags) and the caller provided a revision counter.
        # Finite cut_z or revision=None forces full recompute every frame.
        with _scoped_timer("mc_surface.update"):
            cache_hit = (
                topology_revision is not None
                and self._cached_topology_rev == topology_revision
                and cut_z == float("inf")
                and self._cached_mode == cache_mode
                and self._cached_count > 0
            )

            if needs_mask:
                if self._visible_flags is None or self._visible_flags.shape[0] != particle_flags.shape[0]:
                    self._visible_flags = wp.zeros_like(particle_flags)
                mask_input = material_visible
                if mask_input is None:
                    if self._all_visible is None:
                        import numpy as _np  # noqa: PLC0415
                        self._all_visible = wp.array(
                            _np.ones(int(aux.particle_material.shape[0]) + 1, dtype=_np.int32),
                            dtype=wp.int32,
                            device=self.device,
                        )
                    mask_input = self._all_visible
                wp.launch(
                    compute_visible_flags_kernel,
                    dim=particle_flags.shape[0],
                    inputs=[
                        particle_flags, aux.particle_material, mask_input,
                        particle_q, float(cut_z),
                    ],
                    outputs=[self._visible_flags],
                    device=self.device,
                )
                mc_flags = self._visible_flags

            # Positions always change with deformation - run every frame.
            with _scoped_timer("mc.vertex_positions"):
                compute_mc_vertex_positions(
                    buffers=buffers,
                    particle_q=particle_q,
                    particle_orientation=orientation,
                    mc_factor=mc_factor,
                    device=self.device,
                )

            # Topology: skip on cache hit. On miss, recompute + refresh cache.
            if cache_hit:
                count = self._cached_count
            else:
                count = None
                has_dirty_host_count = int(dirty_particle_count) > 0
                has_dirty_device_count = (
                    dirty_particle_count_device is not None
                    and dirty_particle_capacity is not None
                    and int(dirty_particle_capacity) > 0
                )
                can_try_dirty = (
                    dirty_particle_ids is not None
                    and (has_dirty_host_count or has_dirty_device_count)
                    and topology_revision is not None
                    and dirty_topology_revision == topology_revision
                    and self._cached_topology_rev is not None
                    and self._cached_mode == cache_mode
                    and cut_z == float("inf")
                    and hasattr(aux, "particle_grid_xyz")
                )
                if can_try_dirty:
                    with _scoped_timer("mc.topology_dirty"):
                        count = compute_mc_topology_dirty(
                            buffers=buffers,
                            tables=tables,
                            particle_flags=mc_flags,
                            grid_to_particle=aux.grid_to_particle,
                            particle_grid_xyz=aux.particle_grid_xyz,
                            dirty_particle_ids=dirty_particle_ids,
                            dirty_particle_count=int(dirty_particle_count),
                            dirty_particle_count_device=dirty_particle_count_device,
                            dirty_particle_capacity=dirty_particle_capacity,
                            device=self.device,
                        )
                if count is None:
                    with _scoped_timer("mc.topology"):
                        count = compute_mc_topology(
                            buffers=buffers,
                            tables=tables,
                            particle_flags=mc_flags,
                            grid_to_particle=aux.grid_to_particle,
                            device=self.device,
                        )

            if count == 0:
                # Remember the empty-topology state too so we don't keep
                # retrying emit on the same revision.
                if topology_revision is not None:
                    self._cached_topology_rev = topology_revision
                    self._cached_count = 0
                    self._cached_mode = cache_mode
                    self._cached_flat_uv3_valid = False
                    self._flat_uv3_kind = None
                    self._cached_flat_indices_valid = False
                else:
                    self._cached_flat_uv3_valid = False
                    self._flat_uv3_kind = None
                    self._cached_flat_indices_valid = False
                return 0

            # Refresh cache fields on miss. Do this before returning early so
            # the compute_only caller still benefits from the topology cache.
            if not cache_hit and topology_revision is not None:
                self._cached_topology_rev = topology_revision
                self._cached_count = count
                self._cached_mode = cache_mode
                # flat_uv3 is refreshed below in the textured path; mark stale
                # here and validate after the expand. Same for flat indices,
                # which are rebuilt below in the non-compute_only path only.
                self._cached_flat_uv3_valid = False
                self._flat_uv3_kind = None
                self._cached_flat_indices_valid = False
            elif not cache_hit:
                # Without an external topology revision the caller opted out of
                # cache validation, so every topology run must refresh derived
                # flat buffers.
                self._cached_flat_uv3_valid = False
                self._flat_uv3_kind = None
                self._cached_flat_indices_valid = False

            # compute_only skips Taubin / normals / textured expansion / log_mesh
            # AND the flat-index rebuild: diagnostic overlays read from
            # buffers.tri_indices directly, so there's no reason to pay the CPU
            # sync here. The next non-compute_only call will see
            # _cached_flat_indices_valid=False and rebuild the scratch.
            if compute_only:
                return count

            # Flat index buffer: topology-constant, so only refresh on miss or
            # when the scratch was left stale (e.g. by a prior compute_only call
            # that bumped the topology revision without rebuilding it).
            indices_refreshed = False
            if self._cached_flat_indices_valid and self._indices_scratch is not None:
                idx = self._indices_scratch
                flat_size = count * 3
            else:
                with _scoped_timer("mc.flat_indices"):
                    flat_size = count * 3
                    idx = self._ensure_index_capacity(flat_size)
                    wp.launch(
                        _flatten_triangle_indices_kernel,
                        dim=count,
                        inputs=[buffers.tri_indices, int(count)],
                        outputs=[idx],
                        device=self.device,
                    )
                    self._cached_flat_indices_valid = True
                    indices_refreshed = True

            # Geometry smoothing must happen on the indexed MC vertices before the
            # textured path expands them into flat triangle soup; otherwise the
            # duplicated vertices cannot share neighbourhood information.
            render_vertex_pos = buffers.vertex_pos
            if taubin_iterations > 0:
                with _scoped_timer("mc.taubin"):
                    render_vertex_pos = self._run_taubin_smoothing(
                        tri_indices=buffers.tri_indices,
                        vertex_pos=buffers.vertex_pos,
                        num_triangles=int(count),
                        iterations=int(taubin_iterations),
                        lambda_coeff=float(taubin_lambda),
                        mu_coeff=float(taubin_mu),
                    )

            render_vertex_normals = None
            if smooth_normals:
                self._ensure_vertex_capacity(int(render_vertex_pos.shape[0]))
                assert self._smoothed_normals is not None
                with _scoped_timer("mc.normals"):
                    render_vertex_normals = compute_vertex_normals(
                        render_vertex_pos,
                        idx[:flat_size],
                        normals=self._smoothed_normals,
                        device=self.device,
                    )

            # Two output paths:
            #  * Untextured: log the indexed MC mesh as-is, optionally with
            #    explicit smoothed normals.
            #  * Atlas textured: expand into flat triangle soup for the
            #    per-triangle baked atlas, but carry the indexed smoothed
            #    normals across so the shaded result no longer facets at every
            #    triangle edge.
            #  * Direct volume textured: keep the indexed MC mesh and attach
            #    static UV3s as a separate GL attribute. The shader samples the
            #    uploaded 3D texture directly, so cuts do not rebake/pull a 2D
            #    atlas.
            stress_textured = stress_atlas is not None and cell_stretch is not None
            segmentation_volume_textured = bool(
                not stress_textured
                and segmentation_volume_direct
                and segmentation_texture_host is not None
            )
            segmentation_textured = (
                not stress_textured
                and not segmentation_volume_textured
                and segmentation_atlas is not None
                and segmentation_material_colors is not None
            )
            volume_textured = bool(
                cryo_volume_direct
                and cryo_texture_host is not None
                and not stress_textured
                and not segmentation_volume_textured
                and not segmentation_textured
            )
            direct_volume_textured = segmentation_volume_textured or volume_textured
            cryo_atlas_textured = (
                cryo_atlas is not None
                and cryo_texture_3d is not None
                and not direct_volume_textured
                and not stress_textured
                and not segmentation_textured
            )
            atlas_textured = stress_textured or segmentation_textured or cryo_atlas_textured
            if stress_textured:
                active_atlas = stress_atlas
            elif segmentation_textured:
                active_atlas = segmentation_atlas
            else:
                active_atlas = cryo_atlas

            flat_textured = atlas_textured or segmentation_volume_textured
            if flat_textured:
                with _scoped_timer("mc.textured_expand"):
                    # Allocate flat buffers on demand sized to the MC triangle budget.
                    max_tris = buffers.max_triangles
                    self._ensure_flat_surface_capacity(max_tris)
                    assert self._flat_pos is not None
                    assert self._flat_uv3 is not None
                    flat_uv3_kind = "segmentation_majority" if segmentation_volume_textured else "vertex"
                    if self._cached_flat_uv3_valid and self._flat_uv3_kind != flat_uv3_kind:
                        self._cached_flat_uv3_valid = False
                        self._flat_uv3_kind = None
                    # Positions move every frame -> always expand into flat_pos.
                    # UV3 is topology-constant -> skip expand when cached valid.
                    if self._cached_flat_uv3_valid:
                        wp.launch(
                            _expand_triangle_positions_kernel,
                            dim=count,
                            inputs=[
                                buffers.tri_indices,
                                render_vertex_pos,
                                int(count),
                            ],
                            outputs=[self._flat_pos],
                            device=self.device,
                        )
                    elif segmentation_volume_textured:
                        wp.launch(
                            _expand_triangle_vertices_majority_uv3_kernel,
                            dim=count,
                            inputs=[
                                buffers.tri_indices,
                                render_vertex_pos,
                                buffers.particle_uv3,
                                aux.particle_material,
                                int(count),
                            ],
                            outputs=[self._flat_pos, self._flat_uv3],
                            device=self.device,
                        )
                        self._cached_flat_uv3_valid = True
                        self._flat_uv3_kind = flat_uv3_kind
                        self._flat_uv3_revision += 1
                    else:
                        wp.launch(
                            _expand_triangle_vertices_kernel,
                            dim=count,
                            inputs=[
                                buffers.tri_indices,
                                render_vertex_pos,
                                buffers.vertex_uv3,
                                int(count),
                            ],
                            outputs=[self._flat_pos, self._flat_uv3],
                            device=self.device,
                        )
                        self._cached_flat_uv3_valid = True
                        self._flat_uv3_kind = flat_uv3_kind
                        self._flat_uv3_revision += 1
                    if render_vertex_normals is not None:
                        assert self._flat_normals is not None
                        wp.launch(
                            _expand_triangle_normals_kernel,
                            dim=count,
                            inputs=[buffers.tri_indices, render_vertex_normals, int(count)],
                            outputs=[self._flat_normals],
                            device=self.device,
                        )
                if stress_textured:
                    assert active_atlas is not None
                    assert cell_stretch is not None
                    with _scoped_timer("mc.stress_atlas_rebuild"):
                        active_atlas.rebuild_stress(
                            tri_indices=buffers.tri_indices,
                            num_triangles=int(count),
                            cell_stretch=cell_stretch,
                            color_scale=float(stress_color_scale),
                        )
                elif segmentation_textured:
                    assert active_atlas is not None
                    with _scoped_timer("mc.segmentation_atlas_rebuild"):
                        active_atlas.rebuild_materials(
                            tri_indices=buffers.tri_indices,
                            num_triangles=int(count),
                            particle_material=aux.particle_material,
                            material_colors=segmentation_material_colors,
                            topology_revision=topology_revision,
                        )
                elif cryo_atlas_textured:
                    assert active_atlas is not None
                    assert cryo_texture_3d is not None
                    # Rebuild the per-triangle atlas. With topology_revision passed,
                    # this is a no-op when neither topology nor UI params changed.
                    with _scoped_timer("mc.atlas_rebuild"):
                        active_atlas.rebuild(
                            flat_vertex_uv3=self._flat_uv3,
                            num_triangles=int(count),
                            texture_3d=cryo_texture_3d,
                            flip_x=cryo_flip_x,
                            flip_y=cryo_flip_y,
                            flip_z=cryo_flip_z,
                            src_x=int(cryo_src_x),
                            src_y=int(cryo_src_y),
                            src_z=int(cryo_src_z),
                            scale_x=float(cryo_scale_x),
                            scale_y=float(cryo_scale_y),
                            scale_z=float(cryo_scale_z),
                            topology_revision=topology_revision,
                        )

            # ViewerGL's MeshGL.update uploads the index buffer only on first
            # call; subsequent calls refresh vertex positions but keep the
            # original topology. Keep fixed-size indexed meshes alive and
            # refresh their EBO in place on topology misses/count changes.
            # Flat atlas meshes still need recreation when the point count
            # changes because MeshGL's VBO capacity is fixed.
            expected_num_indices = count * 3 if flat_textured else flat_size
            expected_num_points = count * 3 if flat_textured else int(buffers.vertex_pos.shape[0])
            if hasattr(viewer, "objects") and self.name in viewer.objects:
                cached = viewer.objects[self.name]
                cached_num_indices = int(getattr(cached, "num_indices", 0))
                cached_num_points = int(getattr(cached, "num_points", 0))
                destroy_cached = cached_num_points != expected_num_points
                if not destroy_cached and not flat_textured:
                    if cached_num_indices != expected_num_indices or indices_refreshed:
                        destroy_cached = not _update_mesh_indices(cached, idx[:flat_size])
                elif cached_num_indices != expected_num_indices:
                    destroy_cached = True
                if destroy_cached:
                    try:
                        cached.destroy()
                    except Exception:
                        pass
                    del viewer.objects[self.name]

            atlas_texture = None
            atlas_revision = None
            if atlas_textured:
                assert active_atlas is not None
                atlas_texture = active_atlas.texture
                atlas_revision = active_atlas.texture_revision
                if hasattr(viewer, "objects") and self.name in viewer.objects:
                    viewer.objects[self.name]._cutting_next_texture_revision = atlas_revision

            with _scoped_timer("mc.log_mesh"):
                if atlas_textured:
                    mesh_indices = active_atlas.indices(count)
                    if _precreate_mesh_gl_for_gpu_indices(
                        viewer,
                        self.name,
                        expected_num_points,
                        expected_num_indices,
                        hidden=hidden,
                    ):
                        _update_mesh_indices(viewer.objects[self.name], mesh_indices)
                    flat_normals = None
                    if self._flat_normals is not None and render_vertex_normals is not None:
                        flat_normals = self._flat_normals[: count * 3]
                    viewer.log_mesh(
                        name=self.name,
                        points=self._flat_pos[: count * 3],
                        indices=mesh_indices,
                        normals=flat_normals,
                        uvs=active_atlas.uvs(count),
                        texture=atlas_texture,
                        hidden=hidden,
                    )
                elif segmentation_volume_textured:
                    assert self._flat_pos is not None
                    mesh_indices = self._ensure_flat_identity_capacity(count * 3)[: count * 3]
                    if _precreate_mesh_gl_for_gpu_indices(
                        viewer,
                        self.name,
                        expected_num_points,
                        expected_num_indices,
                        hidden=hidden,
                    ):
                        _update_mesh_indices(viewer.objects[self.name], mesh_indices)
                    flat_normals = None
                    if self._flat_normals is not None and render_vertex_normals is not None:
                        flat_normals = self._flat_normals[: count * 3]
                    viewer.log_mesh(
                        name=self.name,
                        points=self._flat_pos[: count * 3],
                        indices=mesh_indices,
                        normals=flat_normals,
                        hidden=hidden,
                    )
                else:
                    mesh_indices = idx[:flat_size]
                    if _precreate_mesh_gl_for_gpu_indices(
                        viewer,
                        self.name,
                        expected_num_points,
                        expected_num_indices,
                        hidden=hidden,
                    ):
                        _update_mesh_indices(viewer.objects[self.name], mesh_indices)
                    viewer.log_mesh(
                        name=self.name,
                        points=render_vertex_pos,
                        indices=mesh_indices,
                        normals=render_vertex_normals,
                        hidden=hidden,
                    )

            # Newton's MeshGL applies ``color`` and ``material`` during its
            # draw call. Patch this instance's ``render()`` to use white
            # albedo and texture_enable only for our textured draw, then
            # restore the mesh state. Also clamp the atlas to level 0 so
            # mipmaps do not blend unrelated neighbouring triangle tiles.
            if atlas_textured and hasattr(viewer, "objects") and self.name in viewer.objects:
                mesh_gl = viewer.objects[self.name]
                _enable_textured_render(mesh_gl)
                _enable_texture_upload_cache(mesh_gl)
                mesh_gl._cutting_next_texture_revision = atlas_revision
                mesh_gl._cutting_texture_revision = atlas_revision
                mesh_gl._cutting_texture_source = atlas_texture
                _configure_atlas_filter(mesh_gl)
            elif direct_volume_textured and hasattr(viewer, "objects") and self.name in viewer.objects:
                volume_host = segmentation_texture_host if segmentation_volume_textured else cryo_texture_host
                assert volume_host is not None
                volume_uv3 = self._flat_uv3[: count * 3] if segmentation_volume_textured else buffers.vertex_uv3
                assert volume_uv3 is not None
                _enable_cryo_volume_render(
                    viewer.objects[self.name],
                    volume_host,
                    volume_uv3,
                    scale=(
                        (1.0, 1.0, 1.0)
                        if segmentation_volume_textured
                        else (float(cryo_scale_x), float(cryo_scale_y), float(cryo_scale_z))
                    ),
                    src_axis=(
                        (0, 1, 2)
                        if segmentation_volume_textured
                        else (int(cryo_src_x), int(cryo_src_y), int(cryo_src_z))
                    ),
                    flip=(
                        (False, False, False)
                        if segmentation_volume_textured
                        else (bool(cryo_flip_x), bool(cryo_flip_y), bool(cryo_flip_z))
                    ),
                    linear_filter=not segmentation_volume_textured,
                    uv3_revision=(self._flat_uv3_revision if segmentation_volume_textured else None),
                    uv3_source_id=(id(self._flat_uv3) if segmentation_volume_textured else None),
                )
            return count

    def log_hidden(self, viewer: newton.viewer.ViewerBase) -> None:
        """Mark the surface mesh hidden without running the MC pipeline.

        No-op when no asset has been logged yet (the viewer has nothing to
        toggle). When an asset exists, this simply sets ``hidden=True`` on
        the cached GL/USD object so the caller can fully skip ``update(...)``
        for as long as the surface stays off.
        """
        objects = getattr(viewer, "objects", None)
        if objects is None or self.name not in objects:
            return
        cached = objects[self.name]
        # ViewerGL stores per-object hidden state; flip it without touching
        # any geometry. Fall back to a re-log with the last buffers if the
        # backend doesn't expose hidden as a mutable attribute.
        if hasattr(cached, "hidden"):
            cached.hidden = True
        else:
            try:
                viewer.log_mesh(name=self.name, points=None, indices=None, hidden=True)
            except Exception:
                pass

    @property
    def tri_centroid_uv3(self) -> wp.array:
        """Per-triangle centroid UV3 computed during the last textured update."""
        return self._tri_centroid_uv3


class CryoTextureAtlas:
    """Per-triangle textured 2D atlas for the deformable MC mesh.

    Newton's GL shader only has a ``sampler2D`` and linearly interpolates
    vertex UVs across each triangle, but it has no native 3D texture path.
    This atlas assigns ONE tile PER TRIANGLE and bakes a small triangle-local
    texture into that tile from the triangle's three UV3 corners. The mesh is
    rendered as flat triangle soup, with each triangle's three vertices mapped
    to that tile's three corners, so the shader can interpolate within the
    tile and recover local texture variation without sharing UVs across
    unrelated triangles.

    Usage::

        atlas = CryoTextureAtlas(max_triangles=200_000, device=dev)
        atlas.rebuild(flat_vertex_uv3, num_triangles, texture_3d, ...)
        viewer.log_mesh(
            name="...",
            points=atlas.flat_pos_view(surface.flat_pos, num_triangles),
            indices=atlas.indices(num_triangles),
            uvs=atlas.uvs(num_triangles),
            texture=atlas.texture,
        )
    """

    def __init__(
        self,
        max_triangles: int,
        device: wp.context.Device,
        tile_size: int = 2,
    ):
        if tile_size < 1:
            raise ValueError("tile_size must be >= 1")
        self.device = device
        self.tile_size = int(tile_size)
        self.max_triangles = 0
        self.atlas_width = 0
        self.atlas_height = 0
        self._flat_uvs: wp.array | None = None
        self._atlas: wp.array | None = None
        self._flat_indices: wp.array | None = None
        self._atlas_host: np.ndarray | None = None
        self._last_num_triangles: int = 0
        self._texture_revision: int = 0
        # Monotonic counter bumped on every ``_allocate`` call so a capacity
        # growth invalidates any cached bake key (the tile UV layout and the
        # atlas dimensions both change on resize).
        self._atlas_generation: int = 0
        # Last-seen bake key used to short-circuit ``rebuild``. None -> always
        # rebake. See :meth:`rebuild` for the key shape.
        self._last_bake_key: tuple | None = None
        self._allocate(max(1, int(max_triangles)))

    def _allocate(self, max_triangles: int) -> None:
        import math  # noqa: PLC0415

        self.max_triangles = int(max(1, max_triangles))
        tiles_per_row = int(math.ceil(math.sqrt(max(self.max_triangles, 1))))
        tiles_per_row = max(1, 1 << max(tiles_per_row - 1, 0).bit_length())
        tiles_per_col = int(math.ceil(self.max_triangles / max(tiles_per_row, 1)))
        tiles_per_col = max(1, 1 << max(tiles_per_col - 1, 0).bit_length())
        self.atlas_width = tiles_per_row * self.tile_size
        self.atlas_height = tiles_per_col * self.tile_size

        self._flat_uvs = wp.zeros(self.max_triangles * 3, dtype=wp.vec2, device=self.device)
        self._atlas = wp.zeros((self.atlas_height, self.atlas_width), dtype=wp.vec3, device=self.device)
        self._flat_indices = wp.array(
            np.arange(self.max_triangles * 3, dtype=np.int32),
            dtype=wp.int32,
            device=self.device,
        )
        wp.launch(
            _compute_triangle_tile_uvs_kernel,
            dim=self.max_triangles,
            inputs=[self.atlas_width, self.atlas_height, self.tile_size, self.max_triangles],
            outputs=[self._flat_uvs],
            device=self.device,
        )
        self._atlas_host = None
        self._atlas_generation += 1
        self._last_bake_key = None

    def ensure_capacity(self, num_triangles: int) -> None:
        needed = int(max(1, num_triangles))
        if needed <= self.max_triangles:
            return
        new_cap = max(needed, self.max_triangles * 2)
        self._allocate(new_cap)

    def uvs(self, num_triangles: int) -> wp.array:
        """Slice of the precomputed flat UVs sized to the active triangle count."""
        self.ensure_capacity(num_triangles)
        return self._flat_uvs[: num_triangles * 3]

    def indices(self, num_triangles: int) -> wp.array:
        """Identity index array sized to the active triangle count."""
        self.ensure_capacity(num_triangles)
        return self._flat_indices[: num_triangles * 3]

    @property
    def texture(self) -> np.ndarray:
        """Atlas as ``(H, W, 3)`` uint8 numpy, cached after last rebuild."""
        if self._atlas_host is None:
            self._pull_host_copy()
        return self._atlas_host

    @property
    def texture_revision(self) -> int:
        """Monotonic revision for the cached host atlas image."""
        return self._texture_revision

    def _pull_host_copy(self) -> None:
        host = self._atlas.numpy()
        self._atlas_host = np.ascontiguousarray(np.clip(host * 255.0, 0.0, 255.0).astype(np.uint8))
        self._texture_revision += 1

    def save(self, path: str | Path, compact: bool = True) -> None:
        """Write the current baked atlas to an image file."""
        from PIL import Image  # noqa: PLC0415

        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image = self.texture
        if compact and self._last_num_triangles > 0:
            tiles_per_row = max(1, self.atlas_width // self.tile_size)
            used_rows = (self._last_num_triangles + tiles_per_row - 1) // tiles_per_row
            used_height = min(self.atlas_height, used_rows * self.tile_size)
            used_width = self.atlas_width
            if used_rows <= 1:
                used_cols = min(tiles_per_row, self._last_num_triangles)
                used_width = min(self.atlas_width, max(1, used_cols * self.tile_size))
            image = image[:used_height, :used_width]
        Image.fromarray(image).save(out_path)

    def rebuild(
        self,
        flat_vertex_uv3: wp.array,
        num_triangles: int,
        texture_3d: wp.array,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        src_x: int = 0,
        src_y: int = 1,
        src_z: int = 2,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
        topology_revision: int | None = None,
    ) -> None:
        """Bake per-triangle textures from the flat triangle soup UV3s.

        ``flat_vertex_uv3`` stores three UV3 corners per triangle, in the same
        flat-triangle order as :meth:`uvs` and :meth:`indices`.

        When ``topology_revision`` is supplied and matches the prior call
        along with all UI/texture parameters, the bake kernel and the full
        atlas D->H copy are both skipped. ``self.texture`` continues to
        return the cached host numpy.
        """
        if num_triangles == 0:
            # Empty-topology transition: clear the atlas once, then cache
            # the key so subsequent empty frames are free.
            empty_key = (
                "empty",
                int(topology_revision) if topology_revision is not None else None,
                self._atlas_generation,
            )
            if self._last_bake_key == empty_key:
                self._last_num_triangles = 0
                return
            self._last_num_triangles = 0
            self._atlas.zero_()
            self._pull_host_copy()
            self._last_bake_key = empty_key
            return

        self.ensure_capacity(num_triangles)

        # Cache key: topology_revision gates the UV3 layout; UI params gate
        # the sampler; ``id(texture_3d)`` catches caller-swapped volumes
        # (in-place mutation is not detected - document as such). When
        # ``topology_revision`` is None the caller opted out of caching, so
        # key comparison always misses.
        bake_key: tuple | None
        if topology_revision is None:
            bake_key = None
        else:
            bake_key = (
                int(topology_revision),
                int(self._atlas_generation),
                int(num_triangles),
                float(scale_x), float(scale_y), float(scale_z),
                bool(flip_x), bool(flip_y), bool(flip_z),
                int(src_x), int(src_y), int(src_z),
                id(texture_3d),
            )
        if bake_key is not None and bake_key == self._last_bake_key:
            self._last_num_triangles = int(num_triangles)
            return

        self._last_num_triangles = int(num_triangles)
        tex_nx, tex_ny, tex_nz = (int(texture_3d.shape[i]) for i in range(3))
        self._atlas.zero_()
        wp.launch(
            _bake_triangle_texture_kernel,
            dim=(
                self.atlas_width // self.tile_size,
                self.atlas_height // self.tile_size,
                self.tile_size * self.tile_size,
            ),
            inputs=[
                flat_vertex_uv3,
                texture_3d,
                tex_nx, tex_ny, tex_nz,
                int(src_x), int(src_y), int(src_z),
                int(bool(flip_x)), int(bool(flip_y)), int(bool(flip_z)),
                float(scale_x), float(scale_y), float(scale_z),
                self.atlas_width, self.atlas_height, self.tile_size,
                int(num_triangles),
            ],
            outputs=[self._atlas],
            device=self.device,
        )
        self._pull_host_copy()
        self._last_bake_key = bake_key

    def rebuild_materials(
        self,
        tri_indices: wp.array,
        num_triangles: int,
        particle_material: wp.array,
        material_colors: wp.array,
        topology_revision: int | None = None,
    ) -> None:
        """Bake a flat material-color tile for each active MC triangle."""
        if num_triangles == 0:
            empty_key = (
                "materials_empty",
                int(topology_revision) if topology_revision is not None else None,
                self._atlas_generation,
            )
            if self._last_bake_key == empty_key:
                self._last_num_triangles = 0
                return
            self._last_num_triangles = 0
            self._atlas.zero_()
            self._pull_host_copy()
            self._last_bake_key = empty_key
            return

        self.ensure_capacity(num_triangles)
        if topology_revision is None:
            bake_key = None
        else:
            bake_key = (
                "materials",
                int(topology_revision),
                int(self._atlas_generation),
                int(num_triangles),
                id(particle_material),
                id(material_colors),
            )
        if bake_key is not None and bake_key == self._last_bake_key:
            self._last_num_triangles = int(num_triangles)
            return

        self._last_num_triangles = int(num_triangles)
        self._atlas.zero_()
        wp.launch(
            _fill_triangle_material_atlas_kernel,
            dim=(
                self.atlas_width // self.tile_size,
                self.atlas_height // self.tile_size,
                self.tile_size * self.tile_size,
            ),
            inputs=[
                tri_indices,
                particle_material,
                material_colors,
                self.atlas_width,
                self.atlas_height,
                self.tile_size,
                int(num_triangles),
            ],
            outputs=[self._atlas],
            device=self.device,
        )
        self._pull_host_copy()
        self._last_bake_key = bake_key

    def rebuild_stress(
        self,
        tri_indices: wp.array,
        num_triangles: int,
        cell_stretch: wp.array,
        color_scale: float = 1.0,
    ) -> None:
        """Bake dynamic per-cell stretch colours into each active triangle tile."""
        if num_triangles == 0:
            empty_key = ("stress_empty", int(self._atlas_generation))
            if self._last_bake_key == empty_key:
                self._last_num_triangles = 0
                return
            self._last_num_triangles = 0
            self._atlas.zero_()
            self._pull_host_copy()
            self._last_bake_key = empty_key
            return

        self.ensure_capacity(num_triangles)
        self._last_num_triangles = int(num_triangles)
        self._atlas.zero_()
        wp.launch(
            _fill_triangle_stress_atlas_kernel,
            dim=(
                self.atlas_width // self.tile_size,
                self.atlas_height // self.tile_size,
                self.tile_size * self.tile_size,
            ),
            inputs=[
                tri_indices,
                cell_stretch,
                float(color_scale),
                self.atlas_width,
                self.atlas_height,
                self.tile_size,
                int(num_triangles),
            ],
            outputs=[self._atlas],
            device=self.device,
        )
        self._pull_host_copy()
        self._last_bake_key = None


class ColoredParticleOverlay:
    """Per-material coloured overlay of the active particle set.

    Overrides Newton's default ``/model/particles`` draw (uniform tan points)
    by calling :meth:`~newton.viewer.ViewerBase.log_points` after
    :meth:`~newton.viewer.ViewerBase.log_state`. The Newton viewer's
    ``show_particles`` attribute still gates visibility.

    Only :attr:`newton.ParticleFlags.ACTIVE` particles are emitted, so cut
    tissue frozen in place by the cutting kernels does not show up as a
    dead debris cloud.
    """

    def __init__(
        self,
        num_particles: int,
        materials,  # MaterialTable
        device: wp.context.Device,
        name: str = "/model/particles",
    ):
        self.name = name
        self.device = device
        self.num_particles = num_particles
        self._colors = wp.array(materials.color, dtype=wp.vec3, device=device)
        self._out_points = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        self._out_colors = wp.zeros(num_particles, dtype=wp.vec3, device=device)
        self._out_radii = wp.zeros(num_particles, dtype=wp.float32, device=device)
        self._counter = wp.zeros(1, dtype=wp.int32, device=device)
        self._last_radius: float | None = None

    def update(
        self,
        viewer: newton.viewer.ViewerBase,
        particle_q: wp.array,
        particle_flags: wp.array,
        particle_material: wp.array,
        radius: float,
        hidden: bool = False,
        cut_z: float = float("inf"),
    ) -> int:
        """Gather, then log active particles with per-material colours.

        ``cut_z`` hides particles whose world-space Z is above it.
        Returns the number of active particles emitted this frame. When
        ``hidden=True`` the gather kernel and its counter sync are skipped
        and the overlay is marked hidden in the viewer directly.
        """
        if hidden:
            viewer.log_points(name=self.name, points=None, hidden=True)
            return 0
        self._counter.zero_()
        wp.launch(
            _gather_active_particles_kernel,
            dim=self.num_particles,
            inputs=[
                particle_q,
                particle_flags,
                particle_material,
                self._colors,
                float(cut_z),
            ],
            outputs=[self._counter, self._out_points, self._out_colors],
            device=self.device,
        )
        return self._log(viewer, radius, hidden)

    def update_stress(
        self,
        viewer: newton.viewer.ViewerBase,
        particle_q: wp.array,
        particle_flags: wp.array,
        cell_stretch: wp.array,
        color_scale: float,
        radius: float,
        hidden: bool = False,
        cut_z: float = float("inf"),
    ) -> int:
        """Gather active cell-centre particles with cold-to-warm stretch colours."""
        if hidden:
            viewer.log_points(name=self.name, points=None, hidden=True)
            return 0
        self._counter.zero_()
        wp.launch(
            _gather_stress_colored_particles_kernel,
            dim=self.num_particles,
            inputs=[
                particle_q,
                particle_flags,
                cell_stretch,
                float(color_scale),
                float(cut_z),
            ],
            outputs=[self._counter, self._out_points, self._out_colors],
            device=self.device,
        )
        return self._log(viewer, radius, hidden)

    def update_cryo(
        self,
        viewer: newton.viewer.ViewerBase,
        particle_q: wp.array,
        particle_flags: wp.array,
        particle_grid_xyz: wp.array,
        texture_3d: wp.array,
        grid_shape: tuple[int, int, int],
        radius: float,
        hidden: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        src_x: int = 0,
        src_y: int = 1,
        src_z: int = 2,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
        cut_z: float = float("inf"),
    ) -> int:
        """Gather active particles with colours sampled from a 3D volume.

        Drop-in replacement for :meth:`update` when you want the particle
        swarm tinted directly by the cryosection atlas rather than by the
        material palette - a quick way to eyeball whether 3D texture
        sampling is working end-to-end before staring at the MC surface.

        ``flip_{x,y,z}`` invert each axis independently so the axis
        convention between the cryosection stack and the atlas can be
        calibrated interactively.
        """
        if hidden:
            viewer.log_points(name=self.name, points=None, hidden=True)
            return 0
        gx, gy, gz = grid_shape
        inv_nx = 1.0 / max(gx - 1, 1)
        inv_ny = 1.0 / max(gy - 1, 1)
        inv_nz = 1.0 / max(gz - 1, 1)
        tex_nx, tex_ny, tex_nz = int(texture_3d.shape[0]), int(texture_3d.shape[1]), int(texture_3d.shape[2])
        self._counter.zero_()
        wp.launch(
            _gather_cryo_colored_particles_kernel,
            dim=self.num_particles,
            inputs=[
                particle_q,
                particle_flags,
                particle_grid_xyz,
                texture_3d,
                float(inv_nx),
                float(inv_ny),
                float(inv_nz),
                tex_nx,
                tex_ny,
                tex_nz,
                int(src_x), int(src_y), int(src_z),
                int(bool(flip_x)), int(bool(flip_y)), int(bool(flip_z)),
                float(scale_x), float(scale_y), float(scale_z),
                float(cut_z),
            ],
            outputs=[self._counter, self._out_points, self._out_colors],
            device=self.device,
        )
        return self._log(viewer, radius, hidden)

    def _log(self, viewer, radius: float, hidden: bool) -> int:
        count = int(self._counter.numpy()[0])
        if count == 0:
            viewer.log_points(name=self.name, points=None, hidden=True)
            return 0
        # ViewerGL.log_points expects ``radii`` as a wp.array despite the
        # docstring permitting a bare float; fill the scratch buffer once per
        # radius change.
        if self._last_radius != radius:
            self._out_radii.fill_(float(radius))
            self._last_radius = float(radius)
        viewer.log_points(
            name=self.name,
            points=self._out_points[:count],
            radii=self._out_radii[:count],
            colors=self._out_colors[:count],
            hidden=hidden,
        )
        return count


class CryoMeshVertexOverlay:
    """Diagnostic point overlay for MC vertices sampled from the cryo volume.

    Newton's mesh path lacks a per-vertex colour attribute, so this renders the
    emitted MC triangle corners as points coloured directly in Warp. That lets
    us validate the cryo sampling independently from the textured-triangle path.
    """

    def __init__(
        self,
        max_triangles: int,
        device: wp.context.Device,
        name: str = "cutting/mc_vertices",
    ):
        self.name = name
        self.device = device
        self.max_triangles = int(max_triangles)
        self._points = wp.zeros(self.max_triangles * 3, dtype=wp.vec3, device=device)
        self._uv3 = wp.zeros(self.max_triangles * 3, dtype=wp.vec3, device=device)
        self._colors = wp.zeros(self.max_triangles * 3, dtype=wp.vec3, device=device)
        self._radii = wp.zeros(self.max_triangles * 3, dtype=wp.float32, device=device)
        self._last_radius: float | None = None

    def update_cryo(
        self,
        viewer: newton.viewer.ViewerBase,
        tri_indices: wp.array,
        vertex_pos: wp.array,
        vertex_uv3: wp.array,
        num_triangles: int,
        texture_3d: wp.array,
        radius: float,
        hidden: bool = False,
        flip_x: bool = False,
        flip_y: bool = False,
        flip_z: bool = False,
        src_x: int = 0,
        src_y: int = 1,
        src_z: int = 2,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
        scale_z: float = 1.0,
    ) -> int:
        """Render MC triangle corners as cryo-coloured points."""
        if num_triangles <= 0:
            viewer.log_points(name=self.name, points=None, hidden=True)
            return 0

        point_count = int(num_triangles) * 3
        wp.launch(
            _expand_triangle_vertices_kernel,
            dim=int(num_triangles),
            inputs=[tri_indices, vertex_pos, vertex_uv3, int(num_triangles)],
            outputs=[self._points, self._uv3],
            device=self.device,
        )
        tex_nx, tex_ny, tex_nz = (int(texture_3d.shape[i]) for i in range(3))
        wp.launch(
            _sample_3d_texture_kernel,
            dim=point_count,
            inputs=[
                self._uv3,
                texture_3d,
                tex_nx, tex_ny, tex_nz,
                int(src_x), int(src_y), int(src_z),
                int(bool(flip_x)), int(bool(flip_y)), int(bool(flip_z)),
                float(scale_x), float(scale_y), float(scale_z),
            ],
            outputs=[self._colors],
            device=self.device,
        )
        if self._last_radius != radius:
            self._radii.fill_(float(radius))
            self._last_radius = float(radius)
        viewer.log_points(
            name=self.name,
            points=self._points[:point_count],
            radii=self._radii[:point_count],
            colors=self._colors[:point_count],
            hidden=hidden,
        )
        return point_count


class GrabConstraintOverlay:
    """Draws pull-point grab constraints and their grabbed particles."""

    def __init__(self, capacity: int, device: wp.context.Device, name: str = "cutting/grab_constraints"):
        self.name = name
        self.points_name = f"{name}/particles"
        self.device = device
        self.capacity = int(capacity)
        self._starts = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        self._ends = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        self._radii = wp.zeros(self.capacity, dtype=wp.float32, device=device)
        self._colors = wp.zeros(self.capacity, dtype=wp.vec3, device=device)
        self._last_radius: float | None = None
        self._last_color: tuple[float, float, float] | None = None

    def update(
        self,
        viewer: newton.viewer.ViewerBase,
        *,
        grab_indices: wp.array,
        particle_q: wp.array,
        pull_target,
        grab_count: int,
        hidden: bool = False,
        color: tuple[float, float, float] = (1.0, 0.9, 0.1),
        width: float = 0.0005,
        point_radius: float = 0.001,
    ) -> None:
        count = max(0, min(int(grab_count), self.capacity))
        if hidden or count <= 0 or pull_target is None:
            viewer.log_lines(name=self.name, starts=None, ends=None, colors=None, hidden=True)
            viewer.log_points(name=self.points_name, points=None, hidden=True)
            return

        target = np.asarray(pull_target, dtype=np.float32).reshape(3)
        wp.launch(
            _gather_grab_constraint_lines_kernel,
            dim=count,
            inputs=[
                grab_indices,
                particle_q,
                wp.vec3(float(target[0]), float(target[1]), float(target[2])),
            ],
            outputs=[self._starts, self._ends],
            device=self.device,
        )
        viewer.log_lines(
            name=self.name,
            starts=self._starts[:count],
            ends=self._ends[:count],
            colors=color,
            width=width,
            hidden=False,
        )

        radius = max(0.0, float(point_radius))
        if radius > 0.0:
            if self._last_radius != radius:
                self._radii.fill_(radius)
                self._last_radius = radius
            point_color = (float(color[0]), float(color[1]), float(color[2]))
            if self._last_color != point_color:
                self._colors.fill_(wp.vec3(*point_color))
                self._last_color = point_color
            viewer.log_points(
                name=self.points_name,
                points=self._ends[:count],
                radii=self._radii[:count],
                colors=self._colors[:count],
                hidden=False,
            )
        else:
            viewer.log_points(name=self.points_name, points=None, hidden=True)


class ShapeMatchingClusterOverlay:
    """Draws the 12 box edges for uniform 8-node shape-matching clusters."""

    EDGES_PER_CLUSTER = 12

    def __init__(self, device: wp.context.Device, name: str = "cutting/shape_clusters"):
        self.name = name
        self.device = device
        self._capacity = 0
        self._starts: wp.array | None = None
        self._ends: wp.array | None = None

    def _ensure_capacity(self, line_count: int) -> None:
        if self._capacity >= line_count and self._starts is not None and self._ends is not None:
            return
        self._capacity = int(line_count)
        self._starts = wp.zeros(self._capacity, dtype=wp.vec3, device=self.device)
        self._ends = wp.zeros(self._capacity, dtype=wp.vec3, device=self.device)

    def update(
        self,
        viewer: newton.viewer.ViewerBase,
        *,
        indices_by_slot: wp.array | None,
        cluster_active: wp.array | None,
        num_clusters: int,
        particle_q: wp.array,
        particle_flags: wp.array,
        hidden: bool = False,
        color: tuple[float, float, float] = (0.5, 0.9, 1.0),
        width: float = 0.0005,
    ) -> None:
        if hidden or indices_by_slot is None or cluster_active is None or num_clusters <= 0:
            viewer.log_lines(name=self.name, starts=None, ends=None, colors=None, hidden=True)
            return

        line_count = int(num_clusters) * self.EDGES_PER_CLUSTER
        self._ensure_capacity(line_count)
        assert self._starts is not None
        assert self._ends is not None
        wp.launch(
            _gather_cluster_box_edges_kernel,
            dim=line_count,
            inputs=[
                indices_by_slot,
                cluster_active,
                particle_q,
                particle_flags,
                int(num_clusters),
            ],
            outputs=[self._starts, self._ends],
            device=self.device,
        )
        viewer.log_lines(
            name=self.name,
            starts=self._starts[:line_count],
            ends=self._ends[:line_count],
            colors=color,
            width=width,
            hidden=False,
        )


def log_tool_overlay(
    viewer: newton.viewer.ViewerBase,
    name: str,
    tool: Tool,
    width_scale: float = 2.0,
) -> None:
    """Render a tool's capsule segments as viewer lines.

    The line width is set from the first segment's radius scaled by
    ``width_scale`` so the stroke roughly matches the capsule cross-section
    even though viewers typically render lines as unshaded strokes.

    Colour flips between idle (grey) and active (heat-red) based on
    ``tool.active``.
    """
    colour = _TOOL_COLOR_ACTIVE if tool.active else _TOOL_COLOR_IDLE
    # Grab the first segment's radius for line width; a mixed-radius tool
    # would need per-segment widths which log_lines does not support.
    radius = float(tool._radius[0])
    viewer.log_lines(
        name=name,
        starts=tool.segments.p0,
        ends=tool.segments.p1,
        colors=colour,
        width=radius * width_scale,
    )
