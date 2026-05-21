"""Render-state tests for textured surgical surfaces."""

from __future__ import annotations

import numpy as np
import warp as wp
from newton._src.viewer.gl import shaders as gl_shaders
from newton._src.viewer.gl.opengl import RendererGL

import omnisurg.hex.render as render_mod


class _FakeGL:
    def __init__(self) -> None:
        self.attrib3: list[tuple[int, float, float, float]] = []
        self.attrib4: list[tuple[int, float, float, float, float]] = []
        self.uniform1i: list[tuple[int, int]] = []

    def glVertexAttrib3f(self, index: int, x: float, y: float, z: float) -> None:
        self.attrib3.append((index, x, y, z))

    def glVertexAttrib4f(self, index: int, x: float, y: float, z: float, w: float) -> None:
        self.attrib4.append((index, x, y, z, w))

    def glUniform1i(self, location: int, value: int) -> None:
        self.uniform1i.append((location, value))


class _FakeMesh:
    def __init__(self) -> None:
        self.hidden = False
        self.vao = object()
        self.color = (0.7, 0.5, 0.3)
        self.material = (0.5, 0.0, 0.0, 0.0)
        self.render_seen: list[tuple[tuple[float, ...], tuple[float, ...]]] = []

    def render(self) -> None:
        self.render_seen.append((tuple(self.color), tuple(self.material)))


def test_textured_render_sets_mesh_state_before_newton_render(monkeypatch):
    fake_gl = _FakeGL()
    monkeypatch.setattr(RendererGL, "gl", fake_gl)
    mesh = _FakeMesh()

    render_mod._enable_textured_render(mesh)
    mesh.render()

    assert mesh.render_seen == [((1.0, 1.0, 1.0), (0.5, 0.0, 0.0, 1.0))]
    assert mesh.color == (0.7, 0.5, 0.3)
    assert mesh.material == (0.5, 0.0, 0.0, 0.0)
    assert fake_gl.attrib3[-1] == (7, 0.7, 0.5, 0.3)
    assert fake_gl.attrib4[-1] == (8, 0.5, 0.0, 0.0, 0.0)


def test_cryo_volume_render_sets_white_mesh_state_before_newton_render(monkeypatch):
    fake_gl = _FakeGL()
    uniform_events: list[bool] = []
    monkeypatch.setattr(RendererGL, "gl", fake_gl)
    monkeypatch.setattr(render_mod, "_ensure_cryo_volume_destroy_patch", lambda mesh: None)
    upload_kwargs: list[dict] = []
    monkeypatch.setattr(
        render_mod,
        "_upload_cryo_volume_texture",
        lambda mesh, host, **kwargs: upload_kwargs.append(kwargs),
    )
    monkeypatch.setattr(render_mod, "_upload_cryo_uv3_attrib", lambda mesh, uv3: None)
    monkeypatch.setattr(
        render_mod,
        "_set_cryo_volume_uniforms",
        lambda mesh, enabled: uniform_events.append(bool(enabled)),
    )
    mesh = _FakeMesh()
    host = np.zeros((1, 1, 1, 3), dtype=np.uint8)

    render_mod._enable_cryo_volume_render(mesh, host, object(), linear_filter=False)
    mesh.render()

    assert upload_kwargs == [{"linear_filter": False}]
    assert mesh.render_seen == [((1.0, 1.0, 1.0), (0.5, 0.0, 0.0, 0.0))]
    assert mesh.color == (0.7, 0.5, 0.3)
    assert mesh.material == (0.5, 0.0, 0.0, 0.0)
    assert uniform_events == [True, False]
    assert fake_gl.attrib3[-1] == (7, 0.7, 0.5, 0.3)


def test_cryo_shape_shader_update_assigns_distinct_sampler_unit():
    class _FakeShaderShape:
        def __init__(self) -> None:
            self._gl = _FakeGL()
            self.update_calls = 0

        def _get_uniform_location(self, name: str) -> int:
            return {"cryo_volume": 11, "cryo_volume_enabled": 12}.get(name, -1)

        def update(self) -> str:
            self.update_calls += 1
            return "updated"

    class _FakeShaders:
        ShaderShape = _FakeShaderShape

    render_mod._install_cryo_shape_shader_defaults(_FakeShaders)
    render_mod._install_cryo_shape_shader_defaults(_FakeShaders)

    shader = _FakeShaders.ShaderShape()

    assert shader.update() == "updated"
    assert shader.update_calls == 1
    assert shader._gl.uniform1i == [(11, 3), (12, 0)]


def test_build_segmentation_color_texture_maps_labels_to_rgb_volume():
    labels = np.asarray(
        [
            [[0, 1], [2, 1]],
        ],
        dtype=np.uint8,
    )
    colors = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.5, 0.0],
            [2.0, -1.0, 0.25],
        ],
        dtype=np.float32,
    )

    host, device = render_mod.build_segmentation_color_texture(labels, colors, device="cpu")

    assert host.shape == (1, 2, 2, 3)
    assert host.dtype == np.uint8
    np.testing.assert_array_equal(host[0, 0, 1], np.asarray([255, 128, 0], dtype=np.uint8))
    np.testing.assert_array_equal(host[0, 1, 0], np.asarray([255, 0, 64], dtype=np.uint8))
    np.testing.assert_allclose(device.numpy(), host.astype(np.float32) / 255.0)


def test_segmentation_majority_uv3_kernel_keeps_triangle_categorical():
    tri_indices = wp.array(np.asarray([[0, 6, 12]], dtype=np.int32), dtype=wp.int32, device="cpu")
    vertex_pos = wp.array(np.zeros((18, 3), dtype=np.float32), dtype=wp.vec3, device="cpu")
    particle_uv3 = wp.array(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device="cpu",
    )
    particle_material = wp.array(np.asarray([1, 2, 2], dtype=np.int32), dtype=wp.int32, device="cpu")
    flat_pos = wp.zeros(3, dtype=wp.vec3, device="cpu")
    flat_uv3 = wp.zeros(3, dtype=wp.vec3, device="cpu")

    wp.launch(
        render_mod._expand_triangle_vertices_majority_uv3_kernel,
        dim=1,
        inputs=[tri_indices, vertex_pos, particle_uv3, particle_material, 1],
        outputs=[flat_pos, flat_uv3],
        device="cpu",
    )

    np.testing.assert_allclose(flat_uv3.numpy(), np.asarray([[0.5, 0.0, 0.0]] * 3, dtype=np.float32))


def test_slang_material_state_kernel_uses_triangle_majority_material():
    tri_indices = wp.array(np.asarray([[0, 6, 12]], dtype=np.int32), dtype=wp.int32, device="cpu")
    particle_material = wp.array(np.asarray([3, 4, 4], dtype=np.int32), dtype=wp.int32, device="cpu")
    flat_material = wp.zeros(3, dtype=wp.int32, device="cpu")
    flat_state = wp.zeros(3, dtype=wp.vec4, device="cpu")

    wp.launch(
        render_mod._expand_triangle_material_state_kernel,
        dim=1,
        inputs=[tri_indices, particle_material, 1],
        outputs=[flat_material, flat_state],
        device="cpu",
    )

    np.testing.assert_array_equal(flat_material.numpy(), np.asarray([4, 4, 4], dtype=np.int32))
    np.testing.assert_allclose(flat_state.numpy(), np.zeros((3, 4), dtype=np.float32))


def test_cryo_surface_frame_accepts_deferred_material_state():
    frame = render_mod.CryoSurfaceFrame(
        positions=object(),
        normals=object(),
        uv3=object(),
        material_id=None,
        state_rgba=None,
        vertex_count=0,
        triangle_count=0,
        buffer_generation=2,
    )

    assert frame.material_id is None
    assert frame.state_rgba is None
    assert frame.procedural_coord is None
    assert frame.buffer_generation == 2


def test_surface_renderer_flat_capacity_defers_material_state_buffers():
    renderer = object.__new__(render_mod.SurfaceRenderer)
    renderer.device = "cpu"
    renderer._flat_pos = None
    renderer._flat_uv3 = None
    renderer._flat_procedural_coord = None
    renderer._flat_normals = None
    renderer._flat_material_id = None
    renderer._flat_state_rgba = None
    renderer._tri_centroid_uv3 = None
    renderer._max_triangles = 0
    renderer._cached_flat_uv3_valid = True
    renderer._flat_uv3_kind = "vertex"
    renderer._flat_buffer_generation = 0

    renderer._ensure_flat_surface_capacity(2, include_material_state=False)

    assert renderer._flat_pos is not None
    assert renderer._flat_uv3 is not None
    assert renderer._flat_procedural_coord is not None
    assert renderer._flat_normals is not None
    assert renderer._flat_material_id is None
    assert renderer._flat_state_rgba is None
    assert renderer._flat_buffer_generation == 1

    renderer._ensure_flat_surface_capacity(2, include_material_state=False)
    assert renderer._flat_buffer_generation == 1

    renderer._ensure_flat_surface_capacity(2, include_material_state=True)
    assert renderer._flat_material_id is not None
    assert renderer._flat_state_rgba is not None
    assert renderer._flat_buffer_generation == 2


def test_taubin_smoothing_can_use_separate_procedural_coordinate_scratch():
    renderer = object.__new__(render_mod.SurfaceRenderer)
    renderer.device = "cpu"
    renderer._vertex_cap = 0
    renderer._smoothed_pos_a = None
    renderer._smoothed_pos_b = None
    renderer._smoothed_procedural_coord_a = None
    renderer._smoothed_procedural_coord_b = None
    renderer._smoothed_normals = None
    renderer._neighbour_sum = None
    renderer._neighbour_degree = None
    tri_indices = wp.array(np.asarray([[0, 1, 2]], dtype=np.int32), dtype=wp.int32, device="cpu")
    coords = wp.array(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device="cpu",
    )

    renderer._ensure_vertex_capacity(3)
    pos_scratch = renderer._smoothed_pos_a
    assert pos_scratch is not None
    assert renderer._smoothed_procedural_coord_a is not None
    assert renderer._smoothed_procedural_coord_b is not None
    smoothed = renderer._run_taubin_smoothing(
        tri_indices=tri_indices,
        vertex_pos=coords,
        num_triangles=1,
        iterations=1,
        lambda_coeff=1.0,
        mu_coeff=0.0,
        scratch_a=renderer._smoothed_procedural_coord_a,
        scratch_b=renderer._smoothed_procedural_coord_b,
    )

    np.testing.assert_allclose(
        smoothed.numpy(),
        np.asarray(
            [
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(pos_scratch.numpy(), np.zeros((3, 3), dtype=np.float32))


def test_slang_flat_normal_kernel_writes_geometric_triangle_normal():
    flat_pos = wp.array(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        dtype=wp.vec3,
        device="cpu",
    )
    flat_normals = wp.zeros(3, dtype=wp.vec3, device="cpu")

    wp.launch(
        render_mod._compute_flat_triangle_normals_kernel,
        dim=1,
        inputs=[flat_pos, 1],
        outputs=[flat_normals],
        device="cpu",
    )

    np.testing.assert_allclose(flat_normals.numpy(), np.asarray([[0.0, 0.0, 1.0]] * 3, dtype=np.float32))


def test_cryo_volume_shader_uses_unmodulated_cryo_albedo(monkeypatch):
    vertex_shader = """#version 330 core
layout (location = 2) in vec2 aTexCoord;
out vec2 TexCoord;
void main()
{
    TexCoord = aTexCoord;
}
"""
    fragment_shader = """#version 330 core
in vec2 TexCoord;
uniform sampler2D albedo_map;
const float PI = 3.14159265359;
void main()
{
    if (texture_enable > 0.5)
    {
        vec3 tex_color = texture(albedo_map, TexCoord).rgb;
        albedo *= pow(tex_color, vec3(2.2));
    }
}
"""
    monkeypatch.setattr(gl_shaders, "shape_vertex_shader", vertex_shader)
    monkeypatch.setattr(gl_shaders, "shape_fragment_shader", fragment_shader)

    assert render_mod.install_cryo_volume_shader_patch()

    patched = gl_shaders.shape_fragment_shader
    assert "albedo = pow(cryo_color, vec3(2.2));" in patched
    assert "albedo *= pow(cryo_color" not in patched


def test_cryo_volume_shader_patch_upgrades_existing_multiply_patch(monkeypatch):
    monkeypatch.setattr(
        gl_shaders,
        "shape_fragment_shader",
        """uniform bool cryo_volume_enabled;
void main()
{
    if (cryo_volume_enabled)
    {
        vec3 cryo_color = vec3(1.0);
        albedo *= pow(cryo_color, vec3(2.2));
    }
}
""",
    )

    assert render_mod.install_cryo_volume_shader_patch()

    assert "albedo = pow(cryo_color, vec3(2.2));" in gl_shaders.shape_fragment_shader
    assert "albedo *= pow(cryo_color" not in gl_shaders.shape_fragment_shader
