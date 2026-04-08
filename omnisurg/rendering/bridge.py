import newton
import numpy as np
import warp as wp

from omnisurg.config import ViewerConfig
from omnisurg.rendering.headless import HeadlessRenderer
from omnisurg.rendering.surgsim import SurgSimCompatRenderer
from omnisurg.rendering.textures import enable_persistent_gl_textures


class GPUBuffers:
    """Constant GPU arrays allocated once at startup."""

    def __init__(self, device):
        self.device = device
        self.identity_xform = wp.array([wp.transform()], dtype=wp.transformf, device=device)
        self.unit_scale = wp.array([1.0, 1.0, 1.0], dtype=wp.vec3, device=device)
        self.white_color = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=device)
        self.default_material = wp.array([wp.vec4(0.5, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=device)
        self.textured_material = wp.array([wp.vec4(0.5, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=device)
        self.haptic_radius = wp.array([0.025], dtype=wp.float32, device=device)
        self.haptic_color = wp.array([[0.8, 0.2, 0.2]], dtype=wp.vec3f, device=device)


class RenderBridge:
    """Thin backend wrapper for the supported Phase viewers."""

    def __init__(self, viewer_config: ViewerConfig, model, device):
        self.gpu = GPUBuffers(device)
        self._backend = viewer_config.backend
        self._mesh_created: set[str] = set()
        self._instance_colors: dict[str, wp.array] = {}
        self._point_radii: dict[tuple[str, int, float], wp.array] = {}
        self._point_colors: dict[tuple[str, int, tuple[float, float, float]], wp.array] = {}

        if self._backend in {"gl", "surgsim"}:
            enable_persistent_gl_textures()

        if self._backend == "headless":
            self._renderer = HeadlessRenderer()
        elif self._backend == "surgsim":
            self._renderer = SurgSimCompatRenderer(
                model=model,
                camera_pos=viewer_config.camera_pos,
                vsync=viewer_config.vsync,
            )
        elif self._backend == "rtx":
            self._renderer = newton.viewer.ViewerRTX()
            self._renderer.set_model(model)
            self._renderer.set_camera(wp.vec3f(*viewer_config.camera_pos), 0, -90)
        else:
            self._renderer = newton.viewer.ViewerGL(vsync=viewer_config.vsync)
            self._renderer.set_model(model)
            self._renderer.set_camera(wp.vec3f(*viewer_config.camera_pos), 0, -90)

    def _color_buffer(self, name: str, color: tuple[float, float, float] | None):
        if color is None:
            return self.gpu.white_color
        if name not in self._instance_colors:
            self._instance_colors[name] = wp.array([wp.vec3(*color)], dtype=wp.vec3, device=self.gpu.device)
        return self._instance_colors[name]

    def _material_buffer(self, textured: bool):
        return self.gpu.textured_material if textured else self.gpu.default_material

    def _normalize_point_inputs(
        self,
        name: str,
        points: wp.array,
        radii: wp.array | float,
        colors: wp.array | tuple[float, float, float] | list[float],
    ):
        point_count = len(points)

        if isinstance(radii, (int, float)):
            radius = float(radii)
            radius_key = (name, point_count, radius)
            if radius_key not in self._point_radii:
                self._point_radii[radius_key] = wp.full(point_count, radius, dtype=wp.float32, device=self.gpu.device)
            radii = self._point_radii[radius_key]

        if isinstance(colors, (tuple, list)):
            color_array = np.asarray(colors, dtype=np.float32)
            if color_array.ndim == 1 and color_array.size == 3:
                color_key = (name, point_count, tuple(float(value) for value in color_array.tolist()))
                if color_key not in self._point_colors:
                    tiled = np.repeat(color_array[None, :], point_count, axis=0)
                    self._point_colors[color_key] = wp.array(tiled, dtype=wp.vec3f, device=self.gpu.device)
                colors = self._point_colors[color_key]

        return radii, colors

    def begin_frame(self, time: float):
        self._renderer.begin_frame(time)

    def end_frame(self):
        self._renderer.end_frame()

    def is_running(self) -> bool:
        return self._renderer.is_running()

    def close(self):
        if hasattr(self._renderer, "close"):
            self._renderer.close()

    def draw_mesh(
        self,
        name: str,
        particle_q: wp.array,
        surface_indices: wp.array,
        color: tuple[float, float, float] | None = None,
        uvs: wp.array | None = None,
        texture: str | None = None,
    ):
        if self._backend == "headless":
            return

        textured = uvs is not None and texture is not None
        mesh_texture = texture if textured else None
        instance_color = self.gpu.white_color if textured else self._color_buffer(name, color)
        instance_material = self._material_buffer(textured)

        if self._backend == "surgsim":
            self._renderer.render_mesh_warp(
                name=name,
                points=particle_q,
                indices=surface_indices,
                uvs=uvs,
                texture=mesh_texture,
                color=color,
                update_topology=name not in self._mesh_created,
                smooth_shading=True,
                visible=True,
            )
            self._mesh_created.add(name)
            return

        self._renderer.log_mesh(
            name=name,
            points=particle_q,
            indices=surface_indices,
            uvs=uvs,
            texture=mesh_texture,
            hidden=True,
        )
        self._renderer.log_instances(
            f"{name}_inst",
            name,
            self.gpu.identity_xform,
            self.gpu.unit_scale,
            instance_color,
            materials=instance_material,
        )
        self._mesh_created.add(name)

    def draw_points(
        self,
        name: str,
        points: wp.array,
        radii: wp.array | float,
        colors: wp.array | tuple[float, float, float] | list[float],
    ):
        if self._backend == "headless":
            return

        radii, colors = self._normalize_point_inputs(name, points, radii, colors)
        self._renderer.log_points(name, points, radii, colors)

    def draw_haptic_sphere(self, position: wp.array):
        if self._backend == "headless":
            return

        self._renderer.log_points(
            "haptic_sphere",
            position,
            self.gpu.haptic_radius,
            self.gpu.haptic_color,
        )

    def set_input_callbacks(self, on_key_press=None, on_key_release=None):
        if hasattr(self._renderer, "set_input_callbacks"):
            self._renderer.set_input_callbacks(on_key_press, on_key_release)
            return

        viewer_renderer = getattr(self._renderer, "renderer", None)
        if viewer_renderer is None:
            return

        if on_key_press is not None and hasattr(viewer_renderer, "register_key_press"):
            viewer_renderer.register_key_press(on_key_press)
        if on_key_release is not None and hasattr(viewer_renderer, "register_key_release"):
            viewer_renderer.register_key_release(on_key_release)

    def register_ui_callback(self, callback, position: str = "side"):
        if hasattr(self._renderer, "register_ui_callback"):
            self._renderer.register_ui_callback(callback, position=position)
