import newton
import warp as wp

from omnisurg.config import ViewerConfig
from omnisurg.rendering.headless import HeadlessRenderer
from omnisurg.rendering.surgsim import SurgSimCompatRenderer


class GPUBuffers:
    """Constant GPU arrays allocated once at startup."""

    def __init__(self, device):
        self.identity_xform = wp.array([wp.transform()], dtype=wp.transformf, device=device)
        self.unit_scale = wp.array([1.0, 1.0, 1.0], dtype=wp.vec3, device=device)
        self.white_color = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=device)
        self.default_material = wp.array([wp.vec4(0.0, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=device)
        self.haptic_radius = wp.array([0.025], dtype=wp.float32, device=device)
        self.haptic_color = wp.array([[0.8, 0.2, 0.2]], dtype=wp.vec3f, device=device)


class RenderBridge:
    """Thin backend wrapper for the supported Phase 1 viewers."""

    def __init__(self, viewer_config: ViewerConfig, model, device):
        self.gpu = GPUBuffers(device)
        self._backend = viewer_config.backend
        self._mesh_created: set[str] = set()

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

    def begin_frame(self, time: float):
        self._renderer.begin_frame(time)

    def end_frame(self):
        self._renderer.end_frame()

    def is_running(self) -> bool:
        return self._renderer.is_running()

    def close(self):
        if hasattr(self._renderer, "close"):
            self._renderer.close()

    def draw_mesh(self, name: str, particle_q: wp.array, surface_indices: wp.array):
        if self._backend == "headless":
            return

        if self._backend == "surgsim":
            self._renderer.render_mesh_warp(
                name=name,
                points=particle_q,
                indices=surface_indices,
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
            hidden=True,
        )
        if name not in self._mesh_created:
            self._renderer.log_instances(
                f"{name}_inst",
                name,
                self.gpu.identity_xform,
                self.gpu.unit_scale,
                self.gpu.white_color,
                materials=self.gpu.default_material,
            )
            self._mesh_created.add(name)

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
