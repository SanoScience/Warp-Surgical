import newton
import warp as wp


class SurgSimCompatRenderer:
    """Package-local compatibility path for the legacy `surgsim` viewer option."""

    def __init__(self, model, camera_pos: tuple, vsync: bool = True):
        self._renderer = newton.viewer.ViewerGL(vsync=vsync)
        self._renderer.set_model(model)
        self._renderer.set_camera(wp.vec3f(*camera_pos), 0, -90)
        self._device = model.device
        self._mesh_created: set[str] = set()

        self._identity_xform = wp.array([wp.transform()], dtype=wp.transformf, device=self._device)
        self._unit_scale = wp.array([1.0, 1.0, 1.0], dtype=wp.vec3, device=self._device)
        self._white_color = wp.array([wp.vec3(1.0, 1.0, 1.0)], dtype=wp.vec3, device=self._device)
        self._default_material = wp.array([wp.vec4(0.0, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self._device)

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    def close(self):
        if hasattr(self._renderer, "close"):
            self._renderer.close()

    def render_mesh_warp(
        self,
        *,
        name: str,
        points: wp.array,
        indices: wp.array,
        update_topology: bool = True,
        smooth_shading: bool = True,
        visible: bool = True,
    ):
        self._renderer.log_mesh(
            name=name,
            points=points,
            indices=indices,
            hidden=not visible,
        )
        if name not in self._mesh_created:
            self._renderer.log_instances(
                f"{name}_inst",
                name,
                self._identity_xform,
                self._unit_scale,
                self._white_color,
                materials=self._default_material,
            )
            self._mesh_created.add(name)
