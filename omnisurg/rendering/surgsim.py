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
        self._default_material = wp.array([wp.vec4(0.5, 0.0, 0.0, 0.0)], dtype=wp.vec4, device=self._device)
        self._textured_material = wp.array([wp.vec4(0.5, 0.0, 0.0, 1.0)], dtype=wp.vec4, device=self._device)
        self._instance_colors: dict[str, wp.array] = {}

    def __getattr__(self, name):
        return getattr(self._renderer, name)

    def close(self):
        if hasattr(self._renderer, "close"):
            self._renderer.close()

    def _color_buffer(self, name: str, color: tuple[float, float, float] | None):
        if color is None:
            return self._white_color
        if name not in self._instance_colors:
            self._instance_colors[name] = wp.array([wp.vec3(*color)], dtype=wp.vec3, device=self._device)
        return self._instance_colors[name]

    def set_input_callbacks(self, on_key_press=None, on_key_release=None):
        if on_key_press is not None:
            self._renderer.renderer.register_key_press(on_key_press)
        if on_key_release is not None:
            self._renderer.renderer.register_key_release(on_key_release)

    def render_mesh_warp(
        self,
        *,
        name: str,
        points: wp.array,
        indices: wp.array,
        uvs: wp.array | None = None,
        texture: str | None = None,
        color: tuple[float, float, float] | None = None,
        update_topology: bool = True,
        smooth_shading: bool = True,
        visible: bool = True,
    ):
        textured = uvs is not None and texture is not None
        self._renderer.log_mesh(
            name=name,
            points=points,
            indices=indices,
            uvs=uvs,
            texture=texture if textured else None,
            hidden=not visible,
        )
        self._renderer.log_instances(
            f"{name}_inst",
            name,
            self._identity_xform,
            self._unit_scale,
            self._white_color if textured else self._color_buffer(name, color),
            materials=self._textured_material if textured else self._default_material,
        )
        self._mesh_created.add(name)
