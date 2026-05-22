import newton
import numpy as np
import warp as wp

from omnisurg.config import ViewerConfig
from omnisurg.rendering.headless import HeadlessRenderer
from omnisurg.rendering.slang import SLANG_RENDER_BACKENDS, SlangRenderer
from omnisurg.rendering.surgsim import SurgSimCompatRenderer
from omnisurg.rendering.textures import enable_persistent_gl_textures


_FAST_GL_PATCHED = False
_ZERO_COPY_MESH_PATCHED = False
_STATIC_INSTANCE_UPLOAD_PATCHED = False
_SIDE_PANEL_WIDTH_PATCHED = False
_SIDE_PANEL_WIDTH_SCALE = 1.5


def enable_direct_gl_render():
    global _FAST_GL_PATCHED
    if _FAST_GL_PATCHED:
        return

    from newton._src.viewer.gl.opengl import RendererGL, check_gl_error

    if getattr(RendererGL.render, "__omnisurg_patched__", False):
        _FAST_GL_PATCHED = True
        return

    original_render = RendererGL.render

    def patched_render(self, camera, objects, lines=None, wireframe_shapes=None, arrows=None, *args, **kwargs):
        if not getattr(self, "_omnisurg_direct_render", False):
            return original_render(
                self,
                camera,
                objects,
                lines,
                wireframe_shapes,
                arrows,
                *args,
                **kwargs,
            )

        gl = RendererGL.gl
        self._make_current()

        gl.glClearColor(*self.sky_upper, 1)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthMask(True)
        gl.glDepthRange(0.0, 1.0)

        self.camera = camera

        if self._sun_direction is None:
            sun_dirs = {
                0: np.array((0.8, 0.2, -0.3)),
                1: np.array((0.2, 0.8, -0.3)),
                2: np.array((0.2, -0.3, 0.8)),
            }
            direction = sun_dirs.get(camera.up_axis, sun_dirs[2])
            self._sun_direction = direction / np.linalg.norm(direction)

        self._view_matrix = self.camera.get_view_matrix()
        self._projection_matrix = self.camera.get_projection_matrix()

        if self._env_path is not None and self._env_texture is None:
            try:
                self.set_environment_map(self._env_path)
            except Exception:
                pass
            self._env_path = None

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, self._screen_width, self._screen_height)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glBindVertexArray(0)

        self._render_scene(objects)

        if lines:
            self._render_lines(lines)

        if arrows:
            self._render_arrows(arrows)

        if wireframe_shapes:
            self._render_wireframe_shapes(wireframe_shapes)

        check_gl_error()

    patched_render.__omnisurg_patched__ = True
    RendererGL.render = patched_render
    _FAST_GL_PATCHED = True


def enable_zero_copy_gl_mesh_updates():
    global _ZERO_COPY_MESH_PATCHED
    if _ZERO_COPY_MESH_PATCHED:
        return

    from newton._src.viewer.gl import opengl

    if getattr(opengl.MeshGL.update, "__omnisurg_zero_copy_patched__", False):
        _ZERO_COPY_MESH_PATCHED = True
        return

    original_update = opengl.MeshGL.update

    def patched_update(self, points, indices, normals, uvs, texture=None):
        if not (opengl.ENABLE_CUDA_INTEROP and self.device.is_cuda and self.vertex_cuda_buffer is not None):
            return original_update(self, points, indices, normals, uvs, texture)

        gl = opengl.RendererGL.gl

        if len(points) != len(self.vertices):
            raise RuntimeError("Number of points does not match")

        self._points = points

        if self.indices is None:
            self.indices = wp.clone(indices).view(dtype=wp.uint32)
            self.num_indices = int(len(self.indices))

            host_indices = self.indices.numpy()
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            gl.glBufferData(
                gl.GL_ELEMENT_ARRAY_BUFFER, host_indices.nbytes, host_indices.ctypes.data, gl.GL_STATIC_DRAW
            )

        if points is not None and normals is None:
            self.recompute_normals()
            normals = self.normals

        vbo_vertices = self.vertex_cuda_buffer.map(dtype=opengl.RenderVertex, shape=self.vertices.shape)
        wp.launch(
            opengl.fill_vertex_data,
            dim=len(vbo_vertices),
            inputs=[points, normals, uvs],
            outputs=[vbo_vertices],
            device=self.device,
            record_tape=False,
        )
        self.vertex_cuda_buffer.unmap()
        self.update_texture(texture)

    patched_update.__omnisurg_zero_copy_patched__ = True
    opengl.MeshGL.update = patched_update
    _ZERO_COPY_MESH_PATCHED = True


def enable_static_instance_uploads():
    global _STATIC_INSTANCE_UPLOAD_PATCHED
    if _STATIC_INSTANCE_UPLOAD_PATCHED:
        return

    from newton._src.viewer.gl import opengl

    if getattr(opengl.MeshInstancerGL._update_vbo, "__omnisurg_static_upload_patched__", False):
        _STATIC_INSTANCE_UPLOAD_PATCHED = True
        return

    def patched_update_vbo(self, xforms, colors, materials):
        gl = opengl.RendererGL.gl

        if opengl.ENABLE_CUDA_INTEROP and self.device.is_cuda:
            vbo_transforms = self._instance_transform_cuda_buffer.map(dtype=wp.mat44, shape=(self.num_instances,))
            wp.copy(vbo_transforms, xforms)
            self._instance_transform_cuda_buffer.unmap()
        else:
            host_transforms = xforms.numpy()
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_transform_buffer)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, host_transforms.nbytes, host_transforms.ctypes.data, gl.GL_DYNAMIC_DRAW)

        if colors is not None:
            color_state = (id(colors), len(colors))
            if getattr(self, "_omnisurg_color_state", None) != color_state:
                host_colors = colors.numpy()
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_color_buffer)
                gl.glBufferData(gl.GL_ARRAY_BUFFER, host_colors.nbytes, host_colors.ctypes.data, gl.GL_STATIC_DRAW)
                self._omnisurg_color_state = color_state

        if materials is not None:
            material_state = (id(materials), len(materials))
            if getattr(self, "_omnisurg_material_state", None) != material_state:
                host_materials = materials.numpy()
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.instance_material_buffer)
                gl.glBufferData(
                    gl.GL_ARRAY_BUFFER, host_materials.nbytes, host_materials.ctypes.data, gl.GL_STATIC_DRAW
                )
                self._omnisurg_material_state = material_state

    patched_update_vbo.__omnisurg_static_upload_patched__ = True
    opengl.MeshInstancerGL._update_vbo = patched_update_vbo
    _STATIC_INSTANCE_UPLOAD_PATCHED = True


def enable_wider_gl_side_panel():
    global _SIDE_PANEL_WIDTH_PATCHED
    if _SIDE_PANEL_WIDTH_PATCHED:
        return

    from newton._src.viewer.viewer_gl import ViewerGL

    if getattr(ViewerGL._render_left_panel, "__omnisurg_side_panel_width_patched__", False):
        _SIDE_PANEL_WIDTH_PATCHED = True
        return

    original_render_left_panel = ViewerGL._render_left_panel

    def patched_render_left_panel(self):
        ui = getattr(self, "ui", None)
        imgui = getattr(ui, "imgui", None)
        if imgui is None:
            return original_render_left_panel(self)

        original_set_next_window_size = imgui.set_next_window_size

        def patched_set_next_window_size(size, *args, **kwargs):
            if getattr(size, "x", None) == 300:
                size = imgui.ImVec2(size.x * _SIDE_PANEL_WIDTH_SCALE, size.y)
            return original_set_next_window_size(size, *args, **kwargs)

        imgui.set_next_window_size = patched_set_next_window_size
        try:
            return original_render_left_panel(self)
        finally:
            imgui.set_next_window_size = original_set_next_window_size

    patched_render_left_panel.__omnisurg_side_panel_width_patched__ = True
    ViewerGL._render_left_panel = patched_render_left_panel
    _SIDE_PANEL_WIDTH_PATCHED = True


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

    def __getattr__(self, name):
        renderer = object.__getattribute__(self, "_renderer")
        return getattr(renderer, name)

    @property
    def renderer(self):
        return self._renderer

    @property
    def supports_mouse_interaction(self) -> bool:
        if getattr(self, "_backend", "") == "headless":
            return False
        return self._backend in {"gl", "surgsim"} or bool(getattr(self._renderer, "supports_mouse_interaction", False))

    @property
    def show_particles(self) -> bool:
        return bool(getattr(self._renderer, "show_particles", False))

    @show_particles.setter
    def show_particles(self, value: bool) -> None:
        setattr(self._renderer, "show_particles", bool(value))

    @property
    def show_ui(self) -> bool:
        return bool(getattr(self._renderer, "show_ui", False))

    @show_ui.setter
    def show_ui(self, value: bool) -> None:
        setattr(self._renderer, "show_ui", bool(value))

    @property
    def _paused(self) -> bool:
        return bool(getattr(self._renderer, "_paused", False))

    @_paused.setter
    def _paused(self, value: bool) -> None:
        setattr(self._renderer, "_paused", bool(value))

    @classmethod
    def wrap_existing(cls, viewer, *, backend: str | None = None, device=None) -> "RenderBridge":
        """Wrap an already-created viewer with the RenderBridge API."""
        obj = cls.__new__(cls)
        if device is None:
            model = getattr(viewer, "model", None)
            device = getattr(model, "device", None)
        if device is None:
            device = wp.get_device()
        obj.gpu = GPUBuffers(device)
        if backend is None:
            if viewer.__class__.__name__ == "_HeadlessHexViewer":
                backend = "headless"
            elif viewer.__class__.__name__ == "ViewerUSD":
                backend = "usd"
            elif viewer.__class__.__name__ == "ViewerGL":
                backend = "gl"
            else:
                backend = str(getattr(viewer, "backend", "wrapped"))
        obj._backend = backend
        obj._renderer = viewer
        obj._mesh_created = set()
        obj._mesh_instance_state = {}
        obj._instance_colors = {}
        obj._point_radii = {}
        obj._point_colors = {}
        return obj

    def __init__(self, viewer_config: ViewerConfig, model, device):
        self.gpu = GPUBuffers(device)
        self._backend = viewer_config.backend
        self._mesh_created: set[str] = set()
        self._mesh_instance_state: dict[str, tuple[bool, tuple[float, float, float] | None]] = {}
        self._instance_colors: dict[str, wp.array] = {}
        self._point_radii: dict[tuple[str, int, float], wp.array] = {}
        self._point_colors: dict[tuple[str, int, tuple[float, float, float]], wp.array] = {}

        if self._backend in {"gl", "surgsim"}:
            enable_persistent_gl_textures()
            enable_direct_gl_render()
            enable_zero_copy_gl_mesh_updates()
            enable_static_instance_uploads()
            enable_wider_gl_side_panel()

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
        elif self._backend in SLANG_RENDER_BACKENDS:
            self._renderer = SlangRenderer(viewer_config, model, device)
        else:
            self._renderer = newton.viewer.ViewerGL(vsync=viewer_config.vsync)
            self._renderer.set_model(model)
            self._renderer.set_camera(wp.vec3f(*viewer_config.camera_pos), 0, -90)

        self.configure_render_quality(
            sky_enabled=viewer_config.sky_enabled,
            shadows_enabled=viewer_config.shadows_enabled,
            msaa_samples=viewer_config.msaa_samples,
            direct_render_enabled=viewer_config.direct_render_enabled,
        )

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

    def _viewer_renderer(self):
        return getattr(self._renderer, "renderer", None)

    def _callback_target(self):
        return self._renderer if hasattr(self._renderer, "register_mouse_motion") else self._viewer_renderer()

    def configure_render_quality(
        self,
        *,
        sky_enabled: bool | None = None,
        shadows_enabled: bool | None = None,
        msaa_samples: int | None = None,
        direct_render_enabled: bool | None = None,
    ):
        renderer = self._viewer_renderer()
        if renderer is None:
            return

        if hasattr(renderer, "_make_current"):
            renderer._make_current()

        if sky_enabled is not None and hasattr(renderer, "draw_sky"):
            renderer.draw_sky = bool(sky_enabled)

        if shadows_enabled is not None and hasattr(renderer, "draw_shadows"):
            renderer.draw_shadows = bool(shadows_enabled)
            if not renderer.draw_shadows and not hasattr(renderer, "_light_space_matrix"):
                renderer._light_space_matrix = np.eye(4, dtype=np.float32)

        if msaa_samples is not None and hasattr(renderer, "msaa_samples"):
            samples = max(0, int(msaa_samples))
            renderer.msaa_samples = samples
            gl = getattr(renderer, "gl", None)
            if gl is not None:
                if samples > 0:
                    gl.glEnable(gl.GL_MULTISAMPLE)
                else:
                    gl.glDisable(gl.GL_MULTISAMPLE)

        if direct_render_enabled is not None:
            renderer._omnisurg_direct_render = (
                bool(direct_render_enabled)
                and not bool(getattr(renderer, "draw_shadows", False))
                and int(getattr(renderer, "msaa_samples", 0)) == 0
            )

    def begin_frame(self, time: float):
        self._renderer.begin_frame(time)

    def end_frame(self):
        self._renderer.end_frame()

    def is_running(self) -> bool:
        return self._renderer.is_running()

    def close(self):
        if hasattr(self._renderer, "close"):
            self._renderer.close()

    def set_model(self, model) -> None:
        method = getattr(self._renderer, "set_model", None)
        if callable(method):
            method(model)

    def set_camera(self, *args, **kwargs) -> None:
        method = getattr(self._renderer, "set_camera", None)
        if callable(method):
            method(*args, **kwargs)

    def is_paused(self) -> bool:
        method = getattr(self._renderer, "is_paused", None)
        if callable(method):
            return bool(method())
        return bool(getattr(self._renderer, "_paused", False))

    def is_key_down(self, symbol: int) -> bool:
        for target in (self._renderer, self._viewer_renderer()):
            if target is None:
                continue
            method = getattr(target, "is_key_down", None)
            if callable(method):
                try:
                    return bool(method(int(symbol)))
                except Exception:
                    return False
            key_handler = getattr(target, "_key_handler", None)
            if key_handler is not None:
                try:
                    return bool(key_handler[int(symbol)])
                except Exception:
                    return False
        return False

    def log_scalar(self, name: str, value: float) -> None:
        log_fn = getattr(self._renderer, "log_scalar", None)
        if callable(log_fn):
            log_fn(name, float(value))

    def log_state(self, state) -> None:
        log_fn = getattr(self._renderer, "log_state", None)
        if callable(log_fn):
            log_fn(state)

    def set_tissue_material_params(self, **params) -> None:
        if self._backend not in SLANG_RENDER_BACKENDS:
            return
        set_params = getattr(self._renderer, "set_tissue_material_params", None)
        if callable(set_params):
            set_params(**params)

    def set_postprocess_params(self, **params) -> None:
        if self._backend not in SLANG_RENDER_BACKENDS:
            return
        set_params = getattr(self._renderer, "set_postprocess_params", None)
        if callable(set_params):
            set_params(**params)

    def draw_mesh(
        self,
        name: str,
        particle_q: wp.array,
        surface_indices: wp.array,
        color: tuple[float, float, float] | None = None,
        uvs: wp.array | None = None,
        texture: str | None = None,
        vertex_colors: wp.array | None = None,
        hidden: bool = False,
    ):
        if hidden or particle_q is None or surface_indices is None:
            self.log_mesh(name=name, points=None, indices=None, uvs=None, texture=None, hidden=True)
            return
        try:
            if len(particle_q) == 0 or len(surface_indices) == 0:
                self.log_mesh(name=name, points=None, indices=None, uvs=None, texture=None, hidden=True)
                return
        except Exception:
            pass

        if self._backend == "headless":
            return

        if self._backend in SLANG_RENDER_BACKENDS:
            self._renderer.draw_mesh(
                name=name,
                particle_q=particle_q,
                surface_indices=surface_indices,
                color=color,
                uvs=uvs,
                texture=texture,
                vertex_colors=vertex_colors,
                hidden=hidden,
            )
            self._mesh_created.add(name)
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

        instance_name = f"{name}_inst"
        instance_state = (textured, None if color is None else tuple(float(component) for component in color))
        if self._mesh_instance_state.get(instance_name) != instance_state:
            self._renderer.log_instances(
                instance_name,
                name,
                self.gpu.identity_xform,
                self.gpu.unit_scale,
                instance_color,
                materials=instance_material,
            )
            self._mesh_instance_state[instance_name] = instance_state

        self._mesh_created.add(name)

    def draw_points(
        self,
        name: str,
        points: wp.array | None,
        radii: wp.array | float | None = None,
        colors: wp.array | tuple[float, float, float] | list[float] | None = None,
        hidden: bool = False,
    ):
        if hidden or points is None:
            self.log_points(name=name, points=None, hidden=True)
            return
        try:
            if len(points) == 0:
                self.log_points(name=name, points=None, hidden=True)
                return
        except Exception:
            pass

        if self._backend == "headless":
            return

        if radii is None:
            radii = 0.01
        if colors is None:
            colors = (0.85, 0.18, 0.12)

        if self._backend in SLANG_RENDER_BACKENDS:
            self._renderer.draw_points(name, points, radii, colors, hidden=hidden)
            return

        radii, colors = self._normalize_point_inputs(name, points, radii, colors)
        self._renderer.log_points(name, points, radii, colors)

    def draw_lines(
        self,
        name: str,
        starts=None,
        ends=None,
        colors=None,
        width: float = 1.0,
        hidden: bool = False,
    ):
        if self._backend == "headless":
            return
        if hidden or starts is None or ends is None:
            self.log_lines(name=name, starts=None, ends=None, colors=colors, width=width, hidden=True)
            return
        method = getattr(self._renderer, "draw_lines", None)
        if callable(method):
            method(name=name, starts=starts, ends=ends, colors=colors, width=width, hidden=hidden)
            return
        self.log_lines(name=name, starts=starts, ends=ends, colors=colors, width=width, hidden=hidden)

    def log_mesh(self, *args, **kwargs) -> None:
        if self._backend == "headless":
            return
        method = getattr(self._renderer, "log_mesh", None)
        if callable(method):
            method(*args, **kwargs)

    def log_points(self, *args, **kwargs) -> None:
        if self._backend == "headless":
            return
        method = getattr(self._renderer, "log_points", None)
        if callable(method):
            method(*args, **kwargs)

    def log_lines(self, *args, **kwargs) -> None:
        if self._backend == "headless":
            return
        method = getattr(self._renderer, "log_lines", None)
        if callable(method):
            method(*args, **kwargs)

    def draw_haptic_sphere(self, position: wp.array, radius: float | None = None):
        if self._backend == "headless":
            return

        if self._backend in SLANG_RENDER_BACKENDS:
            self._renderer.draw_points(
                "haptic_sphere",
                position,
                0.025 if radius is None else radius,
                (0.8, 0.2, 0.2),
            )
            return

        haptic_radius = self.gpu.haptic_radius if radius is None else radius
        haptic_radius, haptic_color = self._normalize_point_inputs(
            "haptic_sphere",
            position,
            haptic_radius,
            self.gpu.haptic_color,
        )
        self._renderer.log_points(
            "haptic_sphere",
            position,
            haptic_radius,
            haptic_color,
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

    def set_mouse_callbacks(
        self,
        *,
        on_motion=None,
        on_press=None,
        on_drag=None,
        on_release=None,
    ):
        target = self._callback_target()
        if target is None:
            return

        callbacks = (
            ("register_mouse_motion", on_motion),
            ("register_mouse_press", on_press),
            ("register_mouse_drag", on_drag),
            ("register_mouse_release", on_release),
        )
        for method_name, callback in callbacks:
            if callback is None:
                continue
            method = getattr(target, method_name, None)
            if callable(method):
                method(callback)

    def screen_to_world_ray(self, x: float, y: float):
        for target in (self._renderer, self._viewer_renderer()):
            method = getattr(target, "screen_to_world_ray", None)
            if callable(method):
                return method(x, y)
            to_framebuffer = getattr(target, "_to_framebuffer_coords", None)
            camera = getattr(target, "camera", None)
            if callable(to_framebuffer) and camera is not None and hasattr(camera, "get_world_ray"):
                fb_x, fb_y = to_framebuffer(x, y)
                ray_start, ray_dir = camera.get_world_ray(fb_x, fb_y)
                origin = np.asarray(ray_start, dtype=np.float32).reshape(3)
                try:
                    direction = np.asarray((ray_dir.x, ray_dir.y, ray_dir.z), dtype=np.float32)
                except AttributeError:
                    direction = np.asarray(ray_dir, dtype=np.float32).reshape(3)
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm > 1.0e-8:
                    direction /= direction_norm
                return origin, direction
        raise RuntimeError(f"{self._backend} renderer does not expose screen_to_world_ray")

    def is_ui_capturing(self) -> bool:
        method = getattr(self._renderer, "is_ui_capturing", None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
        ui = getattr(self._renderer, "ui", None)
        if ui is None:
            return False
        method = getattr(ui, "is_capturing", None)
        if callable(method):
            try:
                return bool(method())
            except Exception:
                return False
        return False

    def draw_cryo_surface(self, *args, **kwargs):
        method = getattr(self._renderer, "draw_cryo_surface", None)
        if callable(method):
            return method(*args, **kwargs)
        return None

    def set_environment_path(self, path):
        method = getattr(self._renderer, "set_environment_path", None)
        if callable(method):
            method(path)

    def set_environment_intensity(self, intensity: float):
        method = getattr(self._renderer, "set_environment_intensity", None)
        if callable(method):
            method(intensity)

    def set_environment_background_enabled(self, enabled: bool):
        method = getattr(self._renderer, "set_environment_background_enabled", None)
        if callable(method):
            method(enabled)

    def set_environment_rotation_degrees(self, degrees: float):
        method = getattr(self._renderer, "set_environment_rotation_degrees", None)
        if callable(method):
            method(degrees)

    def set_environment_pitch_degrees(self, degrees: float):
        method = getattr(self._renderer, "set_environment_pitch_degrees", None)
        if callable(method):
            method(degrees)

    def register_ui_callback(self, callback, position: str = "side"):
        if hasattr(self._renderer, "register_ui_callback"):
            self._renderer.register_ui_callback(callback, position=position)
