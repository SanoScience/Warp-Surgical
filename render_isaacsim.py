"""Isaac Sim 6 RTX renderer backend for the surgical simulator.

This module provides an ``IsaacSimRenderer`` class that mirrors the
``OvrtxRenderer`` interface so it can be plugged in via ``_setup_renderer()``.

IMPORTANT: ``isaacsim.SimulationApp`` **must** be created before this module
is imported because Omniverse / pxr imports require a running Kit application.
The ``simulation_app`` instance is passed into the constructor.
"""

from __future__ import annotations

import numpy as np


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a USD prim name."""
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "_unnamed"


# Key-name mapping from carb.input key names to pyglet key constants.
# Built lazily on first use so that pyglet is only imported when needed.
_CARB_TO_PYGLET: dict[str, int] | None = None


def _get_carb_to_pyglet_map() -> dict[str, int]:
    global _CARB_TO_PYGLET
    if _CARB_TO_PYGLET is None:
        from pyglet.window import key

        _CARB_TO_PYGLET = {
            "A": key.A, "B": key.B, "C": key.C, "D": key.D,
            "E": key.E, "F": key.F, "G": key.G, "H": key.H,
            "I": key.I, "J": key.J, "K": key.K, "L": key.L,
            "M": key.M, "N": key.N, "O": key.O, "P": key.P,
            "Q": key.Q, "R": key.R, "S": key.S, "T": key.T,
            "U": key.U, "V": key.V, "W": key.W, "X": key.X,
            "Y": key.Y, "Z": key.Z,
            "0": key._0, "1": key._1, "2": key._2, "3": key._3,
            "4": key._4, "5": key._5, "6": key._6, "7": key._7,
            "8": key._8, "9": key._9,
            "SPACE": key.SPACE, "ENTER": key.RETURN, "ESCAPE": key.ESCAPE,
            "TAB": key.TAB, "BACKSPACE": key.BACKSPACE,
            "UP": key.UP, "DOWN": key.DOWN, "LEFT": key.LEFT, "RIGHT": key.RIGHT,
            "LEFT_SHIFT": key.LSHIFT, "RIGHT_SHIFT": key.RSHIFT,
            "LEFT_CONTROL": key.LCTRL, "RIGHT_CONTROL": key.RCTRL,
            "LEFT_ALT": key.LALT, "RIGHT_ALT": key.RALT,
        }
    return _CARB_TO_PYGLET


class IsaacSimRenderer:
    """Real-time RTX path-traced renderer using Isaac Sim 6."""

    def __init__(self, simulation_app, model=None, path="Warp Surgical Simulation",
                 scaling=1.0, near_plane=0.05, far_plane=25.0):
        self._simulation_app = simulation_app
        self._scaling = scaling
        self._near_plane = near_plane
        self._far_plane = far_plane

        # ------ lazy Isaac Sim imports (Kit is already running) ------
        import carb
        import omni.appwindow
        from pxr import Usd, UsdGeom, UsdLux, UsdShade, Sdf, Vt, Gf
        from isaacsim.core.rendering_manager import RenderingManager, ViewportManager

        self._carb = carb
        self._omni_appwindow = omni.appwindow
        self._Usd = Usd
        self._UsdGeom = UsdGeom
        self._UsdLux = UsdLux
        self._UsdShade = UsdShade
        self._Sdf = Sdf
        self._Vt = Vt
        self._Gf = Gf
        self._RenderingManager = RenderingManager
        self._ViewportManager = ViewportManager

        # Prim tracking
        self._mesh_prims: dict[str, str] = {}       # name -> prim_path
        self._sphere_prims: dict[str, str] = {}      # name -> prim_path
        self._xform_ops: dict[str, object] = {}      # prim_path -> cached xformOp
        self._texture_store: dict[int, str] = {}      # id -> filepath
        self._next_texture_id = 0

        # Keyboard callbacks
        self._on_key_press_callback = None
        self._on_key_release_callback = None
        self._sub_keyboard = None

        # Camera state
        self._cam_pos = [0.2, 1.2, -1.0]
        self._cam_target = [0.0, 0.5, 0.0]
        self._cam_up = [0.0, 1.0, 0.0]

        # ------ build the USD stage ------
        import isaacsim.core.experimental.utils.stage as stage_utils
        stage_utils.create_new_stage()
        self._stage = stage_utils.get_current_stage(backend="usd")

        # Root /World xform
        self._world_xform = UsdGeom.Xform.Define(self._stage, "/World")

        # Camera
        cam = UsdGeom.Camera.Define(self._stage, "/World/Camera")
        cam.GetFocalLengthAttr().Set(18.14)
        cam.GetHorizontalApertureAttr().Set(20.955)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(near_plane, far_plane))

        # Distant light
        distant_light = UsdLux.DistantLight.Define(self._stage, "/World/KeyLight")
        distant_light.GetIntensityAttr().Set(3000.0)
        xform_key = UsdGeom.Xformable(distant_light.GetPrim())
        xform_key.AddRotateXYZOp().Set(Gf.Vec3f(315.0, 45.0, 0.0))

        # Dome light
        dome_light = UsdLux.DomeLight.Define(self._stage, "/World/DomeLight")
        dome_light.GetIntensityAttr().Set(1000.0)

        # Set camera in viewport
        camera_prim = cam.GetPrim()
        ViewportManager.set_camera(camera_prim)
        self._update_camera_view()

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    def _update_camera_view(self):
        """Push the current camera pose into the Isaac Sim viewport."""
        camera = self._ViewportManager.get_camera()
        self._ViewportManager.set_camera_view(
            camera,
            eye=self._cam_pos,
            target=self._cam_target,
        )

    @property
    def _camera_pos(self):
        return self._cam_pos

    @_camera_pos.setter
    def _camera_pos(self, value):
        self._cam_pos = list(value)
        self._update_camera_view()

    def update_view_matrix(self, cam_pos=None, cam_front=None, cam_up=None, stiffness=1.0):
        if cam_pos is not None:
            self._cam_pos = list(cam_pos)
        if cam_front is not None:
            self._cam_target = [
                self._cam_pos[0] + cam_front[0],
                self._cam_pos[1] + cam_front[1],
                self._cam_pos[2] + cam_front[2],
            ]
        if cam_up is not None:
            self._cam_up = list(cam_up)
        self._update_camera_view()

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    def _get_or_create_xform_op(self, prim_path: str):
        """Return a cached UsdGeom.XformOp (translate) for *prim_path*."""
        if prim_path in self._xform_ops:
            return self._xform_ops[prim_path]

        prim = self._stage.GetPrimAtPath(prim_path)
        xformable = self._UsdGeom.Xformable(prim)
        # Clear any existing ops and create a single matrix op
        xformable.ClearXformOpOrder()
        op = xformable.AddTransformOp()
        self._xform_ops[prim_path] = op
        return op

    def _write_prim_transform(self, prim_path: str, pos, rot, scale):
        """Write an SRT transform to a prim via a cached xform op."""
        Gf = self._Gf
        x, y, z, w = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
        tx, ty, tz = float(pos[0]), float(pos[1]), float(pos[2])

        # Rotation matrix from quaternion
        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y - w * z)
        r02 = 2.0 * (x * z + w * y)
        r10 = 2.0 * (x * y + w * z)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z - w * x)
        r20 = 2.0 * (x * z - w * y)
        r21 = 2.0 * (y * z + w * x)
        r22 = 1.0 - 2.0 * (x * x + y * y)

        mat = Gf.Matrix4d(
            sx * r00, sx * r01, sx * r02, 0.0,
            sy * r10, sy * r11, sy * r12, 0.0,
            sz * r20, sz * r21, sz * r22, 0.0,
            tx, ty, tz, 1.0,
        )
        op = self._get_or_create_xform_op(prim_path)
        op.Set(mat)

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def begin_frame(self, t=None):
        """No-op — rendering happens in ``end_frame``."""
        pass

    def end_frame(self):
        """Render the frame and pump the application event loop."""
        self._RenderingManager.render()
        self._simulation_app.update()

    # ------------------------------------------------------------------
    # Mesh rendering
    # ------------------------------------------------------------------

    def render_mesh_warp(
        self,
        name: str,
        points,
        indices,
        texture_coords=None,
        vertex_colors=None,
        diffuse_maps=None,
        normal_maps=None,
        specular_maps=None,
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
        basic_color=(1.0, 1.0, 1.0),
        update_topology=False,
        smooth_shading=True,
        visible=True,
        **kwargs,
    ):
        """Render / update a mesh from Warp arrays."""
        UsdGeom = self._UsdGeom
        Vt = self._Vt

        sanitized = _sanitize_name(name)
        prim_path = f"/World/{sanitized}"

        # Handle invisible meshes
        if not visible:
            if name in self._mesh_prims:
                prim = self._stage.GetPrimAtPath(prim_path)
                imageable = UsdGeom.Imageable(prim)
                imageable.MakeInvisible()
            return 0

        # Create the Mesh prim on first encounter
        if name not in self._mesh_prims:
            mesh = UsdGeom.Mesh.Define(self._stage, prim_path)
            mesh.GetSubdivisionSchemeAttr().Set("none")

            # Create a simple PreviewSurface material
            r, g, b = float(basic_color[0]), float(basic_color[1]), float(basic_color[2])
            mat_path = f"/World/Materials/{sanitized}_mat"
            material = self._UsdShade.Material.Define(self._stage, mat_path)
            shader = self._UsdShade.Shader.Define(self._stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", self._Sdf.ValueTypeNames.Color3f).Set(
                self._Gf.Vec3f(r, g, b)
            )
            shader.CreateInput("roughness", self._Sdf.ValueTypeNames.Float).Set(0.5)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self._UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)

            self._mesh_prims[name] = prim_path

        # Make visible (in case it was previously hidden)
        prim = self._stage.GetPrimAtPath(prim_path)
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeVisible()

        mesh = UsdGeom.Mesh(prim)

        # ------ Convert Warp arrays to lists for USD ------
        points_np = points.numpy().astype(np.float32)
        if points_np.ndim == 1:
            points_np = points_np.reshape(-1, 3)

        indices_np = indices.numpy().astype(np.int32)
        if indices_np.ndim == 2:
            num_triangles = indices_np.shape[0]
            indices_flat = indices_np.reshape(-1)
        else:
            indices_flat = indices_np.reshape(-1)
            num_triangles = len(indices_flat) // 3

        # Update points
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points_np))

        # Update topology
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices_flat))
        face_counts = np.full(num_triangles, 3, dtype=np.int32)
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(face_counts))

        # Update transform
        self._write_prim_transform(prim_path, pos, rot, scale)

        return hash(name) & 0x7FFFFFFF

    def render_mesh_warp_range(
        self,
        name: str,
        points,
        indices,
        texture_coords=None,
        colors=None,
        diffuse_maps=None,
        normal_maps=None,
        specular_maps=None,
        index_start: int = 0,
        index_count: int = -1,
        pos=(0.0, 0.0, 0.0),
        rot=(0.0, 0.0, 0.0, 1.0),
        scale=(1.0, 1.0, 1.0),
        basic_color=(1.0, 1.0, 1.0),
        update_topology=False,
        smooth_shading=True,
        visible=True,
        **kwargs,
    ):
        """Render a mesh using a sub-range of the index array."""
        if index_count == -1:
            index_count = indices.shape[0] - index_start

        indices_range = indices[index_start : index_start + index_count]

        return self.render_mesh_warp(
            name=name,
            points=points,
            indices=indices_range,
            texture_coords=texture_coords,
            vertex_colors=colors,
            diffuse_maps=diffuse_maps,
            normal_maps=normal_maps,
            specular_maps=specular_maps,
            pos=pos,
            rot=rot,
            scale=scale,
            basic_color=basic_color,
            update_topology=update_topology,
            smooth_shading=smooth_shading,
            visible=visible,
        )

    # ------------------------------------------------------------------
    # Sphere rendering
    # ------------------------------------------------------------------

    def render_sphere(self, name, pos, rot, radius, color=None, visible=True, **kwargs):
        """Render / update a sphere primitive."""
        UsdGeom = self._UsdGeom
        sanitized = _sanitize_name(name)
        prim_path = f"/World/{sanitized}"

        if color is None:
            color = (0.5, 0.5, 0.5)

        if name not in self._sphere_prims:
            sphere = UsdGeom.Sphere.Define(self._stage, prim_path)
            sphere.GetRadiusAttr().Set(float(radius))

            # Material
            r, g, b = float(color[0]), float(color[1]), float(color[2])
            mat_path = f"/World/Materials/{sanitized}_mat"
            material = self._UsdShade.Material.Define(self._stage, mat_path)
            shader = self._UsdShade.Shader.Define(self._stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", self._Sdf.ValueTypeNames.Color3f).Set(
                self._Gf.Vec3f(r, g, b)
            )
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            self._UsdShade.MaterialBindingAPI.Apply(sphere.GetPrim()).Bind(material)

            self._sphere_prims[name] = prim_path

        # Update transform (translate only for spheres)
        self._write_prim_transform(prim_path, pos, rot, (1.0, 1.0, 1.0))

        return hash(name) & 0x7FFFFFFF

    # ------------------------------------------------------------------
    # Texture / input / lifecycle
    # ------------------------------------------------------------------

    def load_texture(self, filepath: str, **kwargs) -> int:
        """Store a texture filepath for future use. Returns an incremental ID."""
        tex_id = self._next_texture_id
        self._texture_store[tex_id] = filepath
        self._next_texture_id += 1
        return tex_id

    def set_input_callbacks(self, on_key_press=None, on_key_release=None):
        """Register keyboard callbacks via carb.input, translating events to
        pyglet key constants for compatibility with existing WarpSim handlers.
        """
        self._on_key_press_callback = on_key_press
        self._on_key_release_callback = on_key_release

        carb = self._carb
        appwindow = self._omni_appwindow.get_default_app_window()
        input_iface = carb.input.acquire_input_interface()
        keyboard = appwindow.get_keyboard()

        def _keyboard_event(event, *args, **kwargs):
            key_map = _get_carb_to_pyglet_map()
            key_name = event.input.name if hasattr(event.input, "name") else str(event.input)
            pyglet_key = key_map.get(key_name)
            if pyglet_key is None:
                return True

            modifiers = 0  # simplified — no modifier translation needed for current keys
            if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                if self._on_key_press_callback:
                    self._on_key_press_callback(pyglet_key, modifiers)
            elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                if self._on_key_release_callback:
                    self._on_key_release_callback(pyglet_key, modifiers)
            return True

        self._sub_keyboard = input_iface.subscribe_to_keyboard_events(
            keyboard, _keyboard_event
        )

    def is_running(self) -> bool:
        """Return True while the Isaac Sim application is still running."""
        return self._simulation_app.is_running()

    def save(self):
        """No-op for interactive Isaac Sim sessions."""
        pass
