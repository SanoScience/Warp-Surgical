from __future__ import annotations

import ctypes
import os
import numpy as np

import pyglet
from pyglet import gl

# Allow ovrtx to coexist with the pxr (usd-core) package that the simulation
# already uses for instrument loading.  ovrtx bundles its own USD libraries and
# normally refuses to start when pxr is present; the env-var below tells it to
# skip that safety check.
os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")

import ovrtx  # noqa: E402
from ovrtx import Renderer, RendererConfig  # noqa: E402
from ovrtx._src.dlpack import DLTensor  # noqa: E402
from ovrtx.math import Matrix4d  # noqa: E402

# Point ovrtx at the native library if it isn't already on the search path.
# The hint must be set on the *bindings* module where the loader reads it.
import ovrtx._src.bindings as _ovrtx_bindings  # noqa: E402

if _ovrtx_bindings.OVRTX_LIBRARY_PATH_HINT is None:
    _candidate = os.path.join(
        os.path.dirname(os.path.dirname(ovrtx.__file__)),
        "examples", "c", "_deps", "ovrtx-src", "bin",
    )
    if os.path.isfile(os.path.join(_candidate, "libovrtx-dynamic.so")):
        _ovrtx_bindings.OVRTX_LIBRARY_PATH_HINT = _candidate


def _sanitize_name(name: str) -> str:
    """Sanitize a name for use as a USD prim name."""
    sanitized = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if sanitized and sanitized[0].isdigit():
        sanitized = "_" + sanitized
    return sanitized or "_unnamed"


# Per-organ OmniSurface SSS profiles.  Meshes listed here get OmniSurface
# materials with subsurface scattering; all others fall back to OmniPBR.
TISSUE_SSS_PROFILES: dict[str, dict] = {
    "liver_mesh": {
        "subsurface_weight": 0.7,
        "transmission_color": (0.55, 0.05, 0.02),
        "scattering_color": (0.8, 0.15, 0.08),
        "subsurface_scale": 1.0,
        "subsurface_anisotropy": 0.3,
        "coat_weight": 0.15,
        "coat_roughness": 0.3,
        "coat_color": (1.0, 1.0, 1.0),
        "specular_reflection_roughness": 0.4,
        "specular_reflection_ior": 1.4,
    },
    "fat_mesh": {
        "subsurface_weight": 0.85,
        "transmission_color": (0.9, 0.7, 0.25),
        "scattering_color": (1.0, 0.85, 0.35),
        "subsurface_scale": 3.0,
        "subsurface_anisotropy": 0.5,
        "coat_weight": 0.3,
        "coat_roughness": 0.2,
        "coat_color": (1.0, 1.0, 1.0),
        "specular_reflection_roughness": 0.3,
        "specular_reflection_ior": 1.38,
    },
    "gallbladder_mesh": {
        "subsurface_weight": 0.6,
        "transmission_color": (0.35, 0.55, 0.25),
        "scattering_color": (0.6, 0.7, 0.4),
        "subsurface_scale": 1.5,
        "subsurface_anisotropy": 0.2,
        "coat_weight": 0.25,
        "coat_roughness": 0.25,
        "coat_color": (1.0, 1.0, 1.0),
        "specular_reflection_roughness": 0.35,
        "specular_reflection_ior": 1.4,
    },
    "background_mesh": {
        "subsurface_weight": 0.5,
        "transmission_color": (0.7, 0.35, 0.3),
        "scattering_color": (0.85, 0.5, 0.4),
        "subsurface_scale": 1.2,
        "subsurface_anisotropy": 0.2,
        "coat_weight": 0.1,
        "coat_roughness": 0.4,
        "coat_color": (1.0, 1.0, 1.0),
        "specular_reflection_roughness": 0.45,
        "specular_reflection_ior": 1.38,
    },
}

# Per-organ OmniPBR profiles extracted from abdomen.usda Looks scope.
# Meshes listed here get extra OmniPBR-specific parameters; all others use
# plain OmniPBR defaults (diffuse + normal + spec textures, project_uvw=0).
TISSUE_OMNIPBR_PROFILES: dict[str, dict] = {
    "liver_mesh": {
        "albedo_add": -0.01,
        "ao_to_diffuse": 0,
        "bump_factor": 1,
    },
    "gallbladder_mesh": {
        "ao_to_diffuse": 0,
        "diffuse_color_constant": (0.15, 0.2, 0.2),
        "flip_tangent_u": 0,
    },
}


def _make_float3_dltensor(arr: np.ndarray) -> DLTensor:
    """Create a DLTensor with dtype float32×3 (lanes=3) from an (N, 3) float32 array.

    USD ``point3f[]`` attributes store 12-byte float3 elements, so ovrtx expects
    a 1-D tensor of N elements with ``lanes=3`` rather than a 2-D (N, 3) tensor.
    """
    from ovrtx._src.dlpack import DLDataType, DLDataTypeCode, DLDevice, DLDeviceType

    arr = np.ascontiguousarray(arr, dtype=np.float32)
    assert arr.ndim == 2 and arr.shape[1] == 3

    n = arr.shape[0]
    dl = DLTensor()
    dl.data = arr.ctypes.data
    dl.device = DLDevice(device_type=DLDeviceType(DLDeviceType.kDLCPU), device_id=0)
    dl.ndim = 1
    dl.dtype = DLDataType(code=DLDataTypeCode(DLDataTypeCode.kDLFloat), bits=32, lanes=3)
    dl.byte_offset = 0
    dl.strides = None

    shape_arr = (ctypes.c_int64 * 1)(n)
    dl.shape = ctypes.cast(shape_arr, ctypes.POINTER(ctypes.c_int64))

    # prevent GC of backing data and shape array
    DLTensor._array_storage[dl.data] = {"source_obj": arr, "shape": shape_arr}
    return dl


class OvrtxRenderer:
    """Real-time RTX ray-tracing renderer using ovrtx with a pyglet display window."""

    def __init__(self, model, path, scaling=1.0, near_plane=0.05, far_plane=25, use_sss=True):
        self._width = 1280
        self._height = 720
        self._scaling = scaling
        self._near_plane = near_plane
        self._far_plane = far_plane
        self._use_sss = use_sss

        # Camera state — look along -Z towards the simulation (around Z≈-4)
        self._cam_pos = [0.2, 1.2, -1.0]
        self._cam_front = [0.0, 0.0, -1.0]  # look along -Z
        self._cam_up = [0.0, 1.0, 0.0]
        self._cam_yaw = -90.0    # degrees, -90 = looking along -Z
        self._cam_pitch = 0.0    # degrees
        self._cam_speed = 2.0    # units per second
        self._mouse_sensitivity = 0.15  # degrees per pixel

        # Prim tracking
        self._mesh_prims: dict[str, str] = {}  # name -> prim_path
        self._sphere_prims: dict[str, str] = {}  # name -> prim_path
        self._texture_store: dict[int, str] = {}  # id -> abs filepath
        self._next_texture_id = 0
        self._textured_meshes: set[str] = set()  # names with material (OmniPBR or OmniSurface)

        # Light visibility state (stores original intensity for restore)
        self._light_visibility: dict[str, bool] = {
            "EndoscopeLight": True,
            "KeyLight": False,
            "DomeLight": True,
            "SurgicalLight": True,
        }
        self._light_config: dict[str, dict] = {
            "EndoscopeLight": {"prim": "/World/EndoscopeLight", "attr": "inputs:intensity", "intensity": 150000.0},
            "KeyLight": {"prim": "/World/KeyLight", "attr": "inputs:intensity", "intensity": 500.0},
            "DomeLight": {"prim": "/World/DomeLight", "attr": "inputs:intensity", "intensity": 1000.0},
            "SurgicalLight": {"prim": "/World/SurgicalLight", "attr": "inputs:intensity", "intensity": 5000.0},
        }

        # KeyLight rotation angles (IJKL keys) — pitch (X) and yaw (Y)
        self._keylight_pitch = -45.0  # degrees around X
        self._keylight_yaw = -30.0    # degrees around Y
        self._keylight_angle_speed = 60.0  # degrees per second

        # Keyboard callbacks and state
        self._on_key_press_callback = None
        self._on_key_release_callback = None
        self._keys_held: set[int] = set()

        # Frame timing
        self._frame_time = 0.0
        self._dt = 1.0 / 60.0

        # Create pyglet window
        self._window = pyglet.window.Window(
            width=self._width,
            height=self._height,
            caption=path,
            resizable=False,
        )

        @self._window.event
        def on_draw():
            pass  # Drawing is handled in end_frame

        @self._window.event
        def on_key_press(symbol, modifiers):
            self._keys_held.add(symbol)
            if self._on_key_press_callback:
                self._on_key_press_callback(symbol, modifiers)

        @self._window.event
        def on_key_release(symbol, modifiers):
            self._keys_held.discard(symbol)
            if self._on_key_release_callback:
                self._on_key_release_callback(symbol, modifiers)

        @self._window.event
        def on_mouse_motion(x, y, dx, dy):
            self._cam_yaw += dx * self._mouse_sensitivity
            self._cam_pitch += dy * self._mouse_sensitivity
            self._cam_pitch = max(-89.0, min(89.0, self._cam_pitch))
            self._update_cam_front_from_angles()
            self._update_camera_transform()

        @self._window.event
        def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
            self._cam_yaw += dx * self._mouse_sensitivity
            self._cam_pitch += dy * self._mouse_sensitivity
            self._cam_pitch = max(-89.0, min(89.0, self._cam_pitch))
            self._update_cam_front_from_angles()
            self._update_camera_transform()

        # Create ovrtx renderer
        config = RendererConfig(sync_mode=True)
        self._ovrtx = Renderer(config=config)

        # Add base USD scene (camera, lights, render product)
        self._setup_base_scene()

        # Set initial camera and KeyLight transforms
        self._update_camera_transform()
        self._update_keylight_transform()

    # ------------------------------------------------------------------
    # USD scene setup
    # ------------------------------------------------------------------

    def _setup_base_scene(self):
        """Inject the base USD scene with camera, lights and render product."""
        usda = f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Y"
    metersPerUnit = 0.01
)

def Xform "World" {{
    def Camera "Camera" {{
        float focalLength = 18.14
        float horizontalAperture = 20.955
        float verticalAperture = 15.29
        float2 clippingRange = ({self._near_plane}, {self._far_plane})
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def SphereLight "EndoscopeLight" {{
        float inputs:intensity = 150000
        float radius = 0.05
        color3f inputs:color = (1, 1, 1)
        bool inputs:normalize = 1
        bool inputs:enableColorTemperature = 1
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),({self._cam_pos[0]},{self._cam_pos[1]},{self._cam_pos[2]},1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def DomeLight "DomeLight" (prepend apiSchemas = ["ShapingAPI"]) {{
        float inputs:intensity = 1000
        float inputs:exposure = 1
        token inputs:texture:format = "latlong"
    }}

    def DistantLight "KeyLight" {{
        float inputs:intensity = 0
        float inputs:angle = 2.0
        color3f inputs:color = (1, 0.95, 0.9)
        float3 xformOp:rotateXYZ = ({self._keylight_pitch}, {self._keylight_yaw}, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}

    def SphereLight "SurgicalLight" (prepend apiSchemas = ["ShapingAPI"]) {{
        float inputs:intensity = 5000
        float inputs:radius = 0.1
        color3f inputs:color = (0.99, 1, 1)
        bool inputs:normalize = 0
        bool inputs:enableColorTemperature = 1
        float inputs:shaping:cone:angle = 180
        float3 xformOp:rotateXYZ = (135, 0, 0)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]
    }}
}}

def "Render" (
    hide_in_stage_window = true
    no_delete = true
) {{
    def "OmniverseKit" {{
        def "HydraTextures" (
            hide_in_stage_window = true
            no_delete = true
        ) {{
            def RenderProduct "ViewportTexture0" (
                prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1", "OmniRtxSettingsRtAdvancedAPI_1", "OmniRtxSettingsPtAdvancedAPI_1"]
                hide_in_stage_window = true
                no_delete = true
            ) {{
                rel camera = </World/Camera>
                token omni:rtx:background:source:type = "domeLight"
                token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
                rel orderedVars = [</Render/Vars/LdrColor>]
                int omni:rtx:rt:sss:samples = 4
                bool omni:rtx:scene:hydra:mdlMaterialWarmup = 1
                uniform int2 resolution = ({self._width}, {self._height})
            }}
        }}
    }}

    def RenderSettings "OmniverseGlobalRenderSettings" (
        prepend apiSchemas = ["OmniRtxSettingsGlobalRtAdvancedAPI_1", "OmniRtxSettingsGlobalPtAdvancedAPI_1"]
        no_delete = true
    ) {{
        rel products = </Render/OmniverseKit/HydraTextures/ViewportTexture0>
    }}

    def "Vars" {{
        def RenderVar "LdrColor" (
            hide_in_stage_window = true
            no_delete = true
        ) {{
            uniform string sourceName = "LdrColor"
        }}
    }}
}}
"""
        self._render_product_path = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"
        self._ovrtx.add_usd_layer(usda)

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_look_at_matrix(eye, target, up):
        """Compute a look-at 4x4 matrix in USD convention (translation in last row)."""
        eye = np.asarray(eye, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

        forward = target - eye
        fwd_len = np.linalg.norm(forward)
        if fwd_len < 1e-10:
            return np.eye(4, dtype=np.float64)
        forward /= fwd_len

        right = np.cross(forward, up)
        right_len = np.linalg.norm(right)
        if right_len < 1e-10:
            up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            right = np.cross(forward, up)
            right_len = np.linalg.norm(right)
        right /= right_len

        new_up = np.cross(right, forward)

        # USD camera looks down -Z; rows = camera basis vectors in world space
        return np.array(
            [
                [right[0], right[1], right[2], 0.0],
                [new_up[0], new_up[1], new_up[2], 0.0],
                [-forward[0], -forward[1], -forward[2], 0.0],
                [eye[0], eye[1], eye[2], 1.0],
            ],
            dtype=np.float64,
        )

    def _update_camera_transform(self):
        """Write the current camera transform to ovrtx."""
        target = [
            self._cam_pos[0] + self._cam_front[0],
            self._cam_pos[1] + self._cam_front[1],
            self._cam_pos[2] + self._cam_front[2],
        ]
        np_m = self._compute_look_at_matrix(self._cam_pos, target, self._cam_up)
        m = Matrix4d()
        for i in range(4):
            m[i] = [np_m[i, 0], np_m[i, 1], np_m[i, 2], np_m[i, 3]]
        self._ovrtx.write_attribute(
            prim_paths=["/World/Camera"],
            attribute_name="omni:fabric:localMatrix",
            tensor=m.to_dltensor(),
            semantic="transform_4x4",
        )
        # Move endoscope light to camera position
        light_m = Matrix4d()
        light_m[0] = [1.0, 0.0, 0.0, 0.0]
        light_m[1] = [0.0, 1.0, 0.0, 0.0]
        light_m[2] = [0.0, 0.0, 1.0, 0.0]
        light_m[3] = [float(self._cam_pos[0]), float(self._cam_pos[1]), float(self._cam_pos[2]), 1.0]
        self._ovrtx.write_attribute(
            prim_paths=["/World/EndoscopeLight"],
            attribute_name="omni:fabric:localMatrix",
            tensor=light_m.to_dltensor(),
            semantic="transform_4x4",
        )

    @property
    def _camera_pos(self):
        return self._cam_pos

    @_camera_pos.setter
    def _camera_pos(self, value):
        self._cam_pos = list(value)
        self._update_camera_transform()

    def _update_keylight_transform(self):
        """Write KeyLight rotation from pitch/yaw angles (rotateXYZ order)."""
        import math
        px = math.radians(self._keylight_pitch)
        py = math.radians(self._keylight_yaw)
        cx, sx = math.cos(px), math.sin(px)
        cy, sy = math.cos(py), math.sin(py)
        # Ry * Rx  (USD rotateXYZ applies X then Y then Z, Z is 0)
        m = Matrix4d()
        m[0] = [cy,     sx * sy,  -cx * sy, 0.0]
        m[1] = [0.0,    cx,        sx,      0.0]
        m[2] = [sy,    -sx * cy,   cx * cy, 0.0]
        m[3] = [0.0,    0.0,       0.0,     1.0]
        self._ovrtx.write_attribute(
            prim_paths=["/World/KeyLight"],
            attribute_name="omni:fabric:localMatrix",
            tensor=m.to_dltensor(),
            semantic="transform_4x4",
        )

    def update_view_matrix(self, cam_pos=None, cam_front=None, cam_up=None, stiffness=1.0):
        """Update camera view matrix from position/direction/up."""
        if cam_pos is not None:
            self._cam_pos = list(cam_pos)
        if cam_front is not None:
            self._cam_front = list(cam_front)
        if cam_up is not None:
            self._cam_up = list(cam_up)
        self._update_camera_transform()

    # ------------------------------------------------------------------
    # Light helpers
    # ------------------------------------------------------------------

    def adjust_light_intensity(self, light_name: str, delta: float):
        """Adjust a light's stored intensity by delta and write it (clamped to >= 0)."""
        if light_name not in self._light_config:
            return
        cfg = self._light_config[light_name]
        cfg["intensity"] = max(0.0, cfg["intensity"] + delta)
        if self._light_visibility[light_name]:
            tensor = DLTensor.from_dlpack(np.array([cfg["intensity"]], dtype=np.float32))
            self._ovrtx.write_attribute(
                prim_paths=[cfg["prim"]],
                attribute_name=cfg["attr"],
                tensor=tensor,
            )

    def toggle_light(self, light_name: str):
        """Toggle a light by writing its intensity to 0 or restoring the original value."""
        if light_name not in self._light_visibility:
            return
        self._light_visibility[light_name] = not self._light_visibility[light_name]
        cfg = self._light_config[light_name]
        val = cfg["intensity"] if self._light_visibility[light_name] else 0.0
        tensor = DLTensor.from_dlpack(np.array([val], dtype=np.float32))
        self._ovrtx.write_attribute(
            prim_paths=[cfg["prim"]],
            attribute_name=cfg["attr"],
            tensor=tensor,
        )

    # ------------------------------------------------------------------
    # Transform helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_transform_matrix(pos, rot, scale):
        """Build a Matrix4d SRT matrix (USD convention: translation in last row).

        Args:
            pos: (tx, ty, tz)
            rot: quaternion (x, y, z, w)
            scale: (sx, sy, sz)
        """
        x, y, z, w = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
        tx, ty, tz = float(pos[0]), float(pos[1]), float(pos[2])

        # Rotation matrix from quaternion (column-vector convention, same as USD)
        r00 = 1.0 - 2.0 * (y * y + z * z)
        r01 = 2.0 * (x * y - w * z)
        r02 = 2.0 * (x * z + w * y)
        r10 = 2.0 * (x * y + w * z)
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r12 = 2.0 * (y * z - w * x)
        r20 = 2.0 * (x * z - w * y)
        r21 = 2.0 * (y * z + w * x)
        r22 = 1.0 - 2.0 * (x * x + y * y)

        m = Matrix4d()
        m[0] = [sx * r00, sx * r01, sx * r02, 0.0]
        m[1] = [sy * r10, sy * r11, sy * r12, 0.0]
        m[2] = [sz * r20, sz * r21, sz * r22, 0.0]
        m[3] = [tx, ty, tz, 1.0]
        return m

    def _write_prim_transform(self, prim_path, pos, rot, scale):
        """Write an SRT transform to a prim."""
        m = self._build_transform_matrix(pos, rot, scale)
        self._ovrtx.write_attribute(
            prim_paths=[prim_path],
            attribute_name="omni:fabric:localMatrix",
            tensor=m.to_dltensor(),
            semantic="transform_4x4",
        )

    # ------------------------------------------------------------------
    # Frame lifecycle
    # ------------------------------------------------------------------

    def _update_cam_front_from_angles(self):
        """Recompute _cam_front from yaw/pitch angles."""
        import math
        yaw_r = math.radians(self._cam_yaw)
        pitch_r = math.radians(self._cam_pitch)
        self._cam_front = [
            math.cos(pitch_r) * math.cos(yaw_r),
            math.sin(pitch_r),
            math.cos(pitch_r) * math.sin(yaw_r),
        ]

    def begin_frame(self, t=None):
        """Begin a new frame: pump events and apply WASD camera movement."""
        if t is not None:
            self._frame_time = t
        self._window.dispatch_events()

        from pyglet.window import key

        moved = False
        speed = self._cam_speed * self._dt

        # Forward / backward
        if key.W in self._keys_held or key.UP in self._keys_held:
            for i in range(3):
                self._cam_pos[i] += self._cam_front[i] * speed
            moved = True
        if key.S in self._keys_held or key.DOWN in self._keys_held:
            for i in range(3):
                self._cam_pos[i] -= self._cam_front[i] * speed
            moved = True

        # Strafe left / right
        right = np.cross(self._cam_front, self._cam_up)
        rlen = np.linalg.norm(right)
        if rlen > 1e-8:
            right = right / rlen
        if key.D in self._keys_held or key.RIGHT in self._keys_held:
            for i in range(3):
                self._cam_pos[i] += right[i] * speed
            moved = True
        if key.A in self._keys_held or key.LEFT in self._keys_held:
            for i in range(3):
                self._cam_pos[i] -= right[i] * speed
            moved = True

        # Up / down with Q / E
        if key.E in self._keys_held:
            self._cam_pos[1] += speed
            moved = True
        if key.Q in self._keys_held:
            self._cam_pos[1] -= speed
            moved = True

        if moved:
            self._update_camera_transform()

        # Light toggles: 1 = EndoscopeLight, 2 = KeyLight, 3 = DomeLight
        if key._1 in self._keys_held:
            self._keys_held.discard(key._1)
            self.toggle_light("EndoscopeLight")
        if key._2 in self._keys_held:
            self._keys_held.discard(key._2)
            self.toggle_light("KeyLight")
        if key._3 in self._keys_held:
            self._keys_held.discard(key._3)
            self.toggle_light("DomeLight")
        if key._4 in self._keys_held:
            self._keys_held.discard(key._4)
            self.toggle_light("SurgicalLight")

        # Light intensity controls:
        # 0/9 = EndoscopeLight, 5/6 = KeyLight, 7/8 = DomeLight
        if key._0 in self._keys_held:
            self._keys_held.discard(key._0)
            self.adjust_light_intensity("EndoscopeLight", -15000.0)
        if key._9 in self._keys_held:
            self._keys_held.discard(key._9)
            self.adjust_light_intensity("EndoscopeLight", 15000.0)
        if key._5 in self._keys_held:
            self._keys_held.discard(key._5)
            self.adjust_light_intensity("KeyLight", -50.0)
        if key._6 in self._keys_held:
            self._keys_held.discard(key._6)
            self.adjust_light_intensity("KeyLight", 50.0)
        if key._7 in self._keys_held:
            self._keys_held.discard(key._7)
            self.adjust_light_intensity("DomeLight", -100.0)
        if key._8 in self._keys_held:
            self._keys_held.discard(key._8)
            self.adjust_light_intensity("DomeLight", 100.0)

        # KeyLight angle: I/K = pitch, J/L = yaw
        kl_step = self._keylight_angle_speed * self._dt
        kl_moved = False
        if key.I in self._keys_held:
            self._keylight_pitch -= kl_step
            kl_moved = True
        if key.K in self._keys_held:
            self._keylight_pitch += kl_step
            kl_moved = True
        if key.J in self._keys_held:
            self._keylight_yaw -= kl_step
            kl_moved = True
        if key.L in self._keys_held:
            self._keylight_yaw += kl_step
            kl_moved = True
        if kl_moved:
            self._update_keylight_transform()


    def end_frame(self):
        """End frame: step ovrtx, read pixels, blit to pyglet window."""
        # Step ovrtx renderer
        products = self._ovrtx.step(
            render_products={self._render_product_path},
            delta_time=self._dt,
        )

        # Extract LdrColor pixel buffer
        pixels = None
        for _product_name, product in products.items():
            for frame in product.frames:
                with frame.render_vars["LdrColor"].map(device="cpu") as var:
                    pixels = var.tensor.numpy().copy()  # (H, W, 4) uint8

        if pixels is None:
            self._window.flip()
            return

        # Blit to pyglet window via ImageData (negative pitch = top-to-bottom rows)
        self._window.switch_to()
        self._window.clear()
        image = pyglet.image.ImageData(
            self._width,
            self._height,
            "RGBA",
            pixels.tobytes(),
            pitch=-self._width * 4,
        )
        image.blit(0, 0)
        self._window.flip()

    # ------------------------------------------------------------------
    # Mesh rendering
    # ------------------------------------------------------------------

    def _build_omnisurface_shader_inputs(self, diffuse_path, normal_path, spec_path, profile):
        """Build OmniSurface shader input lines with SSS, coat, and specular parameters."""
        lines = []
        # Texture inputs (OmniSurface naming)
        if diffuse_path:
            lines.append(f'            asset inputs:diffuse_reflection_color_image = @{diffuse_path}@')
        if normal_path:
            lines.append(f'            asset inputs:geometry_normal_image = @{normal_path}@')
        if spec_path:
            lines.append(f'            asset inputs:specular_reflection_roughness_image = @{spec_path}@')
        # SSS parameters
        lines.append('            bool inputs:enable_diffuse_transmission = 1')
        lines.append(f'            float inputs:subsurface_weight = {profile["subsurface_weight"]}')
        tc = profile["transmission_color"]
        lines.append(f'            color3f inputs:subsurface_transmission_color = ({tc[0]}, {tc[1]}, {tc[2]})')
        sc = profile["scattering_color"]
        lines.append(f'            color3f inputs:subsurface_scattering_color = ({sc[0]}, {sc[1]}, {sc[2]})')
        lines.append(f'            float inputs:subsurface_scale = {profile["subsurface_scale"]}')
        lines.append(f'            float inputs:subsurface_anisotropy = {profile["subsurface_anisotropy"]}')
        # Coat parameters (wet sheen)
        lines.append(f'            float inputs:coat_weight = {profile["coat_weight"]}')
        lines.append(f'            float inputs:coat_roughness = {profile["coat_roughness"]}')
        cc = profile["coat_color"]
        lines.append(f'            color3f inputs:coat_color = ({cc[0]}, {cc[1]}, {cc[2]})')
        # Specular parameters
        lines.append(f'            float inputs:specular_reflection_roughness = {profile["specular_reflection_roughness"]}')
        lines.append(f'            float inputs:specular_reflection_ior = {profile["specular_reflection_ior"]}')
        # Thin-walled mode for non-watertight surface meshes
        lines.append('            uniform bool inputs:thin_walled = 1')
        return "\n".join(lines) + "\n"

    def _build_omnipbr_shader_inputs(self, diffuse_path, normal_path, spec_path, profile):
        """Build OmniPBR shader input lines with per-organ parameters from abdomen scene."""
        lines = []
        if diffuse_path:
            lines.append(f'            asset inputs:diffuse_texture = @{diffuse_path}@')
        if normal_path:
            lines.append(f'            asset inputs:normalmap_texture = @{normal_path}@')
        if spec_path:
            lines.append(f'            asset inputs:reflectionroughness_texture = @{spec_path}@')
        lines.append('            bool inputs:project_uvw = 0')
        if profile:
            for key, val in profile.items():
                if isinstance(val, tuple):
                    lines.append(f'            color3f inputs:{key} = ({val[0]}, {val[1]}, {val[2]})')
                elif isinstance(val, float):
                    lines.append(f'            float inputs:{key} = {val}')
                elif isinstance(val, int):
                    if isinstance(val, bool):
                        lines.append(f'            bool inputs:{key} = {int(val)}')
                    else:
                        lines.append(f'            float inputs:{key} = {val}')
        return "\n".join(lines) + "\n"

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
        sanitized = _sanitize_name(name)
        prim_path = f"/World/{sanitized}"

        # Handle invisible meshes
        if not visible:
            if name in self._mesh_prims:
                # Write degenerate geometry to hide it
                empty_pts = np.zeros((1, 3), dtype=np.float32)
                empty_idx = np.array([0, 0, 0], dtype=np.int32)
                empty_counts = np.array([3], dtype=np.int32)
                self._ovrtx.write_array_attribute(
                    prim_paths=[prim_path],
                    attribute_name="points",
                    tensors=[_make_float3_dltensor(empty_pts)],
                )
                self._ovrtx.write_array_attribute(
                    prim_paths=[prim_path],
                    attribute_name="faceVertexIndices",
                    tensors=[DLTensor.from_dlpack(empty_idx)],
                )
                self._ovrtx.write_array_attribute(
                    prim_paths=[prim_path],
                    attribute_name="faceVertexCounts",
                    tensors=[DLTensor.from_dlpack(empty_counts)],
                )
            return 0

        # Create the Mesh prim on first encounter
        if name not in self._mesh_prims:
            has_material = diffuse_maps is not None and len(diffuse_maps) > 0
            if has_material:
                self._textured_meshes.add(name)
                # Resolve texture paths
                diffuse_path = self._texture_store.get(diffuse_maps[0], "")
                normal_path = ""
                if normal_maps and len(normal_maps) > 0:
                    normal_path = self._texture_store.get(normal_maps[0], "")
                spec_path = ""
                if specular_maps and len(specular_maps) > 0:
                    spec_path = self._texture_store.get(specular_maps[0], "")
                # Choose material type based on use_sss flag
                sss_profile = TISSUE_SSS_PROFILES.get(name)
                if self._use_sss and sss_profile is not None:
                    # OmniSurface with subsurface scattering
                    shader_inputs = self._build_omnisurface_shader_inputs(
                        diffuse_path, normal_path, spec_path, sss_profile,
                    )
                    mdl_asset = "OmniSurface.mdl"
                    mdl_subidentifier = "OmniSurface"
                else:
                    # OmniPBR with per-organ parameters from abdomen scene
                    pbr_profile = TISSUE_OMNIPBR_PROFILES.get(name)
                    shader_inputs = self._build_omnipbr_shader_inputs(
                        diffuse_path, normal_path, spec_path, pbr_profile,
                    )
                    mdl_asset = "OmniPBR.mdl"
                    mdl_subidentifier = "OmniPBR"
                # Bake UV coordinates into the USDA (fabric can't create primvars dynamically)
                uv_usda = '    texCoord2f[] primvars:st = [] (\n        interpolation = "vertex"\n    )\n'
                if texture_coords is not None:
                    uv_np = texture_coords.numpy().astype(np.float32)
                    if uv_np.ndim == 1:
                        uv_np = uv_np.reshape(-1, 2)
                    uv_strs = ", ".join(f"({u:.6f}, {v:.6f})" for u, v in uv_np)
                    uv_usda = f'    texCoord2f[] primvars:st = [{uv_strs}] (\n        interpolation = "vertex"\n    )\n'
                # Material as child — use RELATIVE paths (absolute paths break
                # with path_prefix because they don't exist in the raw layer).
                no_shadow = '    bool primvars:doNotCastShadows = 1\n' if name == "background_mesh" else ''
                usda = f"""#usda 1.0
(defaultPrim = "{sanitized}")
def Mesh "{sanitized}" (
    prepend apiSchemas = ["MaterialBindingAPI"]
) {{
    rel material:binding = <Material>
    uniform bool doubleSided = 1
    uniform token subdivisionScheme = "none"
    normal3f[] normals = [] (
        interpolation = "vertex"
    )
{no_shadow}{uv_usda}    matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
    uniform token[] xformOpOrder = ["xformOp:transform"]

    def Material "Material" {{
        token outputs:mdl:surface.connect = <Shader.outputs:out>
        token outputs:mdl:displacement.connect = <Shader.outputs:out>
        token outputs:mdl:volume.connect = <Shader.outputs:out>

        def Shader "Shader" {{
            uniform token info:implementationSource = "sourceAsset"
            uniform asset info:mdl:sourceAsset = @{mdl_asset}@
            uniform token info:mdl:sourceAsset:subIdentifier = "{mdl_subidentifier}"
{shader_inputs}            token outputs:out
        }}
    }}
}}
"""
            else:
                r = float(basic_color[0])
                g = float(basic_color[1])
                b = float(basic_color[2])
                usda = f"""#usda 1.0
(defaultPrim = "{sanitized}")
def Mesh "{sanitized}" {{
    color3f[] primvars:displayColor = [({r}, {g}, {b})]
    uniform bool doubleSided = 1
    uniform token subdivisionScheme = "none"
    normal3f[] normals = [] (
        interpolation = "vertex"
    )
    matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
    uniform token[] xformOpOrder = ["xformOp:transform"]
}}
"""
            self._ovrtx.add_usd_layer(usda, path_prefix=prim_path)
            self._mesh_prims[name] = prim_path

        # ------ Convert Warp arrays to numpy and bake SRT into positions ------
        points_np = points.numpy().astype(np.float32)
        if points_np.ndim == 1:
            points_np = points_np.reshape(-1, 3)

        # Bake scale + rotation + translation into vertex positions
        sx, sy, sz = float(scale[0]), float(scale[1]), float(scale[2])
        tx, ty, tz = float(pos[0]), float(pos[1]), float(pos[2])
        qx, qy, qz, qw = float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])
        has_scale = sx != 1.0 or sy != 1.0 or sz != 1.0
        has_rot = qx != 0.0 or qy != 0.0 or qz != 0.0 or qw != 1.0
        has_trans = tx != 0.0 or ty != 0.0 or tz != 0.0
        if has_scale:
            points_np = points_np * np.array([[sx, sy, sz]], dtype=np.float32)
        if has_rot:
            r00 = 1.0 - 2.0*(qy*qy + qz*qz); r01 = 2.0*(qx*qy - qw*qz); r02 = 2.0*(qx*qz + qw*qy)
            r10 = 2.0*(qx*qy + qw*qz); r11 = 1.0 - 2.0*(qx*qx + qz*qz); r12 = 2.0*(qy*qz - qw*qx)
            r20 = 2.0*(qx*qz - qw*qy); r21 = 2.0*(qy*qz + qw*qx); r22 = 1.0 - 2.0*(qx*qx + qy*qy)
            R = np.array([[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]], dtype=np.float32)
            points_np = points_np @ R
        if has_trans:
            points_np = points_np + np.array([[tx, ty, tz]], dtype=np.float32)

        # Baked position correction for the background/cavity mesh
        if name == "background_mesh":
            points_np = points_np + np.array([[0.233, -1.050, 5.350]], dtype=np.float32)

        indices_np = indices.numpy().astype(np.int32)
        if indices_np.ndim == 2:
            num_triangles = indices_np.shape[0]
            indices_flat = indices_np.reshape(-1)
        else:
            indices_flat = indices_np.reshape(-1)
            num_triangles = len(indices_flat) // 3

        # Flip winding order for the background/cavity mesh so faces and
        # normals point inward (camera is inside the environment mesh).
        if name == "background_mesh":
            tri = indices_flat.reshape(-1, 3)
            tri[:, [1, 2]] = tri[:, [2, 1]]
            indices_flat = tri.reshape(-1)

        face_counts = np.full(num_triangles, 3, dtype=np.int32)

        # ------ Write geometry ------
        # points is point3f[] → needs lanes=3 DLTensor
        self._ovrtx.write_array_attribute(
            prim_paths=[prim_path],
            attribute_name="points",
            tensors=[_make_float3_dltensor(points_np)],
        )
        self._ovrtx.write_array_attribute(
            prim_paths=[prim_path],
            attribute_name="faceVertexIndices",
            tensors=[DLTensor.from_dlpack(np.ascontiguousarray(indices_flat))],
        )
        self._ovrtx.write_array_attribute(
            prim_paths=[prim_path],
            attribute_name="faceVertexCounts",
            tensors=[DLTensor.from_dlpack(np.ascontiguousarray(face_counts))],
        )

        # ------ Compute and write per-vertex smooth normals ------
        if smooth_shading and num_triangles > 0:
            tri_idx = indices_flat.reshape(-1, 3)
            v0 = points_np[tri_idx[:, 0]]
            v1 = points_np[tri_idx[:, 1]]
            v2 = points_np[tri_idx[:, 2]]
            face_normals = np.cross(v1 - v0, v2 - v0)
            fn_len = np.linalg.norm(face_normals, axis=1, keepdims=True)
            face_normals = np.where(fn_len > 1e-8, face_normals / fn_len, 0.0)
            vtx_normals = np.zeros_like(points_np)
            np.add.at(vtx_normals, tri_idx[:, 0], face_normals)
            np.add.at(vtx_normals, tri_idx[:, 1], face_normals)
            np.add.at(vtx_normals, tri_idx[:, 2], face_normals)
            vn_len = np.linalg.norm(vtx_normals, axis=1, keepdims=True)
            vtx_normals = np.where(vn_len > 1e-8, vtx_normals / vn_len, 0.0)
            vtx_normals = np.ascontiguousarray(vtx_normals, dtype=np.float32)
            self._ovrtx.write_array_attribute(
                prim_paths=[prim_path],
                attribute_name="normals",
                tensors=[_make_float3_dltensor(vtx_normals)],
            )

        # ------ Write per-vertex displayColor when vertex_colors are provided ------
        # Skip for textured meshes — displayColor conflicts with OmniPBR material
        if vertex_colors is not None and name not in self._textured_meshes:
            vc_np = vertex_colors.numpy().astype(np.float32)
            # vertex_colors is vec4 (RGBA) — extract RGB for displayColor (color3f[])
            if vc_np.ndim == 2 and vc_np.shape[1] >= 3:
                colors_rgb = np.ascontiguousarray(vc_np[:, :3])
            elif vc_np.ndim == 1:
                colors_rgb = np.ascontiguousarray(vc_np.reshape(-1, 3) if vc_np.size % 3 == 0 else vc_np.reshape(-1, 4)[:, :3])
            else:
                colors_rgb = None
            if colors_rgb is not None and colors_rgb.shape[0] == points_np.shape[0]:
                try:
                    self._ovrtx.write_array_attribute(
                        prim_paths=[prim_path],
                        attribute_name="primvars:displayColor",
                        tensors=[_make_float3_dltensor(colors_rgb)],
                    )
                except RuntimeError:
                    pass  # displayColor write may fail if attribute schema doesn't match

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
        sanitized = _sanitize_name(name)
        prim_path = f"/World/{sanitized}"

        if color is None:
            color = (0.5, 0.5, 0.5)

        # Create the Sphere prim on first encounter
        if name not in self._sphere_prims:
            r = float(color[0])
            g = float(color[1])
            b = float(color[2])
            usda = f"""#usda 1.0
(defaultPrim = "{sanitized}")
def Sphere "{sanitized}" {{
    double radius = {float(radius)}
    color3f[] primvars:displayColor = [({r}, {g}, {b})]
    matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
    uniform token[] xformOpOrder = ["xformOp:transform"]
}}
"""
            self._ovrtx.add_usd_layer(usda, path_prefix=prim_path)
            self._sphere_prims[name] = prim_path

        # Update transform
        self._write_prim_transform(prim_path, pos, rot, (1.0, 1.0, 1.0))

        return hash(name) & 0x7FFFFFFF

    # ------------------------------------------------------------------
    # Texture / input / lifecycle
    # ------------------------------------------------------------------

    def load_texture(self, filepath: str, **kwargs) -> int:
        """Store a texture filepath for future use. Returns an incremental ID."""
        tex_id = self._next_texture_id
        self._texture_store[tex_id] = os.path.abspath(filepath)
        self._next_texture_id += 1
        return tex_id

    def set_input_callbacks(self, on_key_press=None, on_key_release=None):
        """Register keyboard callbacks on the pyglet window."""
        self._on_key_press_callback = on_key_press
        self._on_key_release_callback = on_key_release

    def is_running(self) -> bool:
        """Return True while the pyglet window has not been closed."""
        return not self._window.has_exit

    def save(self):
        """Block until the user closes the pyglet window."""
        while not self._window.has_exit:
            self._window.dispatch_events()
            pyglet.clock.tick()
