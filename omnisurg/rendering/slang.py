from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import warp as wp

from omnisurg.config import ViewerConfig


SLANG_RENDER_BACKENDS = frozenset({"slang", "slang-d3d12", "slang-vulkan", "slang-vk"})
SLANG_SHADER_DIR = Path(__file__).with_name("slang_shaders")
DEFAULT_WINDOWS_SLANG_BIN = Path(r"G:\warp\slang-2026.5.1-windows-x86_64\bin")
_SLANG_BIN_ENV = os.environ.get("OMNISURG_SLANG_BIN")
DEFAULT_SLANG_BIN = Path(_SLANG_BIN_ENV) if _SLANG_BIN_ENV else (
    DEFAULT_WINDOWS_SLANG_BIN if os.name == "nt" else None
)

_FALLBACK_MESH_COLORS = {
    "liver": (0.63, 0.29, 0.24),
    "fat": (0.90, 0.82, 0.42),
    "gallbladder": (0.19, 0.54, 0.22),
    "tissue": (0.87, 0.83, 0.78),
}
_TISSUE_LAYERS = ("base", "damage", "coag", "blood")
TISSUE_DEBUG_MODE_LABELS = (
    "Final",
    "Vertex Blend RGB",
    "Masked Layer Weights",
    "Blended Diffuse",
    "Blended Normal",
    "Spec/Roughness",
    "Heat/Blood Masks",
)
_TISSUE_MATERIAL_PARAM_RANGES = {
    "wetness": (0.0, 1.0),
    "wet_spec_scale": (0.0, 4.0),
    "wet_roughness": (0.02, 0.6),
    "blood_wetness": (0.0, 2.0),
}


@dataclass
class TissueMaterialParams:
    debug_mode: int = 0
    normal_strength: float = 0.65
    specular_scale: float = 0.35
    roughness_bias: float = 0.45
    ambient: float = 0.22
    rim_strength: float = 0.12
    wetness: float = 0.0
    wet_spec_scale: float = 1.0
    wet_roughness: float = 0.18
    subsurface_color: tuple[float, float, float] = (0.8, 0.22, 0.16)
    subsurface_strength: float = 0.0
    blood_wetness: float = 1.0


def is_slang_backend(backend: str) -> bool:
    return backend in SLANG_RENDER_BACKENDS


def _prepare_slang_library_path() -> None:
    if DEFAULT_SLANG_BIN is None or not DEFAULT_SLANG_BIN.exists():
        return

    bin_dir = str(DEFAULT_SLANG_BIN)
    if os.name == "nt":
        try:
            os.add_dll_directory(bin_dir)
        except (AttributeError, FileNotFoundError, OSError):
            pass

    path = os.environ.get("PATH", "")
    if bin_dir.lower() not in {entry.lower() for entry in path.split(os.pathsep) if entry}:
        os.environ["PATH"] = bin_dir + os.pathsep + path

    if os.name != "nt":
        # This helps child tools and some dlopen paths. For Linux, setting this
        # before launching Python is still the most reliable way to expose Slang
        # shared libraries that are not packaged inside the slangpy wheel.
        ld_path = os.environ.get("LD_LIBRARY_PATH", "")
        if bin_dir not in {entry for entry in ld_path.split(os.pathsep) if entry}:
            os.environ["LD_LIBRARY_PATH"] = bin_dir + os.pathsep + ld_path


def _load_slangpy():
    _prepare_slang_library_path()
    try:
        import slangpy as spy
    except BaseException as exc:  # pragma: no cover
        if isinstance(exc, KeyboardInterrupt):
            raise
        raise RuntimeError(
            "Slang renderer requires slangpy. Install it in this environment, for example "
            "`uv pip install slangpy`, and keep the Slang binary folder on PATH or set "
            "OMNISURG_SLANG_BIN. On Linux Vulkan, also ensure the NVIDIA driver, CUDA, "
            "Vulkan loader, and Slang shared libraries are visible to the process."
        ) from exc

    return spy


class _CudaWin32Handle(ctypes.Structure):
    _fields_ = [
        ("handle", ctypes.c_void_p),
        ("name", ctypes.c_void_p),
    ]


class _CudaExternalMemoryHandleUnion(ctypes.Union):
    _fields_ = [
        ("fd", ctypes.c_int),
        ("win32", _CudaWin32Handle),
        ("nv_sci_buf_object", ctypes.c_void_p),
    ]


class _CudaExternalMemoryHandleDesc(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_int),
        ("handle", _CudaExternalMemoryHandleUnion),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _CudaExternalMemoryBufferDesc(ctypes.Structure):
    _fields_ = [
        ("offset", ctypes.c_ulonglong),
        ("size", ctypes.c_ulonglong),
        ("flags", ctypes.c_uint),
        ("reserved", ctypes.c_uint * 16),
    ]


class _CudaDriver:
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5
    CUDA_EXTERNAL_MEMORY_DEDICATED = 0x1

    def __init__(self):
        self._cuda = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        self._cuda.cuInit.argtypes = [ctypes.c_uint]
        self._cuda.cuInit.restype = ctypes.c_int
        self._cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        self._cuda.cuCtxSetCurrent.restype = ctypes.c_int
        self._cuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self._cuda.cuCtxGetCurrent.restype = ctypes.c_int
        self._cuda.cuImportExternalMemory.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(_CudaExternalMemoryHandleDesc),
        ]
        self._cuda.cuImportExternalMemory.restype = ctypes.c_int
        self._cuda.cuExternalMemoryGetMappedBuffer.argtypes = [
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.c_void_p,
            ctypes.POINTER(_CudaExternalMemoryBufferDesc),
        ]
        self._cuda.cuExternalMemoryGetMappedBuffer.restype = ctypes.c_int
        self._cu_mem_free = getattr(self._cuda, "cuMemFree_v2", self._cuda.cuMemFree)
        self._cu_mem_free.argtypes = [ctypes.c_ulonglong]
        self._cu_mem_free.restype = ctypes.c_int
        self._cuda.cuDestroyExternalMemory.argtypes = [ctypes.c_void_p]
        self._cuda.cuDestroyExternalMemory.restype = ctypes.c_int
        self._cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._cuda.cuGetErrorName.restype = ctypes.c_int
        self._cuda.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        self._cuda.cuGetErrorString.restype = ctypes.c_int
        self._check(self._cuda.cuInit(0), "cuInit")

    def _error_text(self, result: int) -> str:
        name = ctypes.c_char_p()
        desc = ctypes.c_char_p()
        self._cuda.cuGetErrorName(result, ctypes.byref(name))
        self._cuda.cuGetErrorString(result, ctypes.byref(desc))
        name_text = name.value.decode("utf-8", "replace") if name.value else str(result)
        desc_text = desc.value.decode("utf-8", "replace") if desc.value else ""
        return f"{name_text}: {desc_text}" if desc_text else name_text

    def _check(self, result: int, operation: str) -> None:
        if result != 0:
            raise RuntimeError(f"{operation} failed: {self._error_text(result)}")

    def set_current_context(self, context: int) -> None:
        self._check(self._cuda.cuCtxSetCurrent(ctypes.c_void_p(int(context))), "cuCtxSetCurrent")

    def current_context(self) -> int:
        context = ctypes.c_void_p()
        self._check(self._cuda.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
        return int(context.value or 0)

    def import_external_memory(self, handle_type: int, shared_handle: int, size: int) -> ctypes.c_void_p:
        desc = _CudaExternalMemoryHandleDesc()
        desc.type = int(handle_type)
        if handle_type in (
            self.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32,
            self.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE,
        ):
            desc.handle.win32.handle = ctypes.c_void_p(int(shared_handle))
            desc.handle.win32.name = None
        elif handle_type == self.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD:
            desc.handle.fd = int(shared_handle)
        else:
            raise RuntimeError(f"Unsupported CUDA external memory handle type: {handle_type}")
        desc.size = int(size)
        desc.flags = self.CUDA_EXTERNAL_MEMORY_DEDICATED

        external_memory = ctypes.c_void_p()
        self._check(
            self._cuda.cuImportExternalMemory(ctypes.byref(external_memory), ctypes.byref(desc)),
            "cuImportExternalMemory",
        )
        return external_memory

    def map_external_memory(self, external_memory: ctypes.c_void_p, size: int) -> int:
        desc = _CudaExternalMemoryBufferDesc()
        desc.offset = 0
        desc.size = int(size)
        ptr = ctypes.c_ulonglong()
        self._check(
            self._cuda.cuExternalMemoryGetMappedBuffer(
                ctypes.byref(ptr),
                external_memory,
                ctypes.byref(desc),
            ),
            "cuExternalMemoryGetMappedBuffer",
        )
        return int(ptr.value)

    def free_mapped_pointer(self, ptr: int) -> None:
        if ptr:
            self._check(self._cu_mem_free(ctypes.c_ulonglong(int(ptr))), "cuMemFree")

    def destroy_external_memory(self, external_memory: ctypes.c_void_p) -> None:
        if external_memory:
            self._check(self._cuda.cuDestroyExternalMemory(external_memory), "cuDestroyExternalMemory")


_CUDA_DRIVER: _CudaDriver | None = None


def _cuda_driver() -> _CudaDriver:
    global _CUDA_DRIVER
    if _CUDA_DRIVER is None:
        _CUDA_DRIVER = _CudaDriver()
    return _CUDA_DRIVER


class _CudaMappedExternalMemory:
    def __init__(
        self,
        cuda: _CudaDriver,
        external_memory: ctypes.c_void_p,
        ptr: int,
        context: int,
    ):
        self._cuda = cuda
        self._external_memory = external_memory
        self.ptr = int(ptr)
        self._context = int(context)

    def close(self) -> None:
        ptr = self.ptr
        external_memory = self._external_memory
        self.ptr = 0
        self._external_memory = ctypes.c_void_p()
        if ptr or external_memory:
            self._cuda.set_current_context(self._context)
        if ptr:
            self._cuda.free_mapped_pointer(ptr)
        if external_memory:
            self._cuda.destroy_external_memory(external_memory)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _camera_fit_from_model(model: Any) -> tuple[np.ndarray, float]:
    particle_q = getattr(model, "particle_q", None)
    if particle_q is None:
        return np.array([0.0, 1.2, -4.0], dtype=np.float32), 3.0

    try:
        positions = np.asarray(particle_q.numpy(), dtype=np.float32)
    except Exception:
        return np.array([0.0, 1.2, -4.0], dtype=np.float32), 3.0

    if positions.size == 0:
        return np.array([0.0, 1.2, -4.0], dtype=np.float32), 3.0

    bounds_min = np.min(positions, axis=0)
    bounds_max = np.max(positions, axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 0.5, 0.25)
    return center.astype(np.float32), radius


def _normalize_or(value: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length <= 1.0e-6:
        return np.array(fallback, dtype=np.float32)
    return (value / length).astype(np.float32)


@wp.kernel
def _clear_mesh_normal_scratch(
    normal_accum: wp.array(dtype=wp.vec3f),
    normal_counts: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    normal_accum[tid] = wp.vec3f(0.0, 0.0, 0.0)
    normal_counts[tid] = 0


@wp.kernel
def _accumulate_mesh_normals(
    indices: wp.array(dtype=wp.int32),
    positions: wp.array(dtype=wp.vec3f),
    normal_accum: wp.array(dtype=wp.vec3f),
    normal_counts: wp.array(dtype=wp.int32),
):
    tri = wp.tid()
    base = tri * 3
    i = indices[base + 0]
    j = indices[base + 1]
    k = indices[base + 2]

    v0 = positions[i]
    v1 = positions[j]
    v2 = positions[k]
    n = wp.cross(v1 - v0, v2 - v0)
    if wp.length(n) <= 1.0e-12:
        return

    wp.atomic_add(normal_accum, i, n)
    wp.atomic_add(normal_counts, i, 1)
    wp.atomic_add(normal_accum, j, n)
    wp.atomic_add(normal_counts, j, 1)
    wp.atomic_add(normal_accum, k, n)
    wp.atomic_add(normal_counts, k, 1)


@wp.kernel
def _finalize_mesh_normals(
    normal_accum: wp.array(dtype=wp.vec3f),
    normal_counts: wp.array(dtype=wp.int32),
    normals: wp.array(dtype=wp.vec3f),
):
    tid = wp.tid()
    n = normal_accum[tid]
    if normal_counts[tid] > 0 and wp.length(n) > 1.0e-12:
        normals[tid] = wp.normalize(n)
    else:
        normals[tid] = wp.vec3f(0.0, 1.0, 0.0)


@dataclass
class _SharedPositionBuffer:
    buffer: Any
    mapping: _CudaMappedExternalMemory
    warp_array: wp.array
    count: int
    last_updated_frame: int = -1

    def close(self) -> None:
        self.mapping.close()


@dataclass
class _SharedNormalBuffer:
    buffer: Any
    mapping: _CudaMappedExternalMemory
    warp_array: wp.array
    accum: wp.array
    counts: wp.array
    count: int
    position_source_id: int = 0
    last_updated_frame: int = -1

    def close(self) -> None:
        self.mapping.close()


@dataclass
class _SharedVertexColorBuffer:
    buffer: Any
    mapping: _CudaMappedExternalMemory
    warp_array: wp.array
    count: int
    last_updated_frame: int = -1

    def close(self) -> None:
        self.mapping.close()


@dataclass
class _MeshResource:
    index_buffer: Any
    index_count: int
    index_source_id: int
    normals: _SharedNormalBuffer | None = None
    uv_buffer: Any | None = None
    uv_count: int = 0
    uv_source_id: int = 0

    def close(self) -> None:
        if self.normals is not None:
            self.normals.close()
            self.normals = None


@dataclass
class _TissueLayerPaths:
    base: Path
    damage: Path
    coag: Path
    blood: Path

    def as_tuple(self) -> tuple[Path, Path, Path, Path]:
        return (self.base, self.damage, self.coag, self.blood)


@dataclass
class _TissueMaterialPaths:
    diffuse: _TissueLayerPaths
    normal: _TissueLayerPaths
    spec: _TissueLayerPaths

    def cache_key(self) -> tuple[str, ...]:
        return tuple(
            str(path)
            for layer_paths in (self.diffuse, self.normal, self.spec)
            for path in layer_paths.as_tuple()
        )


@dataclass
class _LayerTextureSet:
    base: Any | None
    damage: Any | None
    coag: Any | None
    blood: Any | None


@dataclass
class _MaterialResource:
    diffuse: _LayerTextureSet | None
    normal: _LayerTextureSet | None
    spec: _LayerTextureSet | None
    layer_masks: _LayerTextureSet | None
    blood_mask_texture: Any | None
    heat_mask_texture: Any | None
    sampler: Any | None
    valid: bool


@dataclass
class _UiValueRecord:
    value: Any
    changed: bool = False
    clicked: bool = False
    widget: Any | None = None


class _UiEnumValue:
    def __init__(self, value: int = 0):
        self.value = int(value)

    def __int__(self) -> int:
        return self.value


class _UiImVec2:
    def __init__(self, x: float = 0.0, y: float = 0.0):
        self.x = float(x)
        self.y = float(y)


class _SlangImmediateUi:
    """Small Newton-style immediate UI facade over SlangPy's retained widgets."""

    ImVec2 = _UiImVec2
    Cond_ = SimpleNamespace(appearing=_UiEnumValue(1), once=_UiEnumValue(2))
    WindowFlags_ = SimpleNamespace(
        no_resize=_UiEnumValue(1),
        no_decoration=_UiEnumValue(2),
        always_auto_resize=_UiEnumValue(4),
        no_saved_settings=_UiEnumValue(8),
        no_focus_on_appearing=_UiEnumValue(16),
        no_nav=_UiEnumValue(32),
        no_move=_UiEnumValue(64),
    )
    TreeNodeFlags_ = SimpleNamespace(default_open=_UiEnumValue(1))

    def __init__(self, spy: Any, sui: Any, screen: Any):
        self._spy = spy
        self._sui = sui
        self._screen = screen
        self._records: dict[tuple[str, str, int], _UiValueRecord] = {}
        self._occurrences: dict[tuple[str, str], int] = {}
        self._used_keys: set[tuple[str, str, int]] = set()
        self._parent_stack: list[Any] = [screen]
        self._next_window_pos: _UiImVec2 | None = None
        self._next_window_size: _UiImVec2 | None = None
        self.io = SimpleNamespace(display_size=(0.0, 0.0))

    def begin_frame(self) -> None:
        self._used_keys.clear()

    def finish_frame(self) -> None:
        for key, record in self._records.items():
            if record.widget is not None:
                record.widget.visible = key in self._used_keys

    def reset(self, parent: Any, width: int, height: int) -> None:
        self._occurrences.clear()
        self._parent_stack = [parent]
        self._next_window_pos = None
        self._next_window_size = None
        self.io.display_size = (float(width), float(height))

    @property
    def _parent(self) -> Any:
        return self._parent_stack[-1]

    def _key(self, kind: str, label: str) -> tuple[str, str, int]:
        base = (f"{id(self._parent)}:{kind}", str(label))
        occurrence = self._occurrences.get(base, 0)
        self._occurrences[base] = occurrence + 1
        return (base[0], base[1], occurrence)

    def _record(self, kind: str, label: str, value: Any) -> tuple[_UiValueRecord, bool, Any]:
        key = self._key(kind, label)
        self._used_keys.add(key)
        record = self._records.get(key)
        if record is None:
            record = _UiValueRecord(value=value)
            self._records[key] = record
        changed = record.changed
        if changed:
            current = record.value
            record.changed = False
        else:
            record.value = value
            current = value
        return record, changed, current

    def _sync_widget(self, record: _UiValueRecord) -> Any | None:
        widget = record.widget
        if widget is None:
            return None
        if getattr(widget, "parent", self._parent) is not self._parent:
            widget.parent = self._parent
        widget.visible = True
        return widget

    def _float2(self, value: _UiImVec2 | tuple[float, float] | None, fallback: tuple[float, float]):
        if value is None:
            x, y = fallback
        else:
            x = getattr(value, "x", fallback[0])
            y = getattr(value, "y", fallback[1])
        return self._spy.float2(float(x), float(y))

    def text(self, text: object) -> None:
        record, _changed, _current = self._record("text", "", str(text))
        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.Text(self._parent, str(text))
            record.widget = widget
        else:
            widget.text = str(text)

    def separator(self) -> None:
        record, _changed, _current = self._record("separator", "", "")
        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.Text(self._parent, "------------------------------")
            record.widget = widget

    def spacing(self) -> None:
        record, _changed, _current = self._record("spacing", "", "")
        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.Text(self._parent, "")
            record.widget = widget

    def collapsing_header(self, label: str, *args, **kwargs) -> bool:
        del args, kwargs
        record, _changed, _current = self._record("group", label, str(label))
        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.Group(self._parent, str(label))
            record.widget = widget
        else:
            widget.label = str(label)
        return True

    def set_next_item_open(self, *args, **kwargs) -> None:
        del args, kwargs

    def set_next_window_pos(self, pos: _UiImVec2, *args, **kwargs) -> None:
        del args, kwargs
        self._next_window_pos = pos

    def set_next_window_size(self, size: _UiImVec2, *args, **kwargs) -> None:
        del args, kwargs
        self._next_window_size = size

    def begin(self, title: str, *args, **kwargs) -> bool:
        del args, kwargs
        record, _changed, _current = self._record("window", title, str(title))
        window = self._sync_widget(record)
        position = self._float2(self._next_window_pos, (460.0, 10.0))
        size = self._float2(self._next_window_size, (420.0, 360.0))
        if window is None:
            window = self._sui.Window(self._screen, str(title), position=position, size=size)
            record.widget = window
        else:
            window.title = str(title)
            window.position = position
            window.size = size
        self._next_window_pos = None
        self._next_window_size = None
        self._parent_stack.append(window)
        return True

    def end(self) -> None:
        if len(self._parent_stack) > 1:
            self._parent_stack.pop()

    def checkbox(self, label: str, value: bool) -> tuple[bool, bool]:
        record, changed, current = self._record("checkbox", label, bool(value))

        def _callback(updated: bool, rec=record) -> None:
            rec.value = bool(updated)
            rec.changed = True

        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.CheckBox(self._parent, str(label), bool(current), _callback)
            record.widget = widget
        else:
            widget.label = str(label)
            widget.value = bool(current)
            widget.callback = _callback
        return changed, bool(current)

    def slider_float(
        self,
        label: str,
        value: float,
        min_value: float,
        max_value: float,
        fmt: str = "%.3f",
        *args,
        **kwargs,
    ) -> tuple[bool, float]:
        del args, kwargs
        record, changed, current = self._record("slider_float", label, float(value))

        def _callback(updated: float, rec=record) -> None:
            rec.value = float(updated)
            rec.changed = True

        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.SliderFloat(
                self._parent,
                str(label),
                float(current),
                _callback,
                float(min_value),
                float(max_value),
                str(fmt),
            )
            record.widget = widget
        else:
            widget.label = str(label)
            widget.value = float(current)
            widget.callback = _callback
            widget.min = float(min_value)
            widget.max = float(max_value)
            widget.format = str(fmt)
        return changed, float(current)

    def slider_int(
        self,
        label: str,
        value: int,
        min_value: int,
        max_value: int,
        *args,
        **kwargs,
    ) -> tuple[bool, int]:
        del args, kwargs
        record, changed, current = self._record("slider_int", label, int(value))

        def _callback(updated: int, rec=record) -> None:
            rec.value = int(updated)
            rec.changed = True

        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.SliderInt(self._parent, str(label), int(current), _callback, int(min_value), int(max_value))
            record.widget = widget
        else:
            widget.label = str(label)
            widget.value = int(current)
            widget.callback = _callback
            widget.min = int(min_value)
            widget.max = int(max_value)
        return changed, int(current)

    def button(self, label: str, *args, **kwargs) -> bool:
        del args, kwargs
        key = self._key("button", label)
        self._used_keys.add(key)
        record = self._records.get(key)
        if record is None:
            record = _UiValueRecord(value=False)
            self._records[key] = record
        clicked = bool(record.clicked)
        record.clicked = False

        def _callback(rec=record) -> None:
            rec.clicked = True

        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.Button(self._parent, str(label), _callback)
            record.widget = widget
        else:
            widget.label = str(label)
            widget.callback = _callback
        return clicked

    def combo(self, label: str, value: int, items: list[str] | tuple[str, ...], *args, **kwargs) -> tuple[bool, int]:
        del args, kwargs
        if not hasattr(self._sui, "ComboBox"):
            return self.slider_int(label, value, 0, max(0, len(items) - 1))

        record, changed, current = self._record("combo", label, int(value))

        def _callback(updated: int, rec=record) -> None:
            rec.value = int(updated)
            rec.changed = True

        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.ComboBox(self._parent, str(label), int(current), _callback, list(items))
            record.widget = widget
        else:
            widget.label = str(label)
            widget.value = int(current)
            widget.callback = _callback
            widget.items = list(items)
        return changed, int(current)

    def input_text(self, label: str, value: str, *args, **kwargs) -> tuple[bool, str]:
        del args, kwargs
        record, changed, current = self._record("input_text", label, str(value))

        def _callback(updated: str, rec=record) -> None:
            rec.value = str(updated)
            rec.changed = True

        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.InputText(self._parent, str(label), str(current), _callback)
            record.widget = widget
        else:
            widget.label = str(label)
            widget.value = str(current)
            widget.callback = _callback
        return changed, str(current)

    def same_line(self, *args, **kwargs) -> None:
        del args, kwargs

    def push_item_width(self, *args, **kwargs) -> None:
        del args, kwargs

    def pop_item_width(self, *args, **kwargs) -> None:
        del args, kwargs

    def is_item_hovered(self, *args, **kwargs) -> bool:
        del args, kwargs
        return False

    def set_tooltip(self, *args, **kwargs) -> None:
        del args, kwargs


class SlangRenderer:
    """SlangPy raster renderer backed by graphics-owned shared buffers.

    D3D12/Vulkan own the vertex buffers. SlangPy exposes those buffers through
    CUDA, and Warp writes simulation positions into the mapped CUDA memory.
    """

    supports_implot = False

    def __init__(self, viewer_config: ViewerConfig, model: Any, device: wp.context.Device):
        if not device.is_cuda:
            raise RuntimeError(
                "Slang renderer requires a CUDA Warp device for shared-buffer interop."
            )

        self._spy = _load_slangpy()
        self._warp_device = device
        self._backend = viewer_config.backend
        self._vsync = bool(viewer_config.vsync)
        self._cuda = _cuda_driver()
        self._frame_id = 0
        self._closed = False
        self._surface_texture = None
        self._command_encoder = None
        self._pass_encoder = None
        self._depth_texture = None
        self._pending_resize: tuple[int, int] | None = None
        self._shared_positions: dict[tuple[int, int], _SharedPositionBuffer] = {}
        self._shared_vertex_colors: dict[tuple[int, int], _SharedVertexColorBuffer] = {}
        self._meshes: dict[str, _MeshResource] = {}
        self._materials: dict[tuple[str, ...], _MaterialResource] = {}
        self._default_textures: dict[tuple[str, bool], Any] = {}
        self._material_sampler = None
        self._tissue_material_params = TissueMaterialParams()
        self._logs: dict[str, float] = {}
        self._warnings: set[str] = set()
        self._sui = None
        self._ui_context = None
        self._ui_adapter: _SlangImmediateUi | None = None
        self._ui_windows: dict[str, Any] = {}
        self._ui_enabled = True
        self._ui_callbacks: dict[str, list[Callable[[Any], None]]] = {
            "side": [],
            "stats": [],
            "free": [],
            "panel": [],
        }
        self._failed_ui_callbacks: set[tuple[str, int]] = set()
        self._on_key_press_callback: Callable[[int, int], None] | None = None
        self._on_key_release_callback: Callable[[int, int], None] | None = None

        self._cuda.set_current_context(int(device.context))

        device_type = self._device_type_from_backend(viewer_config.backend)
        self._device_type = device_type
        self._device = self._spy.create_device(
            device_type,
            include_paths=[SLANG_SHADER_DIR],
            enable_cuda_interop=True,
            existing_device_handles=self._spy.get_cuda_current_context_native_handles(),
        )
        if not bool(getattr(self._device, "supports_cuda_interop", False)):
            raise RuntimeError(
                f"Slang {viewer_config.backend} device does not support CUDA interop."
            )

        self._window = self._spy.Window(
            width=1600,
            height=1000,
            title=f"OmniSurg {viewer_config.backend}",
            resizable=True,
        )
        self._window.on_resize = self._on_resize
        self._surface = self._device.create_surface(self._window)
        self._surface.configure(
            width=self._window.width,
            height=self._window.height,
            vsync=self._vsync,
        )

        self._color_format = self._surface.info.preferred_format
        self._depth_format = self._spy.Format.d32_float
        self._viewport = self._spy.Viewport.from_size(self._window.width, self._window.height)
        self._scissor = self._spy.ScissorRect.from_size(self._window.width, self._window.height)

        self._init_ui()
        self._window.on_keyboard_event = self._on_keyboard_event
        self._window.on_mouse_event = self._on_mouse_event

        self._init_camera(viewer_config.camera_pos, model)
        self._init_pipeline()

    def _device_type_from_backend(self, backend: str):
        if backend == "slang":
            if os.name == "nt":
                return self._spy.DeviceType.d3d12
            return self._spy.DeviceType.vulkan
        if backend == "slang-d3d12":
            if os.name != "nt":
                raise RuntimeError("slang-d3d12 is only available on Windows; use slang-vulkan on Linux.")
            return self._spy.DeviceType.d3d12
        if backend in {"slang-vulkan", "slang-vk"}:
            return self._spy.DeviceType.vulkan
        raise RuntimeError(f"Unsupported Slang backend: {backend}")

    def _init_camera(self, camera_pos: tuple[float, float, float], model: Any) -> None:
        target, radius = _camera_fit_from_model(model)
        eye = np.array(camera_pos, dtype=np.float32)
        forward = _normalize_or(target - eye, (0.0, 0.0, -1.0))
        right = _normalize_or(
            np.cross(forward, np.array([0.0, 1.0, 0.0], dtype=np.float32)),
            (1.0, 0.0, 0.0),
        )
        up = _normalize_or(np.cross(right, forward), (0.0, 1.0, 0.0))

        distance = float(np.linalg.norm(target - eye))
        near = max(0.01, distance - radius * 3.0)
        far = max(near + 1.0, distance + radius * 5.0)

        self._camera_pos = eye
        self._camera_right = right
        self._camera_up = up
        self._camera_forward = forward
        self._camera_near = float(near)
        self._camera_far = float(far)
        self._camera_inv_tan_half_fovy = float(1.0 / np.tan(np.deg2rad(45.0) * 0.5))

    def _init_pipeline(self) -> None:
        self._flat_program = self._device.load_program(
            "omnisurg_mesh.slang",
            ["vertex_main", "fragment_main"],
        )
        self._point_program = self._device.load_program(
            "omnisurg_mesh.slang",
            ["point_vertex_main", "point_fragment_main"],
        )
        self._tissue_program = self._device.load_program(
            "omnisurg_tissue.slang",
            ["vertex_main", "fragment_main"],
        )
        self._flat_input_layout = self._device.create_input_layout(
            input_elements=[
                {
                    "semantic_name": "POSITION",
                    "semantic_index": 0,
                    "format": self._spy.Format.rgb32_float,
                    "buffer_slot_index": 0,
                },
                {
                    "semantic_name": "NORMAL",
                    "semantic_index": 0,
                    "format": self._spy.Format.rgb32_float,
                    "buffer_slot_index": 1,
                }
            ],
            vertex_streams=[{"stride": 12}, {"stride": 12}],
        )
        self._point_input_layout = self._device.create_input_layout(
            input_elements=[
                {
                    "semantic_name": "POSITION",
                    "semantic_index": 0,
                    "format": self._spy.Format.rgb32_float,
                    "buffer_slot_index": 0,
                }
            ],
            vertex_streams=[{"stride": 12}],
        )
        self._tissue_input_layout = self._device.create_input_layout(
            input_elements=[
                {
                    "semantic_name": "POSITION",
                    "semantic_index": 0,
                    "format": self._spy.Format.rgb32_float,
                    "buffer_slot_index": 0,
                },
                {
                    "semantic_name": "NORMAL",
                    "semantic_index": 0,
                    "format": self._spy.Format.rgb32_float,
                    "buffer_slot_index": 1,
                },
                {
                    "semantic_name": "TEXCOORD",
                    "semantic_index": 0,
                    "format": self._spy.Format.rg32_float,
                    "buffer_slot_index": 2,
                },
                {
                    "semantic_name": "COLOR",
                    "semantic_index": 0,
                    "format": self._spy.Format.rgba32_float,
                    "buffer_slot_index": 3,
                },
            ],
            vertex_streams=[{"stride": 12}, {"stride": 12}, {"stride": 8}, {"stride": 16}],
        )
        common = {
            "targets": [{"format": self._color_format}],
            "depth_stencil": {
                "format": self._depth_format,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": self._spy.ComparisonFunc.less,
            },
            "rasterizer": {"cull_mode": self._spy.CullMode.none},
        }
        self._flat_triangle_pipeline = self._device.create_render_pipeline(
            program=self._flat_program,
            input_layout=self._flat_input_layout,
            primitive_topology=self._spy.PrimitiveTopology.triangle_list,
            label="omnisurg-flat-triangles",
            **common,
        )
        self._point_pipeline = self._device.create_render_pipeline(
            program=self._point_program,
            input_layout=self._point_input_layout,
            primitive_topology=self._spy.PrimitiveTopology.point_list,
            label="omnisurg-points",
            **common,
        )
        self._tissue_triangle_pipeline = self._device.create_render_pipeline(
            program=self._tissue_program,
            input_layout=self._tissue_input_layout,
            primitive_topology=self._spy.PrimitiveTopology.triangle_list,
            label="omnisurg-tissue-triangles",
            **common,
        )

    def _init_ui(self) -> None:
        try:
            import slangpy.ui as sui

            ui_context = sui.Context(self._device)
        except Exception as exc:
            self._ui_enabled = False
            self._sui = None
            self._ui_context = None
            self._ui_adapter = None
            self._warn_once("slang-ui:init", f"Slang ImGui UI disabled: {exc}")
            return

        self._sui = sui
        self._ui_context = ui_context
        self._ui_adapter = _SlangImmediateUi(self._spy, sui, ui_context.screen)

    def _ui_window(
        self,
        key: str,
        title: str,
        position: tuple[float, float],
        size: tuple[float, float],
    ) -> Any:
        if self._sui is None or self._ui_context is None:
            return None

        window = self._ui_windows.get(key)
        spy_position = self._spy.float2(float(position[0]), float(position[1]))
        spy_size = self._spy.float2(float(size[0]), float(size[1]))
        if window is None:
            window = self._sui.Window(
                self._ui_context.screen,
                str(title),
                position=spy_position,
                size=spy_size,
            )
            self._ui_windows[key] = window
        else:
            window.title = str(title)
            window.position = spy_position
            window.size = spy_size
            window.visible = True
        return window

    def _pyglet_symbol_from_slang_key(self, key: Any) -> int | None:
        if key is None:
            return None

        name = str(getattr(key, "name", "")).lower()
        fallback = getattr(key, "value", None)
        fallback_symbol = int(fallback) if fallback is not None else None

        try:
            import pyglet

            pyglet_key = pyglet.window.key
            if len(name) == 1 and name.isalpha():
                return int(getattr(pyglet_key, name.upper(), fallback_symbol))

            key_map = {
                "space": "SPACE",
                "escape": "ESCAPE",
                "tab": "TAB",
                "enter": "ENTER",
                "backspace": "BACKSPACE",
                "insert": "INSERT",
                "delete": "DELETE",
                "right": "RIGHT",
                "left": "LEFT",
                "down": "DOWN",
                "up": "UP",
                "page_up": "PAGEUP",
                "page_down": "PAGEDOWN",
                "home": "HOME",
                "end": "END",
            }
            mapped = key_map.get(name)
            if mapped is not None:
                return int(getattr(pyglet_key, mapped, fallback_symbol))

            if name.startswith("f") and name[1:].isdigit():
                return int(getattr(pyglet_key, name.upper(), fallback_symbol))
        except Exception:
            pass

        return fallback_symbol

    def _pyglet_modifiers_from_slang_event(self, event: Any) -> int:
        mods = getattr(event, "mods", 0)
        value = int(getattr(mods, "value", mods) or 0)
        try:
            import pyglet

            pyglet_key = pyglet.window.key
            result = 0
            if value & 1:
                result |= int(pyglet_key.MOD_SHIFT)
            if value & 2:
                result |= int(pyglet_key.MOD_CTRL)
            if value & 4:
                result |= int(pyglet_key.MOD_ALT)
            return result
        except Exception:
            return value

    def _on_keyboard_event(self, event: Any) -> None:
        if self._ui_enabled and self._ui_context is not None:
            try:
                if bool(self._ui_context.handle_keyboard_event(event)):
                    return
            except Exception as exc:
                self._warn_once("slang-ui:keyboard", f"Slang ImGui keyboard handling disabled: {exc}")

        if event.is_key_press():
            callback = self._on_key_press_callback
        elif event.is_key_release():
            callback = self._on_key_release_callback
        else:
            return

        if callback is None:
            return

        symbol = self._pyglet_symbol_from_slang_key(getattr(event, "key", None))
        if symbol is None:
            return

        try:
            callback(symbol, self._pyglet_modifiers_from_slang_event(event))
        except Exception as exc:
            self._warn_once("slang-key-callback", f"Slang key callback raised: {exc}")

    def _on_mouse_event(self, event: Any) -> None:
        if self._ui_enabled and self._ui_context is not None:
            try:
                self._ui_context.handle_mouse_event(event)
            except Exception as exc:
                self._warn_once("slang-ui:mouse", f"Slang ImGui mouse handling disabled: {exc}")

    def _invoke_ui_callbacks(
        self,
        callbacks: list[Callable[[Any], None]],
        adapter: _SlangImmediateUi,
        position: str,
    ) -> None:
        for callback in callbacks:
            callback_key = (position, id(callback))
            if callback_key in self._failed_ui_callbacks:
                continue

            stack_depth = len(adapter._parent_stack)
            try:
                callback(adapter)
            except Exception as exc:
                adapter._parent_stack = adapter._parent_stack[:stack_depth]
                self._failed_ui_callbacks.add(callback_key)
                self._warn_once(
                    f"slang-ui:callback:{position}:{id(callback)}",
                    f"Slang ImGui callback at '{position}' failed and was disabled: {exc}",
                )

    def _render_ui(self, width: int, height: int) -> None:
        if (
            not self._ui_enabled
            or self._sui is None
            or self._ui_context is None
            or self._ui_adapter is None
            or self._command_encoder is None
            or self._surface_texture is None
        ):
            return

        width = max(1, int(width))
        height = max(1, int(height))
        screen = self._ui_context.screen
        adapter = self._ui_adapter

        try:
            for window in self._ui_windows.values():
                window.visible = False
            adapter.begin_frame()

            if self._ui_callbacks["panel"] or self._ui_callbacks["side"]:
                side_width = min(430.0, max(220.0, float(width) - 20.0))
                side_height = max(180.0, float(height) - 20.0)
                side_window = self._ui_window(
                    "side",
                    "OmniSurg",
                    (10.0, 10.0),
                    (side_width, side_height),
                )
                adapter.reset(side_window, width, height)
                self._invoke_ui_callbacks(self._ui_callbacks["panel"], adapter, "panel")
                self._invoke_ui_callbacks(self._ui_callbacks["side"], adapter, "side")

            if self._logs or self._ui_callbacks["stats"]:
                stats_width = min(360.0, max(220.0, float(width) - 20.0))
                stats_x = max(10.0, float(width) - stats_width - 10.0)
                stats_window = self._ui_window(
                    "stats",
                    "Stats",
                    (stats_x, 10.0),
                    (
                        stats_width,
                        min(
                            max(160.0, 42.0 + 22.0 * len(self._logs)),
                            max(80.0, float(height) - 20.0),
                        ),
                    ),
                )
                adapter.reset(stats_window, width, height)
                for name, value in sorted(self._logs.items()):
                    adapter.text(f"{name}: {value:.3f}")
                if self._logs and self._ui_callbacks["stats"]:
                    adapter.separator()
                self._invoke_ui_callbacks(self._ui_callbacks["stats"], adapter, "stats")

            if self._ui_callbacks["free"]:
                adapter.reset(screen, width, height)
                self._invoke_ui_callbacks(self._ui_callbacks["free"], adapter, "free")

            adapter.finish_frame()
            self._ui_context.begin_frame(width, height)
            self._ui_context.end_frame(self._surface_texture, self._command_encoder)
        except Exception as exc:
            self._ui_enabled = False
            try:
                screen.remove_all_children()
            except Exception:
                pass
            self._warn_once("slang-ui:render", f"Slang ImGui UI disabled after render failure: {exc}")

    def _on_resize(self, width: int, height: int) -> None:
        self._pending_resize = (int(width), int(height))

    def _apply_pending_resize(self) -> None:
        if self._pending_resize is None:
            return

        width, height = self._pending_resize
        self._pending_resize = None
        self._device.wait()
        if width <= 0 or height <= 0:
            self._surface.unconfigure()
            self._depth_texture = None
            return

        self._surface.configure(width=width, height=height, vsync=self._vsync)
        self._viewport = self._spy.Viewport.from_size(width, height)
        self._scissor = self._spy.ScissorRect.from_size(width, height)
        self._depth_texture = None

    def _ensure_depth_texture(self, width: int, height: int):
        if (
            self._depth_texture is None
            or self._depth_texture.width != width
            or self._depth_texture.height != height
        ):
            self._depth_texture = self._device.create_texture(
                format=self._depth_format,
                width=width,
                height=height,
                usage=self._spy.TextureUsage.depth_stencil,
                label="omnisurg-depth",
            )
        return self._depth_texture

    def _cuda_stream_handle(self):
        stream = wp.get_stream(self._warp_device)
        return self._spy.NativeHandle.from_cuda_stream(int(stream.cuda_stream))

    def _shared_position_buffer(self, points: wp.array, label: str) -> _SharedPositionBuffer:
        key = (int(points.ptr), int(len(points)))
        shared = self._shared_positions.get(key)
        if shared is not None:
            return shared

        count = int(len(points))
        size = count * 12
        buffer = self._device.create_buffer(
            size=size,
            usage=(
                self._spy.BufferUsage.shared
                | self._spy.BufferUsage.vertex_buffer
                | self._spy.BufferUsage.shader_resource
                | self._spy.BufferUsage.copy_destination
            ),
            label=f"{label}-shared-position",
        )

        self._cuda.set_current_context(int(self._warp_device.context))
        handle_type = self._cuda_external_memory_handle_type()
        shared_handle = int(buffer.shared_handle.value)
        if shared_handle == 0:
            raise RuntimeError(f"Slang shared buffer handle creation failed for {label}.")

        external_memory = self._cuda.import_external_memory(
            handle_type=handle_type,
            shared_handle=shared_handle,
            size=size,
        )
        try:
            ptr = self._cuda.map_external_memory(external_memory, size)
        except Exception:
            self._cuda.destroy_external_memory(external_memory)
            raise
        mapping_context = self._cuda.current_context()
        mapping = _CudaMappedExternalMemory(
            self._cuda,
            external_memory,
            ptr,
            mapping_context,
        )
        warp_array = wp.array(
            ptr=ptr,
            dtype=wp.vec3f,
            shape=(count,),
            capacity=size,
            device=self._warp_device,
            copy=False,
        )
        shared = _SharedPositionBuffer(
            buffer=buffer,
            mapping=mapping,
            warp_array=warp_array,
            count=count,
        )
        self._shared_positions[key] = shared
        return shared

    def _cuda_external_memory_handle_type(self) -> int:
        if self._device_type == self._spy.DeviceType.d3d12:
            return self._cuda.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
        if self._device_type == self._spy.DeviceType.vulkan:
            if os.name != "nt":
                return self._cuda.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
            return self._cuda.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
        raise RuntimeError(f"Unsupported Slang CUDA interop device type: {self._device_type}")

    def _update_shared_positions(self, points: wp.array, label: str) -> _SharedPositionBuffer:
        shared = self._shared_position_buffer(points, label)
        if shared.last_updated_frame != self._frame_id:
            wp.copy(shared.warp_array, points)
            shared.last_updated_frame = self._frame_id
        return shared

    def _shared_vertex_color_buffer(
        self,
        vertex_colors: wp.array,
        label: str,
    ) -> _SharedVertexColorBuffer:
        key = (int(vertex_colors.ptr), int(len(vertex_colors)))
        shared = self._shared_vertex_colors.get(key)
        if shared is not None:
            return shared

        count = int(len(vertex_colors))
        size = count * 16
        buffer = self._device.create_buffer(
            size=size,
            usage=(
                self._spy.BufferUsage.shared
                | self._spy.BufferUsage.vertex_buffer
                | self._spy.BufferUsage.shader_resource
                | self._spy.BufferUsage.copy_destination
            ),
            label=f"{label}-shared-vertex-colors",
        )

        self._cuda.set_current_context(int(self._warp_device.context))
        handle_type = self._cuda_external_memory_handle_type()
        shared_handle = int(buffer.shared_handle.value)
        if shared_handle == 0:
            raise RuntimeError(f"Slang shared vertex color buffer handle creation failed for {label}.")

        external_memory = self._cuda.import_external_memory(
            handle_type=handle_type,
            shared_handle=shared_handle,
            size=size,
        )
        try:
            ptr = self._cuda.map_external_memory(external_memory, size)
        except Exception:
            self._cuda.destroy_external_memory(external_memory)
            raise

        mapping_context = self._cuda.current_context()
        mapping = _CudaMappedExternalMemory(
            self._cuda,
            external_memory,
            ptr,
            mapping_context,
        )
        warp_array = wp.array(
            ptr=ptr,
            dtype=wp.vec4f,
            shape=(count,),
            capacity=size,
            device=self._warp_device,
            copy=False,
        )
        shared = _SharedVertexColorBuffer(
            buffer=buffer,
            mapping=mapping,
            warp_array=warp_array,
            count=count,
        )
        self._shared_vertex_colors[key] = shared
        return shared

    def _update_shared_vertex_colors(
        self,
        vertex_colors: wp.array,
        label: str,
    ) -> _SharedVertexColorBuffer:
        shared = self._shared_vertex_color_buffer(vertex_colors, label)
        if shared.last_updated_frame != self._frame_id:
            wp.copy(shared.warp_array, vertex_colors)
            shared.last_updated_frame = self._frame_id
        return shared

    def _shared_normal_buffer(self, mesh: _MeshResource, count: int, label: str) -> _SharedNormalBuffer:
        if mesh.normals is not None and mesh.normals.count == int(count):
            return mesh.normals

        if mesh.normals is not None:
            mesh.normals.close()
            mesh.normals = None

        size = int(count) * 12
        buffer = self._device.create_buffer(
            size=size,
            usage=(
                self._spy.BufferUsage.shared
                | self._spy.BufferUsage.vertex_buffer
                | self._spy.BufferUsage.shader_resource
                | self._spy.BufferUsage.copy_destination
            ),
            label=f"{label}-shared-normals",
        )

        self._cuda.set_current_context(int(self._warp_device.context))
        handle_type = self._cuda_external_memory_handle_type()
        shared_handle = int(buffer.shared_handle.value)
        if shared_handle == 0:
            raise RuntimeError(f"Slang shared normal buffer handle creation failed for {label}.")

        external_memory = self._cuda.import_external_memory(
            handle_type=handle_type,
            shared_handle=shared_handle,
            size=size,
        )
        try:
            ptr = self._cuda.map_external_memory(external_memory, size)
        except Exception:
            self._cuda.destroy_external_memory(external_memory)
            raise

        mapping_context = self._cuda.current_context()
        mapping = _CudaMappedExternalMemory(
            self._cuda,
            external_memory,
            ptr,
            mapping_context,
        )
        warp_array = wp.array(
            ptr=ptr,
            dtype=wp.vec3f,
            shape=(int(count),),
            capacity=size,
            device=self._warp_device,
            copy=False,
        )
        mesh.normals = _SharedNormalBuffer(
            buffer=buffer,
            mapping=mapping,
            warp_array=warp_array,
            accum=wp.empty(int(count), dtype=wp.vec3f, device=self._warp_device),
            counts=wp.empty(int(count), dtype=wp.int32, device=self._warp_device),
            count=int(count),
        )
        return mesh.normals

    def _update_mesh_normals(
        self,
        mesh: _MeshResource,
        name: str,
        particle_q: wp.array,
        surface_indices: wp.array,
    ) -> _SharedNormalBuffer:
        normals = self._shared_normal_buffer(mesh, int(len(particle_q)), name)
        position_source_id = int(particle_q.ptr)
        if (
            normals.last_updated_frame == self._frame_id
            and normals.position_source_id == position_source_id
        ):
            return normals

        # Warp's Windows precompiled-header path uses a temporary directory that
        # can be inaccessible under sandboxed launches. These small kernels are
        # cheap to compile without PCH, and cached after the first launch.
        previous_pch = wp.config.use_precompiled_headers
        wp.config.use_precompiled_headers = False
        try:
            wp.launch(
                _clear_mesh_normal_scratch,
                dim=normals.count,
                inputs=[normals.accum, normals.counts],
                device=self._warp_device,
            )
            triangle_count = mesh.index_count // 3
            if triangle_count > 0:
                wp.launch(
                    _accumulate_mesh_normals,
                    dim=triangle_count,
                    inputs=[surface_indices, particle_q, normals.accum, normals.counts],
                    device=self._warp_device,
                )
            wp.launch(
                _finalize_mesh_normals,
                dim=normals.count,
                inputs=[normals.accum, normals.counts, normals.warp_array],
                device=self._warp_device,
            )
        finally:
            wp.config.use_precompiled_headers = previous_pch
        normals.position_source_id = position_source_id
        normals.last_updated_frame = self._frame_id
        return normals

    def _mesh_resource(self, name: str, indices: wp.array) -> _MeshResource:
        source_id = int(indices.ptr)
        existing = self._meshes.get(name)
        if (
            existing is not None
            and existing.index_count == int(len(indices))
            and existing.index_source_id == source_id
        ):
            return existing
        if existing is not None:
            existing.close()

        host_indices = np.asarray(indices.numpy(), dtype=np.uint32)
        index_buffer = self._device.create_buffer(
            usage=self._spy.BufferUsage.index_buffer | self._spy.BufferUsage.shader_resource,
            label=f"{name}-indices",
            data=host_indices,
        )
        resource = _MeshResource(
            index_buffer=index_buffer,
            index_count=int(host_indices.size),
            index_source_id=source_id,
        )
        self._meshes[name] = resource
        return resource

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warnings:
            return
        self._warnings.add(key)
        print(message)

    def _ensure_uv_buffer(
        self,
        mesh: _MeshResource,
        name: str,
        particle_q: wp.array,
        uvs: wp.array | None,
        texture: str | None,
    ) -> Any | None:
        if texture is None:
            return None
        if uvs is None:
            self._warn_once(
                f"{name}:missing-uvs",
                f"Slang tissue shader fallback for {name}: mesh has a texture but no UV buffer.",
            )
            return None
        if int(len(uvs)) != int(len(particle_q)):
            self._warn_once(
                f"{name}:uv-count:{int(len(uvs))}:{int(len(particle_q))}",
                (
                    f"Slang tissue shader fallback for {name}: UV count {int(len(uvs))} "
                    f"does not match vertex count {int(len(particle_q))}."
                ),
            )
            return None

        uv_source_id = int(uvs.ptr)
        if (
            mesh.uv_buffer is not None
            and mesh.uv_count == int(len(uvs))
            and mesh.uv_source_id == uv_source_id
        ):
            return mesh.uv_buffer

        host_uvs = np.asarray(uvs.numpy(), dtype=np.float32)
        try:
            host_uvs = np.ascontiguousarray(host_uvs.reshape((int(len(uvs)), 2)))
        except ValueError:
            self._warn_once(
                f"{name}:uv-shape:{host_uvs.shape}",
                f"Slang tissue shader fallback for {name}: UV data shape {host_uvs.shape} is not Nx2.",
            )
            return None

        mesh.uv_buffer = self._device.create_buffer(
            usage=self._spy.BufferUsage.vertex_buffer | self._spy.BufferUsage.shader_resource,
            label=f"{name}-uvs",
            data=host_uvs,
        )
        mesh.uv_count = int(len(uvs))
        mesh.uv_source_id = uv_source_id
        return mesh.uv_buffer

    def _ensure_vertex_color_buffer(
        self,
        name: str,
        particle_q: wp.array,
        vertex_colors: wp.array | None,
        texture: str | None,
    ) -> _SharedVertexColorBuffer | None:
        if texture is None:
            return None
        if vertex_colors is None:
            self._warn_once(
                f"{name}:missing-vertex-colors",
                f"Slang tissue shader fallback for {name}: mesh has a texture but no vertex-color buffer.",
            )
            return None
        if int(len(vertex_colors)) != int(len(particle_q)):
            self._warn_once(
                f"{name}:vertex-color-count:{int(len(vertex_colors))}:{int(len(particle_q))}",
                (
                    f"Slang tissue shader fallback for {name}: vertex-color count {int(len(vertex_colors))} "
                    f"does not match vertex count {int(len(particle_q))}."
                ),
            )
            return None

        if vertex_colors.device != self._warp_device:
            vertex_colors = vertex_colors.to(self._warp_device)

        return self._update_shared_vertex_colors(vertex_colors, name)

    def _material_paths(self, diffuse_path: str) -> _TissueMaterialPaths:
        diffuse = Path(diffuse_path)
        if not diffuse.is_absolute():
            diffuse = Path.cwd() / diffuse
        diffuse = diffuse.resolve(strict=False)

        def layer_paths(kind: str) -> _TissueLayerPaths:
            return _TissueLayerPaths(
                base=diffuse.with_name(f"{kind}-base.png") if kind != "diffuse" else diffuse,
                damage=diffuse.with_name(f"{kind}-damage.png"),
                coag=diffuse.with_name(f"{kind}-coag.png"),
                blood=diffuse.with_name(f"{kind}-blood.png"),
            )

        return _TissueMaterialPaths(
            diffuse=layer_paths("diffuse"),
            normal=layer_paths("normal"),
            spec=layer_paths("spec"),
        )

    def _material_cache_key(self, paths: _TissueMaterialPaths) -> tuple[str, ...]:
        return paths.cache_key()

    def _load_texture_data(self, path: Path) -> np.ndarray:
        try:
            from PIL import Image, ImageOps
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow is required to load Slang material textures.") from exc

        with Image.open(path) as image:
            rgba = ImageOps.flip(image.convert("RGBA"))
            return np.array(rgba, dtype=np.uint8, copy=True, order="C")

    def _create_texture_from_data(self, label: str, data: np.ndarray, srgb: bool) -> Any:
        return self._device.create_texture(
            format=self._spy.Format.rgba8_unorm_srgb if srgb else self._spy.Format.rgba8_unorm,
            width=int(data.shape[1]),
            height=int(data.shape[0]),
            usage=self._spy.TextureUsage.shader_resource,
            default_state=self._spy.ResourceState.shader_resource,
            label=label,
            data=data,
        )

    def _default_texture(self, kind: str, srgb: bool) -> Any:
        key = (kind, bool(srgb))
        existing = self._default_textures.get(key)
        if existing is not None:
            return existing

        if kind == "normal":
            pixel = (128, 128, 255, 255)
        elif kind == "spec":
            pixel = (128, 128, 128, 255)
        elif kind in {"black", "mask-black", "blood-mask", "heat-mask"}:
            pixel = (0, 0, 0, 255)
        else:
            pixel = (255, 255, 255, 255)
        data = np.ascontiguousarray(np.asarray([[pixel]], dtype=np.uint8))
        texture = self._create_texture_from_data(f"omnisurg-default-{kind}", data, srgb)
        self._default_textures[key] = texture
        return texture

    def _linear_wrap_sampler(self) -> Any:
        if self._material_sampler is None:
            self._material_sampler = self._device.create_sampler(
                min_filter=self._spy.TextureFilteringMode.linear,
                mag_filter=self._spy.TextureFilteringMode.linear,
                mip_filter=self._spy.TextureFilteringMode.linear,
                address_u=self._spy.TextureAddressingMode.wrap,
                address_v=self._spy.TextureAddressingMode.wrap,
                address_w=self._spy.TextureAddressingMode.wrap,
                label="omnisurg-material-linear-wrap",
            )
        return self._material_sampler

    def _load_material_texture(
        self,
        name: str,
        label: str,
        path: Path,
        srgb: bool,
        required: bool,
        default_kind: str | None = None,
        fallback_texture: Any | None = None,
    ) -> Any | None:
        if not path.exists():
            if required:
                self._warn_once(
                    f"{name}:{label}:missing:{path}",
                    f"Slang tissue shader fallback for {name}: missing required texture {path}.",
                )
                return None
            if fallback_texture is not None:
                return fallback_texture
            default_kind = default_kind or label
            self._warn_once(
                f"{name}:{label}:default:{path}",
                f"Slang tissue shader using default {default_kind} texture for {name}: {path} is missing.",
            )
            return self._default_texture(default_kind, srgb)

        try:
            data = self._load_texture_data(path)
        except Exception as exc:
            if required:
                raise
            if fallback_texture is not None:
                self._warn_once(
                    f"{name}:{label}:load-fallback:{path}",
                    f"Slang tissue shader using fallback {label} texture for {name}: failed to load {path} ({exc}).",
                )
                return fallback_texture
            default_kind = default_kind or label
            self._warn_once(
                f"{name}:{label}:load-default:{path}",
                f"Slang tissue shader using default {default_kind} texture for {name}: failed to load {path} ({exc}).",
            )
            return self._default_texture(default_kind, srgb)
        return self._create_texture_from_data(f"{name}-{label}", data, srgb)

    def _load_layer_textures(
        self,
        name: str,
        kind: str,
        paths: _TissueLayerPaths,
        srgb: bool,
        base_required: bool,
        default_kind: str,
        missing_layer_fallback_to_base: bool = False,
    ) -> _LayerTextureSet | None:
        base = self._load_material_texture(
            name,
            f"{kind}-base",
            paths.base,
            srgb,
            base_required,
            default_kind=default_kind,
        )
        if base is None:
            return None

        def load_layer(layer: str, path: Path) -> Any | None:
            fallback = base if missing_layer_fallback_to_base else None
            return self._load_material_texture(
                name,
                f"{kind}-{layer}",
                path,
                srgb,
                False,
                default_kind=default_kind,
                fallback_texture=fallback,
            )

        return _LayerTextureSet(
            base=base,
            damage=load_layer("damage", paths.damage),
            coag=load_layer("coag", paths.coag),
            blood=load_layer("blood", paths.blood),
        )

    def _material_resource(self, name: str, diffuse_path: str | None) -> _MaterialResource:
        if diffuse_path is None:
            return _MaterialResource(None, None, None, None, None, None, None, False)

        paths = self._material_paths(diffuse_path)
        key = self._material_cache_key(paths)
        existing = self._materials.get(key)
        if existing is not None:
            return existing

        try:
            diffuse = self._load_layer_textures(
                name,
                "diffuse",
                paths.diffuse,
                True,
                True,
                "diffuse",
                missing_layer_fallback_to_base=True,
            )
            if diffuse is None:
                resource = _MaterialResource(None, None, None, None, None, None, None, False)
            else:
                normal = self._load_layer_textures(
                    name,
                    "normal",
                    paths.normal,
                    False,
                    False,
                    "normal",
                )
                spec = self._load_layer_textures(
                    name,
                    "spec",
                    paths.spec,
                    False,
                    False,
                    "spec",
                )
                layer_mask = self._default_texture("mask-white", False)
                resource = _MaterialResource(
                    diffuse=diffuse,
                    normal=normal,
                    spec=spec,
                    layer_masks=_LayerTextureSet(
                        base=layer_mask,
                        damage=layer_mask,
                        coag=layer_mask,
                        blood=layer_mask,
                    ),
                    blood_mask_texture=self._default_texture("blood-mask", False),
                    heat_mask_texture=self._default_texture("heat-mask", False),
                    sampler=self._linear_wrap_sampler(),
                    valid=normal is not None and spec is not None,
                )
        except Exception as exc:
            self._warn_once(
                f"{name}:material-error:{key}",
                f"Slang tissue shader fallback for {name}: material load failed ({exc}).",
            )
            resource = _MaterialResource(None, None, None, None, None, None, None, False)

        self._materials[key] = resource
        return resource

    def _fallback_color(self, name: str) -> tuple[float, float, float]:
        base_name = name.split("_", 1)[0]
        if base_name in _FALLBACK_MESH_COLORS:
            return _FALLBACK_MESH_COLORS[base_name]
        return _FALLBACK_MESH_COLORS.get(name, _FALLBACK_MESH_COLORS["tissue"])

    def _color4(self, color: Any, name: str) -> tuple[float, float, float, float]:
        if isinstance(color, wp.array):
            return (0.8, 0.2, 0.2, 1.0)
        if color is None:
            color = self._fallback_color(name)
        values = np.asarray(color, dtype=np.float32).reshape(-1)
        if values.size < 3:
            values = np.asarray(self._fallback_color(name), dtype=np.float32)
        return (float(values[0]), float(values[1]), float(values[2]), 1.0)

    def _set_shader_uniforms(
        self,
        shader_object: Any,
        color: tuple[float, float, float, float],
        material: _MaterialResource | None = None,
    ) -> None:
        cursor = self._spy.ShaderCursor(shader_object)
        width = max(1, int(self._surface_texture.width))
        height = max(1, int(self._surface_texture.height))
        cursor.camera_pos = self._spy.float3(*self._camera_pos.tolist())
        cursor.camera_right = self._spy.float3(*self._camera_right.tolist())
        cursor.camera_up = self._spy.float3(*self._camera_up.tolist())
        cursor.camera_forward = self._spy.float3(*self._camera_forward.tolist())
        cursor.camera_inv_tan_half_fovy = self._camera_inv_tan_half_fovy
        cursor.camera_aspect = float(width) / float(height)
        cursor.camera_near = self._camera_near
        cursor.camera_far = self._camera_far
        cursor.mesh_color = self._spy.float4(*color)

        key_dir = _normalize_or(
            -self._camera_forward + self._camera_up * 0.42 - self._camera_right * 0.20,
            (0.0, 1.0, 0.0),
        )
        fill_dir = _normalize_or(
            -self._camera_forward - self._camera_up * 0.15 + self._camera_right * 0.55,
            (1.0, 0.0, 0.0),
        )
        back_dir = _normalize_or(
            self._camera_forward + self._camera_up * 0.48,
            (0.0, 1.0, 0.0),
        )
        cursor.light_key_dir = self._spy.float3(*key_dir.tolist())
        cursor.light_fill_dir = self._spy.float3(*fill_dir.tolist())
        cursor.light_back_dir = self._spy.float3(*back_dir.tolist())
        cursor.light_key_color = self._spy.float3(1.00, 0.92, 0.82)
        cursor.light_fill_color = self._spy.float3(0.30, 0.36, 0.46)
        cursor.light_ambient_color = self._spy.float3(0.22, 0.20, 0.18)
        cursor.light_rim_color = self._spy.float3(0.58, 0.46, 0.36)
        cursor.specular_scale = 0.35
        cursor.roughness_bias = 0.45
        cursor.ambient = 0.22
        cursor.rim_strength = 0.12
        if material is not None and material.valid:
            tissue_params = self._tissue_material_params
            cursor.diffuse_base_tex = material.diffuse.base
            cursor.diffuse_damage_tex = material.diffuse.damage
            cursor.diffuse_coag_tex = material.diffuse.coag
            cursor.diffuse_blood_tex = material.diffuse.blood
            cursor.normal_base_tex = material.normal.base
            cursor.normal_damage_tex = material.normal.damage
            cursor.normal_coag_tex = material.normal.coag
            cursor.normal_blood_tex = material.normal.blood
            cursor.spec_base_tex = material.spec.base
            cursor.spec_damage_tex = material.spec.damage
            cursor.spec_coag_tex = material.spec.coag
            cursor.spec_blood_tex = material.spec.blood
            cursor.layer_mask_damage_tex = material.layer_masks.damage
            cursor.layer_mask_coag_tex = material.layer_masks.coag
            cursor.layer_mask_blood_tex = material.layer_masks.blood
            cursor.blood_mask_tex = material.blood_mask_texture
            cursor.heat_mask_tex = material.heat_mask_texture
            cursor.material_sampler = material.sampler
            cursor.normal_strength = tissue_params.normal_strength
            cursor.specular_scale = tissue_params.specular_scale
            cursor.roughness_bias = tissue_params.roughness_bias
            cursor.ambient = tissue_params.ambient
            cursor.rim_strength = tissue_params.rim_strength
            cursor.blend_damage = 1.0
            cursor.blend_coag = 1.0
            cursor.blend_blood = 1.0
            cursor.debug_mode = int(tissue_params.debug_mode)
            cursor.wetness = tissue_params.wetness
            cursor.wet_spec_scale = tissue_params.wet_spec_scale
            cursor.wet_roughness = tissue_params.wet_roughness
            cursor.subsurface_color = self._spy.float3(*tissue_params.subsurface_color)
            cursor.subsurface_strength = tissue_params.subsurface_strength
            cursor.blood_wetness = tissue_params.blood_wetness

    def _draw_buffer(
        self,
        *,
        source: _SharedPositionBuffer,
        pipeline: Any,
        color: tuple[float, float, float, float],
        index_buffer: Any | None = None,
        extra_vertex_buffers: list[Any] | None = None,
        material: _MaterialResource | None = None,
        vertex_count: int = 0,
    ) -> None:
        if self._pass_encoder is None:
            return
        vertex_buffers = [source.buffer]
        if extra_vertex_buffers is not None:
            vertex_buffers.extend(extra_vertex_buffers)

        render_state: dict[str, Any] = {
            "vertex_buffers": vertex_buffers,
            "viewports": [self._viewport],
            "scissor_rects": [self._scissor],
        }
        if index_buffer is not None:
            render_state["index_buffer"] = index_buffer
            render_state["index_format"] = self._spy.IndexFormat.uint32

        self._pass_encoder.set_render_state(render_state)
        shader_object = self._pass_encoder.bind_pipeline(pipeline)
        self._set_shader_uniforms(shader_object, color, material)
        if index_buffer is not None:
            self._pass_encoder.draw_indexed({"vertex_count": int(vertex_count)})
        else:
            self._pass_encoder.draw({"vertex_count": int(vertex_count)})

    def begin_frame(self, time: float) -> None:
        del time
        if self._closed:
            return

        self._window.process_events()
        self._apply_pending_resize()
        if self._window.should_close() or not self._surface.config:
            return

        self._surface_texture = self._surface.acquire_next_image()
        if not self._surface_texture:
            self._surface_texture = None
            return

        depth_texture = self._ensure_depth_texture(
            self._surface_texture.width,
            self._surface_texture.height,
        )
        self._command_encoder = self._device.create_command_encoder()
        self._pass_encoder = self._command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": self._surface_texture.create_view({}),
                        "clear_value": [0.025, 0.030, 0.034, 1.0],
                        "load_op": self._spy.LoadOp.clear,
                        "store_op": self._spy.StoreOp.store,
                    }
                ],
                "depth_stencil_attachment": {
                    "view": depth_texture.create_view({}),
                    "depth_load_op": self._spy.LoadOp.clear,
                    "depth_store_op": self._spy.StoreOp.store,
                    "depth_clear_value": 1.0,
                },
            }
        )

    def end_frame(self) -> None:
        if (
            self._pass_encoder is None
            or self._command_encoder is None
            or self._surface_texture is None
        ):
            self._frame_id += 1
            return

        self._pass_encoder.end()
        self._pass_encoder = None
        self._render_ui(
            int(self._surface_texture.width),
            int(self._surface_texture.height),
        )
        command_buffer = self._command_encoder.finish()
        self._command_encoder = None
        self._device.submit_command_buffer(command_buffer, cuda_stream=self._cuda_stream_handle())
        del self._surface_texture
        self._surface_texture = None
        self._surface.present()
        self._frame_id += 1

    def is_running(self) -> bool:
        return not self._closed and not self._window.should_close()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._device.wait()
        finally:
            for shared in self._shared_positions.values():
                shared.close()
            self._shared_positions.clear()
            for shared in self._shared_vertex_colors.values():
                shared.close()
            self._shared_vertex_colors.clear()
            for mesh in self._meshes.values():
                mesh.close()
            self._meshes.clear()
            self._ui_context = None
            self._ui_adapter = None
            self._ui_windows.clear()
            self._sui = None
            if hasattr(self._window, "close"):
                self._window.close()
            self._device.close()

    def log_scalar(self, name: str, value: float) -> None:
        self._logs[name] = float(value)

    def set_tissue_material_params(self, **params: Any) -> None:
        current = self._tissue_material_params
        for key, value in params.items():
            if not hasattr(current, key):
                raise ValueError(f"Unknown tissue material parameter: {key}")

            if key == "debug_mode":
                value = int(np.clip(int(value), 0, len(TISSUE_DEBUG_MODE_LABELS) - 1))
            elif key == "subsurface_color":
                color = np.asarray(value, dtype=np.float32).reshape(-1)
                if color.size < 3:
                    color = np.asarray(current.subsurface_color, dtype=np.float32)
                value = (
                    float(np.clip(color[0], 0.0, 1.0)),
                    float(np.clip(color[1], 0.0, 1.0)),
                    float(np.clip(color[2], 0.0, 1.0)),
                )
            else:
                value = float(value)
                value_range = _TISSUE_MATERIAL_PARAM_RANGES.get(key)
                if value_range is not None:
                    value = float(np.clip(value, value_range[0], value_range[1]))

            setattr(current, key, value)

    def draw_mesh(
        self,
        name: str,
        particle_q: wp.array,
        surface_indices: wp.array,
        color: tuple[float, float, float] | None = None,
        uvs: wp.array | None = None,
        texture: str | None = None,
        vertex_colors: wp.array | None = None,
    ) -> None:
        if self._pass_encoder is None:
            return

        source = self._update_shared_positions(particle_q, name)
        mesh = self._mesh_resource(name, surface_indices)
        normals = self._update_mesh_normals(mesh, name, particle_q, surface_indices)
        material = self._material_resource(name, texture) if texture is not None else None
        uv_buffer = (
            self._ensure_uv_buffer(mesh, name, particle_q, uvs, texture)
            if material is not None and material.valid
            else None
        )
        vertex_color_buffer = (
            self._ensure_vertex_color_buffer(name, particle_q, vertex_colors, texture)
            if material is not None and material.valid and uv_buffer is not None
            else None
        )
        if material is not None and material.valid and uv_buffer is not None and vertex_color_buffer is not None:
            self._draw_buffer(
                source=source,
                pipeline=self._tissue_triangle_pipeline,
                color=(1.0, 1.0, 1.0, 1.0) if color is None else self._color4(color, name),
                index_buffer=mesh.index_buffer,
                extra_vertex_buffers=[normals.buffer, uv_buffer, vertex_color_buffer.buffer],
                material=material,
                vertex_count=mesh.index_count,
            )
            return

        self._draw_buffer(
            source=source,
            pipeline=self._flat_triangle_pipeline,
            color=self._color4(color, name),
            index_buffer=mesh.index_buffer,
            extra_vertex_buffers=[normals.buffer],
            vertex_count=mesh.index_count,
        )

    def draw_points(
        self,
        name: str,
        points: wp.array,
        radii: wp.array | float,
        colors: wp.array | tuple[float, float, float] | list[float],
    ) -> None:
        del radii
        if self._pass_encoder is None or len(points) == 0:
            return

        source = self._update_shared_positions(points, name)
        self._draw_buffer(
            source=source,
            pipeline=self._point_pipeline,
            color=self._color4(colors, name),
            vertex_count=len(points),
        )

    def set_input_callbacks(self, on_key_press=None, on_key_release=None) -> None:
        self._on_key_press_callback = on_key_press
        self._on_key_release_callback = on_key_release

    def register_ui_callback(self, callback, position: str = "side") -> None:
        if not callable(callback):
            raise TypeError("callback must be callable")
        if position not in self._ui_callbacks:
            valid_positions = list(self._ui_callbacks.keys())
            raise ValueError(
                f"Invalid position '{position}'. Must be one of: {valid_positions}"
            )
        self._ui_callbacks[position].append(callback)
