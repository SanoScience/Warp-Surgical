# SPDX-License-Identifier: Apache-2.0
"""Slang cryo/procedural surface helpers for OmniSurg renderers."""

from __future__ import annotations

import ctypes
import os
import time as _time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import warp as wp

from .material_maker_slang import (
    MaterialMakerParameterSpec,
    MaterialMakerSlangError,
    MaterialMakerSlangInfo,
    material_source_changed,
    texture_path_for,
    write_material_bridge,
)

SLANG_RENDER_BACKENDS = frozenset({"slang", "slang-vulkan", "slang-vk", "slang-d3d12"})
SLANG_SHADER_DIR = Path(__file__).with_name("slang_shaders")
MATERIAL_MAKER_SLANG_WORK_DIR = SLANG_SHADER_DIR.parent.parent / ".cache" / "material_maker_slang"
_SLANG_BIN_ENV = os.environ.get("OMNISURG_SLANG_BIN")
DEFAULT_SLANG_BIN = Path(_SLANG_BIN_ENV) if _SLANG_BIN_ENV else None
PROCEDURAL_MATERIAL_PARAM_NAMES = ("scale", "grain", "tint", "roughness", "wetness", "height", "normal")
PROCEDURAL_MATERIAL_PARAM_DEFAULTS: dict[str, float] = {
    "scale": 48.0,
    "grain": 0.72,
    "tint": 0.62,
    "roughness": 0.55,
    "wetness": 0.14,
    "height": 0.10,
    "normal": 0.60,
}
PROCEDURAL_MATERIAL_PARAM_RANGES: dict[str, tuple[float, float]] = {
    "scale": (1.0, 500.0),
    "grain": (0.0, 1.0),
    "tint": (0.0, 1.0),
    "roughness": (0.0, 1.0),
    "wetness": (0.0, 1.0),
    "height": (0.0, 1.0),
    "normal": (0.0, 1.0),
}
_DYNAMIC_SHADER_BUFFER_RING_SIZE = 4
MATERIAL_MAKER_MAX_MATERIAL_CLASSES = 256
POINT_SPHERE_LATITUDES = 8
POINT_SPHERE_LONGITUDES = 12
SLANG_SURFACE_DEBUG_VIEW_LABELS = (
    "Off",
    "Height",
    "Roughness",
    "Wetness",
    "Material Normal Map",
    "Final Shade Normal",
    "Material Index",
)
SLANG_SURFACE_DEBUG_VIEW_OFF = 0
SLANG_SURFACE_DEBUG_VIEW_HEIGHT = 1
SLANG_SURFACE_DEBUG_VIEW_ROUGHNESS = 2
SLANG_SURFACE_DEBUG_VIEW_WETNESS = 3
SLANG_SURFACE_DEBUG_VIEW_MATERIAL_NORMAL_MAP = 4
SLANG_SURFACE_DEBUG_VIEW_FINAL_SHADE_NORMAL = 5
SLANG_SURFACE_DEBUG_VIEW_MATERIAL_INDEX = 6


def is_slang_backend(backend: str) -> bool:
    return str(backend) in SLANG_RENDER_BACKENDS


def clamp_slang_surface_debug_view(value: Any, *, height_debug: bool | None = None) -> int:
    """Return a valid Slang surface debug view, mapping legacy height-debug state."""
    if value is None:
        return SLANG_SURFACE_DEBUG_VIEW_HEIGHT if height_debug else SLANG_SURFACE_DEBUG_VIEW_OFF
    try:
        debug_view = int(value)
    except (TypeError, ValueError):
        debug_view = SLANG_SURFACE_DEBUG_VIEW_HEIGHT if height_debug else SLANG_SURFACE_DEBUG_VIEW_OFF
    return int(np.clip(debug_view, 0, len(SLANG_SURFACE_DEBUG_VIEW_LABELS) - 1))


def clamp_procedural_material_params(values: dict[str, Any] | None = None) -> dict[str, float]:
    """Return a complete, finite procedural material parameter dictionary."""
    values = {} if values is None else dict(values)
    clamped: dict[str, float] = {}
    for name in PROCEDURAL_MATERIAL_PARAM_NAMES:
        raw = values.get(name, PROCEDURAL_MATERIAL_PARAM_DEFAULTS[name])
        try:
            value = float(raw)
        except (TypeError, ValueError):
            value = PROCEDURAL_MATERIAL_PARAM_DEFAULTS[name]
        if not np.isfinite(value):
            value = PROCEDURAL_MATERIAL_PARAM_DEFAULTS[name]
        lo, hi = PROCEDURAL_MATERIAL_PARAM_RANGES[name]
        clamped[name] = float(np.clip(value, lo, hi))
    return clamped


def make_default_procedural_materials(count: int) -> list[dict[str, float]]:
    """Build default procedural settings for ``count`` material slots."""
    return [clamp_procedural_material_params() for _ in range(max(0, int(count)))]


def build_procedural_uv3_noise_scale(grid_shape: object) -> tuple[float, float, float]:
    """Return aspect correction for normalized UV3 noise sampling.

    UV3 is normalized independently along each grid axis, so sampling noise
    directly in UV3 space stretches features along the longest physical/data
    axis. Multiplying by extent/max_extent keeps one global scale control while
    making the noise cell size uniform in grid units.
    """
    try:
        shape = np.asarray(grid_shape, dtype=np.float64).reshape(-1)
    except Exception:
        return (1.0, 1.0, 1.0)
    if shape.size < 3:
        return (1.0, 1.0, 1.0)
    extents = shape[:3] - 1.0
    extents = np.where(np.isfinite(extents), extents, 1.0)
    extents = np.maximum(extents, 1.0)
    max_extent = float(np.max(extents))
    if not np.isfinite(max_extent) or max_extent <= 0.0:
        return (1.0, 1.0, 1.0)
    scaled = extents / max_extent
    return (float(scaled[0]), float(scaled[1]), float(scaled[2]))


def build_procedural_material_param_buffer(
    materials: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    material_count: int | None = None,
) -> np.ndarray:
    """Pack procedural material controls as two ``float4`` rows per material."""
    if material_count is None:
        material_count = len(materials) if materials is not None else 1
    count = max(1, int(material_count))
    packed = np.zeros((count * 2, 4), dtype=np.float32)
    source = [] if materials is None else list(materials)
    for idx in range(count):
        values = source[idx] if idx < len(source) and isinstance(source[idx], dict) else None
        params = clamp_procedural_material_params(values)
        packed[idx * 2 + 0] = (
            params["scale"],
            params["grain"],
            params["tint"],
            params["roughness"],
        )
        packed[idx * 2 + 1] = (
            params["wetness"],
            params["height"],
            params["normal"],
            0.0,
        )
    return np.ascontiguousarray(packed)


def _sanitize_untyped_material_maker_params(values: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(values, dict):
        return {}
    result: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, (list, tuple)):
            components: list[float] = []
            for component in list(value)[:4]:
                try:
                    number = float(component)
                except (TypeError, ValueError):
                    number = 0.0
                if not np.isfinite(number):
                    number = 0.0
                components.append(number)
            if components:
                result[str(key)] = components
        else:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(number):
                result[str(key)] = number
    return result


def material_maker_parameter_row(
    spec: MaterialMakerParameterSpec,
    value: Any | None = None,
    *,
    clamp: bool = True,
) -> tuple[float, float, float, float]:
    row = list(spec.default_row)
    if value is not None:
        if spec.value_type == "vec4" and isinstance(value, (list, tuple, np.ndarray)):
            for idx, component in enumerate(list(value)[:4]):
                try:
                    row[idx] = float(component)
                except (TypeError, ValueError):
                    pass
        elif spec.value_type == "float":
            try:
                row[0] = float(value)
            except (TypeError, ValueError):
                pass
    row = [component if np.isfinite(component) else spec.default_row[idx] for idx, component in enumerate(row)]
    if clamp:
        row = [float(np.clip(component, spec.min_value, spec.max_value)) for component in row]
    return (float(row[0]), float(row[1]), float(row[2]), float(row[3]))


def material_maker_setting_value(spec: MaterialMakerParameterSpec, row: tuple[float, float, float, float]) -> Any:
    if spec.value_type == "vec4":
        return [float(row[0]), float(row[1]), float(row[2]), float(row[3])]
    return float(row[0])


def clamp_material_maker_params(
    values: dict[str, Any] | None,
    specs: tuple[MaterialMakerParameterSpec, ...] | list[MaterialMakerParameterSpec] = (),
) -> dict[str, Any]:
    if not specs:
        return _sanitize_untyped_material_maker_params(values)
    source = values if isinstance(values, dict) else {}
    return {
        spec.raw_name: material_maker_setting_value(spec, material_maker_parameter_row(spec, source.get(spec.raw_name)))
        for spec in specs
    }


def make_default_material_maker_params(
    count: int,
    specs: tuple[MaterialMakerParameterSpec, ...] | list[MaterialMakerParameterSpec],
) -> list[dict[str, Any]]:
    return [clamp_material_maker_params(None, specs) for _ in range(max(0, int(count)))]


def build_material_maker_param_buffer(
    specs: tuple[MaterialMakerParameterSpec, ...] | list[MaterialMakerParameterSpec],
    materials: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
    *,
    material_count: int | None = None,
    max_material_classes: int = MATERIAL_MAKER_MAX_MATERIAL_CLASSES,
) -> np.ndarray:
    ordered_specs = tuple(sorted(specs, key=lambda spec: spec.row_index))
    if material_count is None:
        material_count = len(materials) if materials is not None else 1
    class_count = max(1, min(int(max_material_classes), int(material_count)))
    packed = np.zeros((len(ordered_specs) * int(max_material_classes), 4), dtype=np.float32)
    source = [] if materials is None else list(materials)
    for param_idx, spec in enumerate(ordered_specs):
        default_row = material_maker_parameter_row(spec)
        start = param_idx * int(max_material_classes)
        packed[start : start + int(max_material_classes)] = default_row
        for class_idx in range(class_count):
            values = source[class_idx] if class_idx < len(source) and isinstance(source[class_idx], dict) else None
            if values is not None and spec.raw_name in values:
                packed[start + class_idx] = material_maker_parameter_row(spec, values.get(spec.raw_name))
    return np.ascontiguousarray(packed)


def build_unit_sphere_mesh(
    latitudes: int = POINT_SPHERE_LATITUDES,
    longitudes: int = POINT_SPHERE_LONGITUDES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a compact unit-sphere mesh for instanced point rendering."""
    latitudes = max(3, int(latitudes))
    longitudes = max(3, int(longitudes))
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]
    for lat in range(1, latitudes):
        theta = np.pi * float(lat) / float(latitudes)
        z = float(np.cos(theta))
        r = float(np.sin(theta))
        for lon in range(longitudes):
            phi = 2.0 * np.pi * float(lon) / float(longitudes)
            vertices.append((r * float(np.cos(phi)), r * float(np.sin(phi)), z))
    south = len(vertices)
    vertices.append((0.0, 0.0, -1.0))

    indices: list[int] = []
    first_ring = 1
    for lon in range(longitudes):
        indices.extend([0, first_ring + ((lon + 1) % longitudes), first_ring + lon])

    ring_count = latitudes - 1
    for ring in range(ring_count - 1):
        a = 1 + ring * longitudes
        b = a + longitudes
        for lon in range(longitudes):
            next_lon = (lon + 1) % longitudes
            indices.extend([a + lon, b + lon, a + next_lon])
            indices.extend([a + next_lon, b + lon, b + next_lon])

    last_ring = 1 + (ring_count - 1) * longitudes
    for lon in range(longitudes):
        indices.extend([south, last_ring + lon, last_ring + ((lon + 1) % longitudes)])

    return np.asarray(vertices, dtype=np.float32), np.asarray(indices, dtype=np.uint32)


def _is_warp_array(value: Any) -> bool:
    return value is not None and hasattr(value, "ptr") and hasattr(value, "device") and hasattr(value, "dtype")


def _is_warp_vec3_array(value: Any) -> bool:
    if not _is_warp_array(value):
        return False
    dtype_text = str(getattr(value, "dtype", ""))
    return "vec3" in dtype_text


def _is_warp_float_array(value: Any) -> bool:
    if not _is_warp_array(value):
        return False
    dtype_text = str(getattr(value, "dtype", ""))
    return "float" in dtype_text and "vec" not in dtype_text


def _rgb_tuple_or(value: Any, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if value is None:
        return fallback
    try:
        return (float(value.x), float(value.y), float(value.z))
    except Exception:
        pass
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except Exception:
        return fallback


def _float_or(value: Any, fallback: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    return result if np.isfinite(result) else fallback


def build_material_color_buffer(
    colors: list[tuple[float, float, float]] | np.ndarray | None,
    *,
    material_count: int | None = None,
) -> np.ndarray:
    """Pack material palette colors as clamped sRGB ``float4`` rows."""
    source = np.asarray([[1.0, 1.0, 1.0]] if colors is None else colors, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] < 3:
        raise ValueError(f"expected material colors with shape (n, >=3), got {source.shape}")
    if material_count is None:
        material_count = int(source.shape[0])
    count = max(1, int(material_count))
    packed = np.ones((count, 4), dtype=np.float32)
    copy_count = min(count, int(source.shape[0]))
    rgb = source[:copy_count, :3]
    rgb = np.where(np.isfinite(rgb), rgb, 1.0)
    packed[:copy_count, :3] = np.clip(rgb, 0.0, 1.0)
    return np.ascontiguousarray(packed)


def _clamp_unit_float(value: float, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    if not np.isfinite(result):
        result = float(default)
    return float(np.clip(result, 0.0, 1.0))


def _finite_float_or(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = float(default)
    return result if np.isfinite(result) else float(default)


def prepare_cryo_texture_upload(host_volume: np.ndarray) -> np.ndarray:
    """Return Slang upload data as contiguous ``(nz, ny, nx, 4)`` RGBA8."""
    host = np.asarray(host_volume)
    if host.ndim != 4 or host.shape[-1] != 3 or host.dtype != np.uint8:
        raise ValueError(
            f"expected cryo host volume (nx, ny, nz, 3) uint8, got shape={host.shape} dtype={host.dtype}"
        )
    transposed = np.ascontiguousarray(host.transpose(2, 1, 0, 3))
    alpha = np.full((*transposed.shape[:3], 1), 255, dtype=np.uint8)
    return np.ascontiguousarray(np.concatenate((transposed, alpha), axis=-1))


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
            "Slang viewer requires slangpy. Install it in this environment, and keep the "
            "Slang binary folder on PATH or set OMNISURG_SLANG_BIN. On Linux Vulkan, also "
            "ensure the NVIDIA driver, CUDA, Vulkan loader, and Slang shared libraries are visible."
        ) from exc
    return spy


class _CudaWin32Handle(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_void_p), ("name", ctypes.c_void_p)]


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
    def __init__(self, cuda: _CudaDriver, external_memory: ctypes.c_void_p, ptr: int, context: int):
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


def _close_slang_resource(resource: Any) -> None:
    for method_name in ("close", "destroy", "release"):
        method = getattr(resource, method_name, None)
        if not callable(method):
            continue
        try:
            method()
        except Exception:
            pass
        return


def _shared_buffer_cache_key(source: Any, dtype: Any, generation: int | None = None) -> tuple[int, int, str, int]:
    cache_generation = -1 if generation is None else int(generation)
    return (int(source.ptr), int(len(source)), str(dtype), cache_generation)


@dataclass
class _MaterialTextureBinding:
    path: Path
    mtime_ns: int
    texture: Any


@dataclass
class _ExternalMaterialReloadResult:
    changed: bool
    program: Any | None = None
    pipeline: Any | None = None
    info: MaterialMakerSlangInfo | None = None


@dataclass
class _SharedBuffer:
    buffer: Any
    mapping: _CudaMappedExternalMemory
    warp_array: wp.array
    count: int
    dtype: Any
    last_updated_frame: int = -1

    def close(self) -> None:
        try:
            self.mapping.close()
        finally:
            _close_slang_resource(self.buffer)


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


class _UiImVec4:
    def __init__(self, x: float = 0.0, y: float = 0.0, z: float = 0.0, w: float = 0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.w = float(w)


class _SlangImmediateUi:
    ImVec2 = _UiImVec2
    ImVec4 = _UiImVec4
    Cond_ = SimpleNamespace(appearing=_UiEnumValue(1), once=_UiEnumValue(2), always=_UiEnumValue(4))
    WindowFlags_ = SimpleNamespace(
        no_resize=_UiEnumValue(1),
        no_decoration=_UiEnumValue(2),
        always_auto_resize=_UiEnumValue(4),
        no_saved_settings=_UiEnumValue(8),
        no_focus_on_appearing=_UiEnumValue(16),
        no_nav=_UiEnumValue(32),
        no_move=_UiEnumValue(64),
        no_collapse=_UiEnumValue(128),
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
        self._id_stack: list[str] = []
        self._next_window_pos: tuple[_UiImVec2, int] | None = None
        self._next_window_size: tuple[_UiImVec2, int] | None = None
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
        self._id_stack = []
        self._next_window_pos = None
        self._next_window_size = None
        self.io.display_size = (float(width), float(height))

    @property
    def _parent(self) -> Any:
        return self._parent_stack[-1]

    def _key(self, kind: str, label: str) -> tuple[str, str, int]:
        id_scope = "/".join(self._id_stack)
        base = (f"{id(self._parent)}:{id_scope}:{kind}", str(label))
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

    def _float2(self, value: _UiImVec2 | None, fallback: tuple[float, float]):
        if value is None:
            x, y = fallback
        else:
            x = getattr(value, "x", fallback[0])
            y = getattr(value, "y", fallback[1])
        return self._spy.float2(float(x), float(y))

    def _rgb_tuple(self, value: Any, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
        try:
            x = value.x
            y = value.y
            z = value.z
            return (float(x), float(y), float(z))
        except Exception:
            pass
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except Exception:
            return fallback

    def _float3(self, value: Any, fallback: tuple[float, float, float]):
        x, y, z = self._rgb_tuple(value, fallback)
        return self._spy.float3(float(x), float(y), float(z))

    def _condition_value(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
        if "cond" in kwargs:
            cond = kwargs["cond"]
        elif args:
            cond = args[0]
        else:
            cond = self.Cond_.always.value
        try:
            value = int(getattr(cond, "value", cond))
        except Exception:
            value = int(self.Cond_.always.value)
        return int(self.Cond_.always.value) if value == 0 else value

    def _should_apply_window_condition(self, condition: int, is_new_window: bool) -> bool:
        if condition & int(self.Cond_.always.value):
            return True
        if condition & int(self.Cond_.once.value):
            return is_new_window
        if condition & int(self.Cond_.appearing.value):
            return is_new_window
        return is_new_window

    def text(self, text: object) -> None:
        record, _changed, _current = self._record("text", "", str(text))
        widget = self._sync_widget(record)
        if widget is None:
            record.widget = self._sui.Text(self._parent, str(text))
        else:
            widget.text = str(text)

    def text_colored(self, _color: _UiImVec4, text: object) -> None:
        self.text(text)

    def separator(self) -> None:
        record, _changed, _current = self._record("separator", "", "")
        widget = self._sync_widget(record)
        if widget is None:
            record.widget = self._sui.Text(self._parent, "------------------------------")

    def spacing(self) -> None:
        self.text("")

    def set_next_window_pos(self, pos: _UiImVec2, *args, **kwargs) -> None:
        self._next_window_pos = (pos, self._condition_value(args, kwargs))

    def set_next_window_size(self, size: _UiImVec2, *args, **kwargs) -> None:
        self._next_window_size = (size, self._condition_value(args, kwargs))

    def begin(self, title: str, *args, **kwargs) -> bool:
        del args, kwargs
        record, _changed, _current = self._record("window", title, str(title))
        window = self._sync_widget(record)
        position_spec = self._next_window_pos
        size_spec = self._next_window_size
        position = self._float2(None if position_spec is None else position_spec[0], (460.0, 10.0))
        size = self._float2(None if size_spec is None else size_spec[0], (420.0, 360.0))
        if window is None:
            window = self._sui.Window(self._screen, str(title), position=position, size=size)
            record.widget = window
        else:
            window.title = str(title)
            if position_spec is not None and self._should_apply_window_condition(position_spec[1], False):
                window.position = position
            if size_spec is not None and self._should_apply_window_condition(size_spec[1], False):
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

    def slider_float(self, label: str, value: float, min_value: float, max_value: float, fmt: str = "%.3f", *args, **kwargs):
        del args, kwargs
        record, changed, current = self._record("slider_float", label, float(value))

        def _callback(updated: float, rec=record) -> None:
            rec.value = float(updated)
            rec.changed = True

        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.SliderFloat(
                self._parent, str(label), float(current), _callback, float(min_value), float(max_value), str(fmt)
            )
            record.widget = widget
        else:
            widget.label = str(label)
            widget.value = float(current)
            widget.callback = _callback
            widget.min = float(min_value)
            widget.max = float(max_value)
        return changed, float(current)

    def slider_int(self, label: str, value: int, min_value: int, max_value: int, *args, **kwargs):
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

    def color_edit3(self, label: str, r: Any, g: Any = None, b: Any = None, *args, **kwargs):
        del args, kwargs
        value = r if g is None and b is None else (r, g, b)
        fallback = self._rgb_tuple(value, (1.0, 1.0, 1.0))
        record, changed, current = self._record("color_edit3", label, fallback)

        def _callback(updated: Any, rec=record) -> None:
            rec.value = self._rgb_tuple(updated, fallback)
            rec.changed = True

        current_tuple = self._rgb_tuple(current, fallback)
        widget = self._sync_widget(record)
        if widget is None:
            widget = self._sui.SliderFloat3(
                self._parent,
                str(label),
                self._float3(current_tuple, fallback),
                _callback,
                0.0,
                1.0,
                "%.2f",
            )
            record.widget = widget
        else:
            widget.label = str(label)
            widget.value = self._float3(current_tuple, fallback)
            widget.callback = _callback
            widget.min = 0.0
            widget.max = 1.0
            widget.format = "%.2f"
        return changed, tuple(float(np.clip(v, 0.0, 1.0)) for v in current_tuple)

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

    def same_line(self, *args, **kwargs) -> None:
        del args, kwargs

    def push_id(self, *args, **kwargs) -> None:
        del kwargs
        self._id_stack.append(":".join(str(arg) for arg in args))

    def pop_id(self, *args, **kwargs) -> None:
        del args, kwargs
        if self._id_stack:
            self._id_stack.pop()

    def collapsing_header(self, *args, **kwargs) -> bool:
        del args, kwargs
        return True

    def set_next_item_open(self, *args, **kwargs) -> None:
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


class _SlangUiState:
    def __init__(self, viewer: SlangCryoMixin):
        self._viewer = viewer
        self.io = SimpleNamespace(display_size=(0.0, 0.0))

    @property
    def is_available(self) -> bool:
        return bool(self._viewer._ui_enabled)

    def is_capturing(self) -> bool:
        return bool(self._viewer._ui_capturing or getattr(self._viewer, "_ui_mouse_active", False))


def _normalize_or(value: np.ndarray, fallback: tuple[float, float, float]) -> np.ndarray:
    length = float(np.linalg.norm(value))
    if length <= 1.0e-6:
        return np.array(fallback, dtype=np.float32)
    return (value / length).astype(np.float32)


def _rotate_vector_axis_angle(value: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize_or(axis, (0.0, 0.0, 1.0))
    vector = np.asarray(value, dtype=np.float32)
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    return (vector * c + np.cross(axis, vector) * s + axis * float(np.dot(axis, vector)) * (1.0 - c)).astype(
        np.float32
    )


class SlangCryoMixin:
    """Cryo/procedural surface rendering methods mixed into ``SlangRenderer``."""

    supports_mouse_interaction = True

    def __init__(
        self,
        *,
        backend: str,
        device: wp.context.Device,
        camera_pos: tuple[float, float, float],
        vsync: bool,
        procedural_material_path: str | Path | None = None,
        procedural_material_scale: float = 1.0,
        procedural_material_hot_reload: bool = True,
        environment_path: str | Path | None = None,
        environment_intensity: float = 1.0,
        environment_background: bool = True,
        environment_rotation_degrees: float = 0.0,
        environment_pitch_degrees: float = -90.0,
    ):
        raise TypeError("SlangCryoMixin is not a standalone renderer.")
        if not device.is_cuda:
            raise RuntimeError("Slang viewer requires a CUDA Warp device for shared-buffer interop.")
        self._spy = _load_slangpy()
        self._warp_device = device
        self._backend = str(backend)
        self._vsync = bool(vsync)
        self._procedural_material_path = (
            Path(procedural_material_path).expanduser().resolve() if procedural_material_path else None
        )
        self._procedural_material_scale = max(float(procedural_material_scale), 0.000001)
        self._procedural_material_hot_reload = bool(procedural_material_hot_reload)
        self._environment_path = Path(environment_path).expanduser().resolve() if environment_path else None
        self._environment_intensity = max(float(environment_intensity), 0.0)
        self._environment_background_enabled = bool(environment_background)
        self._environment_rotation_degrees = _finite_float_or(environment_rotation_degrees, 0.0)
        self._environment_pitch_degrees = _finite_float_or(environment_pitch_degrees, -90.0)
        self._procedural_material_info: MaterialMakerSlangInfo | None = None
        self._procedural_material_textures: dict[str, _MaterialTextureBinding] = {}
        self._procedural_material_reload_error: str | None = None
        self._procedural_material_reload_status = ""
        self._external_material_reload_executor: ThreadPoolExecutor | None = None
        self._external_material_reload_future: Future[_ExternalMaterialReloadResult] | None = None
        self._time = 0.0
        self._cuda = _cuda_driver()
        self._cuda.set_current_context(int(device.context))
        self._device_type = self._device_type_from_backend(self._backend)
        include_paths = [SLANG_SHADER_DIR, MATERIAL_MAKER_SLANG_WORK_DIR]
        if self._procedural_material_path is not None:
            include_paths.append(self._procedural_material_path.parent)
        self._device = self._spy.create_device(
            self._device_type,
            include_paths=include_paths,
            enable_cuda_interop=True,
            existing_device_handles=self._spy.get_cuda_current_context_native_handles(),
        )
        if not bool(getattr(self._device, "supports_cuda_interop", False)):
            raise RuntimeError(f"Slang {backend} device does not support CUDA interop.")

        self._window = self._spy.Window(width=1600, height=1000, title=f"OmniSurg Hex {backend}", resizable=True)
        self._surface = self._device.create_surface(self._window)
        self._surface.configure(width=self._window.width, height=self._window.height, vsync=self._vsync)
        self._surface_texture = None
        self._command_encoder = None
        self._pass_encoder = None
        self._scene_color_texture = None
        self._depth_texture = None
        self._point_sphere_vertex_buffer = None
        self._point_sphere_index_buffer = None
        self._point_sphere_index_count = 0
        self._last_submit_id: int | None = None
        self._last_submit_frame = -1
        self._pending_resize: tuple[int, int] | None = None
        self._closed = False
        self._paused = True
        self.show_ui = True
        self.show_particles = False
        self.show_springs = False
        self.renderer = self
        self.ui = _SlangUiState(self)
        self.objects: dict[str, Any] = {}
        self._warnings: set[str] = set()
        self._frame_id = 0
        self._shared_buffers: dict[tuple[int, int, str, int], _SharedBuffer] = {}
        self._host_index_buffers: dict[tuple[int, int], Any] = {}
        self._cryo_texture = None
        self._cryo_source_key: tuple[int, tuple[int, ...], str] | None = None
        self._neutral_cryo_texture = None
        self._linear_sampler = None
        self._fallback_shader_buffers: dict[str, Any] = {}
        self._material_color_buffer = None
        self._material_color_buffer_key: tuple[Any, ...] | None = None
        self._material_color_buffer_ring: list[Any] = []
        self._material_color_buffer_shape_key: tuple[Any, ...] | None = None
        self._material_color_buffer_index = -1
        self._procedural_param_buffer = None
        self._procedural_param_buffer_key: tuple[Any, ...] | None = None
        self._procedural_param_buffer_ring: list[Any] = []
        self._procedural_param_buffer_shape_key: tuple[Any, ...] | None = None
        self._procedural_param_buffer_index = -1
        self._material_maker_param_buffer = None
        self._material_maker_param_buffer_key: tuple[Any, ...] | None = None
        self._material_maker_param_buffer_ring: list[Any] = []
        self._material_maker_param_buffer_shape_key: tuple[Any, ...] | None = None
        self._material_maker_param_buffer_index = -1
        self._environment_texture = None
        self._environment_texture_key: tuple[Path, int] | None = None
        self._neutral_environment_texture = None
        self._retired_shader_buffers: list[Any] = []

        self._ui_enabled = True
        self._ui_capturing = False
        self._ui_mouse_active = False
        self._sui = None
        self._ui_context = None
        self._ui_adapter: _SlangImmediateUi | None = None
        self._ui_windows: dict[str, Any] = {}
        self._ui_callbacks: dict[str, list[Callable[[Any], None]]] = {"side": [], "free": [], "stats": [], "panel": []}
        self._failed_ui_callbacks: set[tuple[str, int]] = set()

        self._key_handler: dict[int, bool] = {}
        self._on_key_press_callback: Callable[[int, int], None] | None = None
        self._on_key_release_callback: Callable[[int, int], None] | None = None
        self._on_mouse_motion_callback: Callable[..., None] | None = None
        self._on_mouse_press_callback: Callable[..., None] | None = None
        self._on_mouse_drag_callback: Callable[..., None] | None = None
        self._on_mouse_release_callback: Callable[..., None] | None = None
        self._mouse_buttons = 0
        self._mouse_pos: tuple[float, float] | None = None
        self._camera_key_state: set[str] = set()
        self._last_camera_update_time = _time.perf_counter()

        self._camera_target = np.zeros(3, dtype=np.float32)
        self._camera_scene_radius = 0.25
        self._camera_world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        self._camera_inv_tan_half_fovy = float(1.0 / np.tan(np.deg2rad(45.0) * 0.5))
        self._set_camera_look_at(np.array(camera_pos, dtype=np.float32), self._camera_target)

        self._window.on_resize = self._on_resize
        self._window.on_keyboard_event = self._on_keyboard_event
        self._window.on_mouse_event = self._on_mouse_event

        self._color_format = self._surface.info.preferred_format
        self._scene_color_format = self._hdr_scene_color_format()
        self._depth_format = self._spy.Format.d32_float
        self._viewport = self._spy.Viewport.from_size(self._window.width, self._window.height)
        self._scissor = self._spy.ScissorRect.from_size(self._window.width, self._window.height)
        self._init_ui()
        self._init_pipeline()

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warnings:
            return
        self._warnings.add(key)
        print(message)

    def _device_type_from_backend(self, backend: str):
        if backend == "slang":
            return self._spy.DeviceType.d3d12 if os.name == "nt" else self._spy.DeviceType.vulkan
        if backend == "slang-d3d12":
            if os.name != "nt":
                raise RuntimeError("slang-d3d12 is only available on Windows; use slang-vulkan on Linux.")
            return self._spy.DeviceType.d3d12
        if backend in {"slang-vulkan", "slang-vk"}:
            return self._spy.DeviceType.vulkan
        raise RuntimeError(f"Unsupported Slang backend: {backend}")

    def _hdr_scene_color_format(self):
        for name in ("rgba16_float", "rgba32_float", "rgba8_unorm"):
            fmt = getattr(self._spy.Format, name, None)
            if fmt is not None:
                return fmt
        return self._color_format

    def _cuda_external_memory_handle_type(self) -> int:
        if self._device_type == self._spy.DeviceType.d3d12:
            return self._cuda.CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE
        if self._device_type == self._spy.DeviceType.vulkan:
            if os.name != "nt":
                return self._cuda.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD
            return self._cuda.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
        raise RuntimeError(f"Unsupported Slang CUDA interop device type: {self._device_type}")

    def _cuda_stream_handle(self):
        return self._spy.NativeHandle.from_cuda_stream(self._cuda_stream_ptr())

    def _cuda_stream_ptr(self) -> int:
        stream = wp.get_stream(self._warp_device)
        return int(stream.cuda_stream)

    def _sync_cuda_to_graphics(self) -> None:
        sync_to_cuda = getattr(self._device, "sync_to_cuda", None)
        if callable(sync_to_cuda):
            sync_to_cuda(self._cuda_stream_ptr())

    def _shared_buffer(
        self,
        source: wp.array,
        label: str,
        dtype: Any,
        stride: int,
        usage: Any,
        *,
        generation: int | None = None,
    ) -> _SharedBuffer:
        key = _shared_buffer_cache_key(source, dtype, generation)
        shared = self._shared_buffers.get(key)
        if shared is not None:
            return shared
        count = int(len(source))
        size = max(1, count * int(stride))
        buffer = None
        mapping = None
        external_memory = ctypes.c_void_p()
        try:
            buffer = self._device.create_buffer(
                size=size,
                usage=(
                    self._spy.BufferUsage.shared
                    | usage
                    | self._spy.BufferUsage.shader_resource
                    | self._spy.BufferUsage.copy_destination
                ),
                label=f"{label}-shared",
            )
            self._cuda.set_current_context(int(self._warp_device.context))
            shared_handle = int(buffer.shared_handle.value)
            if shared_handle == 0:
                raise RuntimeError(f"Slang shared buffer handle creation failed for {label}.")
            external_memory = self._cuda.import_external_memory(
                self._cuda_external_memory_handle_type(),
                shared_handle,
                size,
            )
            ptr = self._cuda.map_external_memory(external_memory, size)
            mapping = _CudaMappedExternalMemory(self._cuda, external_memory, ptr, self._cuda.current_context())
            warp_array = wp.array(ptr=ptr, dtype=dtype, shape=(count,), capacity=size, device=self._warp_device, copy=False)
        except Exception:
            if mapping is not None:
                try:
                    mapping.close()
                except Exception:
                    pass
            elif external_memory:
                try:
                    self._cuda.destroy_external_memory(external_memory)
                except Exception:
                    pass
            if buffer is not None:
                _close_slang_resource(buffer)
            raise
        shared = _SharedBuffer(buffer=buffer, mapping=mapping, warp_array=warp_array, count=count, dtype=dtype)
        self._shared_buffers[key] = shared
        return shared

    def _update_shared_buffer(
        self,
        source: wp.array,
        label: str,
        dtype: Any,
        stride: int,
        usage: Any,
        *,
        generation: int | None = None,
    ) -> _SharedBuffer:
        if source.device != self._warp_device:
            source = source.to(self._warp_device)
        shared = self._shared_buffer(source, label, dtype, stride, usage, generation=generation)
        if shared.last_updated_frame != self._frame_id:
            wp.copy(shared.warp_array, source)
            shared.last_updated_frame = self._frame_id
        return shared

    def _index_buffer(self, indices: wp.array, label: str):
        key = (int(indices.ptr), int(len(indices)))
        existing = self._host_index_buffers.get(key)
        if existing is not None:
            return existing
        host_indices = np.asarray(indices.numpy(), dtype=np.uint32)
        buffer = self._device.create_buffer(
            usage=self._spy.BufferUsage.index_buffer | self._spy.BufferUsage.shader_resource,
            label=f"{label}-indices",
            data=host_indices,
        )
        self._host_index_buffers[key] = buffer
        return buffer

    def _sampler(self):
        if self._linear_sampler is None:
            self._linear_sampler = self._device.create_sampler(
                min_filter=self._spy.TextureFilteringMode.linear,
                mag_filter=self._spy.TextureFilteringMode.linear,
                mip_filter=self._spy.TextureFilteringMode.linear,
                address_u=self._spy.TextureAddressingMode.clamp_to_edge,
                address_v=self._spy.TextureAddressingMode.clamp_to_edge,
                address_w=self._spy.TextureAddressingMode.clamp_to_edge,
                label="hex-linear-clamp",
            )
        return self._linear_sampler

    def _texture_usage_shader_resource(self):
        return self._spy.TextureUsage.shader_resource

    def _cryo_texture_resource(self, host_volume: np.ndarray):
        source_key = (id(host_volume), tuple(int(v) for v in host_volume.shape), str(host_volume.dtype))
        if self._cryo_texture is not None and self._cryo_source_key == source_key:
            return self._cryo_texture
        upload = prepare_cryo_texture_upload(host_volume)
        nz, ny, nx = (int(v) for v in upload.shape[:3])
        kwargs = {
            "type": self._spy.TextureType.texture_3d,
            "format": self._spy.Format.rgba8_unorm,
            "width": nx,
            "height": ny,
            "usage": self._texture_usage_shader_resource(),
            "default_state": self._spy.ResourceState.shader_resource,
            "label": "hex-cryo-volume",
            "data": upload,
        }
        try:
            self._cryo_texture = self._device.create_texture(depth=nz, **kwargs)
        except TypeError:
            kwargs["depth"] = nz
            self._cryo_texture = self._device.create_texture(**kwargs)
        self._cryo_source_key = source_key
        return self._cryo_texture

    def _neutral_cryo_texture_resource(self):
        if self._neutral_cryo_texture is not None:
            return self._neutral_cryo_texture
        upload = prepare_cryo_texture_upload(np.full((1, 1, 1, 3), 128, dtype=np.uint8))
        kwargs = {
            "type": self._spy.TextureType.texture_3d,
            "format": self._spy.Format.rgba8_unorm,
            "width": 1,
            "height": 1,
            "usage": self._texture_usage_shader_resource(),
            "default_state": self._spy.ResourceState.shader_resource,
            "label": "hex-neutral-cryo-volume",
            "data": upload,
        }
        try:
            self._neutral_cryo_texture = self._device.create_texture(depth=1, **kwargs)
        except TypeError:
            kwargs["depth"] = 1
            self._neutral_cryo_texture = self._device.create_texture(**kwargs)
        return self._neutral_cryo_texture

    def _shader_buffer_usage(self):
        usage = self._spy.BufferUsage.shader_resource
        copy_dst = getattr(self._spy.BufferUsage, "copy_destination", None)
        if copy_dst is not None:
            usage = usage | copy_dst
        return usage

    def _static_shader_buffer(self, data: np.ndarray, label: str):
        return self._device.create_buffer(
            usage=self._shader_buffer_usage(),
            label=label,
            data=np.ascontiguousarray(data),
        )

    def _update_static_shader_buffer(self, buffer: Any, data: np.ndarray) -> bool:
        copy_from_numpy = getattr(buffer, "copy_from_numpy", None)
        if not callable(copy_from_numpy):
            return False
        try:
            copy_from_numpy(np.ascontiguousarray(data))
        except Exception:
            return False
        return True

    def _retire_shader_buffer(self, buffer: Any | None) -> None:
        if buffer is None:
            return
        retired = getattr(self, "_retired_shader_buffers", None)
        if retired is None:
            retired = []
            self._retired_shader_buffers = retired
        retired.append(buffer)

    def _shader_buffer_shape_key(self, data: np.ndarray) -> tuple[tuple[int, ...], str]:
        return (tuple(int(v) for v in data.shape), str(data.dtype))

    def _retire_shader_buffer_ring(self, ring_attr: str) -> None:
        for buffer in getattr(self, ring_attr, ()) or ():
            self._retire_shader_buffer(buffer)
        setattr(self, ring_attr, [])

    def _dynamic_shader_buffer(
        self,
        data: np.ndarray,
        label: str,
        *,
        ring_attr: str,
        shape_key_attr: str,
        index_attr: str,
    ):
        shape_key = self._shader_buffer_shape_key(data)
        ring = list(getattr(self, ring_attr, []) or [])
        current_shape_key = getattr(self, shape_key_attr, None)
        if not ring or current_shape_key != shape_key:
            if ring:
                self._retire_shader_buffer_ring(ring_attr)
            ring = [
                self._static_shader_buffer(data, f"{label}[{slot}]")
                for slot in range(_DYNAMIC_SHADER_BUFFER_RING_SIZE)
            ]
            setattr(self, ring_attr, ring)
            setattr(self, shape_key_attr, shape_key)
            setattr(self, index_attr, 0)
            return ring[0]

        current_index = int(getattr(self, index_attr, -1))
        next_index = (current_index + 1) % len(ring)
        buffer = ring[next_index]
        if not self._update_static_shader_buffer(buffer, data):
            self._retire_shader_buffer(buffer)
            buffer = self._static_shader_buffer(data, f"{label}[{next_index}]")
            ring[next_index] = buffer
            setattr(self, ring_attr, ring)
        setattr(self, index_attr, next_index)
        return buffer

    def _fallback_shader_buffer(self, key: str, data: np.ndarray):
        buffers = getattr(self, "_fallback_shader_buffers", None)
        if buffers is None:
            buffers = {}
            self._fallback_shader_buffers = buffers
        buffer = buffers.get(key)
        if buffer is None:
            buffer = self._static_shader_buffer(data, f"hex-{key}-fallback")
            buffers[key] = buffer
        return buffer

    def _revisioned_buffer_key(self, data: np.ndarray, revision: int | None, source: Any = None) -> tuple[Any, ...]:
        revision_key = ("rev", int(revision)) if revision is not None else ("source", id(source))
        return (*revision_key, tuple(int(v) for v in data.shape), str(data.dtype))

    def _material_colors_resource(
        self,
        colors: list[tuple[float, float, float]] | np.ndarray | None,
        *,
        revision: int | None = None,
    ) -> tuple[Any, int]:
        data = build_material_color_buffer(colors)
        key = self._revisioned_buffer_key(data, revision, colors)
        if self._material_color_buffer is not None and self._material_color_buffer_key == key:
            return self._material_color_buffer, int(data.shape[0])
        self._material_color_buffer = self._dynamic_shader_buffer(
            data,
            "hex-material-colors",
            ring_attr="_material_color_buffer_ring",
            shape_key_attr="_material_color_buffer_shape_key",
            index_attr="_material_color_buffer_index",
        )
        self._material_color_buffer_key = key
        return self._material_color_buffer, int(data.shape[0])

    def _procedural_params_resource(
        self,
        params: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
        *,
        material_count: int,
        revision: int | None = None,
    ):
        data = build_procedural_material_param_buffer(params, material_count=material_count)
        key = self._revisioned_buffer_key(data, revision, params)
        if self._procedural_param_buffer is not None and self._procedural_param_buffer_key == key:
            return self._procedural_param_buffer
        self._procedural_param_buffer = self._dynamic_shader_buffer(
            data,
            "hex-procedural-material-params",
            ring_attr="_procedural_param_buffer_ring",
            shape_key_attr="_procedural_param_buffer_shape_key",
            index_attr="_procedural_param_buffer_index",
        )
        self._procedural_param_buffer_key = key
        return self._procedural_param_buffer

    def material_maker_parameter_specs(self) -> tuple[MaterialMakerParameterSpec, ...]:
        info = getattr(self, "_procedural_material_info", None)
        return tuple(getattr(info, "parameter_specs", ()) or ())

    def material_maker_fields(self) -> tuple[str, ...]:
        info = getattr(self, "_procedural_material_info", None)
        return tuple(getattr(info, "fields", ()) or ())

    def _material_maker_params_resource(
        self,
        params: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None,
        *,
        material_count: int,
        revision: int | None = None,
    ):
        specs = self.material_maker_parameter_specs()
        if not specs:
            return None
        data = build_material_maker_param_buffer(specs, params, material_count=material_count)
        key = self._revisioned_buffer_key(data, revision, params)
        if self._material_maker_param_buffer is not None and self._material_maker_param_buffer_key == key:
            return self._material_maker_param_buffer
        self._material_maker_param_buffer = self._dynamic_shader_buffer(
            data,
            "hex-material-maker-params",
            ring_attr="_material_maker_param_buffer_ring",
            shape_key_attr="_material_maker_param_buffer_shape_key",
            index_attr="_material_maker_param_buffer_index",
        )
        self._material_maker_param_buffer_key = key
        return self._material_maker_param_buffer

    def prepare_cryo_material_resources(
        self,
        *,
        material_colors: list[tuple[float, float, float]] | np.ndarray | None = None,
        material_colors_revision: int | None = None,
        procedural_params: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        procedural_params_revision: int | None = None,
        material_maker_params: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        material_maker_params_revision: int | None = None,
    ) -> None:
        """Upload material shader buffers outside the active render pass."""
        _, material_count = self._material_colors_resource(
            material_colors,
            revision=material_colors_revision,
        )
        self._procedural_params_resource(
            procedural_params,
            material_count=material_count,
            revision=procedural_params_revision,
        )
        self._material_maker_params_resource(
            material_maker_params,
            material_count=material_count,
            revision=material_maker_params_revision,
        )

    def _create_cryo_pipeline(self, program: Any, label: str):
        return self._device.create_render_pipeline(
            program=program,
            input_layout=self._cryo_input_layout,
            primitive_topology=self._spy.PrimitiveTopology.triangle_list,
            label=label,
            **self._surface_pipeline_common,
        )

    def _validate_external_material_textures(self, info: MaterialMakerSlangInfo) -> None:
        missing = [texture_path_for(info.path, name) for name in info.texture_names if not texture_path_for(info.path, name).exists()]
        if missing:
            names = ", ".join(str(path) for path in missing)
            raise MaterialMakerSlangError(f"Missing Material Maker texture(s): {names}")

    def _clear_external_material_textures(self) -> None:
        for binding in self._procedural_material_textures.values():
            _close_slang_resource(binding.texture)
        self._procedural_material_textures = {}

    def _retire_external_material_textures(self) -> None:
        for binding in self._procedural_material_textures.values():
            self._retire_shader_buffer(binding.texture)
        self._procedural_material_textures = {}

    def set_procedural_material_scale(self, scale: float) -> None:
        self._procedural_material_scale = max(float(scale), 0.000001)

    def set_environment_path(self, path: str | Path | None) -> None:
        new_path = Path(path).expanduser().resolve() if path else None
        if new_path == getattr(self, "_environment_path", None):
            return
        self._environment_path = new_path
        if getattr(self, "_environment_texture", None) is not None:
            _close_slang_resource(self._environment_texture)
        self._environment_texture = None
        self._environment_texture_key = None

    def set_environment_intensity(self, intensity: float) -> None:
        self._environment_intensity = max(float(intensity), 0.0)

    def set_environment_background_enabled(self, enabled: bool) -> None:
        self._environment_background_enabled = bool(enabled)

    def set_environment_rotation_degrees(self, degrees: float) -> None:
        self._environment_rotation_degrees = _finite_float_or(degrees, 0.0)

    def set_environment_pitch_degrees(self, degrees: float) -> None:
        self._environment_pitch_degrees = _finite_float_or(degrees, 0.0)

    def _environment_rotation_turns(self) -> float:
        return _finite_float_or(getattr(self, "_environment_rotation_degrees", 0.0), 0.0) / 360.0

    def _environment_pitch_radians(self) -> float:
        return float(np.deg2rad(_finite_float_or(getattr(self, "_environment_pitch_degrees", -90.0), -90.0)))

    def _compile_external_material_pipeline(self, *, force: bool = False) -> _ExternalMaterialReloadResult:
        if self._procedural_material_path is None:
            if self._cryo_pipeline is not None and not force:
                return _ExternalMaterialReloadResult(changed=False)
            program = self._device.load_program("hex_cryo_surface.slang", ["vertex_main", "fragment_main"])
            pipeline = self._create_cryo_pipeline(program, "hex-cryo-surface")
            return _ExternalMaterialReloadResult(changed=True, program=program, pipeline=pipeline)

        if not force and not self._procedural_material_hot_reload:
            return _ExternalMaterialReloadResult(changed=False)
        if (
            not force
            and self._procedural_material_info is not None
            and not material_source_changed(self._procedural_material_info)
        ):
            return _ExternalMaterialReloadResult(changed=False)

        info = write_material_bridge(self._procedural_material_path, MATERIAL_MAKER_SLANG_WORK_DIR)
        self._validate_external_material_textures(info)
        program = self._device.load_program(str(info.generated_shader_path), ["vertex_main", "fragment_main"])
        pipeline = self._create_cryo_pipeline(program, "hex-cryo-surface-material-maker")
        return _ExternalMaterialReloadResult(changed=True, program=program, pipeline=pipeline, info=info)

    def _external_material_reload_failure_message(self, exc: Exception) -> str:
        if self._procedural_material_path is None:
            return f"[slang] surface shader reload failed, keeping previous pipeline: {exc}"
        return f"[slang] Material Maker shader reload failed, keeping previous pipeline: {exc}"

    def _apply_external_material_reload_result(
        self,
        result: _ExternalMaterialReloadResult,
        *,
        raise_on_error: bool = False,
    ) -> bool:
        del raise_on_error
        if not result.changed:
            if getattr(self, "_procedural_material_reload_status", "") == "Reloading MM shader...":
                self._procedural_material_reload_status = ""
            return False

        self._cryo_program = result.program
        self._cryo_pipeline = result.pipeline
        if result.info is not None:
            self._procedural_material_info = result.info
            self._procedural_material_reload_error = None
            self._procedural_material_reload_status = f"Loaded {result.info.path.name}"
            self._retire_external_material_textures()
            for warning in result.info.warnings:
                self._warn_once(f"material-maker:{warning}", f"[slang] {warning}")
            print(f"[slang] Loaded Material Maker procedural material: {result.info.path}")
        else:
            self._procedural_material_reload_error = None
            self._procedural_material_reload_status = ""
        return True

    def _poll_external_material_reload(self) -> bool:
        future = getattr(self, "_external_material_reload_future", None)
        if future is None or not future.done():
            return False
        self._external_material_reload_future = None
        try:
            result = future.result()
        except Exception as exc:
            message = self._external_material_reload_failure_message(exc)
            if message != self._procedural_material_reload_error:
                print(message)
                self._procedural_material_reload_error = message
            self._procedural_material_reload_status = f"Reload failed: {exc}"
            return False
        return self._apply_external_material_reload_result(result)

    def request_external_material_reload(self, *, force: bool = False) -> bool:
        self._poll_external_material_reload()
        future = getattr(self, "_external_material_reload_future", None)
        if future is not None and not future.done():
            return False
        executor = getattr(self, "_external_material_reload_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="omnisurg-slang-reload")
            self._external_material_reload_executor = executor
        self._procedural_material_reload_status = "Reloading MM shader..."
        self._external_material_reload_future = executor.submit(self._compile_external_material_pipeline, force=force)
        return True

    def external_material_reload_status(self) -> str:
        self._poll_external_material_reload()
        return str(getattr(self, "_procedural_material_reload_status", ""))

    def reload_external_material(self, *, force: bool = False, raise_on_error: bool = False) -> bool:
        try:
            result = self._compile_external_material_pipeline(force=force)
        except Exception as exc:
            message = self._external_material_reload_failure_message(exc)
            if raise_on_error:
                raise RuntimeError(message) from exc
            if message != self._procedural_material_reload_error:
                print(message)
                self._procedural_material_reload_error = message
            self._procedural_material_reload_status = f"Reload failed: {exc}"
            return False
        return self._apply_external_material_reload_result(result, raise_on_error=raise_on_error)

    def _load_external_material_texture(self, path: Path):
        bitmap = self._spy.Bitmap(str(path)).convert(
            pixel_format=self._spy.Bitmap.PixelFormat.rgba,
            component_type=self._spy.Bitmap.ComponentType.float32,
            srgb_gamma=False,
        )
        data = np.asarray(bitmap)
        return self._device.create_texture(
            format=self._spy.Format.rgba32_float,
            width=bitmap.width,
            height=bitmap.height,
            usage=self._spy.TextureUsage.shader_resource,
            data=data,
            label=path.name,
        )

    def _neutral_environment_texture_resource(self):
        if getattr(self, "_neutral_environment_texture", None) is None:
            data = np.asarray([[[0.0, 0.0, 0.0, 1.0]]], dtype=np.float32)
            self._neutral_environment_texture = self._device.create_texture(
                format=self._spy.Format.rgba32_float,
                width=1,
                height=1,
                usage=self._spy.TextureUsage.shader_resource,
                data=data,
                label="hex-environment-fallback",
            )
        return self._neutral_environment_texture

    def _environment_texture_resource(self):
        path = getattr(self, "_environment_path", None)
        if path is None or not path.exists():
            return None
        key = (path, path.stat().st_mtime_ns)
        if (
            getattr(self, "_environment_texture", None) is not None
            and getattr(self, "_environment_texture_key", None) == key
        ):
            return self._environment_texture
        texture = self._load_external_material_texture(path)
        if getattr(self, "_environment_texture", None) is not None:
            _close_slang_resource(self._environment_texture)
        self._environment_texture = texture
        self._environment_texture_key = key
        return texture

    def _bind_environment_uniforms(self, cursor: Any, environment_texture: Any | None = None) -> Any | None:
        if environment_texture is None:
            environment_texture = self._environment_texture_resource()
        cursor.environment_map = (
            environment_texture if environment_texture is not None else self._neutral_environment_texture_resource()
        )
        cursor.environment_sampler = self._sampler()
        cursor.has_environment_map = int(environment_texture is not None)
        cursor.environment_intensity = float(getattr(self, "_environment_intensity", 1.0))
        cursor.environment_rotation = float(self._environment_rotation_turns())
        cursor.environment_pitch = float(self._environment_pitch_radians())
        return environment_texture

    def _external_material_texture_resource(self, name: str):
        if self._procedural_material_info is None:
            raise MaterialMakerSlangError("No Material Maker material is loaded.")
        path = texture_path_for(self._procedural_material_info.path, name)
        if not path.exists():
            raise MaterialMakerSlangError(f"Missing Material Maker texture: {path}")
        mtime_ns = path.stat().st_mtime_ns
        binding = self._procedural_material_textures.get(name)
        if binding is not None and binding.path == path and binding.mtime_ns == mtime_ns:
            return binding.texture
        texture = self._load_external_material_texture(path)
        if binding is not None:
            _close_slang_resource(binding.texture)
        self._procedural_material_textures[name] = _MaterialTextureBinding(path, mtime_ns, texture)
        return texture

    def _bind_external_material_uniforms(self, cursor: Any) -> None:
        if getattr(self, "_procedural_material_info", None) is None:
            return
        cursor.g_sampler = self._sampler()
        for name in self._procedural_material_info.texture_names:
            setattr(cursor, name, self._external_material_texture_resource(name))

    def _init_pipeline(self) -> None:
        self._flat_program = self._device.load_program("hex_flat.slang", ["vertex_main", "fragment_main"])
        self._point_program = self._device.load_program("hex_flat.slang", ["point_vertex_main", "point_fragment_main"])
        self._point_sphere_program = self._device.load_program(
            "hex_flat.slang",
            ["point_sphere_vertex_main", "point_sphere_fragment_main"],
        )
        self._present_program = self._device.load_program("hex_present.slang", ["vertex_main", "fragment_main"])
        self._background_program = self._device.load_program(
            "hex_present.slang",
            ["background_vertex_main", "background_fragment_main"],
        )
        self._cryo_input_layout = self._device.create_input_layout(
            input_elements=[
                {"semantic_name": "POSITION", "semantic_index": 0, "format": self._spy.Format.rgb32_float, "buffer_slot_index": 0},
                {"semantic_name": "NORMAL", "semantic_index": 0, "format": self._spy.Format.rgb32_float, "buffer_slot_index": 1},
                {"semantic_name": "TEXCOORD", "semantic_index": 0, "format": self._spy.Format.rgb32_float, "buffer_slot_index": 2},
                {"semantic_name": "TEXCOORD", "semantic_index": 1, "format": self._spy.Format.rgb32_float, "buffer_slot_index": 3},
            ],
            vertex_streams=[{"stride": 12}, {"stride": 12}, {"stride": 12}, {"stride": 12}],
        )
        self._flat_input_layout = self._device.create_input_layout(
            input_elements=[
                {"semantic_name": "POSITION", "semantic_index": 0, "format": self._spy.Format.rgb32_float, "buffer_slot_index": 0}
            ],
            vertex_streams=[{"stride": 12}],
        )
        self._point_sphere_input_layout = self._device.create_input_layout(
            input_elements=[
                {"semantic_name": "POSITION", "semantic_index": 0, "format": self._spy.Format.rgb32_float, "buffer_slot_index": 0}
            ],
            vertex_streams=[{"stride": 12}],
        )
        self._surface_pipeline_common = {
            "targets": [{"format": self._scene_color_format}],
            "depth_stencil": {
                "format": self._depth_format,
                "depth_test_enable": True,
                "depth_write_enable": True,
                "depth_func": self._spy.ComparisonFunc.less,
            },
            "rasterizer": {"cull_mode": self._spy.CullMode.none},
        }
        self._cryo_program = None
        self._cryo_pipeline = None
        self.reload_external_material(force=True, raise_on_error=True)
        self._flat_pipeline = self._device.create_render_pipeline(
            program=self._flat_program,
            input_layout=self._flat_input_layout,
            primitive_topology=self._spy.PrimitiveTopology.triangle_list,
            label="hex-flat",
            **self._surface_pipeline_common,
        )
        self._point_pipeline = self._device.create_render_pipeline(
            program=self._point_program,
            input_layout=self._flat_input_layout,
            primitive_topology=self._spy.PrimitiveTopology.point_list,
            label="hex-points",
            **self._surface_pipeline_common,
        )
        self._point_sphere_pipeline = self._device.create_render_pipeline(
            program=self._point_sphere_program,
            input_layout=self._point_sphere_input_layout,
            primitive_topology=self._spy.PrimitiveTopology.triangle_list,
            label="hex-point-spheres",
            **self._surface_pipeline_common,
        )
        self._present_pipeline = self._device.create_render_pipeline(
            program=self._present_program,
            input_layout=None,
            primitive_topology=self._spy.PrimitiveTopology.triangle_list,
            targets=[{"format": self._color_format}],
            rasterizer={"cull_mode": self._spy.CullMode.none},
            label="hex-present",
        )
        self._background_pipeline = self._device.create_render_pipeline(
            program=self._background_program,
            input_layout=None,
            primitive_topology=self._spy.PrimitiveTopology.triangle_list,
            targets=[{"format": self._scene_color_format}],
            rasterizer={"cull_mode": self._spy.CullMode.none},
            label="hex-environment-background",
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

    def _ensure_scene_color_texture(self, width: int, height: int):
        if (
            self._scene_color_texture is None
            or self._scene_color_texture.width != width
            or self._scene_color_texture.height != height
        ):
            self._scene_color_texture = self._device.create_texture(
                format=self._scene_color_format,
                width=width,
                height=height,
                usage=self._spy.TextureUsage.render_target | self._spy.TextureUsage.shader_resource,
                label="hex-scene-color",
            )
        return self._scene_color_texture

    def _ensure_depth_texture(self, width: int, height: int):
        if self._depth_texture is None or self._depth_texture.width != width or self._depth_texture.height != height:
            self._depth_texture = self._device.create_texture(
                format=self._depth_format,
                width=width,
                height=height,
                usage=self._spy.TextureUsage.depth_stencil,
                label="hex-depth",
            )
        return self._depth_texture

    def _set_camera_look_at(self, eye: np.ndarray, target: np.ndarray) -> None:
        forward = _normalize_or(np.asarray(target, dtype=np.float32) - np.asarray(eye, dtype=np.float32), (0.0, 0.0, -1.0))
        right = _normalize_or(np.cross(forward, self._camera_world_up), (1.0, 0.0, 0.0))
        up = _normalize_or(np.cross(right, forward), (0.0, 0.0, 1.0))
        distance = float(np.linalg.norm(np.asarray(target, dtype=np.float32) - np.asarray(eye, dtype=np.float32)))
        scene_radius = max(float(self._camera_scene_radius), 0.25)
        self._camera_pos = np.asarray(eye, dtype=np.float32)
        self._camera_target = np.asarray(target, dtype=np.float32)
        self._camera_forward = forward
        self._camera_right = right
        self._camera_up = up
        self._camera_near = max(0.001, min(0.05, distance * 0.02))
        self._camera_far = max(self._camera_near + 1.0, distance + scene_radius * 8.0)

    def _orbit_camera(self, dx: float, dy: float) -> None:
        offset = np.asarray(self._camera_pos - self._camera_target, dtype=np.float32)
        distance = float(np.linalg.norm(offset))
        if distance <= 1.0e-6:
            return
        radians_per_pixel = 0.006
        yawed = _rotate_vector_axis_angle(offset, self._camera_world_up, -float(dx) * radians_per_pixel)
        pitched = _rotate_vector_axis_angle(yawed, self._camera_right, -float(dy) * radians_per_pixel)
        candidate_forward = _normalize_or(-pitched, tuple(float(v) for v in self._camera_forward))
        if abs(float(np.dot(candidate_forward, self._camera_world_up))) > 0.985:
            pitched = yawed
        self._set_camera_look_at(self._camera_target + pitched, self._camera_target)

    def set_model(self, model: Any) -> None:
        particle_q = getattr(model, "particle_q", None)
        if particle_q is None:
            return
        try:
            positions = np.asarray(particle_q.numpy(), dtype=np.float32)
        except Exception:
            return
        if positions.size == 0:
            return
        bounds_min = np.min(positions, axis=0)
        bounds_max = np.max(positions, axis=0)
        self._camera_target = (0.5 * (bounds_min + bounds_max)).astype(np.float32)
        self._camera_scene_radius = max(float(np.linalg.norm(bounds_max - bounds_min)) * 0.5, 0.25)
        self._set_camera_look_at(self._camera_pos, self._camera_target)

    def set_camera(self, pos=None, pitch: float = 0.0, yaw: float = 0.0, **_kwargs) -> None:
        if pos is not None:
            eye = np.asarray([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)
            del pitch, yaw
            self._set_camera_look_at(eye, self._camera_target)

    def _screen_size(self) -> tuple[int, int]:
        if self._surface_texture is not None:
            return int(self._surface_texture.width), int(self._surface_texture.height)
        return int(self._window.width), int(self._window.height)

    def screen_to_world_ray(self, x: float, y: float) -> tuple[np.ndarray, np.ndarray]:
        width, height = self._screen_size()
        ndc_x = (float(x) / max(float(width), 1.0)) * 2.0 - 1.0
        ndc_y = 1.0 - (float(y) / max(float(height), 1.0)) * 2.0
        tan_half = 1.0 / max(float(self._camera_inv_tan_half_fovy), 1.0e-6)
        aspect = float(width) / max(float(height), 1.0)
        direction = (
            self._camera_forward
            + self._camera_right * (ndc_x * tan_half * aspect)
            + self._camera_up * (ndc_y * tan_half)
        )
        return self._camera_pos.copy(), _normalize_or(direction, tuple(float(v) for v in self._camera_forward))

    def _set_common_uniforms(self, shader_object: Any, color=(1.0, 1.0, 1.0, 1.0)) -> None:
        cursor = self._spy.ShaderCursor(shader_object)
        width, height = self._screen_size()
        cursor.camera_pos = self._spy.float3(*self._camera_pos.tolist())
        cursor.camera_right = self._spy.float3(*self._camera_right.tolist())
        cursor.camera_up = self._spy.float3(*self._camera_up.tolist())
        cursor.camera_forward = self._spy.float3(*self._camera_forward.tolist())
        cursor.camera_inv_tan_half_fovy = self._camera_inv_tan_half_fovy
        cursor.camera_aspect = float(width) / max(float(height), 1.0)
        cursor.camera_near = self._camera_near
        cursor.camera_far = self._camera_far
        cursor.mesh_color = self._spy.float4(*color)

    def _set_cryo_uniforms(
        self,
        shader_object: Any,
        host_volume: np.ndarray | None,
        scale: tuple[float, float, float],
        *,
        material_colors: list[tuple[float, float, float]] | np.ndarray | None = None,
        material_colors_revision: int | None = None,
        procedural_params: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        procedural_params_revision: int | None = None,
        material_maker_params: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        material_maker_params_revision: int | None = None,
        material_id_buffer: Any | None = None,
        state_rgba_buffer: Any | None = None,
        procedural_enabled: bool = False,
        procedural_world_space: bool = False,
        procedural_uv3_noise_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        lighting_enabled: bool = True,
        key_light_enabled: bool = True,
        fill_light_enabled: bool = True,
        ambient_light_enabled: bool = False,
        environment_lighting_enabled: bool = True,
        cryo_mix: float = 1.0,
        state_overlay_strength: float = 1.0,
        height_debug: bool = False,
        debug_view: int | None = None,
    ) -> None:
        self._set_common_uniforms(shader_object)
        cursor = self._spy.ShaderCursor(shader_object)
        has_cryo_volume = host_volume is not None
        cursor.cryo_volume = self._cryo_texture_resource(host_volume) if has_cryo_volume else self._neutral_cryo_texture_resource()
        cursor.cryo_sampler = self._sampler()
        cursor.cryo_volume_scale = self._spy.float3(float(scale[0]), float(scale[1]), float(scale[2]))
        material_color_resource, material_count = self._material_colors_resource(
            material_colors,
            revision=material_colors_revision,
        )
        cursor.material_colors = material_color_resource
        procedural_param_resource = self._procedural_params_resource(
            procedural_params,
            material_count=material_count,
            revision=procedural_params_revision,
        )
        cursor.procedural_params = procedural_param_resource
        cursor.material_id = (
            material_id_buffer
            if material_id_buffer is not None
            else self._fallback_shader_buffer("material-id", np.zeros((1,), dtype=np.int32))
        )
        cursor.state_rgba = (
            state_rgba_buffer
            if state_rgba_buffer is not None
            else self._fallback_shader_buffer("state-rgba", np.zeros((1, 4), dtype=np.float32))
        )
        cursor.material_count = int(material_count)
        cursor.has_material_state = int(material_id_buffer is not None and state_rgba_buffer is not None)
        cursor.procedural_enabled = int(bool(procedural_enabled))
        cursor.procedural_world_space = int(bool(procedural_world_space))
        cursor.procedural_uv3_noise_scale = self._spy.float3(
            float(procedural_uv3_noise_scale[0]),
            float(procedural_uv3_noise_scale[1]),
            float(procedural_uv3_noise_scale[2]),
        )
        cursor.has_cryo_volume = int(bool(has_cryo_volume))
        cursor.lighting_enabled = int(bool(lighting_enabled))
        cursor.key_light_enabled = int(bool(key_light_enabled))
        cursor.fill_light_enabled = int(bool(fill_light_enabled))
        cursor.ambient_light_enabled = int(bool(ambient_light_enabled))
        cursor.environment_lighting_enabled = int(bool(environment_lighting_enabled))
        cursor.debug_view = clamp_slang_surface_debug_view(debug_view, height_debug=height_debug)
        cursor.cryo_mix = _clamp_unit_float(cryo_mix, 1.0)
        cursor.state_overlay_strength = _clamp_unit_float(state_overlay_strength, 1.0)
        cursor.procedural_material_scale = float(getattr(self, "_procedural_material_scale", 1.0))
        cursor.procedural_time = float(getattr(self, "_time", 0.0))
        cursor.light_key_dir = self._spy.float3(0.35, 0.75, 0.55)
        cursor.light_fill_dir = self._spy.float3(-0.45, 0.25, 0.60)
        cursor.light_ambient_color = self._spy.float3(0.18, 0.17, 0.16)
        cursor.light_key_color = self._spy.float3(1.0, 0.92, 0.84)
        cursor.light_fill_color = self._spy.float3(0.28, 0.34, 0.44)
        self._bind_environment_uniforms(cursor)
        material_maker_param_resource = self._material_maker_params_resource(
            material_maker_params,
            material_count=material_count,
            revision=material_maker_params_revision,
        )
        if material_maker_param_resource is not None:
            cursor.material_maker_parameters = material_maker_param_resource
        self._bind_external_material_uniforms(cursor)

    def _set_present_uniforms(self, shader_object: Any, source_texture: Any, width: int, height: int) -> None:
        cursor = self._spy.ShaderCursor(shader_object)
        cursor.source_tex = source_texture
        cursor.post_sampler = self._sampler()
        cursor.output_size = self._spy.float2(float(max(1, width)), float(max(1, height)))

    def _set_background_uniforms(self, shader_object: Any, environment_texture: Any) -> None:
        cursor = self._spy.ShaderCursor(shader_object)
        width, height = self._screen_size()
        cursor.camera_right = self._spy.float3(*self._camera_right.tolist())
        cursor.camera_up = self._spy.float3(*self._camera_up.tolist())
        cursor.camera_forward = self._spy.float3(*self._camera_forward.tolist())
        cursor.camera_inv_tan_half_fovy = self._camera_inv_tan_half_fovy
        cursor.camera_aspect = float(width) / max(float(height), 1.0)
        self._bind_environment_uniforms(cursor, environment_texture)

    def _render_environment_background(self) -> None:
        if (
            not bool(getattr(self, "_environment_background_enabled", False))
            or self._pass_encoder is None
            or getattr(self, "_background_pipeline", None) is None
        ):
            return
        environment_texture = self._environment_texture_resource()
        if environment_texture is None:
            return
        self._pass_encoder.set_render_state(
            {
                "viewports": [self._viewport],
                "scissor_rects": [self._scissor],
            }
        )
        shader_object = self._pass_encoder.bind_pipeline(self._background_pipeline)
        self._set_background_uniforms(shader_object, environment_texture)
        self._pass_encoder.draw({"vertex_count": 3})

    def _render_present(self, source_texture: Any, width: int, height: int) -> None:
        if self._command_encoder is None or self._surface_texture is None or source_texture is None:
            return
        pass_encoder = self._command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": self._surface_texture.create_view({}),
                        "clear_value": [0.0, 0.0, 0.0, 1.0],
                        "load_op": self._spy.LoadOp.clear,
                        "store_op": self._spy.StoreOp.store,
                    }
                ],
            }
        )
        pass_encoder.set_render_state(
            {
                "viewports": [self._viewport],
                "scissor_rects": [self._scissor],
            }
        )
        shader_object = pass_encoder.bind_pipeline(self._present_pipeline)
        self._set_present_uniforms(shader_object, source_texture, width, height)
        pass_encoder.draw({"vertex_count": 3})
        pass_encoder.end()

    def draw_cryo_surface(
        self,
        frame: Any | None,
        host_volume: np.ndarray | None,
        *,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        hidden: bool = False,
        material_colors: list[tuple[float, float, float]] | np.ndarray | None = None,
        material_colors_revision: int | None = None,
        procedural_params: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        procedural_params_revision: int | None = None,
        material_maker_params: list[dict[str, Any]] | tuple[dict[str, Any], ...] | None = None,
        material_maker_params_revision: int | None = None,
        procedural_enabled: bool = False,
        procedural_world_space: bool = False,
        procedural_uv3_noise_scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
        lighting_enabled: bool = True,
        key_light_enabled: bool = True,
        fill_light_enabled: bool = True,
        ambient_light_enabled: bool = False,
        environment_lighting_enabled: bool = True,
        cryo_mix: float = 1.0,
        state_overlay_strength: float = 1.0,
        height_debug: bool = False,
        debug_view: int | None = None,
    ) -> None:
        if hidden or frame is None or self._pass_encoder is None or int(frame.vertex_count) <= 0:
            return
        generation = int(getattr(frame, "buffer_generation", -1))
        positions = self._update_shared_buffer(
            frame.positions,
            "cryo-positions",
            wp.vec3f,
            12,
            self._spy.BufferUsage.vertex_buffer,
            generation=generation,
        )
        normals = self._update_shared_buffer(
            frame.normals,
            "cryo-normals",
            wp.vec3f,
            12,
            self._spy.BufferUsage.vertex_buffer,
            generation=generation,
        )
        uv3 = self._update_shared_buffer(
            frame.uv3,
            "cryo-uv3",
            wp.vec3f,
            12,
            self._spy.BufferUsage.vertex_buffer,
            generation=generation,
        )
        procedural_coord_source = getattr(frame, "procedural_coord", None)
        if procedural_coord_source is None:
            procedural_coord = uv3
        else:
            procedural_coord = self._update_shared_buffer(
                procedural_coord_source,
                "cryo-procedural-coord",
                wp.vec3f,
                12,
                self._spy.BufferUsage.vertex_buffer,
                generation=generation,
            )
        material_id = None
        state_rgba = None
        if procedural_enabled and getattr(frame, "material_id", None) is not None and getattr(frame, "state_rgba", None) is not None:
            material_id = self._update_shared_buffer(
                frame.material_id,
                "cryo-material-id",
                wp.int32,
                4,
                self._spy.BufferUsage.shader_resource,
                generation=generation,
            )
            state_rgba = self._update_shared_buffer(
                frame.state_rgba,
                "cryo-state-rgba",
                wp.vec4,
                16,
                self._spy.BufferUsage.shader_resource,
                generation=generation,
            )
        self._pass_encoder.set_render_state(
            {
                "vertex_buffers": [positions.buffer, normals.buffer, uv3.buffer, procedural_coord.buffer],
                "viewports": [self._viewport],
                "scissor_rects": [self._scissor],
            }
        )
        shader_object = self._pass_encoder.bind_pipeline(self._cryo_pipeline)
        self._set_cryo_uniforms(
            shader_object,
            host_volume,
            scale,
            material_colors=material_colors,
            material_colors_revision=material_colors_revision,
            procedural_params=procedural_params,
            procedural_params_revision=procedural_params_revision,
            material_maker_params=material_maker_params,
            material_maker_params_revision=material_maker_params_revision,
            material_id_buffer=None if material_id is None else material_id.buffer,
            state_rgba_buffer=None if state_rgba is None else state_rgba.buffer,
            procedural_enabled=procedural_enabled,
            procedural_world_space=procedural_world_space,
            procedural_uv3_noise_scale=procedural_uv3_noise_scale,
            lighting_enabled=lighting_enabled,
            key_light_enabled=key_light_enabled,
            fill_light_enabled=fill_light_enabled,
            ambient_light_enabled=ambient_light_enabled,
            environment_lighting_enabled=environment_lighting_enabled,
            cryo_mix=cryo_mix,
            state_overlay_strength=state_overlay_strength,
            height_debug=height_debug,
            debug_view=debug_view,
        )
        self._pass_encoder.draw({"vertex_count": int(frame.vertex_count)})

    def _point_sphere_buffers(self) -> tuple[Any, Any, int]:
        if self._point_sphere_vertex_buffer is not None and self._point_sphere_index_buffer is not None:
            return self._point_sphere_vertex_buffer, self._point_sphere_index_buffer, int(self._point_sphere_index_count)

        vertices, indices = build_unit_sphere_mesh()
        self._point_sphere_vertex_buffer = self._device.create_buffer(
            usage=self._spy.BufferUsage.vertex_buffer | self._spy.BufferUsage.shader_resource,
            label="hex-point-sphere-vertices",
            data=vertices,
        )
        self._point_sphere_index_buffer = self._device.create_buffer(
            usage=self._spy.BufferUsage.index_buffer | self._spy.BufferUsage.shader_resource,
            label="hex-point-sphere-indices",
            data=indices,
        )
        self._point_sphere_index_count = int(indices.size)
        return self._point_sphere_vertex_buffer, self._point_sphere_index_buffer, int(self._point_sphere_index_count)

    def _point_radii_resource(self, name: str, radii: Any, count: int) -> tuple[Any, bool, float]:
        default_radius = 0.1
        if _is_warp_float_array(radii):
            if int(len(radii)) >= count:
                return (
                    self._update_shared_buffer(
                        radii,
                        f"{name}-point-radii",
                        wp.float32,
                        4,
                        self._spy.BufferUsage.shader_resource,
                    ).buffer,
                    True,
                    default_radius,
                )
            self._warn_once(
                f"point-radii-short:{name}",
                f"Slang point radii for {name!r} ignored because the array is shorter than the point count.",
            )
        elif radii is not None:
            default_radius = max(0.0, _float_or(radii, default_radius))
        return (
            self._fallback_shader_buffer("point-radii", np.asarray([default_radius], dtype=np.float32)),
            False,
            default_radius,
        )

    def _point_colors_resource(self, name: str, colors: Any, count: int) -> tuple[Any, bool, tuple[float, float, float]]:
        default_color = _rgb_tuple_or(colors, (0.85, 0.18, 0.12))
        if _is_warp_vec3_array(colors):
            if int(len(colors)) >= count:
                return (
                    self._update_shared_buffer(
                        colors,
                        f"{name}-point-colors",
                        wp.vec3f,
                        12,
                        self._spy.BufferUsage.shader_resource,
                    ).buffer,
                    True,
                    default_color,
                )
            self._warn_once(
                f"point-colors-short:{name}",
                f"Slang point colors for {name!r} ignored because the array is shorter than the point count.",
            )
        elif _is_warp_array(colors):
            self._warn_once(
                f"point-colors-dtype:{name}",
                f"Slang point colors for {name!r} must be wp.vec3; using the default color.",
            )
        return (
            self._fallback_shader_buffer("point-colors", np.asarray([default_color], dtype=np.float32)),
            False,
            default_color,
        )

    def log_mesh(self, name: str, points=None, indices=None, hidden: bool = False, **_kwargs) -> None:
        if hidden or points is None or indices is None or self._pass_encoder is None:
            return
        source = self._update_shared_buffer(points, f"{name}-points", wp.vec3f, 12, self._spy.BufferUsage.vertex_buffer)
        index_buffer = self._index_buffer(indices, name)
        self._pass_encoder.set_render_state(
            {
                "vertex_buffers": [source.buffer],
                "index_buffer": index_buffer,
                "index_format": self._spy.IndexFormat.uint32,
                "viewports": [self._viewport],
                "scissor_rects": [self._scissor],
            }
        )
        shader_object = self._pass_encoder.bind_pipeline(self._flat_pipeline)
        self._set_common_uniforms(shader_object, (0.22, 0.22, 0.22, 1.0))
        self._pass_encoder.draw_indexed({"vertex_count": int(len(indices))})

    def log_points(self, name: str, points=None, radii=None, colors=None, hidden: bool = False, **_kwargs) -> None:
        if hidden or points is None or self._pass_encoder is None or int(len(points)) == 0:
            return
        count = int(len(points))
        sphere_vertices, sphere_indices, sphere_index_count = self._point_sphere_buffers()
        centers = self._update_shared_buffer(
            points,
            f"{name}-point-centers",
            wp.vec3f,
            12,
            self._spy.BufferUsage.shader_resource,
        )
        radii_buffer, has_radii, default_radius = self._point_radii_resource(name, radii, count)
        colors_buffer, has_colors, default_color = self._point_colors_resource(name, colors, count)
        self._pass_encoder.set_render_state(
            {
                "vertex_buffers": [sphere_vertices],
                "index_buffer": sphere_indices,
                "index_format": self._spy.IndexFormat.uint32,
                "viewports": [self._viewport],
                "scissor_rects": [self._scissor],
            }
        )
        shader_object = self._pass_encoder.bind_pipeline(self._point_sphere_pipeline)
        self._set_common_uniforms(shader_object, (*default_color, 1.0))
        cursor = self._spy.ShaderCursor(shader_object)
        cursor.point_positions = centers.buffer
        cursor.point_radii = radii_buffer
        cursor.point_colors = colors_buffer
        cursor.point_count = count
        cursor.point_has_radii = int(has_radii)
        cursor.point_has_colors = int(has_colors)
        cursor.point_default_radius = float(default_radius)
        cursor.point_default_color = self._spy.float3(*default_color)
        self._pass_encoder.draw_indexed({"vertex_count": sphere_index_count, "instance_count": count})

    def log_lines(self, *args, **kwargs) -> None:
        del args, kwargs
        self._warn_once("slang-lines", "Slang viewer line overlays are deferred in this MVP.")

    def log_state(self, *_args, **_kwargs) -> None:
        return

    def begin_frame(self, time: float) -> None:
        self._time = float(time)
        if self._closed:
            return
        self._window.process_events()
        self._apply_pending_resize()
        should_close = bool(self._window.should_close())
        if should_close or not self._surface.config:
            return
        self._poll_external_material_reload()
        if self._procedural_material_path is None:
            self.reload_external_material(force=False, raise_on_error=False)
        elif (
            self._procedural_material_hot_reload
            and self._external_material_reload_future is None
            and (
                self._procedural_material_info is None
                or material_source_changed(self._procedural_material_info)
            )
        ):
            self.request_external_material_reload(force=False)
        self._update_keyboard_camera_motion()
        last_submit_id = getattr(self, "_last_submit_id", None)
        if last_submit_id is not None:
            try:
                submit_finished = bool(self._device.is_submit_finished(int(last_submit_id)))
            except Exception:
                submit_finished = True
            if not submit_finished:
                self._surface_texture = None
                self._command_encoder = None
                self._pass_encoder = None
                return
            self._last_submit_id = None
        self._surface_texture = self._surface.acquire_next_image()
        if not self._surface_texture:
            self._surface_texture = None
            return
        width, height = int(self._surface_texture.width), int(self._surface_texture.height)
        self._viewport = self._spy.Viewport.from_size(width, height)
        self._scissor = self._spy.ScissorRect.from_size(width, height)
        scene_color = self._ensure_scene_color_texture(width, height)
        depth = self._ensure_depth_texture(width, height)
        self._command_encoder = self._device.create_command_encoder()
        self._pass_encoder = self._command_encoder.begin_render_pass(
            {
                "color_attachments": [
                    {
                        "view": scene_color.create_view({}),
                        "clear_value": [0.025, 0.030, 0.034, 1.0],
                        "load_op": self._spy.LoadOp.clear,
                        "store_op": self._spy.StoreOp.store,
                    }
                ],
                "depth_stencil_attachment": {
                    "view": depth.create_view({}),
                    "depth_load_op": self._spy.LoadOp.clear,
                    "depth_store_op": self._spy.StoreOp.store,
                    "depth_clear_value": 1.0,
                },
            }
        )
        self._render_environment_background()

    def end_frame(self) -> None:
        if self._pass_encoder is None or self._command_encoder is None or self._surface_texture is None:
            self._frame_id += 1
            return
        self._pass_encoder.end()
        self._pass_encoder = None
        width, height = int(self._surface_texture.width), int(self._surface_texture.height)
        self._render_ui(width, height, self._scene_color_texture)
        self._render_present(self._scene_color_texture, width, height)
        command_buffer = self._command_encoder.finish()
        self._command_encoder = None
        self._sync_cuda_to_graphics()
        submit_id = self._device.submit_command_buffer(command_buffer)
        self._last_submit_id = int(submit_id)
        self._last_submit_frame = int(self._frame_id)
        self._surface.present()
        self._surface_texture = None
        self._frame_id += 1

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
            self._scene_color_texture = None
            self._depth_texture = None
            return
        self._surface.configure(width=width, height=height, vsync=self._vsync)
        self._scene_color_texture = None
        self._depth_texture = None

    def _ui_window(self, key: str, title: str, position: tuple[float, float], size: tuple[float, float]):
        if self._sui is None or self._ui_context is None:
            return None
        window = self._ui_windows.get(key)
        spy_position = self._spy.float2(float(position[0]), float(position[1]))
        spy_size = self._spy.float2(float(size[0]), float(size[1]))
        if window is None:
            window = self._sui.Window(self._ui_context.screen, str(title), position=spy_position, size=spy_size)
            self._ui_windows[key] = window
        else:
            window.title = str(title)
            window.size = spy_size
            window.visible = True
        return window

    def _invoke_ui_callbacks(self, callbacks: list[Callable[[Any], None]], adapter: _SlangImmediateUi, position: str):
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
                    f"slang-ui-callback:{position}:{id(callback)}",
                    f"Slang UI callback at '{position}' failed and was disabled: {exc}",
                )

    def _render_ui(self, width: int, height: int, target_texture: Any | None = None) -> None:
        render_target = self._surface_texture if target_texture is None else target_texture
        if (
            not self.show_ui
            or not self._ui_enabled
            or self._sui is None
            or self._ui_context is None
            or self._ui_adapter is None
            or self._command_encoder is None
            or render_target is None
        ):
            return
        self.ui.io.display_size = (float(width), float(height))
        adapter = self._ui_adapter
        try:
            for window in self._ui_windows.values():
                window.visible = False
            adapter.begin_frame()
            if self._ui_callbacks["side"] or self._ui_callbacks["panel"]:
                side = self._ui_window("side", "OmniSurg", (10, 10), (430, max(180, height - 20)))
                adapter.reset(side, width, height)
                self._invoke_ui_callbacks(self._ui_callbacks["panel"], adapter, "panel")
                self._invoke_ui_callbacks(self._ui_callbacks["side"], adapter, "side")
            if self._ui_callbacks["free"]:
                adapter.reset(self._ui_context.screen, width, height)
                self._invoke_ui_callbacks(self._ui_callbacks["free"], adapter, "free")
            adapter.finish_frame()
            self._ui_context.begin_frame(width, height)
            self._ui_context.end_frame(render_target, self._command_encoder)
        except Exception as exc:
            self._ui_enabled = False
            self._warn_once("slang-ui-render", f"Slang UI disabled after render failure: {exc}")

    def is_running(self) -> bool:
        return not self._closed and not self._window.should_close()

    def is_paused(self) -> bool:
        return bool(self._paused)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        future = getattr(self, "_external_material_reload_future", None)
        if future is not None and not future.done():
            future.cancel()
            if not future.cancelled():
                try:
                    future.result()
                except Exception:
                    pass
        executor = getattr(self, "_external_material_reload_executor", None)
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
            self._external_material_reload_executor = None
        try:
            self._device.wait()
        finally:
            closed_resources: set[int] = set()

            def _close_once(resource: Any | None) -> None:
                if resource is None:
                    return
                marker = id(resource)
                if marker in closed_resources:
                    return
                closed_resources.add(marker)
                _close_slang_resource(resource)

            for shared in self._shared_buffers.values():
                shared.close()
            self._shared_buffers.clear()
            for buffer in getattr(self, "_fallback_shader_buffers", {}).values():
                _close_once(buffer)
            if hasattr(self, "_fallback_shader_buffers"):
                self._fallback_shader_buffers.clear()
            for ring_name in ("_material_color_buffer_ring", "_procedural_param_buffer_ring", "_material_maker_param_buffer_ring"):
                for buffer in getattr(self, ring_name, ()) or ():
                    _close_once(buffer)
                if hasattr(self, ring_name):
                    getattr(self, ring_name).clear()
            for buffer in getattr(self, "_retired_shader_buffers", ()):
                _close_once(buffer)
            if hasattr(self, "_retired_shader_buffers"):
                self._retired_shader_buffers.clear()
            self._clear_external_material_textures()
            for resource_name in (
                "_material_color_buffer",
                "_procedural_param_buffer",
                "_material_maker_param_buffer",
                "_cryo_texture",
                "_neutral_cryo_texture",
                "_environment_texture",
                "_neutral_environment_texture",
                "_point_sphere_vertex_buffer",
                "_point_sphere_index_buffer",
            ):
                resource = getattr(self, resource_name, None)
                if resource is not None:
                    _close_once(resource)
                    setattr(self, resource_name, None)
            if hasattr(self._window, "close"):
                self._window.close()
            self._device.close()

    def register_ui_callback(self, callback, position: str = "side") -> None:
        if position not in self._ui_callbacks:
            raise ValueError(f"Invalid position {position!r}; expected one of {list(self._ui_callbacks)}")
        self._ui_callbacks[position].append(callback)

    def register_key_press(self, callback) -> None:
        self._on_key_press_callback = callback

    def register_key_release(self, callback) -> None:
        self._on_key_release_callback = callback

    def register_mouse_motion(self, callback) -> None:
        self._on_mouse_motion_callback = callback

    def register_mouse_press(self, callback) -> None:
        self._on_mouse_press_callback = callback

    def register_mouse_drag(self, callback) -> None:
        self._on_mouse_drag_callback = callback

    def register_mouse_release(self, callback) -> None:
        self._on_mouse_release_callback = callback

    def _pyglet_symbol_from_slang_key(self, key: Any) -> int | None:
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
                "left_ctrl": "LCTRL",
                "left_control": "LCTRL",
                "right_ctrl": "RCTRL",
                "right_control": "RCTRL",
                "left_alt": "LALT",
                "right_alt": "RALT",
                "left_shift": "LSHIFT",
                "right_shift": "RSHIFT",
            }
            mapped = key_map.get(name)
            if mapped is not None:
                return int(getattr(pyglet_key, mapped, fallback_symbol))
        except Exception:
            pass
        return fallback_symbol

    def _pyglet_modifiers_from_event(self, event: Any) -> int:
        mods = getattr(event, "mods", 0)
        value = int(getattr(mods, "value", mods) or 0)
        try:
            import pyglet

            result = 0
            if value & 1:
                result |= int(pyglet.window.key.MOD_SHIFT)
            if value & 2:
                result |= int(pyglet.window.key.MOD_CTRL)
            if value & 4:
                result |= int(pyglet.window.key.MOD_ALT)
            return result
        except Exception:
            return value

    def _slang_key_name(self, key: Any) -> str:
        name = getattr(key, "name", None)
        if name is None:
            name = str(key)
        return str(name).lower().replace("-", "_")

    def _update_keyboard_camera_motion(self) -> None:
        now = _time.perf_counter()
        dt = max(0.0, min(now - float(self._last_camera_update_time), 0.1))
        self._last_camera_update_time = now
        if not self._camera_key_state:
            return
        move = np.zeros(3, dtype=np.float32)
        if "w" in self._camera_key_state:
            move += self._camera_forward
        if "s" in self._camera_key_state:
            move -= self._camera_forward
        if "d" in self._camera_key_state:
            move += self._camera_right
        if "a" in self._camera_key_state:
            move -= self._camera_right
        if "e" in self._camera_key_state:
            move += self._camera_world_up
        if "q" in self._camera_key_state:
            move -= self._camera_world_up
        length = float(np.linalg.norm(move))
        if length <= 1.0e-6:
            return
        distance = max(float(np.linalg.norm(self._camera_target - self._camera_pos)), self._camera_scene_radius)
        offset = (move / length) * distance * 0.75 * dt
        self._set_camera_look_at(self._camera_pos + offset, self._camera_target + offset)

    def _on_keyboard_event(self, event: Any) -> None:
        captured = False
        if self._ui_enabled and self._ui_context is not None:
            try:
                captured = bool(self._ui_context.handle_keyboard_event(event))
                self._ui_capturing = captured
            except Exception:
                captured = False
        key_name = self._slang_key_name(getattr(event, "key", None))
        is_press = bool(event.is_key_press())
        is_release = bool(event.is_key_release())
        if key_name in {"w", "a", "s", "d", "q", "e"}:
            if is_press:
                self._camera_key_state.add(key_name)
            elif is_release:
                self._camera_key_state.discard(key_name)
        if is_press and not captured:
            if key_name == "space":
                self._paused = not self._paused
            elif key_name == "escape":
                self.close()
        symbol = self._pyglet_symbol_from_slang_key(getattr(event, "key", None))
        if symbol is not None:
            if is_press:
                self._key_handler[symbol] = True
                if not captured and self._on_key_press_callback is not None:
                    self._on_key_press_callback(symbol, self._pyglet_modifiers_from_event(event))
            elif is_release:
                self._key_handler[symbol] = False
                if self._on_key_release_callback is not None:
                    self._on_key_release_callback(symbol, self._pyglet_modifiers_from_event(event))

    def _mouse_button_name(self, button: Any) -> str:
        name = getattr(button, "name", None)
        if name is None:
            name = str(button)
        return str(name).lower().replace("-", "_")

    def _mouse_pos_tuple(self, pos: Any) -> tuple[float, float]:
        if pos is None:
            return (0.0, 0.0)
        x = getattr(pos, "x", None)
        y = getattr(pos, "y", None)
        if x is not None and y is not None:
            return (float(x), float(y))
        try:
            return (float(pos[0]), float(pos[1]))
        except Exception:
            return (0.0, 0.0)

    def _pyglet_mouse_button(self, button_name: str) -> int:
        try:
            import pyglet

            mouse = pyglet.window.mouse
            if button_name == "left":
                return int(mouse.LEFT)
            if button_name == "right":
                return int(mouse.RIGHT)
            if button_name in {"middle", "center"}:
                return int(mouse.MIDDLE)
        except Exception:
            pass
        return {"left": 1, "right": 4, "middle": 2, "center": 2}.get(button_name, 0)

    def _left_mouse_mask(self) -> int:
        return self._pyglet_mouse_button("left")

    def _on_mouse_event(self, event: Any) -> None:
        is_button_up = bool(event.is_button_up())
        is_button_down = bool(event.is_button_down())
        is_move = bool(event.is_move())
        ui_captured = False
        if self._ui_enabled and self._ui_context is not None:
            try:
                ui_captured = bool(self._ui_context.handle_mouse_event(event))
            except Exception:
                ui_captured = False
        captured = bool(ui_captured or getattr(self, "_ui_mouse_active", False))
        self._ui_capturing = captured
        pos = self._mouse_pos_tuple(getattr(event, "pos", None))
        prev = self._mouse_pos
        self._mouse_pos = pos
        dx = 0.0 if prev is None else pos[0] - prev[0]
        dy = 0.0 if prev is None else pos[1] - prev[1]
        modifiers = self._pyglet_modifiers_from_event(event)

        if bool(event.is_scroll()):
            if captured:
                return
            scroll = getattr(event, "scroll", None)
            sy = float(getattr(scroll, "y", 0.0) if scroll is not None else 0.0)
            distance = float(np.linalg.norm(self._camera_target - self._camera_pos))
            factor = float(np.exp(-sy * 0.12))
            self._set_camera_look_at(
                self._camera_target - self._camera_forward * max(0.01, distance * factor),
                self._camera_target,
            )
            return

        button = self._pyglet_mouse_button(self._mouse_button_name(getattr(event, "button", None)))
        if is_button_down:
            if ui_captured:
                self._ui_mouse_active = True
                self._ui_capturing = True
                return
            self._mouse_buttons |= button
            if self._on_mouse_press_callback is not None:
                self._on_mouse_press_callback(pos[0], pos[1], button, modifiers)
            return
        if is_button_up:
            if getattr(self, "_ui_mouse_active", False) or ui_captured:
                self._ui_mouse_active = False
                self._ui_capturing = False
                self._mouse_buttons &= ~button
                return
            if self._on_mouse_release_callback is not None:
                self._on_mouse_release_callback(pos[0], pos[1], button, modifiers)
            self._mouse_buttons &= ~button
            return
        if is_move:
            if getattr(self, "_ui_mouse_active", False) or ui_captured:
                self._ui_capturing = True
                return
            self._ui_capturing = False
            if self._mouse_buttons & self._left_mouse_mask():
                self._orbit_camera(dx, dy)
            if self._mouse_buttons and self._on_mouse_drag_callback is not None:
                self._on_mouse_drag_callback(pos[0], pos[1], dx, dy, self._mouse_buttons, modifiers)
            elif self._on_mouse_motion_callback is not None:
                self._on_mouse_motion_callback(pos[0], pos[1], dx, dy)
