# SPDX-License-Identifier: Apache-2.0
"""Runtime bridge for Material Maker 3D Slang material exports."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GLSL_COMPAT_PREAMBLE = r"""
typealias vec2 = float2;
typealias vec3 = float3;
typealias vec4 = float4;
typealias ivec2 = int2;
typealias ivec3 = int3;
typealias ivec4 = int4;
typealias bvec2 = bool2;
typealias bvec3 = bool3;
typealias bvec4 = bool4;
typealias mat2 = float2x2;
typealias mat3 = float3x3;
typealias mat4 = float4x4;
typealias sampler2D = Texture2D<float4>;
SamplerState g_sampler;

float fract(float x) { return frac(x); }
float2 fract(float2 x) { return frac(x); }
float3 fract(float3 x) { return frac(x); }
float4 fract(float4 x) { return frac(x); }

float mod(float x, float y) { return x - y * floor(x / y); }
float2 mod(float2 x, float2 y) { return x - y * floor(x / y); }
float2 mod(float2 x, float y) { return mod(x, float2(y)); }
float3 mod(float3 x, float3 y) { return x - y * floor(x / y); }
float3 mod(float3 x, float y) { return mod(x, float3(y)); }
float4 mod(float4 x, float4 y) { return x - y * floor(x / y); }
float4 mod(float4 x, float y) { return mod(x, float4(y)); }

float mix(float x, float y, float a) { return lerp(x, y, a); }
float2 mix(float2 x, float2 y, float a) { return lerp(x, y, float2(a)); }
float2 mix(float2 x, float2 y, float2 a) { return lerp(x, y, a); }
float3 mix(float3 x, float3 y, float a) { return lerp(x, y, float3(a)); }
float3 mix(float3 x, float3 y, float3 a) { return lerp(x, y, a); }
float3 mix(float3 x, float3 y, bool3 a) { return select(a, y, x); }
float4 mix(float4 x, float4 y, float a) { return lerp(x, y, float4(a)); }
float4 mix(float4 x, float4 y, float4 a) { return lerp(x, y, a); }
float4 mix(float4 x, float4 y, bool4 a) { return select(a, y, x); }

bool lessThan(float x, float y) { return x < y; }
bool2 lessThan(float2 x, float2 y) { return x < y; }
bool3 lessThan(float3 x, float3 y) { return x < y; }
bool4 lessThan(float4 x, float4 y) { return x < y; }
bool lessThanEqual(float x, float y) { return x <= y; }
bool2 lessThanEqual(float2 x, float2 y) { return x <= y; }
bool3 lessThanEqual(float3 x, float3 y) { return x <= y; }
bool4 lessThanEqual(float4 x, float4 y) { return x <= y; }
bool greaterThan(float x, float y) { return x > y; }
bool2 greaterThan(float2 x, float2 y) { return x > y; }
bool3 greaterThan(float3 x, float3 y) { return x > y; }
bool4 greaterThan(float4 x, float4 y) { return x > y; }
bool greaterThanEqual(float x, float y) { return x >= y; }
bool2 greaterThanEqual(float2 x, float2 y) { return x >= y; }
bool3 greaterThanEqual(float3 x, float3 y) { return x >= y; }
bool4 greaterThanEqual(float4 x, float4 y) { return x >= y; }

float4 texture(sampler2D sampler, float2 uv) { return sampler.SampleLevel(g_sampler, uv, 0.0); }
float4 textureLod(sampler2D sampler, float2 uv, float lod) { return sampler.SampleLevel(g_sampler, uv, lod); }
"""

TEXTURE_SUFFIXES = {
	"texture_albedo": "albedo",
	"texture_orm": "orm",
	"texture_emission": "emission",
	"texture_normal": "normal",
	"texture_depth": "depth",
	"texture_subsurface_scattering": "sss",
}

SAMPLER_RE = re.compile(
	r"(?m)^\s*(?!//)(?:layout\s*\([^)]*\)\s*)?uniform\s+sampler2D\s+([A-Za-z_][A-Za-z0-9_]*)\b"
)
PBR_FIELDS = (
	"albedo",
	"alpha",
	"metallic",
	"roughness",
	"emission",
	"occlusion",
	"height",
	"depth",
	"normal",
	"specular",
	"wetness",
	"clearcoat",
	"clearcoatRoughness",
	"sssStrength",
	"sssColor",
	"sssDepth",
	"sssBoost",
	"thickness",
	"backlight",
)

PARAMETER_SPEC_RE = re.compile(r"(?m)^\s*//\s*MATERIAL_MAKER_PARAMETER_SPEC\s+(\{.*\})\s*$")
RUNTIME_PARAMETER_BLOCK_RE = re.compile(
	r"#ifndef\s+MATERIAL_MAKER_MAX_MATERIAL_CLASSES\s*\n#define\s+MATERIAL_MAKER_MAX_MATERIAL_CLASSES\s+\d+\s*\n#endif\s*\nstatic\s+const\s+uint\s+MATERIAL_MAKER_PARAMETER_COUNT\s*=\s*(?P<count>\d+)\s*;\s*\nStructuredBuffer<float4>\s+material_maker_parameters\s*;\s*\nfloat4\s+__mm_runtime_parameter\s*\(\s*uint\s+parameter_index\s*,\s*uint\s+material_index\s*\)\s*\{\s*\n\s*return\s+material_maker_parameters\s*\[\s*parameter_index\s*\*\s*uint\s*\(\s*MATERIAL_MAKER_MAX_MATERIAL_CLASSES\s*\)\s*\+\s*material_index\s*\]\s*;\s*\n\}\s*\n",
	re.MULTILINE,
)

@dataclass(frozen=True)
class MaterialMakerParameterSpec:
	raw_name: str
	label: str
	value_type: str
	default_value: float | tuple[float, float, float, float]
	group: str
	row_index: int
	min_value: float
	max_value: float
	step: float

	@property
	def default_row(self) -> tuple[float, float, float, float]:
		if isinstance(self.default_value, tuple):
			return self.default_value
		return (float(self.default_value), 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class MaterialMakerSlangInfo:
	path: Path
	fields: tuple[str, ...]
	texture_names: tuple[str, ...]
	uses_timed_call: bool
	source_mtime_ns: int
	texture_mtime_ns: tuple[tuple[str, int | None], ...]
	generated_shader_path: Path
	generated_source_path: Path
	parameter_specs: tuple[MaterialMakerParameterSpec, ...] = ()
	uses_frame_call: bool = False
	uses_timed_frame_call: bool = False
	uses_indexed_frame_call: bool = False
	uses_timed_indexed_frame_call: bool = False
	warnings: tuple[str, ...] = ()


class MaterialMakerSlangError(RuntimeError):
	pass


def _has_signature(source: str, args: str) -> bool:
	return re.search(r"\bevaluateMaterial\s*\(\s*" + args, source) is not None


def _strip_comments(source: str) -> str:
	return re.sub(r"//.*?$|/\*.*?\*/", "", source, flags=re.DOTALL | re.MULTILINE)


def _frame_signature_prefix() -> str:
	vec3_type = r"(?:vec3|float3)"
	identifier = r"[A-Za-z_][A-Za-z0-9_]*"
	return rf"{vec3_type}\s+{identifier}\s*,\s*{vec3_type}\s+{identifier}\s*,\s*{vec3_type}\s+{identifier}"


def _has_frame_signature(source: str) -> bool:
	return _has_signature(
		_strip_comments(source),
		_frame_signature_prefix() + r"(?:\s*,|\s*\))",
	)


def _has_timed_frame_signature(source: str) -> bool:
	identifier = r"[A-Za-z_][A-Za-z0-9_]*"
	return _has_signature(
		_strip_comments(source),
		_frame_signature_prefix() + rf"\s*,\s*float\s+{identifier}",
	)


def _has_indexed_frame_signature(source: str) -> bool:
	identifier = r"[A-Za-z_][A-Za-z0-9_]*"
	return _has_signature(
		_strip_comments(source),
		_frame_signature_prefix() + rf"\s*,\s*(?:uint|int)\s+{identifier}",
	)


def _has_timed_indexed_frame_signature(source: str) -> bool:
	identifier = r"[A-Za-z_][A-Za-z0-9_]*"
	vec4_type = r"(?:vec4|float4)"
	return _has_signature(
		_strip_comments(source),
		_frame_signature_prefix()
		+ rf"\s*,\s*float\s+{identifier}\s*,\s*float\s+{identifier}\s*,\s*{vec4_type}\s+{identifier}\s*,\s*(?:uint|int)\s+{identifier}",
	)


def parse_material_fields(source: str) -> tuple[str, ...]:
	match = re.search(r"\bstruct\s+MMMaterial\s*\{(?P<body>.*?)\}\s*;", source, re.DOTALL)
	if match is None:
		return ()
	body = _strip_comments(match.group("body"))
	field_pattern = re.compile(
		r"\b(?:float|float[234]|vec[234]|int|uint|bool|bvec[234]|ivec[234]|mat[234])\s+"
		r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]+\])?\s*;"
	)
	return tuple(dict.fromkeys(field_pattern.findall(body)))


def parse_texture_names(source: str) -> tuple[str, ...]:
	return tuple(dict.fromkeys(SAMPLER_RE.findall(source)))


def texture_path_for(source_path: Path, texture_name: str) -> Path:
	stem = source_path.with_suffix("").name
	if re.fullmatch(r"texture_\d+", texture_name):
		suffix = texture_name
	else:
		suffix = TEXTURE_SUFFIXES.get(texture_name, texture_name.removeprefix("texture_"))
	return source_path.with_name(f"{stem}_{suffix}.png")


def _parameter_default_row(default_value: Any) -> tuple[float, float, float, float]:
	if isinstance(default_value, list):
		components = list(default_value[:4])
	else:
		components = [default_value, 0.0, 0.0, 0.0]
	while len(components) < 4:
		components.append(0.0)
	result: list[float] = []
	for component in components[:4]:
		try:
			value = float(component)
		except (TypeError, ValueError):
			value = 0.0
		result.append(value)
	return (result[0], result[1], result[2], result[3])


def _format_float(value: float) -> str:
	return f"{float(value):.9f}"


def _format_float4(row: tuple[float, float, float, float]) -> str:
	return "float4(%s, %s, %s, %s)" % tuple(_format_float(component) for component in row)


def _finite_float(value: Any, default: float) -> float:
	try:
		result = float(value)
	except (TypeError, ValueError):
		return float(default)
	return result if result == result and result not in (float("inf"), float("-inf")) else float(default)


def parse_material_maker_parameter_specs(source: str) -> tuple[MaterialMakerParameterSpec, ...]:
	specs: list[MaterialMakerParameterSpec] = []
	for match in PARAMETER_SPEC_RE.finditer(source):
		try:
			raw = json.loads(match.group(1))
		except json.JSONDecodeError:
			continue
		value_type = str(raw.get("value_type", "float"))
		if value_type not in {"float", "vec4"}:
			continue
		default_row = _parameter_default_row(raw.get("default_value", 0.0))
		default_value: float | tuple[float, float, float, float]
		if value_type == "vec4":
			default_value = default_row
		else:
			default_value = float(default_row[0])
		raw_name = str(raw.get("raw_name", raw.get("shader_name", f"parameter_{len(specs)}")))
		min_value = _finite_float(raw.get("min"), 0.0)
		max_default = max(default_row) if value_type == "vec4" else default_row[0]
		max_value = max(_finite_float(raw.get("max"), 1.0), max_default)
		if value_type == "float" and raw_name == "normal":
			max_value = 1.0
		step = _finite_float(raw.get("step"), 0.01)
		specs.append(
			MaterialMakerParameterSpec(
				raw_name=raw_name,
				label=str(raw.get("label", raw.get("raw_name", f"Parameter {len(specs)}"))),
				value_type=value_type,
				default_value=default_value,
				group=str(raw.get("group", "material")),
				row_index=int(raw.get("row_index", len(specs))),
				min_value=float(min_value),
				max_value=float(max_value),
				step=float(step),
			)
		)
	return tuple(sorted(specs, key=lambda spec: spec.row_index))


def _material_maker_parameter_defaults(source: str) -> tuple[tuple[float, float, float, float], ...]:
	return tuple(spec.default_row for spec in parse_material_maker_parameter_specs(source))


def _sanitize_source_for_slang(source: str) -> str:
	source = re.sub(r"(?m)^\s*import\s+glsl\s*;\s*", "", source)
	return _add_material_maker_runtime_default_fallback(source)


def _add_material_maker_runtime_default_fallback(source: str) -> str:
	if "StructuredBuffer<float4> material_maker_parameters" not in source:
		return source
	if "material_maker_parameter_defaults" in source:
		return source
	defaults = _material_maker_parameter_defaults(source)
	if not defaults:
		return source

	def replace_runtime_block(match: re.Match[str]) -> str:
		count = int(match.group("count"))
		rows = list(defaults[:count])
		while len(rows) < count:
			rows.append((0.0, 0.0, 0.0, 0.0))
		default_lines = ",\n".join(f"\t{_format_float4(row)}" for row in rows)
		return (
			"#ifndef MATERIAL_MAKER_MAX_MATERIAL_CLASSES\n"
			"#define MATERIAL_MAKER_MAX_MATERIAL_CLASSES 256\n"
			"#endif\n"
			f"static const uint MATERIAL_MAKER_PARAMETER_COUNT = {count};\n"
			f"static const float4 material_maker_parameter_defaults[{count}] = {{\n{default_lines}\n}};\n"
			"#ifdef MATERIAL_MAKER_ENABLE_RUNTIME_PARAMETERS\n"
			"StructuredBuffer<float4> material_maker_parameters;\n"
			"float4 __mm_runtime_parameter(uint parameter_index, uint material_index) {\n"
			"\treturn material_maker_parameters[parameter_index*uint(MATERIAL_MAKER_MAX_MATERIAL_CLASSES)+material_index];\n"
			"}\n"
			"#else\n"
			"float4 __mm_runtime_parameter(uint parameter_index, uint material_index) {\n"
			"\treturn material_maker_parameter_defaults[parameter_index];\n"
			"}\n"
			"#endif\n"
		)

	return RUNTIME_PARAMETER_BLOCK_RE.sub(replace_runtime_block, source, count=1)


def _texture_mtimes(source_path: Path, texture_names: tuple[str, ...]) -> tuple[tuple[str, int | None], ...]:
	result: list[tuple[str, int | None]] = []
	for name in texture_names:
		path = texture_path_for(source_path, name)
		result.append((name, path.stat().st_mtime_ns if path.exists() else None))
	return tuple(result)


def analyze_material_source(source_path: Path, work_dir: Path) -> MaterialMakerSlangInfo:
	path = source_path.expanduser().resolve()
	if not path.exists():
		raise MaterialMakerSlangError(f"Material Maker Slang file does not exist: {path}")
	source = path.read_text(encoding="utf-8")
	if not _has_signature(source, r"vec3\s+texcoords_3d(?:\s*,|\s*\))"):
		raise MaterialMakerSlangError(f"Expected evaluateMaterial(vec3 texcoords_3d) in {path}")
	fields = parse_material_fields(source)
	texture_names = parse_texture_names(source)
	parameter_specs = parse_material_maker_parameter_specs(source)
	warnings: list[str] = []
	return MaterialMakerSlangInfo(
		path=path,
		fields=fields,
		texture_names=texture_names,
		uses_timed_call=_has_signature(source, r"vec3\s+texcoords_3d\s*,\s*float\s+elapsed_time"),
		source_mtime_ns=path.stat().st_mtime_ns,
		texture_mtime_ns=_texture_mtimes(path, texture_names),
		generated_shader_path=work_dir / "hex_cryo_surface_material_maker.slang",
		generated_source_path=work_dir / "material_maker_source.slang",
		parameter_specs=parameter_specs,
		uses_frame_call=_has_frame_signature(source),
		uses_timed_frame_call=_has_timed_frame_signature(source),
		uses_indexed_frame_call=_has_indexed_frame_signature(source),
		uses_timed_indexed_frame_call=_has_timed_indexed_frame_signature(source),
		warnings=tuple(warnings),
	)


def material_source_changed(info: MaterialMakerSlangInfo) -> bool:
	if not info.path.exists() or info.path.stat().st_mtime_ns != info.source_mtime_ns:
		return True
	return _texture_mtimes(info.path, info.texture_names) != info.texture_mtime_ns


def _macro_bool(name: str, enabled: bool) -> str:
	return f"#define {name} {1 if enabled else 0}"


def write_material_bridge(source_path: Path, work_dir: Path, base_shader_name: str = "hex_cryo_surface.slang") -> MaterialMakerSlangInfo:
	work_dir.mkdir(parents=True, exist_ok=True)
	info = analyze_material_source(source_path, work_dir)
	source = info.path.read_text(encoding="utf-8")
	info.generated_source_path.write_text(
		f"{GLSL_COMPAT_PREAMBLE}\n\n// Sanitized copy of {info.path.name}.\n{_sanitize_source_for_slang(source)}",
		encoding="utf-8",
	)
	evaluate_macro = (
		"#define MM_EVALUATE_MATERIAL(coord, material_index) evaluateMaterial((coord), procedural_time, 0.0, float4(0.0))"
		if info.uses_timed_call
		else "#define MM_EVALUATE_MATERIAL(coord, material_index) evaluateMaterial((coord))"
	)
	if info.uses_timed_indexed_frame_call:
		evaluate_frame_macro = (
			"#define MM_EVALUATE_MATERIAL_FRAME(coord, tangent, binormal, material_index) "
			"evaluateMaterial((coord), (tangent), (binormal), procedural_time, 0.0, float4(0.0), uint(material_index))"
		)
	elif info.uses_indexed_frame_call:
		evaluate_frame_macro = (
			"#define MM_EVALUATE_MATERIAL_FRAME(coord, tangent, binormal, material_index) "
			"evaluateMaterial((coord), (tangent), (binormal), uint(material_index))"
		)
	elif info.uses_timed_frame_call:
		evaluate_frame_macro = (
			"#define MM_EVALUATE_MATERIAL_FRAME(coord, tangent, binormal, material_index) "
			"evaluateMaterial((coord), (tangent), (binormal), procedural_time, 0.0, float4(0.0))"
		)
	elif info.uses_frame_call:
		evaluate_frame_macro = (
			"#define MM_EVALUATE_MATERIAL_FRAME(coord, tangent, binormal, material_index) "
			"evaluateMaterial((coord), (tangent), (binormal))"
		)
	else:
		evaluate_frame_macro = (
			"#define MM_EVALUATE_MATERIAL_FRAME(coord, tangent, binormal, material_index) "
			"MM_EVALUATE_MATERIAL(coord, material_index)"
		)
	field_macros = "\n".join(_macro_bool(f"MM_HAS_{field.upper()}", field in info.fields) for field in PBR_FIELDS)
	source_include = info.generated_source_path.name.replace("\\", "\\\\").replace('"', '\\"')
	base_include = base_shader_name.replace("\\", "\\\\").replace('"', '\\"')
	info.generated_shader_path.write_text(
		f"""// Generated Material Maker bridge. Do not edit by hand.
#define MM_EXTERNAL_MATERIAL 1
{"#define MATERIAL_MAKER_ENABLE_RUNTIME_PARAMETERS 1" if info.parameter_specs else ""}
{field_macros}
{evaluate_macro}
{evaluate_frame_macro}
#include "{source_include}"
#include "{base_include}"
""",
		encoding="utf-8",
	)
	return info
