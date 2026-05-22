from __future__ import annotations

import ctypes
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import omnisurg.rendering.slang_cryo as slang_mod
from omnisurg.rendering.slang import SlangRenderer
from omnisurg.hex.ui import UiState
from omnisurg.hex.app_runtime import (
    _camera_local_offsets,
    _camera_points_from_local_offsets,
    _camera_transform_points_between_frames,
    _delete_stats_dirty,
    _discover_hdri_maps,
    _hdri_map_choice_index,
    _hdri_map_choice_labels,
    _hdri_map_choice_paths,
    _load_segmentation_panel_settings,
    _save_segmentation_panel_settings,
    build_timer_panel_snapshot,
)
from omnisurg.rendering.material_maker_slang import (
    MaterialMakerSlangInfo,
    material_source_changed,
    parse_material_maker_parameter_specs,
    texture_path_for,
    write_material_bridge,
)
from omnisurg.rendering.slang_cryo import (
    PROCEDURAL_MATERIAL_PARAM_DEFAULTS,
    SLANG_SURFACE_DEBUG_VIEW_HEIGHT,
    SLANG_SURFACE_DEBUG_VIEW_LABELS,
    SLANG_SURFACE_DEBUG_VIEW_MATERIAL_INDEX,
    _ExternalMaterialReloadResult,
    _shared_buffer_cache_key,
    _SlangImmediateUi,
    build_material_color_buffer,
    build_material_maker_param_buffer,
    build_procedural_material_param_buffer,
    build_procedural_uv3_noise_scale,
    build_unit_sphere_mesh,
    clamp_material_maker_params,
    clamp_procedural_material_params,
    clamp_slang_surface_debug_view,
    is_slang_backend,
    make_default_material_maker_params,
    make_default_procedural_materials,
    prepare_cryo_texture_upload,
)

RendererUnderTest = SlangRenderer


def test_prepare_cryo_texture_upload_transposes_xyz_volume_to_slang_depth_layout():
    host = np.zeros((2, 3, 4, 3), dtype=np.uint8)
    for x in range(host.shape[0]):
        for y in range(host.shape[1]):
            for z in range(host.shape[2]):
                host[x, y, z] = [x + 10, y + 20, z + 30]

    upload = prepare_cryo_texture_upload(host)

    assert upload.shape == (4, 3, 2, 4)
    assert upload.flags.c_contiguous
    np.testing.assert_array_equal(upload[3, 2, 1, :3], host[1, 2, 3])
    np.testing.assert_array_equal(upload[..., 3], np.full((4, 3, 2), 255, dtype=np.uint8))


def test_prepare_cryo_texture_upload_rejects_non_rgb_uint8_volume():
    with pytest.raises(ValueError, match="expected cryo host volume"):
        prepare_cryo_texture_upload(np.zeros((2, 3, 4), dtype=np.float32))


def test_material_maker_bridge_generates_height_macros_and_texture_paths(tmp_path: Path):
    source = tmp_path / "organ.slang"
    source.write_text(
        """
import glsl;
uniform sampler2D texture_1;
struct MMMaterial {
    vec3 albedo;
    float alpha;
    float roughness;
    float height;
};
MMMaterial evaluateMaterial(vec3 texcoords_3d, float elapsed_time, float variation, vec4 controlled_variation) {
    MMMaterial material;
    material.albedo = texture(texture_1, texcoords_3d.xy).rgb;
    material.alpha = 1.0;
    material.roughness = 0.5;
    material.height = texcoords_3d.z;
    return material;
}
MMMaterial evaluateMaterial(vec3 texcoords_3d) {
    return evaluateMaterial(texcoords_3d, 0.0, 0.0, vec4(0.0));
}
""",
        encoding="utf-8",
    )

    info = write_material_bridge(source, tmp_path / "generated")

    shader = info.generated_shader_path.read_text(encoding="utf-8")
    sanitized = info.generated_source_path.read_text(encoding="utf-8")
    assert info.fields == ("albedo", "alpha", "roughness", "height")
    assert info.texture_names == ("texture_1",)
    assert texture_path_for(source, "texture_1").name == "organ_texture_1.png"
    assert material_source_changed(info) is False
    assert "#define MM_EXTERNAL_MATERIAL 1" in shader
    assert "#define MM_HAS_HEIGHT 1" in shader
    assert "#define MM_HAS_DEPTH 0" in shader
    assert "procedural_time" in shader
    assert "typealias vec3 = float3;" in sanitized
    assert "import glsl" not in sanitized


def test_material_maker_bridge_adds_defaults_for_unguarded_runtime_parameters(tmp_path: Path):
    source = tmp_path / "runtime_params.slang"
    source.write_text(
        """
import glsl;
// MATERIAL_MAKER_PARAMETERS_BEGIN
// MATERIAL_MAKER_PARAMETER_SPEC {"default_value":0.52,"group":"material","label":"Roughness","raw_name":"roughness","row_index":0,"value_type":"float"}
// MATERIAL_MAKER_PARAMETER_SPEC {"default_value":[1.0,0.5,0.25,1.0],"group":"material","label":"Albedo Color","raw_name":"albedo_color","row_index":1,"value_type":"vec4"}
// MATERIAL_MAKER_PARAMETERS_END
#ifndef MATERIAL_MAKER_MAX_MATERIAL_CLASSES
#define MATERIAL_MAKER_MAX_MATERIAL_CLASSES 256
#endif
static const uint MATERIAL_MAKER_PARAMETER_COUNT = 2;
StructuredBuffer<float4> material_maker_parameters;
float4 __mm_runtime_parameter(uint parameter_index, uint material_index) {
    return material_maker_parameters[parameter_index*uint(MATERIAL_MAKER_MAX_MATERIAL_CLASSES)+material_index];
}
float __mm_runtime_float(uint parameter_index, uint material_index) {
    return __mm_runtime_parameter(parameter_index, material_index).x;
}
float4 __mm_runtime_vec4(uint parameter_index, uint material_index) {
    return __mm_runtime_parameter(parameter_index, material_index);
}
struct MMMaterial {
    vec3 albedo;
    float roughness;
};
MMMaterial evaluateMaterial(vec3 texcoords_3d) {
    MMMaterial material;
    material.albedo = __mm_runtime_vec4(1, 0).rgb + texcoords_3d * 0.0;
    material.roughness = __mm_runtime_float(0, 0);
    return material;
}
""",
        encoding="utf-8",
    )

    info = write_material_bridge(source, tmp_path / "generated")
    sanitized = info.generated_source_path.read_text(encoding="utf-8")

    assert "static const float4 material_maker_parameter_defaults[2]" in sanitized
    assert "float4(0.520000000, 0.000000000, 0.000000000, 0.000000000)" in sanitized
    assert "float4(1.000000000, 0.500000000, 0.250000000, 1.000000000)" in sanitized
    assert "#ifdef MATERIAL_MAKER_ENABLE_RUNTIME_PARAMETERS" in sanitized
    assert "return material_maker_parameter_defaults[parameter_index];" in sanitized


def test_material_maker_bridge_accepts_depth_alias_and_exports_vector_normal(tmp_path: Path):
    source = tmp_path / "depth_alias.slang"
    source.write_text(
        """
struct MMMaterial {
    vec3 albedo;
    float depth;
    vec3 normal;
};
MMMaterial evaluateMaterial(vec3 texcoords_3d) {
    MMMaterial material;
    material.albedo = texcoords_3d;
    material.depth = texcoords_3d.x;
    material.normal = vec3(0.0, 0.0, 1.0);
    return material;
}
""",
        encoding="utf-8",
    )

    info = write_material_bridge(source, tmp_path / "generated")
    shader = info.generated_shader_path.read_text(encoding="utf-8")

    assert "#define MM_HAS_HEIGHT 0" in shader
    assert "#define MM_HAS_DEPTH 1" in shader
    assert "#define MM_HAS_NORMAL 1" in shader
    assert not any("normal is ignored" in warning for warning in info.warnings)


def test_material_maker_bridge_depth_only_legacy_source_keeps_height_fallback(tmp_path: Path):
    source = tmp_path / "legacy_depth.slang"
    source.write_text(
        """
struct MMMaterial {
    vec3 albedo;
    float depth;
};
MMMaterial evaluateMaterial(vec3 texcoords_3d) {
    MMMaterial material;
    material.albedo = texcoords_3d;
    material.depth = texcoords_3d.z;
    return material;
}
""",
        encoding="utf-8",
    )

    info = write_material_bridge(source, tmp_path / "generated")
    shader = info.generated_shader_path.read_text(encoding="utf-8")

    assert "#define MM_HAS_DEPTH 1" in shader
    assert "#define MM_HAS_NORMAL 0" in shader
    assert (
        "#define MM_EVALUATE_MATERIAL_FRAME(coord, tangent, binormal, material_index) "
        "MM_EVALUATE_MATERIAL(coord, material_index)"
    ) in shader


def test_material_maker_bridge_generates_frame_evaluate_macro_for_full_overload(tmp_path: Path):
    source = tmp_path / "framed.slang"
    source.write_text(
        """
struct MMMaterial {
    vec3 albedo;
    vec3 normal;
    float height;
};
MMMaterial evaluateMaterial(
    vec3 texcoords_3d,
    vec3 tangent_3d,
    vec3 binormal_3d,
    float elapsed_time,
    float variation,
    vec4 controlled_variation) {
    MMMaterial material;
    material.albedo = texcoords_3d + tangent_3d * 0.0 + binormal_3d * 0.0;
    material.normal = vec3(0.5, 0.5, 1.0);
    material.height = elapsed_time + variation + controlled_variation.x;
    return material;
}
MMMaterial evaluateMaterial(vec3 texcoords_3d) {
    return evaluateMaterial(texcoords_3d, vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), 0.0, 0.0, vec4(0.0));
}
""",
        encoding="utf-8",
    )

    info = write_material_bridge(source, tmp_path / "generated")
    shader = info.generated_shader_path.read_text(encoding="utf-8")

    assert info.uses_frame_call is True
    assert info.uses_timed_frame_call is True
    assert "#define MM_HAS_NORMAL 1" in shader
    assert (
        "#define MM_EVALUATE_MATERIAL_FRAME(coord, tangent, binormal, material_index) "
        "evaluateMaterial((coord), (tangent), (binormal), procedural_time, 0.0, float4(0.0))"
    ) in shader


def test_material_maker_bridge_uses_material_index_for_indexed_frame_overload(tmp_path: Path):
    source = tmp_path / "indexed_framed.slang"
    source.write_text(
        """
struct MMMaterial {
    vec3 albedo;
};
MMMaterial evaluateMaterial(vec3 texcoords_3d, vec3 tangent_3d, vec3 binormal_3d, uint material_index) {
    MMMaterial material;
    material.albedo = texcoords_3d + tangent_3d * 0.0 + binormal_3d * 0.0 + float(material_index) * 0.0;
    return material;
}
MMMaterial evaluateMaterial(vec3 texcoords_3d) {
    return evaluateMaterial(texcoords_3d, vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0), 0);
}
""",
        encoding="utf-8",
    )

    info = write_material_bridge(source, tmp_path / "generated")
    shader = info.generated_shader_path.read_text(encoding="utf-8")

    assert info.uses_indexed_frame_call is True
    assert (
        "#define MM_EVALUATE_MATERIAL_FRAME(coord, tangent, binormal, material_index) "
        "evaluateMaterial((coord), (tangent), (binormal), uint(material_index))"
    ) in shader


def _material_maker_specs():
    return parse_material_maker_parameter_specs(
        """
// MATERIAL_MAKER_PARAMETER_SPEC {"default_value":0.52,"group":"material","label":"Roughness","max":1.0,"min":0.0,"raw_name":"roughness","row_index":0,"step":0.01,"value_type":"float"}
// MATERIAL_MAKER_PARAMETER_SPEC {"default_value":[1.0,0.5,0.25,1.0],"group":"procedural","label":"Albedo Color","max":1.0,"min":0.0,"raw_name":"albedo_color","row_index":1,"step":0.05,"value_type":"vec4"}
"""
    )


def test_material_maker_parameter_specs_parse_defaults_and_ui_metadata():
    roughness, albedo = _material_maker_specs()

    assert roughness.raw_name == "roughness"
    assert roughness.label == "Roughness"
    assert roughness.value_type == "float"
    assert roughness.group == "material"
    assert roughness.row_index == 0
    assert roughness.default_row == pytest.approx((0.52, 0.0, 0.0, 0.0))
    assert roughness.min_value == pytest.approx(0.0)
    assert roughness.max_value == pytest.approx(1.0)
    assert roughness.step == pytest.approx(0.01)
    assert albedo.raw_name == "albedo_color"
    assert albedo.value_type == "vec4"
    assert albedo.group == "procedural"
    assert albedo.default_row == pytest.approx((1.0, 0.5, 0.25, 1.0))


def test_material_maker_normal_runtime_slider_is_capped_to_unit_range():
    normal, normal_blend = parse_material_maker_parameter_specs(
        """
// MATERIAL_MAKER_PARAMETER_SPEC {"default_value":0.38,"group":"material","label":"Normal","max":10.0,"min":0.0,"raw_name":"normal","row_index":0,"step":0.01,"value_type":"float"}
// MATERIAL_MAKER_PARAMETER_SPEC {"default_value":0.5,"group":"material","label":"Normal Blend","max":10.0,"min":0.0,"raw_name":"normal_blend","row_index":1,"step":0.01,"value_type":"float"}
"""
    )

    assert normal.max_value == pytest.approx(1.0)
    assert normal_blend.max_value == pytest.approx(10.0)
    assert clamp_material_maker_params({"normal": 5.0}, (normal,))["normal"] == pytest.approx(1.0)


def test_material_maker_parameter_buffer_packs_parameter_major_class_rows():
    specs = _material_maker_specs()
    materials = [
        {"roughness": 0.25, "albedo_color": [0.1, 0.2, 0.3, 0.4]},
        {"roughness": 0.75},
    ]

    packed = build_material_maker_param_buffer(specs, materials, material_count=3, max_material_classes=3)

    assert packed.shape == (6, 4)
    assert packed.dtype == np.float32
    np.testing.assert_allclose(packed[0], [0.25, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(packed[1], [0.75, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(packed[2], [0.52, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(packed[3], [0.1, 0.2, 0.3, 0.4])
    np.testing.assert_allclose(packed[4], [1.0, 0.5, 0.25, 1.0])
    np.testing.assert_allclose(packed[5], [1.0, 0.5, 0.25, 1.0])


def test_material_maker_bridge_detects_tissue_fields(tmp_path: Path):
    source = tmp_path / "tissue.slang"
    source.write_text(
        """
struct MMMaterial {
    vec3 albedo;
    float specular;
    float wetness;
    float clearcoat;
    float clearcoatRoughness;
    float sssStrength;
    vec3 sssColor;
    float sssDepth;
    float sssBoost;
    float thickness;
    vec3 backlight;
};
MMMaterial evaluateMaterial(vec3 texcoords_3d) {
    MMMaterial material;
    material.albedo = texcoords_3d;
    material.specular = 0.5;
    material.wetness = 1.0;
    material.clearcoat = 1.0;
    material.clearcoatRoughness = 0.2;
    material.sssStrength = 0.4;
    material.sssColor = vec3(1.0, 0.3, 0.2);
    material.sssDepth = 0.1;
    material.sssBoost = 0.2;
    material.thickness = 0.8;
    material.backlight = vec3(0.1);
    return material;
}
""",
        encoding="utf-8",
    )

    info = write_material_bridge(source, tmp_path / "generated")
    shader = info.generated_shader_path.read_text(encoding="utf-8")

    assert "wetness" in info.fields
    assert "#define MM_HAS_SPECULAR 1" in shader
    assert "#define MM_HAS_WETNESS 1" in shader
    assert "#define MM_HAS_CLEARCOAT 1" in shader
    assert "#define MM_HAS_CLEARCOATROUGHNESS 1" in shader
    assert "#define MM_HAS_SSSSTRENGTH 1" in shader
    assert "#define MM_HAS_BACKLIGHT 1" in shader


def test_procedural_material_params_default_and_clamp_to_shader_buffer_shape():
    clamped = clamp_procedural_material_params(
        {
            "scale": -10.0,
            "grain": 2.0,
            "tint": float("nan"),
            "roughness": -0.5,
            "wetness": 4.0,
            "height": 4.0,
            "normal": -0.5,
        }
    )

    assert clamped["scale"] == pytest.approx(1.0)
    assert clamped["grain"] == pytest.approx(1.0)
    assert clamped["tint"] == pytest.approx(PROCEDURAL_MATERIAL_PARAM_DEFAULTS["tint"])
    assert clamped["roughness"] == pytest.approx(0.0)
    assert clamped["wetness"] == pytest.approx(1.0)
    assert clamped["height"] == pytest.approx(1.0)
    assert clamped["normal"] == pytest.approx(0.0)

    packed = build_procedural_material_param_buffer([clamped], material_count=2)
    assert packed.shape == (4, 4)
    assert packed.dtype == np.float32
    np.testing.assert_allclose(packed[0], [1.0, 1.0, PROCEDURAL_MATERIAL_PARAM_DEFAULTS["tint"], 0.0])
    np.testing.assert_allclose(packed[1], [1.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(
        packed[2],
        [
            PROCEDURAL_MATERIAL_PARAM_DEFAULTS["scale"],
            PROCEDURAL_MATERIAL_PARAM_DEFAULTS["grain"],
            PROCEDURAL_MATERIAL_PARAM_DEFAULTS["tint"],
            PROCEDURAL_MATERIAL_PARAM_DEFAULTS["roughness"],
        ],
    )
    np.testing.assert_allclose(
        packed[3],
        [
            PROCEDURAL_MATERIAL_PARAM_DEFAULTS["wetness"],
            PROCEDURAL_MATERIAL_PARAM_DEFAULTS["height"],
            PROCEDURAL_MATERIAL_PARAM_DEFAULTS["normal"],
            0.0,
        ],
    )


def test_material_color_buffer_clamps_to_rgba_rows():
    packed = build_material_color_buffer(np.asarray([[2.0, -1.0, 0.25]], dtype=np.float32), material_count=2)

    assert packed.shape == (2, 4)
    np.testing.assert_allclose(packed[0], [1.0, 0.0, 0.25, 1.0])
    np.testing.assert_allclose(packed[1], [1.0, 1.0, 1.0, 1.0])


def test_slang_surface_debug_view_clamps_and_maps_legacy_height_flag():
    assert clamp_slang_surface_debug_view(None, height_debug=True) == SLANG_SURFACE_DEBUG_VIEW_HEIGHT
    assert clamp_slang_surface_debug_view(-10) == 0
    assert clamp_slang_surface_debug_view(999) == SLANG_SURFACE_DEBUG_VIEW_MATERIAL_INDEX


def test_procedural_uv3_noise_scale_uses_grid_extent_ratios():
    assert build_procedural_uv3_noise_scale((5, 7, 9)) == pytest.approx((0.5, 0.75, 1.0))
    assert build_procedural_uv3_noise_scale(np.asarray([11, 6, 3], dtype=np.int32)) == pytest.approx((1.0, 0.5, 0.2))
    assert build_procedural_uv3_noise_scale((1, 1, 1)) == pytest.approx((1.0, 1.0, 1.0))
    assert build_procedural_uv3_noise_scale(None) == pytest.approx((1.0, 1.0, 1.0))


def test_delete_stats_dirty_uses_deletion_dirty_domains():
    assert _delete_stats_dirty(SimpleNamespace(_dirty_domains={"stats"})) is True
    assert _delete_stats_dirty(SimpleNamespace(_dirty_domains={"cells"})) is False
    assert _delete_stats_dirty(SimpleNamespace()) is True


def _panel_ui() -> UiState:
    return UiState(
        material_names=["background", "skin"],
        material_stiffness_scale=[1.0, 2.0],
        material_visible=[True, False],
        material_cuttable=[False, True],
        material_locked=[False, True],
        material_colors=[(1.0, 1.0, 1.0), (0.8, 0.4, 0.2)],
        material_procedural=make_default_procedural_materials(2),
    )


def test_hdri_map_choices_scan_folder_and_keep_current_outside_folder(tmp_path: Path):
    folder = tmp_path / "environments"
    folder.mkdir()
    first = folder / "b_studio.hdr"
    second = folder / "a_forest.EXR"
    ignored = folder / "notes.txt"
    outside = tmp_path / "custom.hdr"
    for path in (first, second, ignored, outside):
        path.write_text("", encoding="utf-8")

    discovered = _discover_hdri_maps(folder)
    assert discovered == (second, first)

    choices = _hdri_map_choice_paths(folder, outside)
    assert choices == (None, second, first, outside)
    assert _hdri_map_choice_labels(choices) == ("None", "a_forest.EXR", "b_studio.hdr", "custom.hdr")
    assert _hdri_map_choice_index(choices, first) == 2
    assert _hdri_map_choice_index(choices, "") == 0


def test_segmentation_panel_settings_save_writes_v2_slang_surface(tmp_path: Path):
    ui = _panel_ui()
    ui.slang_procedural_surface = False
    ui.slang_procedural_world_space = True
    ui.slang_surface_lighting = False
    ui.slang_environment_map = "environments/epping_forest_01_1k.hdr"
    ui.slang_debug_view = SLANG_SURFACE_DEBUG_VIEW_HEIGHT
    ui.slang_cryo_mix = 0.25
    ui.slang_state_overlay_strength = 0.75
    assert ui.material_procedural is not None
    ui.material_procedural[1]["scale"] = 42.0
    settings_path = tmp_path / "settings.json"

    _save_segmentation_panel_settings(settings_path, ui)

    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["version"] == 2
    assert payload["slang_surface"] == {
        "cryo_mix": 0.25,
        "environment_background": True,
        "environment_map": "environments/epping_forest_01_1k.hdr",
        "environment_intensity": 1.0,
        "environment_pitch_degrees": -90.0,
        "environment_rotation_degrees": 0.0,
        "ambient_light_enabled": False,
        "environment_lighting_enabled": True,
        "fill_light_enabled": True,
        "debug_view": SLANG_SURFACE_DEBUG_VIEW_HEIGHT,
        "key_light_enabled": True,
        "lighting_enabled": False,
        "procedural_enabled": False,
        "procedural_world_space": True,
        "state_overlay_strength": 0.75,
    }
    assert payload["materials"][1]["color"] == [0.8, 0.4, 0.2]
    assert payload["materials"][1]["procedural"]["scale"] == pytest.approx(42.0)


def test_segmentation_panel_settings_save_load_preserves_material_maker_params(tmp_path: Path):
    specs = _material_maker_specs()
    ui = _panel_ui()
    ui.material_maker_parameter_specs = specs
    ui.material_maker_params = make_default_material_maker_params(2, specs)
    ui.material_maker_params[1]["roughness"] = 0.42
    ui.material_maker_params[1]["albedo_color"] = [0.2, 0.3, 0.4, 0.5]
    settings_path = tmp_path / "settings-mm.json"

    _save_segmentation_panel_settings(settings_path, ui)

    payload = json.loads(settings_path.read_text(encoding="utf-8"))
    assert payload["materials"][1]["material_maker"] == {
        "albedo_color": [0.2, 0.3, 0.4, 0.5],
        "roughness": 0.42,
    }

    loaded = _panel_ui()
    loaded.material_maker_parameter_specs = specs
    applied = _load_segmentation_panel_settings(settings_path, loaded)

    assert applied == 2
    assert loaded.material_maker_params_revision == 1
    assert loaded.material_maker_params is not None
    assert loaded.material_maker_params[0] == clamp_material_maker_params(None, specs)
    assert loaded.material_maker_params[1] == {
        "albedo_color": [0.2, 0.3, 0.4, 0.5],
        "roughness": 0.42,
    }


def test_segmentation_panel_settings_loads_v1_without_procedural_fields(tmp_path: Path):
    ui = _panel_ui()
    assert ui.material_procedural is not None
    original_params = [dict(item) for item in ui.material_procedural]
    settings_path = tmp_path / "settings-v1.json"
    settings_path.write_text(
        json.dumps(
            {
                "version": 1,
                "materials": [
                    {
                        "id": 1,
                        "name": "skin",
                        "visible": True,
                        "cuttable": False,
                        "locked": False,
                        "stiffness_scale": 12.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    applied = _load_segmentation_panel_settings(settings_path, ui)

    assert applied == 1
    assert ui.material_visible == [True, True]
    assert ui.material_cuttable == [False, False]
    assert ui.material_locked == [False, False]
    assert ui.material_stiffness_scale == [1.0, 5.0]
    assert ui.slang_procedural_world_space is False
    assert ui.slang_debug_view == 0
    assert ui.slang_environment_pitch_degrees == pytest.approx(-90.0)
    assert ui.material_procedural == original_params
    assert ui.material_procedural_revision == 0
    assert ui.material_maker_params_revision == 0


def test_segmentation_panel_settings_loads_v2_slang_fields_with_clamping(tmp_path: Path):
    ui = _panel_ui()
    settings_path = tmp_path / "settings-v2.json"
    settings_path.write_text(
        json.dumps(
            {
                "version": 2,
                "slang_surface": {
                    "procedural_enabled": True,
                    "procedural_world_space": True,
                    "lighting_enabled": False,
                    "debug_view": SLANG_SURFACE_DEBUG_VIEW_MATERIAL_INDEX + 100,
                    "cryo_mix": 2.0,
                    "environment_map": "environments/moonless_golf_1k.hdr",
                    "state_overlay_strength": -1.0,
                },
                "materials": [
                    {
                        "name": "skin",
                        "procedural": {
                            "scale": 1000.0,
                            "grain": -0.5,
                            "tint": 0.25,
                            "roughness": 0.5,
                            "wetness": 4.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    applied = _load_segmentation_panel_settings(settings_path, ui)

    assert applied == 1
    assert ui.slang_procedural_world_space is True
    assert ui.slang_surface_lighting is False
    assert ui.slang_environment_map == "environments/moonless_golf_1k.hdr"
    assert ui.slang_debug_view == SLANG_SURFACE_DEBUG_VIEW_MATERIAL_INDEX
    assert ui.slang_cryo_mix == pytest.approx(1.0)
    assert ui.slang_state_overlay_strength == pytest.approx(0.0)
    assert ui.material_procedural_revision == 1
    assert ui.material_procedural is not None
    assert ui.material_procedural[1] == {
        "scale": 500.0,
        "grain": 0.0,
        "tint": 0.25,
        "roughness": 0.5,
        "wetness": 1.0,
        "height": PROCEDURAL_MATERIAL_PARAM_DEFAULTS["height"],
        "normal": PROCEDURAL_MATERIAL_PARAM_DEFAULTS["normal"],
    }


def test_segmentation_panel_settings_matches_by_name_when_class_ids_shift(tmp_path: Path):
    ui = UiState(
        material_names=["background", "procedural", "skin", "bone"],
        material_stiffness_scale=[0.0, 1.0, 1.0, 1.0],
        material_visible=[False, False, False, True],
        material_cuttable=[False, False, False, True],
        material_locked=[False, False, False, False],
        material_colors=[
            (0.0, 0.0, 0.0),
            (0.4, 0.4, 0.4),
            (0.5, 0.5, 0.5),
            (0.6, 0.6, 0.6),
        ],
        material_procedural=make_default_procedural_materials(4),
    )
    settings_path = tmp_path / "settings-shifted.json"
    settings_path.write_text(
        json.dumps(
            {
                "version": 2,
                "materials": [
                    {"id": 0, "name": "background", "visible": False, "stiffness_scale": 0.0},
                    {
                        "id": 1,
                        "name": "skin",
                        "visible": True,
                        "cuttable": True,
                        "stiffness_scale": 2.0,
                        "color": [0.1, 0.2, 0.3],
                        "procedural": {"scale": 42.0},
                    },
                    {
                        "id": 2,
                        "name": "bone",
                        "visible": False,
                        "cuttable": False,
                        "stiffness_scale": 3.0,
                        "color": [0.7, 0.8, 0.9],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    applied = _load_segmentation_panel_settings(settings_path, ui)

    assert applied == 3
    assert ui.material_visible == [False, False, True, False]
    assert ui.material_cuttable == [False, False, True, False]
    assert ui.material_stiffness_scale == pytest.approx([0.0, 1.0, 2.0, 3.0])
    assert ui.material_colors == [
        (0.0, 0.0, 0.0),
        (0.4, 0.4, 0.4),
        (0.1, 0.2, 0.3),
        (0.7, 0.8, 0.9),
    ]
    assert ui.material_procedural is not None
    assert ui.material_procedural[1] == make_default_procedural_materials(1)[0]
    assert ui.material_procedural[2]["scale"] == pytest.approx(42.0)


def test_segmentation_panel_settings_maps_legacy_height_debug_to_debug_view(tmp_path: Path):
    ui = _panel_ui()
    settings_path = tmp_path / "settings-height-debug.json"
    settings_path.write_text(
        json.dumps(
            {
                "version": 2,
                "slang_surface": {"height_debug": True},
                "materials": [{"name": "skin"}],
            }
        ),
        encoding="utf-8",
    )

    applied = _load_segmentation_panel_settings(settings_path, ui)

    assert applied == 1
    assert ui.slang_debug_view == SLANG_SURFACE_DEBUG_VIEW_HEIGHT


def test_segmentation_panel_settings_loads_material_color_with_revision(tmp_path: Path):
    ui = _panel_ui()
    settings_path = tmp_path / "settings-color.json"
    settings_path.write_text(
        json.dumps(
            {
                "version": 2,
                "materials": [
                    {
                        "id": 1,
                        "name": "skin",
                        "color": [2.0, -1.0, 0.25],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    applied = _load_segmentation_panel_settings(settings_path, ui)

    assert applied == 1
    assert ui.material_colors == [(1.0, 1.0, 1.0), (1.0, 0.0, 0.25)]
    assert ui.material_colors_revision == 1


def test_timer_panel_snapshot_sorts_by_average_ms_and_limits_rows():
    snapshot = build_timer_panel_snapshot(
        {
            "small": [1.0, 1.0, 1.0],
            "slow": [5.0, 15.0],
            "spiky": [30.0],
        },
        window_secs=3.0,
        window_frames=30,
        frame_count=42,
        triangle_count=7,
        max_rows=2,
    )

    assert snapshot.fps == pytest.approx(10.0)
    assert snapshot.frame_count == 42
    assert snapshot.triangle_count == 7
    assert [row.name for row in snapshot.rows] == ["spiky", "slow"]
    assert [row.avg_ms for row in snapshot.rows] == [pytest.approx(30.0), pytest.approx(10.0)]


def test_timer_panel_snapshot_includes_optional_gpu_average():
    snapshot = build_timer_panel_snapshot(
        {"physics": [2.0, 4.0], "ui": [1.0]},
        window_secs=0.0,
        window_frames=0,
        frame_count=3,
        triangle_count=11,
        gpu_stats={"physics": [7.0, 9.0]},
    )

    rows = {row.name: row for row in snapshot.rows}
    assert snapshot.fps == 0.0
    assert rows["physics"].gpu_avg_ms == pytest.approx(8.0)
    assert rows["ui"].gpu_avg_ms is None


def test_cryo_texture_resource_requests_3d_texture_type():
    class _FakeTextureType:
        texture_3d = object()

    class _FakeSpy:
        TextureType = _FakeTextureType
        Format = type("Format", (), {"rgba8_unorm": object()})
        TextureUsage = type("TextureUsage", (), {"shader_resource": object()})
        ResourceState = type("ResourceState", (), {"shader_resource": object()})

    class _FakeDevice:
        def __init__(self) -> None:
            self.kwargs = None

        def create_texture(self, **kwargs):
            self.kwargs = kwargs
            return object()

    device = _FakeDevice()
    viewer = object.__new__(RendererUnderTest)
    viewer._spy = _FakeSpy
    viewer._device = device
    viewer._cryo_texture = None
    viewer._cryo_source_key = None

    viewer._cryo_texture_resource(np.zeros((2, 3, 4, 3), dtype=np.uint8))

    assert device.kwargs is not None
    assert device.kwargs["type"] is _FakeTextureType.texture_3d
    assert device.kwargs["width"] == 2
    assert device.kwargs["height"] == 3
    assert device.kwargs["depth"] == 4


def test_slang_camera_basis_is_z_up():
    viewer = object.__new__(RendererUnderTest)
    viewer._camera_world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    viewer._camera_scene_radius = 0.25

    viewer._set_camera_look_at(
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )

    np.testing.assert_allclose(viewer._camera_forward, [-1.0, 0.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(viewer._camera_right, [0.0, 1.0, 0.0], atol=1.0e-6)
    np.testing.assert_allclose(viewer._camera_up, [0.0, 0.0, 1.0], atol=1.0e-6)


def test_slang_orbit_camera_preserves_distance_to_target():
    viewer = object.__new__(RendererUnderTest)
    viewer._camera_world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    viewer._camera_scene_radius = 0.25
    viewer._set_camera_look_at(
        np.asarray([1.0, 0.0, 0.25], dtype=np.float32),
        np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
    )
    initial_distance = float(np.linalg.norm(viewer._camera_pos - viewer._camera_target))

    viewer._orbit_camera(80.0, 25.0)

    final_distance = float(np.linalg.norm(viewer._camera_pos - viewer._camera_target))
    assert final_distance == pytest.approx(initial_distance)
    assert abs(float(np.dot(viewer._camera_forward, viewer._camera_world_up))) < 0.985


def test_camera_local_offsets_keep_instruments_attached_to_camera_frame():
    frame = (
        np.asarray([10.0, 0.0, 2.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
    )
    points = np.asarray([[8.0, -0.25, 1.8], [8.0, 0.25, 1.8]], dtype=np.float32)

    offsets = _camera_local_offsets(points, frame)

    np.testing.assert_allclose(offsets, [[-0.25, -0.2, 2.0], [0.25, -0.2, 2.0]], atol=1.0e-6)

    moved_frame = (
        np.asarray([0.0, 5.0, 3.0], dtype=np.float32),
        np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        np.asarray([0.0, 0.0, 1.0], dtype=np.float32),
        np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
    )
    moved = _camera_points_from_local_offsets(offsets, moved_frame)

    np.testing.assert_allclose(moved, [[-0.25, 7.0, 2.8], [0.25, 7.0, 2.8]], atol=1.0e-6)

    device_moved = points.copy()
    device_moved[:, 1] += 0.1
    followed = _camera_transform_points_between_frames(device_moved, frame, moved_frame)

    np.testing.assert_allclose(followed - moved, [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0]], atol=1.0e-6)


def test_slang_shared_buffer_cache_key_includes_generation():
    class _FakeSource:
        ptr = 1234

        def __len__(self):
            return 5

    source = _FakeSource()

    assert _shared_buffer_cache_key(source, "vec3", None) == (1234, 5, "vec3", -1)
    assert _shared_buffer_cache_key(source, "vec3", 1) != _shared_buffer_cache_key(source, "vec3", 2)


def test_slang_unit_sphere_mesh_is_indexed_triangle_geometry():
    vertices, indices = build_unit_sphere_mesh(latitudes=4, longitudes=6)

    assert vertices.dtype == np.float32
    assert indices.dtype == np.uint32
    assert vertices.shape == (20, 3)
    assert indices.ndim == 1
    assert indices.size % 3 == 0
    assert int(indices.max()) < len(vertices)
    np.testing.assert_allclose(np.linalg.norm(vertices, axis=1), np.ones(len(vertices)), atol=1.0e-6)


def test_slang_log_points_draws_instanced_spheres_with_radii_and_colors():
    class _FakeWarpArray:
        def __init__(self, count: int, dtype: str, ptr: int):
            self._count = int(count)
            self.dtype = dtype
            self.ptr = ptr
            self.device = "cuda:0"

        def __len__(self):
            return self._count

    class _FakeCursor:
        def __init__(self, shader_object):
            object.__setattr__(self, "_shader_object", shader_object)

        def __setattr__(self, name, value):
            self._shader_object.assignments[name] = value

    class _FakeSpy:
        class BufferUsage:
            shader_resource = 1

        class IndexFormat:
            uint32 = "uint32"

        @staticmethod
        def ShaderCursor(shader_object):
            return _FakeCursor(shader_object)

        @staticmethod
        def float3(*values):
            return ("float3", tuple(values))

    class _FakePassEncoder:
        def __init__(self):
            self.state = None
            self.pipeline = None
            self.shader_object = SimpleNamespace(assignments={})
            self.draw_args = None

        def set_render_state(self, state):
            self.state = state

        def bind_pipeline(self, pipeline):
            self.pipeline = pipeline
            return self.shader_object

        def draw_indexed(self, args):
            self.draw_args = args

    viewer = object.__new__(RendererUnderTest)
    viewer._spy = _FakeSpy()
    viewer._pass_encoder = _FakePassEncoder()
    viewer._viewport = "viewport"
    viewer._scissor = "scissor"
    viewer._point_sphere_pipeline = "sphere-pipeline"
    viewer._point_sphere_vertex_buffer = "sphere-vertices"
    viewer._point_sphere_index_buffer = "sphere-indices"
    viewer._point_sphere_index_count = 42
    viewer._set_common_uniforms = lambda shader_object, color: setattr(shader_object, "common_color", color)

    updated: list[tuple[str, str, int]] = []

    def _update_shared_buffer(source, label, _dtype, stride, _usage, **_kwargs):
        updated.append((label, getattr(source, "dtype", ""), int(stride)))
        return SimpleNamespace(buffer=f"{label}-buffer")

    viewer._update_shared_buffer = _update_shared_buffer
    viewer._fallback_shader_buffer = lambda key, _data: f"{key}-fallback"
    viewer._warn_once = lambda *_args, **_kwargs: None

    points = _FakeWarpArray(2, "vec3f", 100)
    radii = _FakeWarpArray(2, "float32", 101)
    colors = _FakeWarpArray(2, "vec3f", 102)

    viewer.log_points("/instruments/spheres", points=points, radii=radii, colors=colors)

    assert viewer._pass_encoder.state["vertex_buffers"] == ["sphere-vertices"]
    assert viewer._pass_encoder.state["index_buffer"] == "sphere-indices"
    assert viewer._pass_encoder.pipeline == "sphere-pipeline"
    assert viewer._pass_encoder.draw_args == {"vertex_count": 42, "instance_count": 2}
    assert updated == [
        ("/instruments/spheres-point-centers", "vec3f", 12),
        ("/instruments/spheres-point-radii", "float32", 4),
        ("/instruments/spheres-point-colors", "vec3f", 12),
    ]
    assignments = viewer._pass_encoder.shader_object.assignments
    assert assignments["point_positions"] == "/instruments/spheres-point-centers-buffer"
    assert assignments["point_radii"] == "/instruments/spheres-point-radii-buffer"
    assert assignments["point_colors"] == "/instruments/spheres-point-colors-buffer"
    assert assignments["point_count"] == 2
    assert assignments["point_has_radii"] == 1
    assert assignments["point_has_colors"] == 1


def test_slang_shared_buffer_setup_cleans_up_when_wrap_fails(monkeypatch):
    closed: list[str] = []

    class _FakeBuffer:
        shared_handle = SimpleNamespace(value=99)

        def close(self):
            closed.append("buffer")

    class _FakeDevice:
        def create_buffer(self, **_kwargs):
            return _FakeBuffer()

    class _FakeBufferUsage:
        shared = 1
        shader_resource = 2
        copy_destination = 4

    class _FakeSpy:
        BufferUsage = _FakeBufferUsage

    class _FakeCuda:
        def __init__(self):
            self.events: list[tuple[str, int]] = []

        def set_current_context(self, context):
            self.events.append(("set", int(context)))

        def import_external_memory(self, handle_type, shared_handle, size):
            self.events.append(("import", int(handle_type)))
            self.events.append(("shared", int(shared_handle)))
            self.events.append(("size", int(size)))
            return ctypes.c_void_p(456)

        def map_external_memory(self, external_memory, size):
            self.events.append(("map", int(external_memory.value)))
            self.events.append(("map_size", int(size)))
            return 789

        def current_context(self):
            return 321

        def free_mapped_pointer(self, ptr):
            self.events.append(("free", int(ptr)))

        def destroy_external_memory(self, external_memory):
            self.events.append(("destroy", int(external_memory.value)))

    class _FakeSource:
        ptr = 555

        def __len__(self):
            return 2

    def _raise_array(*_args, **_kwargs):
        raise RuntimeError("wrap failed")

    fake_cuda = _FakeCuda()
    viewer = object.__new__(RendererUnderTest)
    viewer._device = _FakeDevice()
    viewer._spy = _FakeSpy
    viewer._cuda = fake_cuda
    viewer._warp_device = SimpleNamespace(context=11)
    viewer._shared_buffers = {}
    viewer._cuda_external_memory_handle_type = lambda: 7
    monkeypatch.setattr(slang_mod.wp, "array", _raise_array)

    with pytest.raises(RuntimeError, match="wrap failed"):
        viewer._shared_buffer(_FakeSource(), "bad", "dtype", 4, 8, generation=3)

    assert ("free", 789) in fake_cuda.events
    assert ("destroy", 456) in fake_cuda.events
    assert closed == ["buffer"]
    assert viewer._shared_buffers == {}


def test_slang_imgui_color_edit3_uses_float3_slider_and_reports_changed_value():
    sliders = []

    class _FakeVec3:
        def __init__(self, x, y, z):
            self.x = float(x)
            self.y = float(y)
            self.z = float(z)

    class _FakeSpy:
        @staticmethod
        def float3(x, y, z):
            return _FakeVec3(x, y, z)

    class _FakeSliderFloat3:
        def __init__(self, parent, label, value, callback, min, max, format):
            self.parent = parent
            self.label = label
            self.value = value
            self.callback = callback
            self.min = min
            self.max = max
            self.format = format
            self.visible = True
            sliders.append(self)

    screen = object()
    adapter = _SlangImmediateUi(_FakeSpy, SimpleNamespace(SliderFloat3=_FakeSliderFloat3), screen)

    adapter.reset(screen, 800, 600)
    changed, color = adapter.color_edit3("color", 0.1, 0.2, 0.3)
    assert not changed
    assert color == pytest.approx((0.1, 0.2, 0.3))
    assert len(sliders) == 1
    assert sliders[0].min == 0.0
    assert sliders[0].max == 1.0

    sliders[0].callback(_FakeVec3(1.2, -0.5, 0.4))
    adapter.reset(screen, 800, 600)
    changed, color = adapter.color_edit3("color", 0.1, 0.2, 0.3)

    assert changed
    assert color == pytest.approx((1.0, 0.0, 0.4))


def test_slang_imgui_combo_uses_slang_combobox_for_debug_views():
    combos = []

    class _FakeCombo:
        def __init__(self, parent, label, value, callback, items):
            self.parent = parent
            self.label = label
            self.value = value
            self.callback = callback
            self.items = list(items)
            self.visible = True
            combos.append(self)

    screen = object()
    adapter = _SlangImmediateUi(SimpleNamespace(), SimpleNamespace(ComboBox=_FakeCombo), screen)

    adapter.reset(screen, 800, 600)
    changed, value = adapter.combo("Debug view", 0, SLANG_SURFACE_DEBUG_VIEW_LABELS)
    assert not changed
    assert value == 0
    assert combos[0].items == list(SLANG_SURFACE_DEBUG_VIEW_LABELS)

    combos[0].callback(SLANG_SURFACE_DEBUG_VIEW_HEIGHT)
    adapter.reset(screen, 800, 600)
    changed, value = adapter.combo("Debug view", 0, SLANG_SURFACE_DEBUG_VIEW_LABELS)

    assert changed
    assert value == SLANG_SURFACE_DEBUG_VIEW_HEIGHT
    assert len(combos) == 1


def test_slang_imgui_push_id_keeps_repeated_row_controls_stable_when_rows_insert():
    boxes = []

    class _FakeCheckBox:
        def __init__(self, parent, label, value, callback):
            self.parent = parent
            self.label = label
            self.value = value
            self.callback = callback
            self.visible = True
            boxes.append(self)

    screen = object()
    adapter = _SlangImmediateUi(SimpleNamespace(), SimpleNamespace(CheckBox=_FakeCheckBox), screen)

    adapter.reset(screen, 800, 600)
    for row_id in (1, 2):
        adapter.push_id(row_id)
        changed, value = adapter.checkbox("Visible", False)
        adapter.pop_id()
        assert not changed
        assert value is False

    boxes[1].callback(True)
    adapter.reset(screen, 800, 600)
    values = {}
    changes = {}
    for row_id in (0, 1, 2):
        adapter.push_id(row_id)
        changed, value = adapter.checkbox("Visible", False)
        adapter.pop_id()
        values[row_id] = value
        changes[row_id] = changed

    assert changes == {0: False, 1: False, 2: True}
    assert values == {0: False, 1: False, 2: True}


def _draw_test_viewer():
    class _FakeBufferUsage:
        vertex_buffer = 1
        shader_resource = 2

    class _FakeSpy:
        BufferUsage = _FakeBufferUsage

    class _FakePassEncoder:
        def __init__(self) -> None:
            self.render_states = []
            self.draw_calls = []

        def set_render_state(self, state):
            self.render_states.append(state)

        def bind_pipeline(self, pipeline):
            return {"pipeline": pipeline}

        def draw(self, args):
            self.draw_calls.append(args)

    viewer = object.__new__(RendererUnderTest)
    viewer._spy = _FakeSpy
    viewer._pass_encoder = _FakePassEncoder()
    viewer._viewport = object()
    viewer._scissor = object()
    viewer._cryo_pipeline = object()
    viewer._frame_id = 1
    calls: list[tuple[str, object, int]] = []
    uniforms: list[dict] = []

    def _update_shared_buffer(source, label, dtype, stride, usage, *, generation=None):
        calls.append((label, dtype, int(stride)))
        return SimpleNamespace(buffer=f"{label}-buffer")

    def _set_cryo_uniforms(shader_object, host_volume, scale, **kwargs):
        uniforms.append({"host_volume": host_volume, "scale": scale, **kwargs})

    viewer._update_shared_buffer = _update_shared_buffer
    viewer._set_cryo_uniforms = _set_cryo_uniforms
    return viewer, calls, uniforms


def test_slang_cryo_draw_binds_material_state_only_for_procedural_mode():
    viewer, calls, uniforms = _draw_test_viewer()
    frame = SimpleNamespace(
        positions=object(),
        normals=object(),
        uv3=object(),
        procedural_coord=object(),
        material_id=object(),
        state_rgba=object(),
        vertex_count=3,
        buffer_generation=7,
    )

    viewer.draw_cryo_surface(
        frame,
        None,
        material_colors=[(1.0, 0.0, 0.0)],
        procedural_params=make_default_procedural_materials(1),
        procedural_params_revision=4,
        procedural_enabled=True,
        procedural_world_space=True,
        procedural_uv3_noise_scale=(0.5, 1.0, 0.25),
        lighting_enabled=False,
        cryo_mix=0.0,
        state_overlay_strength=1.0,
        debug_view=SLANG_SURFACE_DEBUG_VIEW_HEIGHT,
    )

    assert [call[0] for call in calls] == [
        "cryo-positions",
        "cryo-normals",
        "cryo-uv3",
        "cryo-procedural-coord",
        "cryo-material-id",
        "cryo-state-rgba",
    ]
    assert uniforms[0]["host_volume"] is None
    assert uniforms[0]["material_id_buffer"] == "cryo-material-id-buffer"
    assert uniforms[0]["state_rgba_buffer"] == "cryo-state-rgba-buffer"
    assert uniforms[0]["procedural_enabled"] is True
    assert uniforms[0]["procedural_world_space"] is True
    assert uniforms[0]["procedural_uv3_noise_scale"] == (0.5, 1.0, 0.25)
    assert uniforms[0]["lighting_enabled"] is False
    assert uniforms[0]["debug_view"] == SLANG_SURFACE_DEBUG_VIEW_HEIGHT
    assert viewer._pass_encoder.render_states[0]["vertex_buffers"] == [
        "cryo-positions-buffer",
        "cryo-normals-buffer",
        "cryo-uv3-buffer",
        "cryo-procedural-coord-buffer",
    ]
    assert viewer._pass_encoder.draw_calls == [{"vertex_count": 3}]


def test_slang_cryo_only_draw_does_not_require_material_state_buffers():
    viewer, calls, uniforms = _draw_test_viewer()
    host_volume = np.zeros((1, 1, 1, 3), dtype=np.uint8)
    frame = SimpleNamespace(
        positions=object(),
        normals=object(),
        uv3=object(),
        material_id=None,
        state_rgba=None,
        vertex_count=3,
        buffer_generation=8,
    )

    viewer.draw_cryo_surface(frame, host_volume, procedural_enabled=False)

    assert [call[0] for call in calls] == ["cryo-positions", "cryo-normals", "cryo-uv3"]
    assert viewer._pass_encoder.render_states[0]["vertex_buffers"] == [
        "cryo-positions-buffer",
        "cryo-normals-buffer",
        "cryo-uv3-buffer",
        "cryo-uv3-buffer",
    ]
    assert uniforms[0]["host_volume"] is host_volume
    assert uniforms[0]["material_id_buffer"] is None
    assert uniforms[0]["state_rgba_buffer"] is None
    assert uniforms[0]["procedural_enabled"] is False


def test_slang_end_frame_renders_ui_offscreen_before_present():
    events: list[tuple[str, object | None]] = []

    class _FakePassEncoder:
        def end(self) -> None:
            events.append(("scene_end", None))

    class _FakeCommandEncoder:
        def finish(self):
            events.append(("finish", None))
            return "command-buffer"

    class _FakeDevice:
        def sync_to_cuda(self, cuda_stream):
            events.append(("sync", cuda_stream))

        def submit_command_buffer(self, command_buffer):
            events.append(("submit", command_buffer))
            return 1

    class _FakeSurface:
        def present(self) -> None:
            events.append(("surface_present", None))

    viewer = object.__new__(RendererUnderTest)
    viewer._pass_encoder = _FakePassEncoder()
    viewer._command_encoder = _FakeCommandEncoder()
    viewer._surface_texture = SimpleNamespace(width=640, height=480)
    viewer._scene_color_texture = "scene-color"
    viewer._frame_id = 8
    viewer._device = _FakeDevice()
    viewer._surface = _FakeSurface()
    viewer._cuda_stream_ptr = lambda: 99
    viewer._last_submit_id = None
    viewer._last_submit_frame = -1
    viewer._render_ui = lambda width, height, target_texture=None: events.append(("ui", target_texture))
    viewer._render_present = lambda source_texture, width, height: events.append(("present", source_texture))

    viewer.end_frame()

    assert events == [
        ("scene_end", None),
        ("ui", "scene-color"),
        ("present", "scene-color"),
        ("finish", None),
        ("sync", 99),
        ("submit", "command-buffer"),
        ("surface_present", None),
    ]
    assert viewer._surface_texture is None
    assert viewer._frame_id == 9
    assert viewer._last_submit_id == 1
    assert viewer._last_submit_frame == 8


def test_slang_begin_frame_skips_acquire_while_previous_submit_pending():
    events: list[str] = []

    class _FakeWindow:
        width = 640
        height = 480

        def process_events(self):
            events.append("events")

        def should_close(self):
            return False

    class _FakeSurface:
        config = object()

        def acquire_next_image(self):
            events.append("acquire")
            return object()

    class _FakeDevice:
        def is_submit_finished(self, submit_id):
            events.append(f"finished:{submit_id}")
            return False

    viewer = object.__new__(RendererUnderTest)
    viewer._time = 0.0
    viewer._closed = False
    viewer._window = _FakeWindow()
    viewer._surface = _FakeSurface()
    viewer._device = _FakeDevice()
    viewer._apply_pending_resize = lambda: events.append("resize")
    viewer._poll_external_material_reload = lambda: events.append("reload")
    viewer._update_keyboard_camera_motion = lambda: events.append("camera")
    viewer._procedural_material_path = object()
    viewer._procedural_material_hot_reload = False
    viewer._external_material_reload_future = None
    viewer._procedural_material_info = object()
    viewer._frame_id = 12
    viewer._last_submit_id = 42
    viewer._last_submit_frame = 11
    viewer._surface_texture = "old"
    viewer._command_encoder = "old"
    viewer._pass_encoder = "old"

    viewer.begin_frame(1.0)

    assert events == ["events", "resize", "reload", "camera", "finished:42"]
    assert viewer._surface_texture is None
    assert viewer._command_encoder is None
    assert viewer._pass_encoder is None
    assert viewer._last_submit_id == 42


def test_set_cryo_uniforms_uses_neutral_texture_when_host_volume_missing():
    class _FakeCursor:
        pass

    class _FakeSpy:
        @staticmethod
        def ShaderCursor(shader_object):
            return shader_object

        @staticmethod
        def float3(x, y, z):
            return (float(x), float(y), float(z))

    viewer = object.__new__(RendererUnderTest)
    viewer._spy = _FakeSpy
    viewer._set_common_uniforms = lambda _shader_object: None
    viewer._neutral_cryo_texture_resource = lambda: "neutral-texture"
    viewer._cryo_texture_resource = lambda _host: "host-texture"
    viewer._sampler = lambda: "sampler"
    viewer._material_colors_resource = lambda _colors, revision=None: ("colors-buffer", 1)
    viewer._procedural_params_resource = lambda _params, material_count, revision=None: "params-buffer"
    viewer._fallback_shader_buffer = lambda key, _data: f"{key}-fallback"
    viewer._environment_path = None
    viewer._neutral_environment_texture = "neutral-environment"
    cursor = _FakeCursor()

    viewer._set_cryo_uniforms(cursor, None, (1.0, 2.0, 3.0), cryo_mix=8.0, state_overlay_strength=-2.0)

    assert cursor.cryo_volume == "neutral-texture"
    assert cursor.has_cryo_volume == 0
    assert cursor.procedural_world_space == 0
    assert cursor.procedural_uv3_noise_scale == (1.0, 1.0, 1.0)
    assert cursor.lighting_enabled == 1
    assert cursor.key_light_enabled == 1
    assert cursor.fill_light_enabled == 1
    assert cursor.ambient_light_enabled == 0
    assert cursor.environment_lighting_enabled == 1
    assert cursor.debug_view == 0
    assert cursor.procedural_material_scale == pytest.approx(1.0)
    assert cursor.procedural_time == pytest.approx(0.0)
    assert cursor.material_id == "material-id-fallback"
    assert cursor.state_rgba == "state-rgba-fallback"
    assert cursor.environment_map == "neutral-environment"
    assert cursor.has_environment_map == 0
    assert cursor.environment_intensity == pytest.approx(1.0)
    assert cursor.environment_rotation == pytest.approx(0.0)
    assert cursor.environment_pitch == pytest.approx(np.deg2rad(-90.0))
    assert cursor.light_key_dir == (0.35, 0.75, 0.55)
    assert cursor.cryo_mix == pytest.approx(1.0)
    assert cursor.state_overlay_strength == pytest.approx(0.0)

    legacy_cursor = _FakeCursor()
    viewer._set_cryo_uniforms(legacy_cursor, None, (1.0, 1.0, 1.0), height_debug=True)
    assert legacy_cursor.debug_view == SLANG_SURFACE_DEBUG_VIEW_HEIGHT


def test_slang_environment_path_setter_retires_cached_texture(tmp_path: Path):
    closed = []

    class _FakeTexture:
        def close(self):
            closed.append("closed")

    first = tmp_path / "first.hdr"
    second = tmp_path / "second.hdr"
    first.write_text("", encoding="utf-8")
    second.write_text("", encoding="utf-8")

    viewer = object.__new__(RendererUnderTest)
    viewer._environment_path = first.resolve()
    viewer._environment_texture = _FakeTexture()
    viewer._environment_texture_key = (first.resolve(), 123)

    viewer.set_environment_path(second)

    assert viewer._environment_path == second.resolve()
    assert viewer._environment_texture is None
    assert viewer._environment_texture_key is None
    assert closed == ["closed"]

    viewer.set_environment_path(None)
    assert viewer._environment_path is None


def test_slang_environment_background_draws_hdr_before_geometry():
    class _FakeCursor:
        pass

    class _FakeSpy:
        @staticmethod
        def ShaderCursor(shader_object):
            return shader_object

        @staticmethod
        def float3(x, y, z):
            return (float(x), float(y), float(z))

    cursors: list[_FakeCursor] = []

    class _FakePassEncoder:
        def __init__(self) -> None:
            self.render_states = []
            self.bound = []
            self.draw_calls = []

        def set_render_state(self, state):
            self.render_states.append(state)

        def bind_pipeline(self, pipeline):
            self.bound.append(pipeline)
            cursor = _FakeCursor()
            cursors.append(cursor)
            return cursor

        def draw(self, args):
            self.draw_calls.append(args)

    viewer = object.__new__(RendererUnderTest)
    viewer._spy = _FakeSpy
    viewer._pass_encoder = _FakePassEncoder()
    viewer._background_pipeline = "background-pipeline"
    viewer._viewport = "viewport"
    viewer._scissor = "scissor"
    viewer._environment_background_enabled = True
    viewer._environment_intensity = 2.0
    viewer._environment_rotation_degrees = 90.0
    viewer._environment_pitch_degrees = 45.0
    viewer._environment_texture_resource = lambda: "hdr-texture"
    viewer._neutral_environment_texture_resource = lambda: "neutral-environment"
    viewer._sampler = lambda: "sampler"
    viewer._surface_texture = SimpleNamespace(width=800, height=400)
    viewer._camera_right = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    viewer._camera_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    viewer._camera_forward = np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    viewer._camera_inv_tan_half_fovy = 2.0

    viewer._render_environment_background()

    assert viewer._pass_encoder.bound == ["background-pipeline"]
    assert viewer._pass_encoder.render_states == [{"viewports": ["viewport"], "scissor_rects": ["scissor"]}]
    assert viewer._pass_encoder.draw_calls == [{"vertex_count": 3}]
    assert cursors[0].environment_map == "hdr-texture"
    assert cursors[0].has_environment_map == 1
    assert cursors[0].environment_intensity == pytest.approx(2.0)
    assert cursors[0].environment_rotation == pytest.approx(0.25)
    assert cursors[0].environment_pitch == pytest.approx(np.deg2rad(45.0))
    assert cursors[0].camera_aspect == pytest.approx(2.0)


def test_slang_environment_background_skips_when_disabled():
    class _FakePassEncoder:
        def bind_pipeline(self, _pipeline):
            raise AssertionError("background should not draw")

    viewer = object.__new__(RendererUnderTest)
    viewer._pass_encoder = _FakePassEncoder()
    viewer._background_pipeline = "background-pipeline"
    viewer._environment_background_enabled = False

    viewer._render_environment_background()


def test_procedural_param_resource_reuses_buffer_until_revision_changes():
    closed: list[str] = []

    class _FakeBuffer:
        def __init__(self, label: str) -> None:
            self.label = label
            self.copied_shapes: list[tuple[int, ...]] = []

        def copy_from_numpy(self, data) -> None:
            self.copied_shapes.append(tuple(int(v) for v in data.shape))

        def close(self) -> None:
            closed.append(self.label)

    class _FakeBufferUsage:
        shader_resource = 1

    class _FakeSpy:
        BufferUsage = _FakeBufferUsage

    class _FakeDevice:
        def __init__(self) -> None:
            self.labels: list[str] = []

        def create_buffer(self, **kwargs):
            self.labels.append(kwargs["label"])
            return _FakeBuffer(kwargs["label"])

    viewer = object.__new__(RendererUnderTest)
    viewer._spy = _FakeSpy
    viewer._device = _FakeDevice()
    viewer._procedural_param_buffer = None
    viewer._procedural_param_buffer_key = None
    viewer._procedural_param_buffer_ring = []
    viewer._procedural_param_buffer_shape_key = None
    viewer._procedural_param_buffer_index = -1
    viewer._retired_shader_buffers = []

    params = make_default_procedural_materials(1)
    first = viewer._procedural_params_resource(params, material_count=1, revision=1)
    second = viewer._procedural_params_resource(params, material_count=1, revision=1)
    third = viewer._procedural_params_resource(params, material_count=1, revision=2)

    assert first is second
    assert third is not first
    assert len(viewer._device.labels) == slang_mod._DYNAMIC_SHADER_BUFFER_RING_SIZE
    assert viewer._device.labels[0] == "hex-procedural-material-params[0]"
    assert third is viewer._procedural_param_buffer_ring[1]
    assert first.copied_shapes == []
    assert third.copied_shapes == [(2, 4)]
    assert viewer._retired_shader_buffers == []
    assert closed == []


def test_material_color_resource_rotates_inactive_buffer_on_revision_change():
    closed: list[str] = []

    class _FakeBuffer:
        def __init__(self, label: str) -> None:
            self.label = label
            self.copied_shapes: list[tuple[int, ...]] = []

        def copy_from_numpy(self, data) -> None:
            self.copied_shapes.append(tuple(int(v) for v in data.shape))

        def close(self) -> None:
            closed.append(self.label)

    class _FakeBufferUsage:
        shader_resource = 1

    class _FakeSpy:
        BufferUsage = _FakeBufferUsage

    class _FakeDevice:
        def __init__(self) -> None:
            self.labels: list[str] = []

        def create_buffer(self, **kwargs):
            self.labels.append(kwargs["label"])
            return _FakeBuffer(kwargs["label"])

    viewer = object.__new__(RendererUnderTest)
    viewer._spy = _FakeSpy
    viewer._device = _FakeDevice()
    viewer._material_color_buffer = None
    viewer._material_color_buffer_key = None
    viewer._material_color_buffer_ring = []
    viewer._material_color_buffer_shape_key = None
    viewer._material_color_buffer_index = -1
    viewer._retired_shader_buffers = []

    first, first_count = viewer._material_colors_resource([(1.0, 0.0, 0.0)], revision=1)
    second, second_count = viewer._material_colors_resource([(0.0, 1.0, 0.0)], revision=2)

    assert first_count == 1
    assert second_count == 1
    assert second is not first
    assert len(viewer._device.labels) == slang_mod._DYNAMIC_SHADER_BUFFER_RING_SIZE
    assert viewer._device.labels[0] == "hex-material-colors[0]"
    assert second is viewer._material_color_buffer_ring[1]
    assert first.copied_shapes == []
    assert second.copied_shapes == [(1, 4)]
    assert viewer._retired_shader_buffers == []
    assert closed == []


def test_prepare_cryo_material_resources_uploads_buffers_before_draw():
    calls: list[tuple[str, int | None, int | None]] = []
    viewer = object.__new__(RendererUnderTest)

    def _material_colors_resource(colors, *, revision=None):
        calls.append(("colors", revision, None if colors is None else len(colors)))
        return "color-buffer", 3

    def _procedural_params_resource(params, *, material_count, revision=None):
        calls.append(("params", revision, material_count))
        return "param-buffer"

    def _material_maker_params_resource(params, *, material_count, revision=None):
        calls.append(("mm", revision, material_count))
        return "mm-param-buffer"

    viewer._material_colors_resource = _material_colors_resource
    viewer._procedural_params_resource = _procedural_params_resource
    viewer._material_maker_params_resource = _material_maker_params_resource

    viewer.prepare_cryo_material_resources(
        material_colors=[(1.0, 0.0, 0.0), (0.0, 1.0, 0.0)],
        material_colors_revision=7,
        procedural_params=make_default_procedural_materials(2),
        procedural_params_revision=9,
        material_maker_params=[{"roughness": 0.5}],
        material_maker_params_revision=11,
    )

    assert calls == [
        ("colors", 7, 2),
        ("params", 9, 3),
        ("mm", 11, 3),
    ]


def test_slang_captured_mouse_drag_does_not_reach_scene_handlers():
    class _FakeUiContext:
        def __init__(self) -> None:
            self.captured = [True, False, True]

        def handle_mouse_event(self, _event) -> bool:
            return self.captured.pop(0)

    class _FakeMouseEvent:
        def __init__(self, kind: str, x: float, y: float) -> None:
            self.kind = kind
            self.pos = SimpleNamespace(x=x, y=y)
            self.button = SimpleNamespace(name="left")
            self.mods = 0

        def is_scroll(self) -> bool:
            return False

        def is_button_down(self) -> bool:
            return self.kind == "down"

        def is_button_up(self) -> bool:
            return self.kind == "up"

        def is_move(self) -> bool:
            return self.kind == "move"

    calls: list[tuple] = []
    viewer = object.__new__(RendererUnderTest)
    viewer._ui_enabled = True
    viewer._ui_context = _FakeUiContext()
    viewer._ui_capturing = False
    viewer._ui_mouse_active = False
    viewer._mouse_pos = None
    viewer._mouse_buttons = 0
    viewer._pyglet_mouse_button = lambda name: 1 if name == "left" else 0
    viewer._on_mouse_press_callback = lambda *args: calls.append(("press", *args))
    viewer._on_mouse_drag_callback = lambda *args: calls.append(("drag", *args))
    viewer._on_mouse_release_callback = lambda *args: calls.append(("release", *args))
    viewer._on_mouse_motion_callback = lambda *args: calls.append(("motion", *args))
    viewer._orbit_camera = lambda *args: calls.append(("orbit", *args))

    viewer._on_mouse_event(_FakeMouseEvent("down", 10.0, 20.0))
    viewer._on_mouse_event(_FakeMouseEvent("move", 15.0, 25.0))
    viewer._on_mouse_event(_FakeMouseEvent("up", 15.0, 25.0))

    assert calls == []
    assert viewer._mouse_buttons == 0
    assert viewer._ui_mouse_active is False


def test_external_material_reload_keeps_previous_pipeline_on_compile_failure(monkeypatch, tmp_path: Path):
    source = tmp_path / "material.slang"
    source.write_text("struct MMMaterial { vec3 albedo; }; MMMaterial evaluateMaterial(vec3 p) { MMMaterial m; m.albedo = p; return m; }")
    generated = tmp_path / "generated.slang"
    generated.write_text("// generated", encoding="utf-8")
    info = MaterialMakerSlangInfo(
        path=source,
        fields=("albedo",),
        texture_names=(),
        uses_timed_call=False,
        source_mtime_ns=source.stat().st_mtime_ns,
        texture_mtime_ns=(),
        generated_shader_path=generated,
        generated_source_path=tmp_path / "source.slang",
    )
    monkeypatch.setattr(slang_mod, "write_material_bridge", lambda _path, _work_dir: info)

    class _FakeDevice:
        def __init__(self) -> None:
            self.fail = False

        def wait(self):
            return None

        def load_program(self, _path, _entries):
            if self.fail:
                raise RuntimeError("compile failed")
            return "program-ok"

    viewer = object.__new__(RendererUnderTest)
    viewer._procedural_material_path = source
    viewer._procedural_material_hot_reload = True
    viewer._procedural_material_info = None
    viewer._procedural_material_textures = {}
    viewer._procedural_material_reload_error = None
    viewer._cryo_pipeline = "old-pipeline"
    viewer._cryo_program = "old-program"
    viewer._device = _FakeDevice()
    viewer._create_cryo_pipeline = lambda program, _label: f"pipeline:{program}"
    viewer._clear_external_material_textures = lambda: None
    viewer._warnings = set()

    assert viewer.reload_external_material(force=True)
    assert viewer._cryo_pipeline == "pipeline:program-ok"
    viewer._device.fail = True

    assert viewer.reload_external_material(force=True) is False
    assert viewer._cryo_pipeline == "pipeline:program-ok"


def test_external_material_reload_request_applies_async_result(tmp_path: Path):
    source = tmp_path / "material.slang"
    source.write_text("struct MMMaterial { vec3 albedo; }; MMMaterial evaluateMaterial(vec3 p) { MMMaterial m; m.albedo = p; return m; }")
    generated = tmp_path / "generated.slang"
    generated.write_text("// generated", encoding="utf-8")
    info = MaterialMakerSlangInfo(
        path=source,
        fields=("albedo",),
        texture_names=(),
        uses_timed_call=False,
        source_mtime_ns=source.stat().st_mtime_ns,
        texture_mtime_ns=(),
        generated_shader_path=generated,
        generated_source_path=tmp_path / "source.slang",
    )

    viewer = object.__new__(RendererUnderTest)
    viewer._external_material_reload_executor = None
    viewer._external_material_reload_future = None
    viewer._procedural_material_reload_status = ""
    viewer._procedural_material_reload_error = None
    viewer._procedural_material_info = None
    viewer._procedural_material_textures = {}
    viewer._retired_shader_buffers = []
    viewer._warnings = set()
    viewer._warn_once = lambda *_args, **_kwargs: None
    viewer._compile_external_material_pipeline = lambda *, force=False: _ExternalMaterialReloadResult(
        changed=True,
        program=f"program:{force}",
        pipeline=f"pipeline:{force}",
        info=info,
    )

    assert viewer.request_external_material_reload(force=True)
    assert viewer._external_material_reload_future is not None
    viewer._external_material_reload_future.result(timeout=1.0)

    assert viewer.external_material_reload_status() == f"Loaded {source.name}"
    assert viewer._cryo_program == "program:True"
    assert viewer._cryo_pipeline == "pipeline:True"
    assert viewer._procedural_material_info is info

    assert viewer._external_material_reload_executor is not None
    viewer._external_material_reload_executor.shutdown(wait=True, cancel_futures=True)


def test_hex_cryo_surface_shader_declares_procedural_material_bindings():
    shader = (Path(__file__).resolve().parents[2] / "omnisurg/rendering/slang_shaders/hex_cryo_surface.slang").read_text(
        encoding="utf-8"
    )

    assert "SV_VertexID" in shader
    assert "StructuredBuffer<int> material_id" in shader
    assert "StructuredBuffer<float4> state_rgba" in shader
    assert "StructuredBuffer<float4> procedural_params" in shader
    assert "float3 procedural_coord : TEXCOORD1" in shader
    assert "material_id[vertex_id]" in shader
    assert "state_rgba[vertex_id]" in shader
    assert "uniform int lighting_enabled" in shader
    assert "uniform int key_light_enabled" in shader
    assert "uniform int fill_light_enabled" in shader
    assert "uniform int ambient_light_enabled" in shader
    assert "uniform int environment_lighting_enabled" in shader
    assert "uniform int debug_view" in shader
    assert "uniform int height_debug" not in shader
    assert "uniform int procedural_world_space" in shader
    assert "uniform float3 procedural_uv3_noise_scale" in shader
    assert "uniform float procedural_material_scale" in shader
    assert "uniform float procedural_time" in shader
    assert "uniform float environment_rotation" in shader
    assert "uniform float environment_pitch" in shader
    assert "pitch_sin" in shader
    assert "atan2(dir.z, dir.x)" in shader
    assert "0.5 + environment_rotation" not in shader
    assert "sample_environment_blurred" in shader
    assert "lerp(0.025, 0.085, effective_roughness)" in shader
    assert "MM_EXTERNAL_MATERIAL" in shader
    assert "MM_HAS_NORMAL" in shader
    assert "MM_HAS_WETNESS" in shader
    assert "MM_HAS_CLEARCOAT" in shader
    assert "MM_HAS_SSSSTRENGTH" in shader
    assert "MM_HAS_BACKLIGHT" in shader
    assert "smooth_rest_gradient" in shader
    assert "diagnostic_gradient_albedo" in shader
    assert "visible_grain" in shader
    assert "material_maker_albedo" in shader
    assert "MM_EVALUATE_MATERIAL_FRAME" in shader
    assert "material_maker_derivative_frame" in shader
    assert "material_space_jacobian_axis" in shader
    assert "material_normal_map_to_world" in shader
    assert "mm_material_wetness" in shader
    assert "mm_material_clearcoat" in shader
    assert "mm_material_sss_strength" in shader
    assert "shade_pbr_surface" in shader
    assert "specular_level * 0.08" in shader
    assert "clearcoat_specular" in shader
    assert "transmitted" in shader
    assert "backlight_tint" in shader
    assert "float procedural_height(" in shader
    assert "params1.y" in shader
    assert "params1.z" in shader
    assert "float3 perturb_normal_from_height(" in shader
    assert "#if MM_EXTERNAL_MATERIAL && MM_HAS_NORMAL" in shader
    assert "ddx(world_position)" in shader
    assert "ddy(world_position)" in shader
    assert "surface_debug_color" in shader
    assert "DEBUG_VIEW_MATERIAL_NORMAL_MAP" in shader
    assert "DEBUG_VIEW_FINAL_SHADE_NORMAL" in shader
    assert "DEBUG_VIEW_MATERIAL_INDEX" in shader
    assert "clamp(roughness, 0.04, 1.0)" in shader
    assert "roughness * (1.0 - wetness" not in shader
    assert "input.world_position * 100.0 : input.procedural_coord * procedural_uv3_noise_scale" in shader
    assert "lighting_enabled == 0" in shader
    assert "key_light_enabled != 0" in shader
    assert "fill_light_enabled != 0" in shader
    assert "ambient_light_enabled != 0" in shader
    assert "environment_lighting_enabled != 0" in shader


def test_hex_present_shader_declares_environment_background_pass():
    shader = (Path(__file__).resolve().parents[2] / "omnisurg/rendering/slang_shaders/hex_present.slang").read_text(
        encoding="utf-8"
    )

    assert "background_vertex_main" in shader
    assert "background_fragment_main" in shader
    assert "Texture2D<float4> environment_map" in shader
    assert "uniform float environment_rotation" in shader
    assert "uniform float environment_pitch" in shader
    assert "pitch_sin" in shader
    assert "atan2(dir.z, dir.x)" in shader
    assert "0.5 + environment_rotation" not in shader
    assert "frac(u)" in shader
    assert "camera_forward" in shader


def test_slang_imgui_appearing_position_does_not_override_dragged_window_position():
    windows = []

    class _FakeSpy:
        @staticmethod
        def float2(x, y):
            return (float(x), float(y))

    class _FakeWindow:
        def __init__(self, parent, title, *, position, size):
            self.parent = parent
            self.title = title
            self.position = position
            self.size = size
            self.visible = True
            windows.append(self)

    screen = object()
    adapter = _SlangImmediateUi(_FakeSpy, SimpleNamespace(Window=_FakeWindow), screen)

    adapter.reset(screen, 800, 600)
    adapter.set_next_window_pos(adapter.ImVec2(10, 20), adapter.Cond_.appearing.value)
    adapter.set_next_window_size(adapter.ImVec2(300, 200), adapter.Cond_.appearing.value)
    assert adapter.begin("Panel")
    adapter.end()
    assert windows[0].position == (10.0, 20.0)

    windows[0].position = (140.0, 90.0)
    adapter.reset(screen, 800, 600)
    adapter.set_next_window_pos(adapter.ImVec2(10, 20), adapter.Cond_.appearing.value)
    adapter.set_next_window_size(adapter.ImVec2(300, 200), adapter.Cond_.appearing.value)
    assert adapter.begin("Panel")
    adapter.end()

    assert len(windows) == 1
    assert windows[0].position == (140.0, 90.0)


def test_slang_imgui_always_position_still_pins_window_position():
    windows = []

    class _FakeSpy:
        @staticmethod
        def float2(x, y):
            return (float(x), float(y))

    class _FakeWindow:
        def __init__(self, parent, title, *, position, size):
            self.parent = parent
            self.title = title
            self.position = position
            self.size = size
            self.visible = True
            windows.append(self)

    screen = object()
    adapter = _SlangImmediateUi(_FakeSpy, SimpleNamespace(Window=_FakeWindow), screen)

    adapter.reset(screen, 800, 600)
    adapter.set_next_window_pos(adapter.ImVec2(10, 20), adapter.Cond_.appearing.value)
    assert adapter.begin("Panel")
    adapter.end()

    windows[0].position = (140.0, 90.0)
    adapter.reset(screen, 800, 600)
    adapter.set_next_window_pos(adapter.ImVec2(25, 35), adapter.Cond_.always.value)
    assert adapter.begin("Panel")
    adapter.end()

    assert windows[0].position == (25.0, 35.0)


def test_slang_side_panel_window_does_not_override_dragged_position():
    windows = []

    class _FakeSpy:
        @staticmethod
        def float2(x, y):
            return (float(x), float(y))

    class _FakeWindow:
        def __init__(self, parent, title, *, position, size):
            self.parent = parent
            self.title = title
            self.position = position
            self.size = size
            self.visible = True
            windows.append(self)

    screen = object()
    viewer = object.__new__(RendererUnderTest)
    viewer._spy = _FakeSpy
    viewer._sui = SimpleNamespace(Window=_FakeWindow)
    viewer._ui_context = SimpleNamespace(screen=screen)
    viewer._ui_windows = {}

    window = viewer._ui_window("side", "Side", (10, 10), (300, 400))
    assert window.position == (10.0, 10.0)
    window.position = (80.0, 70.0)

    same_window = viewer._ui_window("side", "Side", (10, 10), (320, 420))

    assert same_window is window
    assert len(windows) == 1
    assert window.position == (80.0, 70.0)
    assert window.size == (320.0, 420.0)


def test_is_slang_backend_variants():
    assert is_slang_backend("slang")
    assert is_slang_backend("slang-vulkan")
    assert is_slang_backend("slang-vk")
    assert is_slang_backend("slang-d3d12")
    assert not is_slang_backend("gl")
