# Slang Rendering Notes

## Done

- Added the textured Slang tissue shader path for layered tissue rendering.
- Bound tissue vertex colors through the Slang mesh path so RGB channels can drive:
  - damage blend
  - coagulation blend
  - blood blend
- Added layered material texture bindings for each tissue material:
  - diffuse base/damage/coag/blood
  - normal base/damage/coag/blood
  - specular base/damage/coag/blood
  - layer masks, blood mask, and heat mask
- Added runtime tissue debug views:
  - final
  - vertex blend RGB
  - masked layer weights
  - blended diffuse
  - blended normal
  - spec/roughness/wet film
  - heat/blood masks
- Added runtime tissue material controls for:
  - normal strength
  - specular scale
  - roughness bias
  - ambient
  - rim strength
  - wetness
  - wet spec scale
  - wet roughness
  - blood wetness
- Added wet tissue film shading:
  - preserved the broad dry/spec-map lobe
  - added a sharper wet film specular lobe
  - blood blend and blood mask automatically contribute to wet film
  - wet film slightly darkens diffuse tissue
  - `DEBUG_SPEC_ROUGHNESS` now shows red = spec mask, green = dry roughness, blue = wet film amount
- Fixed wet specular stacking:
  - dry specular is attenuated by wet film
  - final specular uses `dry_spec * (1.0 - wet_film) + wet_spec`
- Wired subsurface uniforms into final lighting:
  - `subsurface_color`
  - `subsurface_strength`
  - contribution is driven by the existing backscatter term
- Added API clamping for wet material parameters in `SlangRenderer.set_tissue_material_params()`:
  - `wetness`: `0.0..1.0`
  - `wet_spec_scale`: `0.0..4.0`
  - `wet_roughness`: `0.02..0.6`
  - `blood_wetness`: `0.0..2.0`
- Added focused tests for:
  - wet parameter binding and clamping
  - runtime wetness UI sliders and renderer sync
  - shader source checks proving wet uniforms affect final lighting
  - vertex-color blend plumbing
- Verified the tissue shader compiles with local `slangc.exe`.

## Current Defaults

- `wetness = 0.0`
- `wet_spec_scale = 1.0`
- `wet_roughness = 0.18`
- `blood_wetness = 1.0`
- `subsurface_strength = 0.0`

These defaults are intentionally conservative. With zero vertex colors and default wetness, existing scenes should stay close to their previous appearance.

## Known Limitations

- Wetness is still a global material control plus blood-mask/blood-blend response. There is no spatial wetness accumulator from tool contact, bleeding, or fluid simulation yet.
- Damage, coagulation, and blood are exposed visually, but most runtime state is still slider/debug driven rather than authored by surgical events.
- Blood wetness does not yet account for coagulation state. Coagulated blood should become less glossy.
- Diffuse wet darkening is hard-coded and conservative. There is no `wet_darken` uniform yet.
- The wet roughness range is narrow and tuned for sharp film highlights. It does not cover very matte mucus or clotted films.
- Numeric material parameter clamping is currently asymmetric. Wet parameters are clamped in the renderer API, while older material controls rely mainly on UI ranges.
- Normal layer blending is still linear in tangent space. It has not been upgraded to reoriented normal blending.
- Specular lighting is still Blinn-Phong style. There is no GGX/physical BRDF yet.
- Flat/untextured rendering is unchanged; wetness affects the textured Slang tissue path.

## Todo: Tissue State

- Add persistent per-vertex or per-surface tissue state for:
  - damage
  - coagulation
  - blood staining
  - heat
  - optional wet film
- Drive tissue state from surgical events:
  - cautery increases heat/coagulation
  - tool contact/cutting increases damage
  - bleeding events increase blood stain and wet film
  - heat/coagulation suppresses bleeding and wet gloss
- Add decay and transitions:
  - heat fades over time
  - blood dries and darkens
  - wetness evaporates
  - coagulation reduces gloss and blood response
- Add dedicated debug views for each dynamic tissue state channel.

## Todo: Shading Quality

- Add a configurable `wet_darken` material uniform instead of hard-coded wet diffuse darkening.
- Revisit `wet_roughness` semantics and range:
  - either make it a normalized `0.0..1.0` roughness control
  - or rename it to make the current narrow highlight control explicit
- Suppress blood film by coagulation, for example `blood_film *= (1.0 - coag_blend)`.
- Consolidate all tissue material parameter ranges in one table and clamp them consistently in the renderer API.
- Add a short shader comment or helper function documenting why dry and wet shininess curves differ.
- Upgrade normal blending to reoriented normal mapping.
- Replace the Blinn-Phong specular model with GGX or another clearer roughness model.
- Tune per-organ subsurface color/strength defaults once visual references are available.

## Todo: Postprocessing Realism

Highest-value post effects to add after the tissue shader is stable:

1. Screen-space ambient occlusion or contact occlusion.
2. Tone mapping and color calibration.
3. Temporal anti-aliasing.
4. Subtle bloom for wet tissue and metal highlights.
5. Endoscope/lens pass:
   - mild barrel distortion
   - slight edge darkening
   - very subtle chromatic aberration
   - sensor noise
6. Screen-space contact shadows under tools and tissue folds.
7. Very subtle depth of field for endoscopic camera simulation.
8. Cautery smoke/haze tied to heat or tool activity.

Suggested implementation order:

1. SSAO/contact occlusion.
2. Tone mapping.
3. TAA.
4. Subtle bloom.
5. Endoscope lens/sensor pass.

## Verification Notes

Focused verification used during this slice:

```powershell
python -m unittest tests.test_phase1_runtime.TestPhaseRuntime.test_slang_renderer_tissue_material_params_are_clamped_and_bound tests.test_phase1_runtime.TestPhaseRuntime.test_slang_renderer_clamps_wet_tissue_material_params_to_ui_ranges tests.test_phase1_runtime.TestPhaseRuntime.test_runtime_tissue_material_param_sync_updates_renderer tests.test_phase1_runtime.TestPhaseRuntime.test_runtime_tissue_material_ui_exposes_wet_sliders_and_syncs_renderer tests.test_phase1_runtime.TestPhaseRuntime.test_runtime_tissue_debug_ui_uses_named_combo_and_reset_button tests.test_phase1_runtime.TestPhaseRuntime.test_runtime_tissue_debug_ui_falls_back_to_slider tests.test_phase1_runtime.TestPhaseRuntime.test_slang_tissue_shader_declares_color_and_layer_bindings tests.test_phase1_runtime.TestPhaseRuntime.test_tissue_blend_debug_channels_fill_vertex_colors tests.test_phase1_runtime.TestPhaseRuntime.test_render_bridge_forwards_vertex_colors_to_slang tests.test_phase1_runtime.TestPhaseRuntime.test_render_bridge_accepts_vertex_colors_on_gl_path
```

```powershell
G:\warp\slang-2026.5.1-windows-x86_64\bin\slangc.exe omnisurg\rendering\slang_shaders\omnisurg_tissue.slang -entry vertex_main -stage vertex -entry fragment_main -stage fragment -target spirv -o .tmp_tests\omnisurg_tissue.spv
```

`uv run pytest` was blocked in the sandbox because the existing `.venv` interpreter could not be queried by the sandbox user.
