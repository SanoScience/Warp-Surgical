# Slang Rendering Notes

## Done

- Added the first Slang postprocessing scaffold:
  - geometry now renders into an offscreen HDR scene color target
  - a fullscreen `omnisurg_post.slang` pass writes to an LDR linear target before FXAA/present
  - the pass applies exposure, white balance, ACES tonemapping, endoscope vignette, and a soft circular scope mask
  - added a final present pass that applies sensor grain after antialiasing and manually encodes sRGB only when the swapchain is not sRGB
  - added a depth-only half-resolution AO pass with edge-aware blur for folds and close tissue contact
  - added shader-only FXAA as a lightweight non-temporal antialiasing pass
  - added COD-style multi-scale HDR bloom before tonemapping for wet tissue and metal highlights
  - added bloom-driven lens dirt overlay using `textures/lensdirt/LensDirt00.png`
  - runtime lens dirt selection can switch between the four `textures/lensdirt/LensDirt00..03.png` assets
  - added endoscope optics controls for barrel/pincushion distortion and radial chromatic aberration
  - added post-tonemap color calibration controls for saturation, contrast, gamma, and warmth
  - added subtle luminance-dependent sensor grain
  - added optional endoscope-style auto-exposure/AGC using a 1x1 temporal adaptation pass
  - Slang mesh/tissue final outputs keep HDR highlights instead of final `saturate()` clipping
  - runtime postprocess controls are exposed in the existing Rendering panel
- Added Slang viewport camera controls:
  - left mouse drag orbits around the fitted scene target
  - right or middle mouse drag pans
  - mouse wheel and PageUp/PageDown dolly
  - WASD strafes/flies, Q/E moves vertically, Shift speeds up, Control slows down
  - arrow keys orbit and Home resets the view
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
  - added derivative-based specular antialiasing controls for wet/dry highlight shimmer
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
- `specular_aa_enabled = True`
- `specular_aa_strength = 0.35`
- `specular_aa_min_roughness = 0.04`
- Postprocessing:
  - `enabled = True`
  - `exposure = 1.0`
  - `white_balance = (1.0, 1.0, 1.0)`
  - `auto_exposure_enabled = False`
  - `auto_exposure_target_luma = 0.35`
  - `auto_exposure_min = 0.35`
  - `auto_exposure_max = 1.8`
  - `auto_exposure_speed = 4.0`
  - `auto_exposure_highlight_weight = 0.8`
  - `ao_enabled = True`
  - `ao_intensity = 1.6`
  - `ao_radius = 0.22`
  - `ao_bias = 0.004`
  - `ao_power = 1.6`
  - `fxaa_enabled = True`
  - `fxaa_subpix = 0.75`
  - `fxaa_edge_threshold = 0.125`
  - `fxaa_edge_threshold_min = 0.0312`
  - `bloom_enabled = True`
  - `bloom_threshold = 1.0`
  - `bloom_intensity = 0.6`
  - `bloom_radius = 16.0`
  - `lens_dirt_enabled = True`
  - `lens_dirt_texture_index = 0` (`LensDirt00`)
  - `lens_dirt_intensity = 0.45`
  - `lens_dirt_threshold = 0.20`
  - `lens_dirt_base_opacity = 0.05`
  - `lens_dirt_global_drive = 1.0`
  - `lens_dirt_mask_gamma = 0.60`
  - `lens_distortion_enabled = True`
  - `lens_distortion_strength = 0.08`
  - `lens_distortion_zoom = 1.04`
  - `chromatic_aberration_enabled = True`
  - `chromatic_aberration_strength = 0.6`
  - `sensor_noise_enabled = True`
  - `sensor_noise_strength = 0.008`
  - `sensor_noise_shadow_boost = 1.5`
  - `color_grade_enabled = True`
  - `color_saturation = 1.0`
  - `color_contrast = 1.0`
  - `color_gamma = 1.0`
  - `color_warmth = 0.0`
  - `vignette_strength = 0.45`
  - `vignette_radius = 0.78`
  - `scope_radius = 0.965`
  - `scope_softness = 0.035`

These defaults are intentionally conservative. With zero vertex colors and default wetness, existing scenes should stay close to their previous appearance.

## Known Limitations

- Wetness is still a global material control plus blood-mask/blood-blend response. There is no spatial wetness accumulator from tool contact, bleeding, or fluid simulation yet.
- Damage, coagulation, and blood are exposed visually, but most runtime state is still slider/debug driven rather than authored by surgical events.
- Blood wetness does not yet account for coagulation state. Coagulated blood should become less glossy.
- Diffuse wet darkening is hard-coded and conservative. There is no `wet_darken` uniform yet.
- The wet roughness range is narrow and tuned for sharp film highlights. It does not cover very matte mucus or clotted films.
- Numeric material parameter clamping is currently asymmetric. Wet parameters are clamped in the renderer API, while older material controls rely mainly on UI ranges.
- Bloom uses a fixed five-level HDR pyramid with Karis-weighted downsample and tent upsample. It does not yet expose separate knee, firefly suppression, or chromatic dispersion controls.
- Lens dirt uses selectable static textures with base opacity, bloom-driven local flare, low-resolution global bloom drive, and mask gamma shaping. Dynamic droplets, smears, condensation, and blood accumulation are not implemented yet.
- AO is depth-only and half-resolution. It does not use true geometric ray queries, material thickness, or a full GTAO horizon integration.
- FXAA is intentionally lightweight and shader-only. It does not solve temporal shimmer as well as TAA.
- Auto-exposure samples a fixed 3x3 scene pattern plus the lowest bloom level rather than a full luminance histogram. It is intended as AGC scaffolding, not a finished metering model.
- Distortion, chromatic aberration, grain, and color grading are screen-space approximations. They do not yet use calibrated optics or sensor profiles.
- Smoke/haze is not implemented yet.
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
- Replace the Blinn-Phong specular model with GGX or another clearer roughness model; keep the specular AA roughness boost in that path.
- Tune per-organ subsurface color/strength defaults once visual references are available.

## Todo: Postprocessing Realism

Highest-value post effects to add after the first postprocess scaffold:

1. Dynamic lens contamination: droplets, smears, condensation, and blood masks.
2. Cautery smoke/haze tied to heat or tool activity.
3. Very subtle depth of field for close endoscopic camera simulation.
4. Screen-space subsurface scattering on a diffuse/subsurface target.
5. Temporal antialiasing with history rejection if motion/velocity buffers are added later.
6. Calibrated optics/sensor profiles for distortion, chromatic aberration, AGC, noise, and OR color response.

Suggested implementation order:

1. Dynamic lens dirt masks and droplet/smear authoring.
2. Cautery smoke/haze.
3. Subtle depth of field.
4. Screen-space subsurface scattering.
5. Calibrated endoscope profiles and preset bundles for the existing optics controls.

## Verification Notes

Focused verification used during this slice:

```powershell
python -m unittest tests.test_phase1_runtime.TestPhaseRuntime.test_slang_renderer_tissue_material_params_are_clamped_and_bound tests.test_phase1_runtime.TestPhaseRuntime.test_slang_renderer_clamps_wet_tissue_material_params_to_ui_ranges tests.test_phase1_runtime.TestPhaseRuntime.test_slang_renderer_postprocess_params_are_clamped_and_bound tests.test_phase1_runtime.TestPhaseRuntime.test_slang_renderer_bloom_level_sizes_are_bounded tests.test_phase1_runtime.TestPhaseRuntime.test_slang_renderer_resize_invalidates_bloom_textures tests.test_phase1_runtime.TestPhaseRuntime.test_runtime_tissue_material_param_sync_updates_renderer tests.test_phase1_runtime.TestPhaseRuntime.test_runtime_tissue_material_ui_exposes_wet_sliders_and_syncs_renderer tests.test_phase1_runtime.TestPhaseRuntime.test_runtime_postprocess_param_sync_updates_renderer tests.test_phase1_runtime.TestPhaseRuntime.test_runtime_postprocess_ui_exposes_optics_controls_and_syncs_renderer tests.test_phase1_runtime.TestPhaseRuntime.test_slang_tissue_shader_declares_color_and_layer_bindings tests.test_phase1_runtime.TestPhaseRuntime.test_slang_postprocess_shader_declares_tonemap_and_scope_bindings
```

```powershell
G:\warp\slang-2026.5.1-windows-x86_64\bin\slangc.exe -I omnisurg\rendering\slang_shaders -target spirv -profile glsl_460 -entry vertex_main -entry fragment_main omnisurg\rendering\slang_shaders\omnisurg_post.slang
G:\warp\slang-2026.5.1-windows-x86_64\bin\slangc.exe -I omnisurg\rendering\slang_shaders -target spirv -profile glsl_460 -entry vertex_main -entry ao_fragment omnisurg\rendering\slang_shaders\omnisurg_ao.slang
G:\warp\slang-2026.5.1-windows-x86_64\bin\slangc.exe -I omnisurg\rendering\slang_shaders -target spirv -profile glsl_460 -entry vertex_main -entry fragment_main omnisurg\rendering\slang_shaders\omnisurg_fxaa.slang
G:\warp\slang-2026.5.1-windows-x86_64\bin\slangc.exe -I omnisurg\rendering\slang_shaders -target spirv -profile glsl_460 -entry vertex_main -entry fragment_main omnisurg\rendering\slang_shaders\omnisurg_present.slang
```
