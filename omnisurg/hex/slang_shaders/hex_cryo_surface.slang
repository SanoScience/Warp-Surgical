#ifndef MM_EXTERNAL_MATERIAL
#define MM_EXTERNAL_MATERIAL 0
#endif
#ifndef MM_HAS_ALBEDO
#define MM_HAS_ALBEDO 0
#endif
#ifndef MM_HAS_ALPHA
#define MM_HAS_ALPHA 0
#endif
#ifndef MM_HAS_METALLIC
#define MM_HAS_METALLIC 0
#endif
#ifndef MM_HAS_ROUGHNESS
#define MM_HAS_ROUGHNESS 0
#endif
#ifndef MM_HAS_EMISSION
#define MM_HAS_EMISSION 0
#endif
#ifndef MM_HAS_OCCLUSION
#define MM_HAS_OCCLUSION 0
#endif
#ifndef MM_HAS_HEIGHT
#define MM_HAS_HEIGHT 0
#endif
#ifndef MM_HAS_DEPTH
#define MM_HAS_DEPTH 0
#endif
#ifndef MM_HAS_NORMAL
#define MM_HAS_NORMAL 0
#endif
#ifndef MM_HAS_SPECULAR
#define MM_HAS_SPECULAR 0
#endif
#ifndef MM_HAS_WETNESS
#define MM_HAS_WETNESS 0
#endif
#ifndef MM_HAS_CLEARCOAT
#define MM_HAS_CLEARCOAT 0
#endif
#ifndef MM_HAS_CLEARCOATROUGHNESS
#define MM_HAS_CLEARCOATROUGHNESS 0
#endif
#ifndef MM_HAS_SSSSTRENGTH
#define MM_HAS_SSSSTRENGTH 0
#endif
#ifndef MM_HAS_SSSCOLOR
#define MM_HAS_SSSCOLOR 0
#endif
#ifndef MM_HAS_SSSDEPTH
#define MM_HAS_SSSDEPTH 0
#endif
#ifndef MM_HAS_SSSBOOST
#define MM_HAS_SSSBOOST 0
#endif
#ifndef MM_HAS_THICKNESS
#define MM_HAS_THICKNESS 0
#endif
#ifndef MM_HAS_BACKLIGHT
#define MM_HAS_BACKLIGHT 0
#endif

#define DEBUG_VIEW_OFF 0
#define DEBUG_VIEW_HEIGHT 1
#define DEBUG_VIEW_ROUGHNESS 2
#define DEBUG_VIEW_WETNESS 3
#define DEBUG_VIEW_MATERIAL_NORMAL_MAP 4
#define DEBUG_VIEW_FINAL_SHADE_NORMAL 5
#define DEBUG_VIEW_MATERIAL_INDEX 6

struct V2F
{
    float4 position : SV_Position;
    float3 world_position : WORLD_POSITION;
    float3 world_normal : NORMAL;
    float3 uv3 : TEXCOORD0;
    float3 procedural_coord : TEXCOORD1;
    nointerpolation int material_index : TEXCOORD2;
    nointerpolation float4 surface_state : TEXCOORD3;
};

uniform float3 camera_pos;
uniform float3 camera_right;
uniform float3 camera_up;
uniform float3 camera_forward;
uniform float camera_inv_tan_half_fovy;
uniform float camera_aspect;
uniform float camera_near;
uniform float camera_far;
uniform float4 mesh_color;

uniform float3 cryo_volume_scale;
uniform float3 procedural_uv3_noise_scale;
uniform float3 light_key_dir;
uniform float3 light_fill_dir;
uniform float3 light_ambient_color;
uniform float3 light_key_color;
uniform float3 light_fill_color;
uniform int key_light_enabled;
uniform int fill_light_enabled;
uniform int ambient_light_enabled;
uniform int environment_lighting_enabled;
uniform int has_environment_map;
uniform float environment_intensity;
uniform float environment_rotation;
uniform float environment_pitch;
uniform int material_count;
uniform int has_material_state;
uniform int procedural_enabled;
uniform int procedural_world_space;
uniform int has_cryo_volume;
uniform int lighting_enabled;
uniform int debug_view;
uniform float cryo_mix;
uniform float state_overlay_strength;
uniform float procedural_material_scale;
uniform float procedural_time;

Texture3D<float4> cryo_volume;
Texture2D<float4> environment_map;
SamplerState cryo_sampler;
SamplerState environment_sampler;
StructuredBuffer<int> material_id;
StructuredBuffer<float4> state_rgba;
StructuredBuffer<float4> material_colors;
StructuredBuffer<float4> procedural_params;

float3 project_position(float3 position, out float camera_z)
{
    float3 rel = position - camera_pos;
    float camera_x = dot(rel, camera_right);
    float camera_y = dot(rel, camera_up);
    camera_z = max(dot(rel, camera_forward), camera_near);

    float clip_x = camera_x * camera_inv_tan_half_fovy / max(camera_aspect, 0.0001);
    float clip_y = camera_y * camera_inv_tan_half_fovy;
    float clip_z = camera_z * camera_far / (camera_far - camera_near)
        - camera_near * camera_far / (camera_far - camera_near);
    return float3(clip_x, clip_y, clip_z);
}

float scale_about_center(float v, float scale)
{
    return (v - 0.5) / max(scale, 1.0e-6) + 0.5;
}

float3 cryo_uv(float3 uv3)
{
    return saturate(float3(
        scale_about_center(uv3.x, cryo_volume_scale.x),
        scale_about_center(uv3.y, cryo_volume_scale.y),
        scale_about_center(uv3.z, cryo_volume_scale.z)
    ));
}

[shader("vertex")]
V2F vertex_main(
    float3 position : POSITION,
    float3 normal : NORMAL,
    float3 uv3 : TEXCOORD0,
    float3 procedural_coord : TEXCOORD1,
    uint vertex_id : SV_VertexID)
{
    V2F output;
    float camera_z;
    output.position = float4(project_position(position, camera_z), camera_z);
    output.world_position = position;
    output.world_normal = normal;
    output.uv3 = uv3;
    output.procedural_coord = procedural_coord;
    if (has_material_state != 0)
    {
        output.material_index = clamp(material_id[vertex_id], 0, max(material_count - 1, 0));
        output.surface_state = state_rgba[vertex_id];
    }
    else
    {
        output.material_index = 0;
        output.surface_state = float4(0.0);
    }
    return output;
}

float srgb_channel_to_linear(float c)
{
    return pow(max(c, 0.0), 2.2);
}

float3 srgb_to_linear(float3 color)
{
    return float3(
        srgb_channel_to_linear(color.r),
        srgb_channel_to_linear(color.g),
        srgb_channel_to_linear(color.b)
    );
}

float smooth_rest_gradient(float3 texture_position)
{
    float3 uv = saturate(texture_position);
    return smoothstep(0.0, 1.0, dot(uv, float3(0.46, 0.34, 0.20)));
}

float3 diagnostic_gradient_albedo(float3 texture_position, float3 base, float grain, float tint)
{
    float3 uv = saturate(texture_position);
    float gradient = smooth_rest_gradient(uv);
    float3 axis_gradient = saturate(uv * 0.90 + float3(0.06));
    float3 ramp_gradient = lerp(float3(0.04, 0.18, 0.95), float3(1.0, 0.86, 0.12), gradient);
    float3 gradient_color = srgb_to_linear(lerp(ramp_gradient, axis_gradient, 0.38));
    float3 material_tint = lerp(float3(1.0), base * 1.25, tint * 0.20);
    float gradient_gain = lerp(1.0, lerp(0.72, 1.24, gradient), grain);
    return max(gradient_color * material_tint * gradient_gain, float3(0.0));
}

float hash31(float3 p)
{
    p = frac(p * 0.1031);
    p = p + float3(dot(p, p.yzx + float3(33.33)));
    return frac((p.x + p.y) * p.z);
}

float value_noise(float3 p)
{
    float3 cell = floor(p);
    float3 f = frac(p);
    float3 u = f * f * (3.0 - 2.0 * f);

    float n000 = hash31(cell + float3(0.0, 0.0, 0.0));
    float n100 = hash31(cell + float3(1.0, 0.0, 0.0));
    float n010 = hash31(cell + float3(0.0, 1.0, 0.0));
    float n110 = hash31(cell + float3(1.0, 1.0, 0.0));
    float n001 = hash31(cell + float3(0.0, 0.0, 1.0));
    float n101 = hash31(cell + float3(1.0, 0.0, 1.0));
    float n011 = hash31(cell + float3(0.0, 1.0, 1.0));
    float n111 = hash31(cell + float3(1.0, 1.0, 1.0));

    float nx00 = lerp(n000, n100, u.x);
    float nx10 = lerp(n010, n110, u.x);
    float nx01 = lerp(n001, n101, u.x);
    float nx11 = lerp(n011, n111, u.x);
    float nxy0 = lerp(nx00, nx10, u.y);
    float nxy1 = lerp(nx01, nx11, u.y);
    return lerp(nxy0, nxy1, u.z);
}

float procedural_height(float3 noise_position)
{
    float coarse = value_noise(noise_position);
    float fine = value_noise(noise_position * 2.13 + float3(17.7, 9.2, 3.5));
    return saturate(coarse * 0.68 + fine * 0.32);
}

float visible_grain(float3 texture_position)
{
    return procedural_height(texture_position);
}

float3 normalize_or(float3 value, float3 fallback)
{
    float value_len = length(value);
    return value_len > 0.000001 ? value / value_len : fallback;
}

float3 fallback_surface_tangent(float3 normal)
{
    float3 axis = abs(normal.z) < 0.999 ? float3(0.0, 0.0, 1.0) : float3(0.0, 1.0, 0.0);
    return normalize_or(cross(axis, normal), float3(1.0, 0.0, 0.0));
}

bool material_space_jacobian_axis(
    float3 target_axis,
    float3 dpdx,
    float3 dpdy,
    float3 dtdx,
    float3 dtdy,
    out float3 world_axis,
    out float3 material_axis)
{
    float gram_xx = dot(dtdx, dtdx);
    float gram_xy = dot(dtdx, dtdy);
    float gram_yy = dot(dtdy, dtdy);
    float det = gram_xx * gram_yy - gram_xy * gram_xy;
    if (abs(det) <= 0.0000000001)
    {
        world_axis = float3(0.0);
        material_axis = float3(0.0);
        return false;
    }

    float rhs_x = dot(dtdx, target_axis);
    float rhs_y = dot(dtdy, target_axis);
    float weight_x = (gram_yy * rhs_x - gram_xy * rhs_y) / det;
    float weight_y = (-gram_xy * rhs_x + gram_xx * rhs_y) / det;
    world_axis = dpdx * weight_x + dpdy * weight_y;
    material_axis = dtdx * weight_x + dtdy * weight_y;
    return length(world_axis) > 0.000001 && length(material_axis) > 0.000001;
}

void material_maker_derivative_frame(
    float3 world_position,
    float3 world_normal,
    float3 texture_position,
    out float3 material_tangent,
    out float3 material_binormal,
    out float3 world_tangent,
    out float3 world_binormal)
{
    float3 base_normal = normalize_or(world_normal, float3(0.0, 0.0, 1.0));
    float3 fallback_tangent = fallback_surface_tangent(base_normal);
    float3 fallback_binormal = normalize_or(cross(base_normal, fallback_tangent), float3(0.0, 1.0, 0.0));

    float3 dpdx = ddx(world_position);
    float3 dpdy = ddy(world_position);
    float3 dtdx = ddx(texture_position);
    float3 dtdy = ddy(texture_position);

    float3 fallback_world_tangent = normalize_or(dpdx - base_normal * dot(dpdx, base_normal), fallback_tangent);
    float3 fallback_world_binormal = dpdy - base_normal * dot(dpdy, base_normal);
    fallback_world_binormal = normalize_or(
        fallback_world_binormal - fallback_world_tangent * dot(fallback_world_binormal, fallback_world_tangent),
        fallback_binormal);
    if (dot(cross(fallback_world_tangent, fallback_world_binormal), base_normal) < 0.0)
    {
        fallback_world_binormal = -fallback_world_binormal;
    }

    float3 fallback_material_tangent = normalize_or(dtdx, float3(1.0, 0.0, 0.0));
    float3 fallback_material_binormal = normalize_or(dtdy, float3(0.0, 1.0, 0.0));
    float3 solved_world_tangent;
    float3 solved_world_binormal;
    float3 solved_material_tangent;
    float3 solved_material_binormal;
    bool solved_tangent = material_space_jacobian_axis(
        float3(1.0, 0.0, 0.0),
        dpdx,
        dpdy,
        dtdx,
        dtdy,
        solved_world_tangent,
        solved_material_tangent);
    bool solved_binormal = material_space_jacobian_axis(
        float3(0.0, 1.0, 0.0),
        dpdx,
        dpdy,
        dtdx,
        dtdy,
        solved_world_binormal,
        solved_material_binormal);
    if (!solved_tangent || !solved_binormal)
    {
        world_tangent = fallback_world_tangent;
        world_binormal = fallback_world_binormal;
        material_tangent = fallback_material_tangent;
        material_binormal = fallback_material_binormal;
        return;
    }

    world_tangent = solved_world_tangent - base_normal * dot(solved_world_tangent, base_normal);
    world_tangent = normalize_or(world_tangent, fallback_world_tangent);
    world_binormal = solved_world_binormal - base_normal * dot(solved_world_binormal, base_normal);
    world_binormal = normalize_or(world_binormal - world_tangent * dot(world_binormal, world_tangent), fallback_world_binormal);
    if (dot(cross(world_tangent, world_binormal), base_normal) < 0.0)
    {
        world_binormal = -world_binormal;
        solved_material_binormal = -solved_material_binormal;
    }

    material_tangent = normalize_or(solved_material_tangent, fallback_material_tangent);
    material_binormal = normalize_or(solved_material_binormal, fallback_material_binormal);
}

float3 material_normal_map_to_world(
    float3 normal_map,
    float3 world_normal,
    float3 world_tangent,
    float3 world_binormal)
{
    float3 base_normal = normalize_or(world_normal, float3(0.0, 0.0, 1.0));
    float3 tangent_normal = saturate(normal_map) * 2.0 - float3(1.0);
    tangent_normal = normalize_or(tangent_normal, float3(0.0, 0.0, 1.0));
    float3 perturbed = world_tangent * tangent_normal.x
        + world_binormal * tangent_normal.y
        + base_normal * tangent_normal.z;
    perturbed = normalize_or(perturbed, base_normal);
    return dot(perturbed, base_normal) < 0.0 ? -perturbed : perturbed;
}

float3 procedural_albedo(
    float3 texture_position,
    int material_index,
    out float roughness,
    out float wetness,
    out float height_strength,
    out float normal_strength,
    out float height)
{
    int clamped_index = clamp(material_index, 0, max(material_count - 1, 0));
    float4 color_row = material_colors[clamped_index];
    float4 params0 = procedural_params[clamped_index * 2 + 0];
    float4 params1 = procedural_params[clamped_index * 2 + 1];
    float scale = max(params0.x, 1.0);
    float grain = saturate(params0.y);
    float tint = saturate(params0.z);
    roughness = saturate(params0.w);
    wetness = saturate(params1.x);
    height_strength = saturate(params1.y);
    normal_strength = saturate(params1.z);

    float3 base = srgb_to_linear(saturate(color_row.rgb));
    float3 noise_position = texture_position * scale;
    float albedo_noise = procedural_height(noise_position);
    height = albedo_noise;
    float tint_noise = visible_grain(noise_position * 0.57 + float3(31.4, 5.6, 12.8));
    float albedo_gain = lerp(1.0, lerp(0.62, 1.44, albedo_noise), grain);
    float3 color_variation = lerp(float3(0.74, 0.84, 1.12), float3(1.18, 1.06, 0.82), tint_noise);
    float3 tinted = lerp(base, base * color_variation, tint * grain);
    return max(tinted * albedo_gain, float3(0.0));
}

#if MM_EXTERNAL_MATERIAL
float3 mm_material_albedo(MMMaterial material)
{
#if MM_HAS_ALBEDO
    return max(material.albedo, float3(0.0));
#else
    return float3(1.0);
#endif
}

float mm_material_metallic(MMMaterial material)
{
#if MM_HAS_METALLIC
    return saturate(material.metallic);
#else
    return 0.0;
#endif
}

float mm_material_roughness(MMMaterial material)
{
#if MM_HAS_ROUGHNESS
    return clamp(material.roughness, 0.04, 1.0);
#else
    return 0.5;
#endif
}

float mm_material_occlusion(MMMaterial material)
{
#if MM_HAS_OCCLUSION
    return saturate(material.occlusion);
#else
    return 1.0;
#endif
}

float3 mm_material_emission(MMMaterial material)
{
#if MM_HAS_EMISSION
    return max(material.emission, float3(0.0));
#else
    return float3(0.0);
#endif
}

float3 mm_material_normal(MMMaterial material)
{
#if MM_HAS_NORMAL
    return material.normal;
#else
    return float3(0.5, 0.5, 1.0);
#endif
}

float mm_material_specular(MMMaterial material)
{
#if MM_HAS_SPECULAR
    return saturate(material.specular);
#else
    return 0.5;
#endif
}

float mm_material_wetness(MMMaterial material, float fallback)
{
#if MM_HAS_WETNESS
    return saturate(material.wetness);
#else
    return fallback;
#endif
}

float mm_material_clearcoat(MMMaterial material)
{
#if MM_HAS_CLEARCOAT
    return saturate(material.clearcoat);
#else
    return 0.0;
#endif
}

float mm_material_clearcoat_roughness(MMMaterial material)
{
#if MM_HAS_CLEARCOATROUGHNESS
    return clamp(material.clearcoatRoughness, 0.02, 1.0);
#else
    return 0.12;
#endif
}

float mm_material_sss_strength(MMMaterial material)
{
#if MM_HAS_SSSSTRENGTH
    return saturate(material.sssStrength);
#else
    return 0.0;
#endif
}

float3 mm_material_sss_color(MMMaterial material)
{
#if MM_HAS_SSSCOLOR
    return max(material.sssColor, float3(0.0));
#else
    return float3(1.0, 0.35, 0.2);
#endif
}

float mm_material_sss_depth(MMMaterial material)
{
#if MM_HAS_SSSDEPTH
    return max(material.sssDepth, 0.0);
#else
    return 0.0;
#endif
}

float mm_material_sss_boost(MMMaterial material)
{
#if MM_HAS_SSSBOOST
    return max(material.sssBoost, 0.0);
#else
    return 0.0;
#endif
}

float mm_material_thickness(MMMaterial material)
{
#if MM_HAS_THICKNESS
    return saturate(material.thickness);
#else
    return 1.0;
#endif
}

float3 mm_material_backlight(MMMaterial material, float sss_strength, float3 sss_color, float thickness)
{
#if MM_HAS_BACKLIGHT
    return max(material.backlight, float3(0.0));
#else
    return sss_color * sss_strength * thickness;
#endif
}

float mm_material_height(MMMaterial material)
{
#if MM_HAS_HEIGHT
    return saturate(material.height);
#elif MM_HAS_DEPTH
    return saturate(material.depth);
#else
    return 0.0;
#endif
}

float3 material_maker_albedo(
    float3 texture_position,
    float3 material_tangent,
    float3 material_binormal,
    int material_index,
    out float roughness,
    out float wetness,
    out float height_strength,
    out float normal_strength,
    out float height,
    out float metallic,
    out float occlusion,
    out float3 emission,
    out float3 normal_map,
    out float specular_level,
    out float clearcoat,
    out float clearcoat_roughness,
    out float sss_strength,
    out float3 sss_color,
    out float sss_depth,
    out float sss_boost,
    out float thickness,
    out float3 backlight)
{
    int clamped_index = clamp(material_index, 0, max(material_count - 1, 0));
    float4 params1 = procedural_params[clamped_index * 2 + 1];
    MMMaterial material = MM_EVALUATE_MATERIAL_FRAME(texture_position, material_tangent, material_binormal, uint(clamped_index));
    roughness = mm_material_roughness(material);
    wetness = mm_material_wetness(material, saturate(params1.x));
    height_strength = saturate(params1.y);
    normal_strength = saturate(params1.z);
    height = mm_material_height(material);
    metallic = mm_material_metallic(material);
    occlusion = mm_material_occlusion(material);
    emission = mm_material_emission(material);
    normal_map = mm_material_normal(material);
    specular_level = mm_material_specular(material);
    clearcoat = mm_material_clearcoat(material);
    clearcoat_roughness = mm_material_clearcoat_roughness(material);
    sss_strength = mm_material_sss_strength(material);
    sss_color = mm_material_sss_color(material);
    sss_depth = mm_material_sss_depth(material);
    sss_boost = mm_material_sss_boost(material);
    thickness = mm_material_thickness(material);
    backlight = mm_material_backlight(material, sss_strength, sss_color, thickness);
    return mm_material_albedo(material);
}
#endif

float3 perturb_normal_from_height(
    float3 world_position,
    float3 world_normal,
    float height,
    float height_strength,
    float normal_strength)
{
    float3 base_normal = world_normal;
    float normal_len = length(base_normal);
    if (normal_len <= 0.000001)
    {
        return world_normal;
    }
    base_normal = base_normal / normal_len;

    float3 dpdx = ddx(world_position);
    float3 dpdy = ddy(world_position);
    float dhdx = ddx(height);
    float dhdy = ddy(height);
    float3 r1 = cross(dpdy, base_normal);
    float3 r2 = cross(base_normal, dpdx);
    float det = dot(dpdx, r1);
    if (abs(det) <= 0.0000000001)
    {
        return world_normal;
    }

    float amount = saturate(height_strength) * saturate(normal_strength);
    float3 gradient = (r1 * dhdx + r2 * dhdy) / det;
    float3 perturbed = base_normal - gradient * (amount * 0.02);
    float perturbed_len = length(perturbed);
    if (perturbed_len <= 0.000001)
    {
        return world_normal;
    }
    perturbed = perturbed / perturbed_len;
    return dot(perturbed, base_normal) < 0.0 ? -perturbed : perturbed;
}

float2 environment_equirect_uv(float3 direction)
{
    float3 dir = normalize_or(direction, float3(0.0, 1.0, 0.0));
    float pitch_sin = sin(environment_pitch);
    float pitch_cos = cos(environment_pitch);
    dir = normalize_or(
        float3(
            dir.x,
            dir.y * pitch_cos - dir.z * pitch_sin,
            dir.y * pitch_sin + dir.z * pitch_cos
        ),
        dir
    );
    float u = atan2(dir.z, dir.x) * 0.159154943 + environment_rotation;
    float v = acos(clamp(dir.y, -1.0, 1.0)) * 0.318309886;
    return float2(frac(u), v);
}

float3 sample_environment_uv(float2 uv)
{
    if (has_environment_map == 0 || environment_lighting_enabled == 0)
    {
        return float3(0.0);
    }
    float2 wrapped_uv = float2(frac(uv.x), clamp(uv.y, 0.0, 1.0));
    return max(environment_map.SampleLevel(environment_sampler, wrapped_uv, 0.0).rgb, float3(0.0))
        * max(environment_intensity, 0.0);
}

float3 sample_environment(float3 direction)
{
    return sample_environment_uv(environment_equirect_uv(direction));
}

float3 sample_environment_blurred(float3 direction, float blur_radius)
{
    float2 uv = environment_equirect_uv(direction);
    float radius = clamp(blur_radius, 0.0, 0.12);
    if (radius <= 0.000001)
    {
        return sample_environment_uv(uv);
    }

    float2 u_offset = float2(radius, 0.0);
    float2 v_offset = float2(0.0, radius * 2.0);
    float3 color = sample_environment_uv(uv) * 0.20;
    color += (
        sample_environment_uv(uv + u_offset)
        + sample_environment_uv(uv - u_offset)
        + sample_environment_uv(uv + v_offset)
        + sample_environment_uv(uv - v_offset)
    ) * 0.12;
    color += (
        sample_environment_uv(uv + u_offset + v_offset)
        + sample_environment_uv(uv + u_offset - v_offset)
        + sample_environment_uv(uv - u_offset + v_offset)
        + sample_environment_uv(uv - u_offset - v_offset)
    ) * 0.08;
    return color;
}

float3 shade_surface(float3 world_position, float3 world_normal, float3 base_color, float roughness, float wetness)
{
    float3 view_dir = normalize(camera_pos - world_position);
    float3 normal = world_normal;
    float normal_len = length(normal);
    normal = normal_len > 0.000001 ? normal / normal_len : view_dir;
    normal = dot(normal, view_dir) < 0.0 ? -normal : normal;

    float key_diffuse = saturate(dot(normal, normalize(light_key_dir)));
    float fill_diffuse = saturate(dot(normal, normalize(light_fill_dir)));
    float3 half_dir = normalize(normalize(light_key_dir) + view_dir);
    float gloss = saturate((1.0 - roughness) * (0.72 + wetness * 0.55));
    float spec_power = lerp(18.0, 128.0, gloss);
    float spec_strength = lerp(0.035, 0.18, gloss) + wetness * 0.16;
    float specular = pow(saturate(dot(normal, half_dir)), spec_power) * spec_strength;
    float fresnel = pow(1.0 - saturate(dot(normal, view_dir)), 4.0);

    float3 key_color = key_light_enabled != 0 ? light_key_color : float3(0.0);
    float3 fill_color = fill_light_enabled != 0 ? light_fill_color : float3(0.0);
    float3 ambient_color = has_environment_map != 0 && environment_lighting_enabled != 0
        ? sample_environment_blurred(normal, 0.035) * 0.55
        : (ambient_light_enabled != 0 ? light_ambient_color : float3(0.0));
    float3 diffuse =
        ambient_color
        + key_color * key_diffuse * 0.88
        + fill_color * fill_diffuse * 0.34;
    float3 reflection_color = has_environment_map != 0 && environment_lighting_enabled != 0
        ? sample_environment_blurred(reflect(-view_dir, normal), lerp(0.028, 0.080, saturate(1.0 - gloss)))
            * lerp(0.08, 0.45, gloss)
        : key_color * specular;
    return base_color * diffuse + reflection_color * specular + base_color * fresnel * (0.10 + wetness * 0.10);
}

#if MM_EXTERNAL_MATERIAL
float3 shade_pbr_surface(
    float3 world_position,
    float3 world_normal,
    float3 base_color,
    float roughness,
    float metallic,
    float occlusion,
    float3 emission,
    float specular_level,
    float clearcoat,
    float clearcoat_roughness,
    float sss_strength,
    float3 sss_color,
    float sss_depth,
    float sss_boost,
    float thickness,
    float3 backlight,
    float wetness)
{
    float3 view_dir = normalize(camera_pos - world_position);
    float3 normal = world_normal;
    float normal_len = length(normal);
    normal = normal_len > 0.000001 ? normal / normal_len : view_dir;
    normal = dot(normal, view_dir) < 0.0 ? -normal : normal;

    float3 light_dir = normalize(light_key_dir);
    float3 half_dir = normalize(light_dir + view_dir);
    float n_dot_l = saturate(dot(normal, light_dir));
    float n_dot_v = saturate(dot(normal, view_dir));
    float n_dot_h = saturate(dot(normal, half_dir));
    float v_dot_h = saturate(dot(view_dir, half_dir));
    float effective_roughness = clamp(roughness, 0.04, 1.0);
    float alpha = effective_roughness * effective_roughness;
    float alpha_squared = max(alpha * alpha, 0.001);
    float denominator = n_dot_h * n_dot_h * (alpha_squared - 1.0) + 1.0;
    float distribution = alpha_squared / max(3.14159265 * denominator * denominator, 0.001);
    float k = (effective_roughness + 1.0) * (effective_roughness + 1.0) * 0.125;
    float geometry_l = n_dot_l / max(n_dot_l * (1.0 - k) + k, 0.001);
    float geometry_v = n_dot_v / max(n_dot_v * (1.0 - k) + k, 0.001);
    float dielectric_f0 = clamp(specular_level * 0.08, 0.0, 0.16);
    float3 f0 = lerp(float3(dielectric_f0), base_color, metallic);
    float3 fresnel = f0 + (float3(1.0) - f0) * pow(1.0 - v_dot_h, 5.0);
    float3 specular = distribution * geometry_l * geometry_v * fresnel / max(4.0 * n_dot_l * n_dot_v, 0.001);
    float3 diffuse = (float3(1.0) - fresnel) * (1.0 - metallic) * base_color;
    float fill_diffuse = saturate(dot(normal, normalize(light_fill_dir)));
    float3 key_color = key_light_enabled != 0 ? light_key_color : float3(0.0);
    float3 fill_color = fill_light_enabled != 0 ? light_fill_color : float3(0.0);
    float clearcoat_gloss = saturate(1.0 - clearcoat_roughness);
    float clearcoat_power = lerp(32.0, 384.0, clearcoat_gloss);
    float clearcoat_specular = pow(n_dot_h, clearcoat_power) * saturate(clearcoat) * (0.04 + wetness * 0.08);
    float3 direct = (diffuse + specular) * key_color * n_dot_l
        + key_color * clearcoat_specular
        + base_color * fill_color * fill_diffuse * 0.22;
    float3 ambient_color = has_environment_map != 0 && environment_lighting_enabled != 0
        ? sample_environment_blurred(normal, 0.035) * 0.55
        : (ambient_light_enabled != 0 ? light_ambient_color : float3(0.0));
    float3 ambient = base_color * ambient_color * occlusion;
    float3 environment_specular = has_environment_map != 0 && environment_lighting_enabled != 0
        ? sample_environment_blurred(reflect(-view_dir, normal), lerp(0.025, 0.085, effective_roughness))
            * fresnel * (1.0 - effective_roughness) * 0.35
        : float3(0.0);
    float back_wrap = saturate(dot(-normal, light_dir) * 0.5 + 0.5);
    float view_wrap = pow(1.0 - n_dot_v, 2.0);
    float transmission = saturate(sss_strength) * saturate(thickness) * (0.25 + saturate(sss_boost));
    float depth_attenuation = 1.0 / (1.0 + max(sss_depth, 0.0));
    float3 transmitted = max(sss_color, float3(0.0)) * key_color * back_wrap * transmission * depth_attenuation;
    float3 backlight_tint = max(backlight, float3(0.0)) * fill_color * (0.15 + 0.65 * view_wrap);
    return direct + ambient + environment_specular + emission + transmitted + backlight_tint;
}
#endif

float4 surface_debug_color(
    int debug_mode,
    float height,
    float roughness,
    float wetness,
    float3 material_normal_map,
    float3 shade_normal,
    int material_index)
{
    if (debug_mode == DEBUG_VIEW_HEIGHT)
    {
        return float4(float3(saturate(height)), 1.0);
    }
    if (debug_mode == DEBUG_VIEW_ROUGHNESS)
    {
        return float4(float3(saturate(roughness)), 1.0);
    }
    if (debug_mode == DEBUG_VIEW_WETNESS)
    {
        return float4(float3(saturate(wetness)), 1.0);
    }
    if (debug_mode == DEBUG_VIEW_MATERIAL_NORMAL_MAP)
    {
        return float4(saturate(material_normal_map), 1.0);
    }
    if (debug_mode == DEBUG_VIEW_FINAL_SHADE_NORMAL)
    {
        return float4(normalize_or(shade_normal, float3(0.0, 0.0, 1.0)) * 0.5 + float3(0.5), 1.0);
    }
    if (debug_mode == DEBUG_VIEW_MATERIAL_INDEX)
    {
        int clamped_index = clamp(material_index, 0, max(material_count - 1, 0));
        float denom = float(max(material_count - 1, 1));
        float value = material_count > 1 ? saturate(float(clamped_index) / denom) : 0.0;
        return float4(float3(value), 1.0);
    }
    return float4(float3(0.0), 1.0);
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    float3 sampled = has_cryo_volume != 0 ? cryo_volume.Sample(cryo_sampler, cryo_uv(input.uv3)).rgb : float3(0.5);
    float3 cryo_color = srgb_to_linear(sampled);
    float roughness = 0.50;
    float wetness = 0.0;
    float height_strength = 0.0;
    float normal_strength = 0.0;
    float height = 0.0;
    float metallic = 0.0;
    float occlusion = 1.0;
    float3 emission = float3(0.0);
#if MM_EXTERNAL_MATERIAL
    float3 material_normal_map = float3(0.5, 0.5, 1.0);
    float specular_level = 0.5;
    float clearcoat = 0.0;
    float clearcoat_roughness = 0.12;
    float sss_strength = 0.0;
    float3 sss_color = float3(1.0, 0.35, 0.2);
    float sss_depth = 0.0;
    float sss_boost = 0.0;
    float thickness = 1.0;
    float3 backlight = float3(0.0);
    float3 material_tangent = float3(1.0, 0.0, 0.0);
    float3 material_binormal = float3(0.0, 1.0, 0.0);
    float3 world_tangent = float3(1.0, 0.0, 0.0);
    float3 world_binormal = float3(0.0, 1.0, 0.0);
#endif
    float3 base_color = cryo_color;
    if (procedural_enabled != 0)
    {
        float3 procedural_position =
            procedural_world_space != 0 ? input.world_position * 100.0 : input.procedural_coord * procedural_uv3_noise_scale;
        procedural_position *= max(procedural_material_scale, 0.000001);
#if MM_EXTERNAL_MATERIAL
        material_maker_derivative_frame(
            input.world_position,
            input.world_normal,
            procedural_position,
            material_tangent,
            material_binormal,
            world_tangent,
            world_binormal);
        base_color = material_maker_albedo(
            procedural_position,
            material_tangent,
            material_binormal,
            input.material_index,
            roughness,
            wetness,
            height_strength,
            normal_strength,
            height,
            metallic,
            occlusion,
            emission,
            material_normal_map,
            specular_level,
            clearcoat,
            clearcoat_roughness,
            sss_strength,
            sss_color,
            sss_depth,
            sss_boost,
            thickness,
            backlight);
#else
        base_color = procedural_albedo(
            procedural_position,
            input.material_index,
            roughness,
            wetness,
            height_strength,
            normal_strength,
            height);
#endif
        if (has_cryo_volume != 0)
        {
            base_color = lerp(base_color, cryo_color, saturate(cryo_mix));
        }
    }
    float3 shade_normal = input.world_normal;
    if (procedural_enabled != 0)
    {
#if MM_EXTERNAL_MATERIAL && MM_HAS_NORMAL
        shade_normal = material_normal_map_to_world(
            material_normal_map,
            input.world_normal,
            world_tangent,
            world_binormal);
#else
        shade_normal = perturb_normal_from_height(
            input.world_position,
            input.world_normal,
            height,
            height_strength,
            normal_strength);
#endif
    }
    float3 debug_material_normal_map = float3(0.5, 0.5, 1.0);
#if MM_EXTERNAL_MATERIAL
    debug_material_normal_map = material_normal_map;
#endif
    int clamped_debug_view = clamp(debug_view, DEBUG_VIEW_OFF, DEBUG_VIEW_MATERIAL_INDEX);
    if (clamped_debug_view != DEBUG_VIEW_OFF)
    {
        return surface_debug_color(
            clamped_debug_view,
            height,
            roughness,
            wetness,
            debug_material_normal_map,
            shade_normal,
            input.material_index);
    }
    float state_alpha = saturate(input.surface_state.a * state_overlay_strength);
    base_color = lerp(base_color, srgb_to_linear(saturate(input.surface_state.rgb)), state_alpha);
    if (lighting_enabled == 0)
    {
        return float4(max(base_color, float3(0.0)), 1.0);
    }
#if MM_EXTERNAL_MATERIAL
    float3 shaded = procedural_enabled != 0
        ? shade_pbr_surface(
            input.world_position,
            shade_normal,
            base_color,
            roughness,
            metallic,
            occlusion,
            emission,
            specular_level,
            clearcoat,
            clearcoat_roughness,
            sss_strength,
            sss_color,
            sss_depth,
            sss_boost,
            thickness,
            backlight,
            wetness)
        : shade_surface(input.world_position, shade_normal, base_color, roughness, wetness);
#else
    float3 shaded = shade_surface(input.world_position, shade_normal, base_color, roughness, wetness);
#endif
    return float4(max(shaded, float3(0.0)), 1.0);
}
