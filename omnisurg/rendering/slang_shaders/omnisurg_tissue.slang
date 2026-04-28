struct V2F
{
    float4 position : SV_Position;
    float3 world_position : WORLD_POSITION;
    float3 world_normal : NORMAL;
    float2 uv : TEXCOORD;
    float4 vertex_color : COLOR;
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
uniform float3 light_key_dir;
uniform float3 light_fill_dir;
uniform float3 light_back_dir;
uniform float3 light_key_color;
uniform float3 light_fill_color;
uniform float3 light_ambient_color;
uniform float3 light_rim_color;

Texture2D<float4> diffuse_base_tex;
Texture2D<float4> diffuse_damage_tex;
Texture2D<float4> diffuse_coag_tex;
Texture2D<float4> diffuse_blood_tex;
Texture2D<float4> normal_base_tex;
Texture2D<float4> normal_damage_tex;
Texture2D<float4> normal_coag_tex;
Texture2D<float4> normal_blood_tex;
Texture2D<float4> spec_base_tex;
Texture2D<float4> spec_damage_tex;
Texture2D<float4> spec_coag_tex;
Texture2D<float4> spec_blood_tex;
Texture2D<float4> layer_mask_damage_tex;
Texture2D<float4> layer_mask_coag_tex;
Texture2D<float4> layer_mask_blood_tex;
Texture2D<float4> blood_mask_tex;
Texture2D<float4> heat_mask_tex;
SamplerState material_sampler;

uniform float normal_strength;
uniform float specular_scale;
uniform float roughness_bias;
uniform float ambient;
uniform float rim_strength;
uniform float blend_damage;
uniform float blend_coag;
uniform float blend_blood;
uniform int debug_mode;
uniform float wetness;
uniform float wet_spec_scale;
uniform float wet_roughness;
uniform float3 subsurface_color;
uniform float subsurface_strength;
uniform float blood_wetness;

static const int DEBUG_FINAL = 0;
static const int DEBUG_VERTEX_BLEND_RGB = 1;
static const int DEBUG_MASKED_LAYER_WEIGHTS = 2;
static const int DEBUG_BLENDED_DIFFUSE = 3;
static const int DEBUG_BLENDED_NORMAL = 4;
static const int DEBUG_SPEC_ROUGHNESS = 5;
static const int DEBUG_HEAT_BLOOD_MASKS = 6;
static const int DEBUG_RESERVED_MATERIAL_PARAMS = 99;

[shader("vertex")]
V2F vertex_main(float3 position : POSITION, float3 normal : NORMAL, float2 uv : TEXCOORD, float4 color : COLOR)
{
    V2F output;

    float3 rel = position - camera_pos;
    float camera_x = dot(rel, camera_right);
    float camera_y = dot(rel, camera_up);
    float camera_z = max(dot(rel, camera_forward), camera_near);

    float clip_x = camera_x * camera_inv_tan_half_fovy / max(camera_aspect, 0.0001);
    float clip_y = camera_y * camera_inv_tan_half_fovy;
    float clip_z = camera_z * camera_far / (camera_far - camera_near)
        - camera_near * camera_far / (camera_far - camera_near);

    output.position = float4(clip_x, clip_y, clip_z, camera_z);
    output.world_position = position;
    output.world_normal = normal;
    output.uv = uv;
    output.vertex_color = color;
    return output;
}

float3 fallback_tangent(float3 normal)
{
    float3 axis = abs(normal.y) < 0.95 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0);
    return normalize(cross(axis, normal));
}

void build_tangent_basis(
    float3 world_position,
    float3 smooth_normal,
    float2 uv,
    float3 view_dir,
    out float3 normal,
    out float3 tangent,
    out float3 bitangent)
{
    float3 dpdx = ddx(world_position);
    float3 dpdy = ddy(world_position);
    float2 duvdx = ddx(uv);
    float2 duvdy = ddy(uv);

    normal = smooth_normal;
    float normal_len = length(normal);
    normal = normal_len > 0.000001 ? normal / normal_len : cross(dpdx, dpdy);
    normal_len = length(normal);
    normal = normal_len > 0.000001 ? normal / normal_len : -camera_forward;
    normal = dot(normal, view_dir) < 0.0 ? -normal : normal;

    float det = duvdx.x * duvdy.y - duvdx.y * duvdy.x;
    if (abs(det) > 0.0000001)
    {
        float inv_det = 1.0 / det;
        tangent = (dpdx * duvdy.y - dpdy * duvdx.y) * inv_det;
        bitangent = (-dpdx * duvdy.x + dpdy * duvdx.x) * inv_det;
    }
    else
    {
        tangent = dpdx - normal * dot(normal, dpdx);
        bitangent = cross(normal, tangent);
    }

    tangent = tangent - normal * dot(normal, tangent);
    float tangent_len = length(tangent);
    tangent = tangent_len > 0.000001 ? tangent / tangent_len : fallback_tangent(normal);

    bitangent = bitangent - normal * dot(normal, bitangent) - tangent * dot(tangent, bitangent);
    float bitangent_len = length(bitangent);
    bitangent = bitangent_len > 0.000001 ? bitangent / bitangent_len : normalize(cross(normal, tangent));
}

float3 blend_rgb_layers(float2 uv, float damage_blend, float coag_blend, float blood_blend)
{
    float3 value = diffuse_base_tex.Sample(material_sampler, uv).rgb;
    value = lerp(value, diffuse_damage_tex.Sample(material_sampler, uv).rgb, damage_blend);
    value = lerp(value, diffuse_coag_tex.Sample(material_sampler, uv).rgb, coag_blend);
    value = lerp(value, diffuse_blood_tex.Sample(material_sampler, uv).rgb, blood_blend);
    return value;
}

float3 blend_normal_layers(float2 uv, float damage_blend, float coag_blend, float blood_blend)
{
    float3 value = normal_base_tex.Sample(material_sampler, uv).xyz * 2.0 - 1.0;
    float3 damage_normal = normal_damage_tex.Sample(material_sampler, uv).xyz * 2.0 - 1.0;
    float3 coag_normal = normal_coag_tex.Sample(material_sampler, uv).xyz * 2.0 - 1.0;
    float3 blood_normal = normal_blood_tex.Sample(material_sampler, uv).xyz * 2.0 - 1.0;
    value = lerp(value, damage_normal, damage_blend);
    value = lerp(value, coag_normal, coag_blend);
    value = lerp(value, blood_normal, blood_blend);
    return value;
}

float blend_spec_layers(float2 uv, float damage_blend, float coag_blend, float blood_blend)
{
    float value = spec_base_tex.Sample(material_sampler, uv).r;
    value = lerp(value, spec_damage_tex.Sample(material_sampler, uv).r, damage_blend);
    value = lerp(value, spec_coag_tex.Sample(material_sampler, uv).r, coag_blend);
    value = lerp(value, spec_blood_tex.Sample(material_sampler, uv).r, blood_blend);
    return value;
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    float3 view_dir = normalize(camera_pos - input.world_position);

    float3 geom_normal;
    float3 tangent;
    float3 bitangent;
    build_tangent_basis(
        input.world_position,
        input.world_normal,
        input.uv,
        view_dir,
        geom_normal,
        tangent,
        bitangent);

    float heat_mask = heat_mask_tex.Sample(material_sampler, input.uv).r;
    float blood_mask = blood_mask_tex.Sample(material_sampler, input.uv).r;
    float damage_layer_mask = max(layer_mask_damage_tex.Sample(material_sampler, input.uv).r, heat_mask);
    float coag_layer_mask = max(layer_mask_coag_tex.Sample(material_sampler, input.uv).r, heat_mask);
    float blood_layer_mask = max(layer_mask_blood_tex.Sample(material_sampler, input.uv).r, blood_mask);
    float damage_blend = saturate(input.vertex_color.r * blend_damage * damage_layer_mask);
    float coag_blend = saturate(input.vertex_color.g * blend_coag * coag_layer_mask);
    float blood_blend = saturate(input.vertex_color.b * blend_blood * blood_layer_mask);

    float3 sampled_normal = blend_normal_layers(input.uv, damage_blend, coag_blend, blood_blend);
    sampled_normal.xy *= normal_strength;
    sampled_normal = normalize(sampled_normal);
    float3 normal = normalize(
        tangent * sampled_normal.x
        + bitangent * sampled_normal.y
        + geom_normal * sampled_normal.z);
    normal = dot(normal, view_dir) < 0.0 ? -normal : normal;

    float3 blended_diffuse = blend_rgb_layers(input.uv, damage_blend, coag_blend, blood_blend);
    float3 base_color = blended_diffuse * mesh_color.rgb;
    float spec_mask = blend_spec_layers(input.uv, damage_blend, coag_blend, blood_blend);

    float3 key_light = normalize(light_key_dir);
    float3 fill_light = normalize(light_fill_dir);
    float3 back_light = normalize(light_back_dir);
    float key_diffuse = saturate(dot(normal, key_light));
    float fill_diffuse = saturate(dot(normal, fill_light));
    float backscatter = pow(saturate(dot(-normal, back_light)), 2.0);

    float3 half_dir = normalize(key_light + view_dir);
    float roughness = saturate(roughness_bias + (1.0 - spec_mask) * 0.35);
    float shininess = lerp(96.0, 18.0, roughness);
    float wet_spec = pow(saturate(dot(normal, half_dir)), shininess)
        * specular_scale
        * (0.25 + spec_mask);

    if (debug_mode == DEBUG_VERTEX_BLEND_RGB)
        return float4(saturate(input.vertex_color.rgb), mesh_color.a);
    if (debug_mode == DEBUG_MASKED_LAYER_WEIGHTS)
        return float4(saturate(float3(damage_blend, coag_blend, blood_blend)), mesh_color.a);
    if (debug_mode == DEBUG_BLENDED_DIFFUSE)
        return float4(saturate(blended_diffuse), mesh_color.a);
    if (debug_mode == DEBUG_BLENDED_NORMAL)
        return float4(saturate(normal * 0.5 + 0.5), mesh_color.a);
    if (debug_mode == DEBUG_SPEC_ROUGHNESS)
        return float4(saturate(float3(spec_mask, roughness, 0.0)), mesh_color.a);
    if (debug_mode == DEBUG_HEAT_BLOOD_MASKS)
        return float4(saturate(float3(heat_mask, blood_mask, blood_layer_mask)), mesh_color.a);
    if (debug_mode == DEBUG_RESERVED_MATERIAL_PARAMS)
        return float4(
            saturate(
                float3(
                    wetness,
                    wet_spec_scale * wet_roughness,
                    blood_wetness)
                + subsurface_color * subsurface_strength),
            mesh_color.a);

    float ndv = saturate(dot(normal, view_dir));
    float rim = pow(1.0 - ndv, 2.0) * rim_strength;

    float3 diffuse_light =
        light_ambient_color * ambient
        + light_key_color * key_diffuse * 0.88
        + light_fill_color * fill_diffuse * 0.42
        + light_rim_color * backscatter * 0.16;
    float3 color =
        base_color * diffuse_light
        + light_key_color * wet_spec
        + base_color * light_rim_color * rim;
    return float4(saturate(color), mesh_color.a);
}
