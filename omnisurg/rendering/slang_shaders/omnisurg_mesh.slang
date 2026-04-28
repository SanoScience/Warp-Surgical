struct V2F
{
    float4 position : SV_Position;
    float3 world_position : WORLD_POSITION;
    float3 world_normal : NORMAL;
};

struct PointV2F
{
    float4 position : SV_Position;
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
uniform float specular_scale;
uniform float roughness_bias;
uniform float ambient;
uniform float rim_strength;

[shader("vertex")]
V2F vertex_main(float3 position : POSITION, float3 normal : NORMAL)
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
    return output;
}

float3 shade_surface(float3 world_position, float3 world_normal, float3 base_color)
{
    float3 view_dir = normalize(camera_pos - world_position);

    float3 normal = world_normal;
    float normal_len = length(normal);
    normal = normal_len > 0.000001 ? normal / normal_len : view_dir;
    normal = dot(normal, view_dir) < 0.0 ? -normal : normal;

    float key_diffuse = saturate(dot(normal, normalize(light_key_dir)));
    float fill_diffuse = saturate(dot(normal, normalize(light_fill_dir)));
    float back_wrap = pow(saturate(dot(-normal, normalize(light_back_dir))), 2.0);

    float3 half_dir = normalize(normalize(light_key_dir) + view_dir);
    float roughness = saturate(roughness_bias);
    float shininess = lerp(96.0, 22.0, roughness);
    float spec = pow(saturate(dot(normal, half_dir)), shininess) * specular_scale * 0.55;

    float ndv = saturate(dot(normal, view_dir));
    float rim = pow(1.0 - ndv, 2.0) * rim_strength;

    float3 diffuse_light =
        light_ambient_color * ambient
        + light_key_color * key_diffuse * 0.95
        + light_fill_color * fill_diffuse * 0.48
        + light_rim_color * back_wrap * 0.20;
    return base_color * diffuse_light
        + light_key_color * spec
        + base_color * light_rim_color * rim;
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    float3 color = shade_surface(input.world_position, input.world_normal, mesh_color.rgb);
    return float4(saturate(color), mesh_color.a);
}

[shader("vertex")]
PointV2F point_vertex_main(float3 position : POSITION)
{
    PointV2F output;

    float3 rel = position - camera_pos;
    float camera_x = dot(rel, camera_right);
    float camera_y = dot(rel, camera_up);
    float camera_z = max(dot(rel, camera_forward), camera_near);

    float clip_x = camera_x * camera_inv_tan_half_fovy / max(camera_aspect, 0.0001);
    float clip_y = camera_y * camera_inv_tan_half_fovy;
    float clip_z = camera_z * camera_far / (camera_far - camera_near)
        - camera_near * camera_far / (camera_far - camera_near);

    output.position = float4(clip_x, clip_y, clip_z, camera_z);
    return output;
}

[shader("fragment")]
float4 point_fragment_main(PointV2F input) : SV_Target
{
    return float4(mesh_color.rgb, mesh_color.a);
}
