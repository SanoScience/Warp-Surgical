struct V2F
{
    float4 position : SV_Position;
};

struct PointV2F
{
    float4 position : SV_Position;
};

struct PointSphereV2F
{
    float4 position : SV_Position;
    float3 normal : NORMAL;
    nointerpolation float3 color : COLOR0;
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
uniform int point_count;
uniform int point_has_radii;
uniform int point_has_colors;
uniform float point_default_radius;
uniform float3 point_default_color;

StructuredBuffer<float3> point_positions;
StructuredBuffer<float> point_radii;
StructuredBuffer<float3> point_colors;

float4 project_position(float3 position)
{
    float3 rel = position - camera_pos;
    float camera_x = dot(rel, camera_right);
    float camera_y = dot(rel, camera_up);
    float camera_z = max(dot(rel, camera_forward), camera_near);

    float clip_x = camera_x * camera_inv_tan_half_fovy / max(camera_aspect, 0.0001);
    float clip_y = camera_y * camera_inv_tan_half_fovy;
    float clip_z = camera_z * camera_far / (camera_far - camera_near)
        - camera_near * camera_far / (camera_far - camera_near);
    return float4(clip_x, clip_y, clip_z, camera_z);
}

[shader("vertex")]
V2F vertex_main(float3 position : POSITION)
{
    V2F output;
    output.position = project_position(position);
    return output;
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    return float4(mesh_color.rgb, mesh_color.a);
}

float3 normalize_or(float3 value, float3 fallback)
{
    float value_len = length(value);
    return value_len > 0.000001 ? value / value_len : fallback;
}

[shader("vertex")]
PointSphereV2F point_sphere_vertex_main(float3 unit_position : POSITION, uint instance_id : SV_InstanceID)
{
    PointSphereV2F output;
    int point_index = clamp(int(instance_id), 0, max(point_count - 1, 0));
    float radius = point_has_radii != 0 ? point_radii[point_index] : point_default_radius;
    radius = max(radius, 0.0);
    float3 center = point_positions[point_index];
    float3 normal = normalize_or(unit_position, float3(0.0, 0.0, 1.0));
    float3 world_position = center + normal * radius;
    output.position = project_position(world_position);
    output.normal = normal;
    output.color = point_has_colors != 0 ? point_colors[point_index] : point_default_color;
    return output;
}

[shader("fragment")]
float4 point_sphere_fragment_main(PointSphereV2F input) : SV_Target
{
    float3 normal = normalize_or(input.normal, float3(0.0, 0.0, 1.0));
    float3 light_dir = normalize_or(-camera_forward + camera_up * 0.45 + camera_right * 0.25, float3(0.0, 0.0, 1.0));
    float diffuse = saturate(dot(normal, light_dir));
    float shade = 0.35 + 0.65 * diffuse;
    return float4(saturate(input.color * shade), mesh_color.a);
}

[shader("vertex")]
PointV2F point_vertex_main(float3 position : POSITION)
{
    PointV2F output;
    output.position = project_position(position);
    return output;
}

[shader("fragment")]
float4 point_fragment_main(PointV2F input) : SV_Target
{
    return float4(mesh_color.rgb, mesh_color.a);
}
