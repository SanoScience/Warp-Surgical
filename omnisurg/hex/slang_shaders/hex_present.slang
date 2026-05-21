struct V2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

Texture2D<float4> source_tex;
SamplerState post_sampler;
uniform float2 output_size;

Texture2D<float4> environment_map;
SamplerState environment_sampler;
uniform float3 camera_right;
uniform float3 camera_up;
uniform float3 camera_forward;
uniform float camera_inv_tan_half_fovy;
uniform float camera_aspect;
uniform int has_environment_map;
uniform float environment_intensity;
uniform float environment_rotation;
uniform float environment_pitch;

float3 normalize_or(float3 value, float3 fallback)
{
    float value_len = length(value);
    return value_len > 0.000001 ? value / value_len : fallback;
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

[shader("vertex")]
V2F vertex_main(uint vertex_id : SV_VertexID)
{
    V2F output;
    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(output.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    return float4(saturate(source_tex.Sample(post_sampler, input.uv).rgb), 1.0);
}

[shader("vertex")]
V2F background_vertex_main(uint vertex_id : SV_VertexID)
{
    V2F output;
    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(output.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 background_fragment_main(V2F input) : SV_Target
{
    if (has_environment_map == 0)
    {
        return float4(0.0, 0.0, 0.0, 1.0);
    }
    float tan_half_fovy = 1.0 / max(camera_inv_tan_half_fovy, 0.000001);
    float2 ndc = float2(input.uv.x * 2.0 - 1.0, 1.0 - input.uv.y * 2.0);
    float3 direction = normalize_or(
        camera_forward
        + camera_right * (ndc.x * tan_half_fovy * max(camera_aspect, 0.0001))
        + camera_up * (ndc.y * tan_half_fovy),
        camera_forward
    );
    float3 color = max(environment_map.SampleLevel(environment_sampler, environment_equirect_uv(direction), 0.0).rgb, float3(0.0))
        * max(environment_intensity, 0.0);
    return float4(color, 1.0);
}
