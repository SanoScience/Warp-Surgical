struct V2F
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD;
};

Texture2D<float4> scene_color_tex;
Texture2D<float> depth_tex;
Texture2D<float4> bloom_tex;
Texture2D<float4> lens_dirt_tex;
SamplerState post_sampler;

uniform float2 output_size;
uniform int postprocess_enabled;
uniform float exposure;
uniform float3 white_balance;
uniform int bloom_enabled;
uniform float bloom_threshold;
uniform float bloom_intensity;
uniform float bloom_radius;
uniform int lens_dirt_enabled;
uniform float lens_dirt_intensity;
uniform float lens_dirt_threshold;
uniform float vignette_strength;
uniform float vignette_radius;
uniform float scope_radius;
uniform float scope_softness;

[shader("vertex")]
V2F vertex_main(uint vertex_id : SV_VertexID)
{
    V2F output;
    output.uv = float2((vertex_id << 1) & 2, vertex_id & 2);
    output.position = float4(output.uv * float2(2.0, -2.0) + float2(-1.0, 1.0), 0.0, 1.0);
    return output;
}

float3 aces_tonemap(float3 color)
{
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return saturate((color * (a * color + b)) / (color * (c * color + d) + e));
}

float radial_distance(float2 uv)
{
    float aspect = output_size.x / max(output_size.y, 1.0);
    float2 centered = uv * 2.0 - 1.0;
    centered.x *= aspect;
    return length(centered);
}

float endoscope_vignette(float radius)
{
    float start = max(vignette_radius, 0.0001);
    float falloff = saturate((radius - start) / max(1.0 - start, 0.0001));
    return lerp(1.0, 1.0 - vignette_strength, falloff * falloff);
}

float scope_mask(float radius)
{
    float softness = max(scope_softness, 0.0001);
    return 1.0 - smoothstep(scope_radius - softness, scope_radius, radius);
}

float3 apply_lens_dirt(float3 color, float2 uv, float3 bloom_drive)
{
    float3 dirt = lens_dirt_tex.Sample(post_sampler, uv).rgb;
    float bloom_luminance = dot(bloom_drive, float3(0.2126, 0.7152, 0.0722));
    float drive = saturate((bloom_luminance - lens_dirt_threshold) * 2.5);
    float3 warmed_dirt = dirt * float3(1.0, 0.92, 0.82);
    return color + warmed_dirt * drive * lens_dirt_intensity;
}

[shader("fragment")]
float4 fragment_main(V2F input) : SV_Target
{
    float depth = depth_tex.Sample(post_sampler, input.uv);
    float3 color = max(scene_color_tex.Sample(post_sampler, input.uv).rgb, float3(0.0));
    color += depth * 0.0;

    if (postprocess_enabled != 0)
    {
        color *= white_balance * exposure;
        bool bloom_active = bloom_enabled != 0 && bloom_intensity > 0.0 && bloom_radius > 0.0;
        bool dirt_active = lens_dirt_enabled != 0 && lens_dirt_intensity > 0.0;
        float3 bloom_signal = max(bloom_tex.Sample(post_sampler, input.uv).rgb, float3(0.0));
        if (!bloom_active)
            bloom_signal = float3(0.0);
        float3 bloom_drive = bloom_signal * bloom_intensity;
        if (bloom_active)
            color += bloom_drive;
        color = aces_tonemap(color);
        if (dirt_active)
            color = apply_lens_dirt(color, input.uv, bloom_drive);

        float radius = radial_distance(input.uv);
        color *= endoscope_vignette(radius);
        color *= scope_mask(radius);
    }
    else
    {
        color = saturate(color);
    }

    return float4(color, 1.0);
}
