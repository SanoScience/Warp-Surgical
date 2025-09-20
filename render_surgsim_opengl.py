# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import ctypes
import sys
import time
from collections import defaultdict
from typing import List, Union

import numpy as np
import warp as wp
from warp.render import OpenGLRenderer, UsdRenderer
#from newton._src.viewer.gl.opengl import RendererGL as OpenGLRenderer
from warp.render.utils import solidify_mesh, tab10_color_map
from pyglet.graphics.shader import Shader, ShaderProgram

import newton

from render_opengl import str_buffer

shape_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec3 aNormal;
layout (location = 2) in vec3 aTangent;
layout (location = 3) in vec2 aTexCoord;
layout (location = 4) in vec4 aVertexColor;

// column vectors of the instance transform matrix
layout (location = 5) in vec4 aInstanceTransform0;
layout (location = 6) in vec4 aInstanceTransform1;
layout (location = 7) in vec4 aInstanceTransform2;
layout (location = 8) in vec4 aInstanceTransform3;

uniform mat4 view;
uniform mat4 model;
uniform mat4 projection;

out vec3 Normal;
out vec3 Tangent;
out vec3 FragPos;
out vec2 TexCoord;
out vec4 VertexColor;

void main()
{
    mat4 transform = model * mat4(aInstanceTransform0, aInstanceTransform1, aInstanceTransform2, aInstanceTransform3);
    vec4 worldPos = transform * vec4(aPos, 1.0);
    gl_Position = projection * view * worldPos;
    FragPos = vec3(worldPos);
    Normal = mat3(transpose(inverse(transform))) * aNormal;
    Tangent = mat3(transpose(inverse(transform))) * aTangent;
    TexCoord = aTexCoord;
    VertexColor = aVertexColor;
}
"""

shape_fragment_shader = """
#version 330 core
out vec4 FragColor;

in vec3 Normal;
in vec3 Tangent;
in vec3 FragPos;
in vec2 TexCoord;
in vec4 VertexColor;

uniform vec3 viewPos;
uniform vec3 lightColor;
uniform vec3 sunDirection;
uniform int numTextures;
uniform vec3 startColor;

uniform sampler2D diffuseMaps[4];
uniform sampler2D normalMaps[4];
uniform sampler2D specularMaps[4];

void main()
{
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * lightColor;

    // Construct TBN matrix
    vec3 norm = normalize(Normal);
    vec3 tangent = normalize(Tangent);
    vec3 bitangent = normalize(cross(norm, tangent));
    mat3 TBN = mat3(tangent, bitangent, norm);

    vec4 vCol = VertexColor;
    float burnBlend = vCol.y;
    float stretchBlend = vCol.z;
    float bloodBlend = vCol.w;
    float mainBlend   = max(0.0, 1.0 - (burnBlend + stretchBlend + bloodBlend));

    // Normalize if sum > 1
    float total = burnBlend + stretchBlend + bloodBlend + mainBlend;
    if (total > 1.0)
    {
        burnBlend    /= total;
        stretchBlend /= total;
        bloodBlend   /= total;
        mainBlend    /= total;
    }
    

    // Sample normal map and transform to world space
    vec3 n = norm;
    if (numTextures > 0)
    {
        vec3 baseNorm =      texture(normalMaps[0], TexCoord).rgb;
        vec3 burnNorm =      texture(normalMaps[1], TexCoord).rgb;
        vec3 stretchNorm =   texture(normalMaps[2], TexCoord).rgb;
        vec3 bloodNorm =     texture(normalMaps[3], TexCoord).rgb;

        vec3 sampledNormal = mainBlend * baseNorm +
            burnBlend    * burnNorm +
            stretchBlend * stretchNorm +
            bloodBlend   * bloodNorm;

        sampledNormal = normalize(sampledNormal * 2.0 - 1.0); // [-1,1] range
        n = normalize(TBN * sampledNormal);
    }

    float diff = max(dot(n, sunDirection), 0.0);
    vec3 diffuse = diff * lightColor;

    vec3 lightDir2 = normalize(vec3(1.0, 0.3, -0.3));
    diff = max(dot(n, lightDir2), 0.0);
    diffuse += diff * lightColor * 0.3;

    float specularStrength = 1.0;
    if (numTextures > 0)
    {
        float baseSpec =      texture(specularMaps[0], TexCoord).r;
        float burnSpec =      texture(specularMaps[1], TexCoord).r * 0.5;
        float stretchSpec =   texture(specularMaps[2], TexCoord).r;
        float bloodSpec =     texture(specularMaps[3], TexCoord).r * 2;


        float specVal = mainBlend * baseSpec +
            burnBlend    * burnSpec +
            stretchBlend * stretchSpec +
            bloodBlend   * bloodSpec;
        specularStrength *= specVal;
    }

    vec3 viewDir = normalize(viewPos - FragPos);

    vec3 reflectDir = reflect(-sunDirection, n);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular = specularStrength * spec * lightColor;

    reflectDir = reflect(-lightDir2, n);
    spec = pow(max(dot(viewDir, reflectDir), 0.0), 64);
    specular += specularStrength * spec * lightColor * 0.3;

    vec3 baseColor = startColor;
    if (numTextures > 0)
    {
        vec3 baseCol =      texture(diffuseMaps[0], TexCoord).rgb;
        vec3 burnCol =      texture(diffuseMaps[1], TexCoord).rgb;
        vec3 stretchCol =   texture(diffuseMaps[2], TexCoord).rgb;
        vec3 bloodCol =     texture(diffuseMaps[3], TexCoord).rgb;


        baseColor = mainBlend * baseCol +
            burnBlend    * burnCol +
            stretchBlend * stretchCol +
            bloodBlend   * bloodCol;
    }

    // Use per-vertex color when no textures are bound
    if (numTextures == 0)
    {
        baseColor = VertexColor.rgb;
    }

    // Environment hack
    if (numTextures == 1)
        baseColor *= 0.5;

    vec3 result = (ambient + diffuse + specular) * baseColor; // Multiply lighting by chosen base color
    FragColor = vec4(result, 1.0);
}
"""

post_vertex_shader = """
#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;

out vec2 TexCoord;

void main() {
    gl_Position = vec4(aPos, 1.0);
    TexCoord = aTexCoord;
}
"""

post_fragment_shader = """
#version 330 core
in vec2 TexCoord;
out vec4 FragColor;

uniform sampler2D colorTexture;
uniform sampler2D depthTexture;
uniform mat4 invProjection;
uniform vec2 screenSize;

vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x*(a*x+b))/(x*(c*x+d)+e), 0.0, 1.0);
}

vec3 agx_tonemap(vec3 x) {
    const float agx_a = 0.8;
    const float agx_b = 0.15;
    const float agx_c = 0.5;
    x = max(x, 0.0);
    vec3 t = pow(x, vec3(agx_a));
    t = t / (t + vec3(agx_b));
    t = pow(t, vec3(agx_c));
    return clamp(t, 0.0, 1.0);
}

float sample_depth(vec2 uv)
{
    return texture(depthTexture, uv).r;
}

float linearize_depth(float depth) {
    float near = 0.05;
    float far = 25;
    
    float z = depth * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

vec3 getViewPosition(vec2 uv, float depth) {
    float z = depth * 2.0 - 1.0;
    vec4 clipPos = vec4(uv * 2.0 - 1.0, z, 1.0);
    vec4 viewPos = invProjection * clipPos;
    return viewPos.xyz / viewPos.w;
}

vec3 reconstructNormal(vec2 uv, float depth) {
    float dx = 1.0 / screenSize.x;
    float dy = 1.0 / screenSize.y;

    // Sample neighboring depths
    float depthC = depth;
    float depthR = sample_depth(uv + vec2(dx, 0.0));
    float depthU = sample_depth(uv + vec2(0.0, dy));

    // Reconstruct view-space positions
    vec3 pC = getViewPosition(uv, depthC);
    vec3 pR = getViewPosition(uv + vec2(dx, 0.0), depthR);
    vec3 pU = getViewPosition(uv + vec2(0.0, dy), depthU);

    // Compute normal from cross product of tangent vectors
    vec3 dX = pR - pC;
    vec3 dY = pU - pC;
    vec3 normal = normalize(cross(dX, dY));
    return normal;
}

float computeSSAO1(vec2 uv, float depth) {
    float radius = 4.0; // pixel radius
    float occlusion = 0.0;
    int samples = 8;
    float total = 0.0;

    for (int i = 0; i < samples; ++i) {
        float angle = 6.2831853 * float(i) / float(samples);
        vec2 offset = vec2(cos(angle), sin(angle)) * radius / screenSize;
        float sampleDepth = sample_depth(uv + offset);
        float rangeCheck = smoothstep(0.0, 1.0, radius / abs(depth - sampleDepth + 0.0001));
        if (sampleDepth < depth)
            occlusion += rangeCheck;
        total += 1.0;
    }
    return 1.0 - (occlusion / total) * 0.5; // 0.5 = strength
}

float computeSSAO2(vec2 uv, float depth) {
    float radius = 8.0; // pixel radius
    int samples = 8;
    float occlusion = 0.0;
    float total = 0.0;

    vec3 normal = reconstructNormal(uv, depth);

    for (int i = 0; i < samples; ++i) {
        float angle = 6.2831853 * float(i) / float(samples);
        vec2 offset = vec2(cos(angle), sin(angle)) * radius / screenSize;
        float sampleDepth = sample_depth(uv + offset);
        vec3 sampleNormal = reconstructNormal(uv + offset, sampleDepth);

        // Reconstruct view-space positions
        vec3 p = getViewPosition(uv, depth);
        vec3 pSample = getViewPosition(uv + offset, sampleDepth);

        float dist = length(pSample - p);
        float rangeCheck = smoothstep(0.0, radius * 0.02, dist);

        // Angle between normals (optional, for smoother occlusion)
        float normalWeight = max(dot(normal, sampleNormal), 0.0);

        if (sampleDepth < depth && normalWeight > 0.5)
            occlusion += rangeCheck * normalWeight;
        total += 1.0;
    }
    return 1.0 - (occlusion / total) * 0.5; // 0.5 = strength
}

float random(vec2 uv) {
    // Simple hash based on UV
    return fract(sin(dot(uv, vec2(12.9898,78.233))) * 43758.5453);
}

float computeSSAO3(vec2 uv, float depth) {
    float base_radius = 32.0;
    float radius = base_radius * depth;
    int samples = 12;
    float occlusion = 0.0;
    float total = 0.0;
    vec3 normal = reconstructNormal(uv, depth);

    // Random rotation per pixel
    float rand = random(uv);
    float angle_offset = rand * 6.2831853; // [0, 2pi]

    for (int i = 0; i < samples; ++i) {
        float phi = float(i) * 2.399963229728653 + angle_offset; // golden angle + random offset
        float r = radius * sqrt(float(i) / float(samples));
        vec2 offset = vec2(cos(phi), sin(phi)) * r / screenSize;
        float sampleDepth = sample_depth(uv + offset);
        vec3 sampleNormal = reconstructNormal(uv + offset, sampleDepth);
        vec3 p = getViewPosition(uv, depth);
        vec3 pSample = getViewPosition(uv + offset, sampleDepth);
        float dist = length(pSample - p);
        float rangeCheck = smoothstep(0.0, radius * 0.02, dist);
        float normalWeight = max(dot(normal, sampleNormal), 0.0);
        if (sampleDepth < depth && normalWeight > 0.5)
            occlusion += rangeCheck * normalWeight;
        total += 1.0;
    }
    return 1.0 - (occlusion / total);
}

void main() {
    vec3 color = texture(colorTexture, TexCoord).rgb;
    float depth = sample_depth(TexCoord);

    float ao = 1.0;
    if(depth < 0.999)
    {
        ao = computeSSAO3(TexCoord, depth);
        float contrast = 2.5;
        ao = pow(ao, contrast);

        color *= ao;
    }


    color = aces_tonemap(color);


    //color = pow(color, vec3(1.0/2.2));
    FragColor = vec4(color, 1.0);
}
"""

@wp.kernel
def assemble_gfx_vertices_with_colors(
    vertices: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    tangents: wp.array(dtype=wp.vec3),
    faces_per_vertex: wp.array(dtype=wp.int32),
    texture_coords: wp.array(dtype=wp.vec2),
    vertex_colors: wp.array(dtype=wp.vec4),
    scale: wp.vec3,
    gfx_vertices: wp.array(dtype=wp.float32, ndim=2)
):
    tid = wp.tid()
    if tid >= len(vertices):
        return
        
    pos = vertices[tid]
    normal = normals[tid]
    tangent = tangents[tid]
    
    # Normalize normal and tangent
    if faces_per_vertex[tid] > 0:
        normal = normal / wp.float32(faces_per_vertex[tid])
        normal = wp.normalize(normal)
        tangent = tangent / wp.float32(faces_per_vertex[tid])
        tangent = wp.normalize(tangent)
    
    uv = texture_coords[tid]
    color = vertex_colors[tid]
    
    # Pack vertex data: position (3) + normal (3) + tangent (3) + uv (2) + color (4)
    gfx_vertices[tid, 0] = pos[0] * scale[0]
    gfx_vertices[tid, 1] = pos[1] * scale[1]
    gfx_vertices[tid, 2] = pos[2] * scale[2]
    gfx_vertices[tid, 3] = normal[0]
    gfx_vertices[tid, 4] = normal[1]
    gfx_vertices[tid, 5] = normal[2]
    gfx_vertices[tid, 6] = tangent[0]
    gfx_vertices[tid, 7] = tangent[1]
    gfx_vertices[tid, 8] = tangent[2]
    gfx_vertices[tid, 9] = uv[0]
    gfx_vertices[tid, 10] = uv[1]
    gfx_vertices[tid, 11] = color[0]
    gfx_vertices[tid, 12] = color[1]
    gfx_vertices[tid, 13] = color[2]
    gfx_vertices[tid, 14] = color[3]

@wp.kernel
def compute_gfx_vertices_with_colors(
    indices: wp.array(dtype=wp.int32, ndim=2),
    vertices: wp.array(dtype=wp.vec3),
    texture_coords: wp.array(dtype=wp.vec2),
    vertex_colors: wp.array(dtype=wp.vec4),
    scale: wp.vec3,
    gfx_vertices: wp.array(dtype=wp.float32, ndim=2)
):
    tid = wp.tid()
    if tid >= len(indices):
        return
    
    # Get triangle indices
    i0 = indices[tid, 0]
    i1 = indices[tid, 1] 
    i2 = indices[tid, 2]
    
    # Get positions
    p0 = vertices[i0] * scale[0]
    p1 = vertices[i1] * scale[1]
    p2 = vertices[i2] * scale[2]
    
    # Compute face normal
    edge1 = p1 - p0
    edge2 = p2 - p0
    normal = wp.normalize(wp.cross(edge1, edge2))
    
    # Get texture coordinates and vertex colors
    uv0 = texture_coords[i0]
    uv1 = texture_coords[i1]
    uv2 = texture_coords[i2]
    
    # Tangent
    deltaUV1 = uv1 - uv0
    deltaUV2 = uv2 - uv0
    f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1] + 1e-8)
    tangent = f * (deltaUV2[1] * edge1 - deltaUV1[1] * edge2)
    tangent = wp.normalize(tangent)
    
    color0 = vertex_colors[i0]
    color1 = vertex_colors[i1]
    color2 = vertex_colors[i2]
    
    # Store vertices for this triangle
    base_idx = tid * 3
    
    # Vertex 0
    gfx_vertices[base_idx + 0, 0] = p0[0]
    gfx_vertices[base_idx + 0, 1] = p0[1]
    gfx_vertices[base_idx + 0, 2] = p0[2]
    gfx_vertices[base_idx + 0, 3] = normal[0]
    gfx_vertices[base_idx + 0, 4] = normal[1]
    gfx_vertices[base_idx + 0, 5] = normal[2]
    gfx_vertices[base_idx + 0, 6] = tangent[0]
    gfx_vertices[base_idx + 0, 7] = tangent[1]
    gfx_vertices[base_idx + 0, 8] = tangent[2]
    gfx_vertices[base_idx + 0, 9] = uv0[0]
    gfx_vertices[base_idx + 0, 10] = uv0[1]
    gfx_vertices[base_idx + 0, 11] = color0[0]
    gfx_vertices[base_idx + 0, 12] = color0[1]
    gfx_vertices[base_idx + 0, 13] = color0[2]
    gfx_vertices[base_idx + 0, 14] = color0[3]

    # Vertex 1
    gfx_vertices[base_idx + 1, 0] = p1[0]
    gfx_vertices[base_idx + 1, 1] = p1[1]
    gfx_vertices[base_idx + 1, 2] = p1[2]
    gfx_vertices[base_idx + 1, 3] = normal[0]
    gfx_vertices[base_idx + 1, 4] = normal[1]
    gfx_vertices[base_idx + 1, 5] = normal[2]
    gfx_vertices[base_idx + 1, 6] = tangent[0]
    gfx_vertices[base_idx + 1, 7] = tangent[1]
    gfx_vertices[base_idx + 1, 8] = tangent[2]
    gfx_vertices[base_idx + 1, 9] = uv1[0]
    gfx_vertices[base_idx + 1, 10] = uv1[1]
    gfx_vertices[base_idx + 1, 11] = color1[0]
    gfx_vertices[base_idx + 1, 12] = color1[1]
    gfx_vertices[base_idx + 1, 13] = color1[2]
    gfx_vertices[base_idx + 1, 14] = color1[3]

    # Vertex 2
    gfx_vertices[base_idx + 2, 0] = p2[0]
    gfx_vertices[base_idx + 2, 1] = p2[1]
    gfx_vertices[base_idx + 2, 2] = p2[2]
    gfx_vertices[base_idx + 2, 3] = normal[0]
    gfx_vertices[base_idx + 2, 4] = normal[1]
    gfx_vertices[base_idx + 2, 5] = normal[2]
    gfx_vertices[base_idx + 2, 6] = tangent[0]
    gfx_vertices[base_idx + 2, 7] = tangent[1]
    gfx_vertices[base_idx + 2, 8] = tangent[2]
    gfx_vertices[base_idx + 2, 9] = uv2[0]
    gfx_vertices[base_idx + 2, 10] = uv2[1]
    gfx_vertices[base_idx + 2, 11] = color2[0]
    gfx_vertices[base_idx + 2, 12] = color2[1]
    gfx_vertices[base_idx + 2, 13] = color2[2]
    gfx_vertices[base_idx + 2, 14] = color2[3]


@wp.kernel
def update_mesh_vertices_optimized_with_colors(
    points: wp.array(dtype=wp.vec3),
    texture_coords: wp.array(dtype=wp.vec2),
    vertex_colors: wp.array(dtype=wp.vec4),
    scale: wp.vec3,
    gfx_vertices: wp.array(dtype=wp.float32, ndim=2)
):
    tid = wp.tid()
    if tid >= len(points):
        return
        
    pos = points[tid]
    uv = texture_coords[tid]
    color = vertex_colors[tid]
    
    # Update position, texture coordinates, and vertex colors (keep existing normals + tangents)
    gfx_vertices[tid, 0] = pos[0] * scale[0]
    gfx_vertices[tid, 1] = pos[1] * scale[1]
    gfx_vertices[tid, 2] = pos[2] * scale[2]
    gfx_vertices[tid, 9] = uv[0]
    gfx_vertices[tid, 10] = uv[1]
    gfx_vertices[tid, 11] = color[0]
    gfx_vertices[tid, 12] = color[1]
    gfx_vertices[tid, 13] = color[2]
    gfx_vertices[tid, 14] = color[3]



def check_gl_error():
    from pyglet import gl

    error = gl.glGetError()
    if error != gl.GL_NO_ERROR:
        print(f"OpenGL error: {error}")



@wp.kernel
def compute_average_normals(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3),
    scale: wp.vec3,
    # outputs
    normals: wp.array(dtype=wp.vec3),
    faces_per_vertex: wp.array(dtype=int),
):
    tid = wp.tid()
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    v0 = vertices[i] * scale[0]
    v1 = vertices[j] * scale[1]
    v2 = vertices[k] * scale[2]
    n = wp.normalize(wp.cross(v1 - v0, v2 - v0))
    wp.atomic_add(normals, i, n)
    wp.atomic_add(faces_per_vertex, i, 1)
    wp.atomic_add(normals, j, n)
    wp.atomic_add(faces_per_vertex, j, 1)
    wp.atomic_add(normals, k, n)
    wp.atomic_add(faces_per_vertex, k, 1)

@wp.kernel
def compute_average_tangents(
    indices: wp.array(dtype=int, ndim=2),
    vertices: wp.array(dtype=wp.vec3),
    texture_coords: wp.array(dtype=wp.vec2),
    scale: wp.vec3,
    # outputs
    tangents: wp.array(dtype=wp.vec3),
    faces_per_vertex: wp.array(dtype=int),
):
    tid = wp.tid()
    i = indices[tid, 0]
    j = indices[tid, 1]
    k = indices[tid, 2]
    v0 = vertices[i] * scale[0]
    v1 = vertices[j] * scale[1]
    v2 = vertices[k] * scale[2]
    uv0 = texture_coords[i]
    uv1 = texture_coords[j]
    uv2 = texture_coords[k]

    edge1 = v1 - v0
    edge2 = v2 - v0
    deltaUV1 = uv1 - uv0
    deltaUV2 = uv2 - uv0

    f = 1.0 / (deltaUV1[0] * deltaUV2[1] - deltaUV2[0] * deltaUV1[1] + 1e-8)
    tangent = f * (deltaUV2[1] * edge1 - deltaUV1[1] * edge2)

    tangent = wp.normalize(tangent)
    wp.atomic_add(tangents, i, tangent)
    wp.atomic_add(faces_per_vertex, i, 1)
    wp.atomic_add(tangents, j, tangent)
    wp.atomic_add(faces_per_vertex, j, 1)
    wp.atomic_add(tangents, k, tangent)
    wp.atomic_add(faces_per_vertex, k, 1)


def CreateSurgSimRenderer(renderer):
    class SimRenderer(renderer):
        use_unique_colors = True

        def __init__(
            self,
            model: newton.Model,
            path: str,
            scaling: float = 1.0,
            fps: int = 60,
            up_axis: newton.AxisType | None = None,
            show_joints: bool = False,
            show_particles: bool = True,
            **render_kwargs,
        ):
            if up_axis is None:
                up_axis = model.up_axis
            up_axis = newton.Axis.from_any(up_axis)    

            super().__init__(path, scaling=scaling, fps=fps, up_axis=str(up_axis), **render_kwargs)
            self._replace_shape_fragment_shader()
            self.scaling = scaling
            self.cam_axis = up_axis.value
            self.show_joints = show_joints
            self.show_particles = show_particles
            self._instance_key_count = {}
            self.populate(model)

            self._contact_points0 = None
            self._contact_points1 = None

            gl = self.gl
            # Postprocess pipeline (can be disabled via env)
            self.enable_postprocess = True
            try:
                import os
                if os.environ.get('WARP_DISABLE_POSTPROCESS', '0') in ('1', 'true', 'True'):
                    self.enable_postprocess = False
            except Exception:
                pass

            self._postprocess_shader = ShaderProgram(
                Shader(post_vertex_shader, "vertex"),
                Shader(post_fragment_shader, "fragment")
            )
            self._loc_postprocess_color = gl.glGetUniformLocation(self._postprocess_shader.id, str_buffer("colorTexture"))
            self._loc_postprocess_depth = gl.glGetUniformLocation(self._postprocess_shader.id, str_buffer("depthTexture"))
            self._loc_postprocess_invproj = gl.glGetUniformLocation(self._postprocess_shader.id, str_buffer("invProjection"))
            self._loc_postprocess_screensize = gl.glGetUniformLocation(self._postprocess_shader.id, str_buffer("screenSize"))

        def set_input_callbacks(self, on_key_press=None, on_key_release=None):
            """Set callback functions for input events."""
            self.on_key_press_callback = on_key_press
            self.on_key_release_callback = on_key_release
            # If window exists, set pyglet event handlers
            if hasattr(self, "window"):
                self.window.on_key_press = self._on_key_press
                self.window.on_key_release = self._on_key_release

        def _on_key_press(self, symbol, modifiers):
            if self.on_key_press_callback:
                self.on_key_press_callback(symbol, modifiers)

        def _on_key_release(self, symbol, modifiers):
            if self.on_key_release_callback:
                self.on_key_release_callback(symbol, modifiers)

        def load_texture(self, filepath: str, **kwargs) -> int:
            """Load a texture from file"""
            if not hasattr(self, '_texture_manager'):
                from texture_loader import OpenGLTextureManager
                self._texture_manager = OpenGLTextureManager(self)
            return self._texture_manager.load_texture(filepath, **kwargs)

        def _replace_shape_fragment_shader(self):
            # Access the OpenGL context and shader utilities
            gl = self.gl
            from pyglet.graphics.shader import Shader, ShaderProgram

            # Create new Shader objects
            vert_shader = Shader(shape_vertex_shader, 'vertex')
            frag_shader = Shader(shape_fragment_shader, 'fragment')

            # Create a new ShaderProgram
            new_program = ShaderProgram(vert_shader, frag_shader)

            # Replace the renderer's shape shader program
            self._shape_shader = new_program

            # Reinitialize uniforms
            with self._shape_shader:
                gl.glUniform3f(
                gl.glGetUniformLocation(self._shape_shader.id, str_buffer("sunDirection")), *self._sun_direction)
                gl.glUniform3f(gl.glGetUniformLocation(self._shape_shader.id, str_buffer("lightColor")), 1, 1, 1)
                gl.glUniform1i(gl.glGetUniformLocation(self._shape_shader.id, str_buffer("textureSampler")), 0)
                gl.glUniform1i(gl.glGetUniformLocation(self._shape_shader.id, str_buffer("hasTexture")), 0)  # Default to no texture
                self._loc_shape_model = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("model"))
                self._loc_shape_view = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("view"))
                self._loc_shape_projection = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("projection"))
                self._loc_shape_view_pos = gl.glGetUniformLocation(self._shape_shader.id, str_buffer("viewPos"))
                gl.glUniform3f(self._loc_shape_view_pos, 0, 0, 10)

        def render_mesh_warp(
            self,
            name: str,
            points: wp.array,
            indices: wp.array,
            texture_coords: wp.array = None,
            vertex_colors: wp.array = None,
            diffuse_maps: list[int] = None,
            normal_maps: list[int] = None,
            specular_maps: list[int] = None,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
            basic_color = (1.0, 1.0, 1.0),
            update_topology=False,
            parent_body: str | None = None,
            is_template: bool = False,
            smooth_shading: bool = True,
            visible: bool = True,
        ):
            """Add a mesh for visualization using Warp arrays directly

            Args:
                name: A name for the mesh instance
                points: Warp array of mesh vertices (dtype=wp.vec3)
                indices: Warp array of mesh face indices (dtype=int, shape=(-1, 3) or flat)
                texture_coords: Warp array of texture coordinates (dtype=wp.vec2 or 1D array with shape=(vertex_count*2,), optional)
                vertex_colors: Warp array of per-vertex colors (dtype=wp.vec4, optional)
                textures: List of OpenGL texture IDs (optional)
                pos: The position of the mesh
                rot: The rotation of the mesh
                scale: The scale of the mesh
                update_topology: Whether the mesh topology may have changed
                parent_body: Parent body name
                is_template: Whether this is a template shape
                smooth_shading: Whether to average face normals at each vertex
                visible: Whether the shape is visible
            """
            
            # Ensure arrays are on the correct device
            if points.device != self._device:
                points = points.to(self._device)
            if indices.device != self._device:
                indices = indices.to(self._device)
            if vertex_colors is not None and vertex_colors.device != self._device:
                vertex_colors = vertex_colors.to(self._device)
            if texture_coords is not None and texture_coords.device != self._device:
                texture_coords = texture_coords.to(self._device)

            point_count = points.shape[0]
            
            # Create default vertex colors if not provided
            if vertex_colors is None:
                vertex_colors = wp.zeros(point_count, dtype=wp.vec4, device=self._device)
            
            # Create default texture coordinates if not provided
            if texture_coords is None:
                texture_coords = wp.zeros(point_count, dtype=wp.vec2, device=self._device)

            texture_coords_2d = texture_coords

            indices_reshaped = indices

            if len(indices.shape) == 1:
                # Flat array, reshape to (-1, 3)
                assert indices.shape[0] % 3 == 0, "Flat indices array must be divisible by 3. Actual count: " + str(indices.shape[0])
                indices_reshaped = indices.reshape((indices.shape[0] // 3, 3))

            idx_count = indices_reshaped.shape[0]

            # Use a simplified hash based on array metadata instead of data content
            geo_hash = hash(name)

            if name in self._instances:
                shape_id = self._instances[name][2]
            else:
                shape_id = None

            textures = [diffuse_maps, normal_maps, specular_maps]

            # Fast path: update existing shape without topology changes
            if not update_topology and shape_id is not None:
                self._update_shape_vertices_gpu_direct_with_colors(shape_id, points, texture_coords_2d, vertex_colors, scale)
                if diffuse_maps is not None:
                    self._update_shape_texture(shape_id, diffuse_maps)
                return shape_id

            # Slow path: create new shape or topology changed
            if smooth_shading:
                # Compute averaged normals
                normals = wp.zeros(point_count, dtype=wp.vec3, device=self._device)
                faces_per_vertex = wp.zeros(point_count, dtype=wp.int32, device=self._device)

                wp.launch(
                    compute_average_normals,
                    dim=idx_count,
                    inputs=[indices_reshaped, points, wp.vec3(scale)],
                    outputs=[normals, faces_per_vertex],
                    device=self._device,
                )

                tangents = wp.zeros(point_count, dtype=wp.vec3, device=self._device)
                wp.launch(
                    compute_average_tangents,
                    dim=idx_count,
                    inputs=[indices_reshaped, points, texture_coords_2d, wp.vec3(scale)],
                    outputs=[tangents, faces_per_vertex],
                    device=self._device,
                )

                # Assemble vertex data with 15 components: pos(3) + normal(3) + tangent(3) + uv(2) + color(4)
                gfx_vertices = wp.zeros((point_count, 15), dtype=wp.float32, device=self._device)
                wp.launch(
                    assemble_gfx_vertices_with_colors,
                    dim=point_count,
                    inputs=[points, normals, tangents, faces_per_vertex, texture_coords_2d, vertex_colors, wp.vec3(scale)],
                    outputs=[gfx_vertices],
                    device=self._device,
                )

                gfx_indices_wp = indices_reshaped.flatten()
            else:
                # Generate per-face vertices with 15 components: pos(3) + normal(3) + tangent(3) + uv(2) + color(4)
                gfx_vertices = wp.zeros((idx_count * 3, 15), dtype=wp.float32, device=self._device)
                wp.launch(
                    compute_gfx_vertices_with_colors,
                    dim=idx_count,
                    inputs=[indices_reshaped, points, texture_coords_2d, vertex_colors, wp.vec3(scale)],
                    outputs=[gfx_vertices],
                    device=self._device,
                )

                gfx_indices_wp = wp.arange(idx_count * 3, dtype=wp.int32, device=self._device)

            if shape_id is not None:
                self.deregister_shape(shape_id)
            if name in self._instances:
                self.remove_shape_instance(name)

            # Register shape with GPU data and textures
            shape_id = self._register_shape_gpu_direct_with_colors(geo_hash, shape_id, gfx_vertices, gfx_indices_wp, textures, basic_color)

            if textures is not None:
                self._update_shape_texture(shape_id, textures)

            if not is_template:
                self.add_shape_instance(
                    name=name,
                    shape=shape_id,
                    body=parent_body,
                    pos=pos,
                    rot=rot,
                    scale=scale,
                    color1=basic_color,
                    color2=basic_color,
                    visible=visible,
                )

            return shape_id

        def render_mesh_warp_range(
            self,
            name: str,
            points: wp.array,
            indices: wp.array,
            texture_coords: wp.array = None,
            colors: wp.array = None,
            diffuse_maps: list[int] = None,
            normal_maps: list[int] = None,
            specular_maps: list[int] = None,
            index_start: int = 0,
            index_count: int = -1,
            pos=(0.0, 0.0, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
            scale=(1.0, 1.0, 1.0),
            basic_color=(1.0, 1.0, 1.0),
            update_topology=False,
            parent_body: str | None = None,
            is_template: bool = False,
            smooth_shading: bool = True,
            visible: bool = True,
        ):
            """
            Render a mesh using a specific range of indices from the full mesh
            """
            
            # Extract the range of indices
            if index_count == -1:
                index_count = indices.shape[0] - index_start
            
            # Create a slice of the indices array for this range
            indices_range = indices[index_start:index_start + index_count]
            
            # Use the existing render_mesh_warp method with the sliced indices
            return self.render_mesh_warp(
                name=name,
                points=points,
                indices=indices_range,
                texture_coords=texture_coords,
                vertex_colors=colors,
                diffuse_maps=diffuse_maps,
                normal_maps=normal_maps,
                specular_maps=specular_maps,
                pos=pos,
                rot=rot,
                scale=scale,
                basic_color=basic_color,
                update_topology=update_topology,
                parent_body=parent_body,
                is_template=is_template,
                smooth_shading=smooth_shading,
                visible=visible,
            )


        def _register_shape_gpu_direct_with_colors(self, geo_hash, shape_id, gfx_vertices: wp.array, gfx_indices: wp.array, textures: list[list[int]] = None, color = None):
            """Register shape using GPU arrays directly with vertex color and texture support"""
            gl = SurgSimRendererOpenGL.gl
            self._switch_context()

            if shape_id is None:
                new_shape = True
                shape_id = len(self._shapes)
            else:
                new_shape = False

            if color is None:
                color1 = self._get_default_color(len(self._shape_geo_hash))
                color2 = np.clip(np.array(color1) + 0.25, 0.0, 1.0)
            else:
                color1 = color
                color2 = color

            # Store shape data with textures info (list)
            shape = (None, None, color1, color2, geo_hash, textures)
            if new_shape:
                self._shapes.append(shape)
            else:
                self._shapes[shape_id] = shape

            self._shape_geo_hash[geo_hash] = shape_id

            gl.glUseProgram(self._shape_shader.id)

            # Create VAO, VBO, and EBO
            vao = gl.GLuint()
            gl.glGenVertexArrays(1, vao)
            gl.glBindVertexArray(vao)

            # Create VBO and upload GPU data directly
            vbo = gl.GLuint()
            gl.glGenBuffers(1, vbo)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
            
            # Get buffer size and allocate
            vertex_count = gfx_vertices.shape[0]
            vertex_size = gfx_vertices.shape[1] * 4  # 4 bytes per float
            total_size = vertex_count * vertex_size
            
            gl.glBufferData(gl.GL_ARRAY_BUFFER, total_size, None, gl.GL_DYNAMIC_DRAW)
            
            # Create CUDA buffer and copy GPU data directly
            vertex_cuda_buffer = wp.RegisteredGLBuffer(int(vbo.value), self._device)
            mapped_buffer = vertex_cuda_buffer.map(dtype=wp.float32, shape=gfx_vertices.shape)
            wp.copy(mapped_buffer, gfx_vertices)
            vertex_cuda_buffer.unmap()

            # Create EBO and upload indices
            ebo = gl.GLuint()
            gl.glGenBuffers(1, ebo)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            
            indices_count = gfx_indices.shape[0]
            indices_size = indices_count * 4  # 4 bytes per int32
            
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices_size, None, gl.GL_DYNAMIC_DRAW)
            
            # Copy indices to GPU buffer
            indices_cuda_buffer = wp.RegisteredGLBuffer(int(ebo.value), self._device)
            mapped_indices = indices_cuda_buffer.map(dtype=wp.int32, shape=(indices_count,))
            wp.copy(mapped_indices, gfx_indices)
            indices_cuda_buffer.unmap()

            # Set up vertex attributes for 15-component format: pos(3) + normal(3) + tangent(3) + uv(2) + color(4)
            stride = 15 * 4  # 15 floats * 4 bytes per float
            
            # positions (location 0)
            gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(0))
            gl.glEnableVertexAttribArray(0)
            
            # normals (location 1)
            gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(12))
            gl.glEnableVertexAttribArray(1)
     
            # tangents (location 2)
            gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(24))
            gl.glEnableVertexAttribArray(2)
            
            
            # texture coordinates (location 3)
            gl.glVertexAttribPointer(3, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(36))
            gl.glEnableVertexAttribArray(3)
            
            # vertex colors (location 4)
            gl.glVertexAttribPointer(4, 4, gl.GL_FLOAT, gl.GL_FALSE, stride, ctypes.c_void_p(44))
            gl.glEnableVertexAttribArray(4)

            gl.glBindVertexArray(0)

            self._shape_gl_buffers[shape_id] = (vao, vbo, ebo, indices_count, vertex_cuda_buffer)
            return shape_id

        def _update_shape_vertices_gpu_direct_with_colors(self, shape, points: wp.array, texture_coords: wp.array = None, vertex_colors: wp.array = None, scale=(1.0, 1.0, 1.0)):
                """Update vertices, texture coordinates, and vertex colors using direct GPU-to-GPU copy"""
                if shape not in self._shape_gl_buffers:
                    return
                    
                cuda_buffer = self._shape_gl_buffers[shape][4]
                vertex_count = points.shape[0]
                
                try:
                    # Map with 15 components: pos(3) + normal(3) + tangent(3) + uv(2) + color(4)
                    vbo_vertices = cuda_buffer.map(dtype=wp.float32, shape=(vertex_count, 15))
                    
                    wp.launch(
                        update_mesh_vertices_optimized_with_colors,
                        dim=vertex_count,
                        inputs=[points, texture_coords, vertex_colors, wp.vec3(scale)],
                        outputs=[vbo_vertices],
                        device=self._device,
                    )
                    
                    cuda_buffer.unmap()
                except Exception as e:
                    print(f"Warning: Could not update vertices with colors: {e}")
                    return

        def _update_shape_texture(self, shape: int, textures: list[list[int]]):
            """Update the texture bindings for a shape"""
            if shape >= len(self._shapes):
                return
            shape_data = list(self._shapes[shape])
            if len(shape_data) >= 6:
                shape_data[5] = textures
            else:
                shape_data.append(textures)
            self._shapes[shape] = tuple(shape_data) 

        def allocate_shape_instances(self):
                gl = SurgSimRendererOpenGL.gl

                self._switch_context()

                self._add_shape_instances = False
                self._wp_instance_transforms = wp.array(
                    [instance[3] for instance in self._instances.values()], dtype=wp.transform, device=self._device
                )
                self._wp_instance_scalings = wp.array(
                    [instance[4] for instance in self._instances.values()], dtype=wp.vec3, device=self._device
                )
                self._wp_instance_bodies = wp.array(
                    [instance[1] for instance in self._instances.values()], dtype=wp.int32, device=self._device
                )

                gl.glUseProgram(self._shape_shader.id)
                if self._instance_transform_gl_buffer is None:
                    # create instance buffer and bind it as an instanced array
                    self._instance_transform_gl_buffer = gl.GLuint()
                    gl.glGenBuffers(1, self._instance_transform_gl_buffer)
                gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_transform_gl_buffer)

                transforms = np.tile(np.diag(np.ones(4, dtype=np.float32)), (len(self._instances), 1, 1))
                gl.glBufferData(gl.GL_ARRAY_BUFFER, transforms.nbytes, transforms.ctypes.data, gl.GL_DYNAMIC_DRAW)

                # create CUDA buffer for instance transforms
                self._instance_transform_cuda_buffer = wp.RegisteredGLBuffer(
                    int(self._instance_transform_gl_buffer.value), self._device
                )

                self.update_instance_colors()

                # set up instance attribute pointers
                matrix_size = transforms[0].nbytes

                instance_ids = []
                instance_custom_ids = []
                instance_visible = []
                instances = list(self._instances.values())
                inverse_instance_ids = {}
                instance_count = 0
                colors_size = np.zeros(3, dtype=np.float32).nbytes
                for shape, (vao, _vbo, _ebo, _tri_count, _vertex_cuda_buffer) in self._shape_gl_buffers.items():
                    gl.glBindVertexArray(vao)

                    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._instance_transform_gl_buffer)

                    # we can only send vec4s to the shader, so we need to split the instance transforms matrix into its column vectors
                    # locations 5-8 for instance transforms
                    for i in range(4):
                        gl.glVertexAttribPointer(
                            5 + i, 4, gl.GL_FLOAT, gl.GL_FALSE, matrix_size, ctypes.c_void_p(i * matrix_size // 4)
                        )
                        gl.glEnableVertexAttribArray(5 + i)
                        gl.glVertexAttribDivisor(5 + i, 1)

                    instance_ids.extend(self._shape_instances[shape])
                    for i in self._shape_instances[shape]:
                        instance_custom_ids.append(self._instance_custom_ids[i])
                        instance_visible.append(instances[i][7])
                        inverse_instance_ids[i] = instance_count
                        instance_count += 1

                # trigger update to the instance transforms
                self._update_shape_instances = True

                self._wp_instance_ids = wp.array(instance_ids, dtype=wp.int32, device=self._device)
                self._wp_instance_custom_ids = wp.array(instance_custom_ids, dtype=wp.int32, device=self._device)
                self._np_instance_visible = np.array(instance_visible)
                self._instance_ids = instance_ids
                self._inverse_instance_ids = inverse_instance_ids

                gl.glBindVertexArray(0)

        def deregister_shape(self, shape):
            gl = OpenGLRenderer.gl

            self._switch_context()

            if shape not in self._shape_gl_buffers:
                return

            vao, vbo, ebo, _, vertex_cuda_buffer = self._shape_gl_buffers[shape]
            try:
                gl.glDeleteVertexArrays(1, vao)
                gl.glDeleteBuffers(1, vbo)
                gl.glDeleteBuffers(1, ebo)
            except gl.GLException:
                pass

            _, _, _, _, geo_hash, _ = self._shapes[shape]
            assert(self._shape_geo_hash[geo_hash] == shape)

            del self._shape_geo_hash[geo_hash]
            #del self._shape_gl_buffers[shape]
            #self._shapes.pop(shape)

        def populate(self, model: newton.Model):
            self.skip_rendering = False

            self.model = model
            self.num_envs = model.num_envs
            self.body_names = []

            self.body_env = []  # mapping from body index to its environment index
            env_id = 0
            self.bodies_per_env = model.body_count // self.num_envs

            # create rigid body nodes
            for b in range(model.body_count):
                body_name = f"body_{b}_{self.model.body_key[b].replace(' ', '_')}"
                self.body_names.append(body_name)
                self.register_body(body_name)
                if b > 0 and b % self.bodies_per_env == 0:
                    env_id += 1
                self.body_env.append(env_id)

            # create rigid shape children
            if self.model.shape_count:
                # mapping from hash of geometry to shape ID
                self.geo_shape = {}

                self.instance_count = 0

                self.body_name = {}  # mapping from body name to its body ID
                self.body_shapes = defaultdict(list)  # mapping from body index to its shape IDs

                shape_body = model.shape_body.numpy()
                shape_geo_src = model.shape_source
                shape_geo_type = model.shape_type.numpy()
                shape_geo_scale = model.shape_scale.numpy()
                shape_geo_thickness = model.shape_thickness.numpy()
                shape_geo_is_solid = model.shape_is_solid.numpy()
                shape_transform = model.shape_transform.numpy()
                shape_flags = model.shape_flags.numpy()

                p = np.zeros(3, dtype=np.float32)
                q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                scale = np.ones(3)
                color = (1.0, 1.0, 1.0)
                # loop over shapes excluding the ground plane
                for s in range(model.shape_count):
                    geo_type = shape_geo_type[s]
                    geo_scale = [float(v) for v in shape_geo_scale[s]]
                    geo_thickness = float(shape_geo_thickness[s])
                    geo_is_solid = bool(shape_geo_is_solid[s])
                    geo_src = shape_geo_src[s]
                    name = model.shape_key[s]
                    count = self._instance_key_count.get(name, 0)
                    if count > 0:
                        self._instance_key_count[name] += 1
                        # ensure unique name for the shape instance
                        name = f"{name}_{count + 1}"
                    else:
                        self._instance_key_count[name] = 1
                    add_shape_instance = True

                    # shape transform in body frame
                    body = int(shape_body[s])
                    if body >= 0 and body < len(self.body_names):
                        body = self.body_names[body]
                    else:
                        body = None

                    if self.use_unique_colors and body is not None:
                        color = self._get_new_color()

                    # shape transform in body frame
                    X_bs = wp.transform_expand(shape_transform[s])
                    # check whether we can instance an already created shape with the same geometry
                    geo_hash = hash((int(geo_type), geo_src, *geo_scale, geo_thickness, geo_is_solid))
                    if geo_hash in self.geo_shape:
                        shape = self.geo_shape[geo_hash]
                    else:
                        if geo_type == newton.GeoType.PLANE:
                            # plane mesh
                            width = geo_scale[0] if geo_scale[0] > 0.0 else 100.0
                            length = geo_scale[1] if geo_scale[1] > 0.0 else 100.0

                            if name == "ground_plane":
                                normal = wp.quat_rotate(X_bs.q, wp.vec3(0.0, 1.0, 0.0))
                                offset = wp.dot(normal, X_bs.p)
                                shape = self.render_ground(plane=[*normal, offset])
                                add_shape_instance = False
                            else:
                                shape = self.render_plane(
                                    name, p, q, width, length, color, parent_body=body, is_template=True
                                )

                        elif geo_type == newton.GeoType.SPHERE:
                            shape = self.render_sphere(
                                name, p, q, geo_scale[0], parent_body=body, is_template=True, color=color
                            )

                        elif geo_type == newton.GeoType.CAPSULE:
                            shape = self.render_capsule(
                                name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True, color=color
                            )

                        elif geo_type == newton.GeoType.CYLINDER:
                            shape = self.render_cylinder(
                                name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True, color=color
                            )

                        elif geo_type == newton.GeoType.CONE:
                            shape = self.render_cone(
                                name, p, q, geo_scale[0], geo_scale[1], parent_body=body, is_template=True, color=color
                            )

                        elif geo_type == newton.GeoType.BOX:
                            shape = self.render_box(
                                name, p, q, geo_scale, parent_body=body, is_template=True, color=color
                            )

                        elif geo_type == newton.GeoType.MESH:
                            if not geo_is_solid:
                                faces, vertices = solidify_mesh(geo_src.indices, geo_src.vertices, geo_thickness)
                            else:
                                faces, vertices = geo_src.indices, geo_src.vertices

                            shape = self.render_mesh(
                                name,
                                vertices,
                                faces,
                                pos=p,
                                rot=q,
                                scale=geo_scale,
                                colors=color,
                                parent_body=body,
                                is_template=True,
                            )

                        elif geo_type == newton.GeoType.SDF:
                            continue

                        self.geo_shape[geo_hash] = shape

                    if add_shape_instance and shape_flags[s] & int(newton._src.geometry.flags.ShapeFlags.VISIBLE):
                        # TODO support dynamic visibility
                        self.add_shape_instance(name, shape, body, X_bs.p, X_bs.q, scale, custom_index=s, visible=True)
                    self.instance_count += 1

                if self.show_joints and model.joint_count:
                    joint_type = model.joint_type.numpy()
                    joint_axis = model.joint_axis.numpy()
                    joint_qd_start = model.joint_qd_start.numpy()
                    joint_dof_dim = model.joint_dof_dim.numpy()
                    joint_parent = model.joint_parent.numpy()
                    joint_child = model.joint_child.numpy()
                    joint_tf = model.joint_X_p.numpy()
                    shape_collision_radius = model.shape_collision_radius.numpy()
                    y_axis = wp.vec3(0.0, 1.0, 0.0)
                    color = (1.0, 0.0, 1.0)

                    shape = self.render_arrow(
                        "joint_arrow",
                        None,
                        None,
                        base_radius=0.01,
                        base_height=0.4,
                        cap_radius=0.02,
                        cap_height=0.1,
                        parent_body=None,
                        is_template=True,
                        color=color,
                    )
                    for i, t in enumerate(joint_type):
                        if t not in {
                            newton.JOINT_REVOLUTE,
                            # newton.JOINT_PRISMATIC,
                            newton.JOINT_D6,
                        }:
                            continue
                        tf = joint_tf[i]
                        body = int(joint_parent[i])
                        if body >= 0 and body < len(self.body_names):
                            body = self.body_names[body]
                        else:
                            body = None
                        # if body == -1:
                        #     continue
                        num_linear_axes = int(joint_dof_dim[i][0])
                        num_angular_axes = int(joint_dof_dim[i][1])

                        # find a good scale for the arrow based on the average radius
                        # of the shapes attached to the joint child body
                        scale = np.ones(3)
                        child = int(joint_child[i])
                        if child >= 0:
                            radii = []
                            bs = model.body_shapes.get(child, [])
                            for s in bs:
                                radii.append(shape_collision_radius[s])
                            if len(radii) > 0:
                                scale *= np.mean(radii) * 2.0

                        for a in range(num_linear_axes, num_linear_axes + num_angular_axes):
                            index = joint_qd_start[i] + a
                            axis = joint_axis[index]
                            if np.linalg.norm(axis) < 1e-6:
                                continue
                            p = wp.vec3(tf[:3])
                            q = wp.quat(tf[3:])
                            # compute rotation between axis and y
                            axis = axis / np.linalg.norm(axis)
                            q = q * wp.quat_between_vectors(wp.vec3(axis), y_axis)
                            name = f"joint_{i}_{a}"
                            self.add_shape_instance(name, shape, body, p, q, scale, color1=color, color2=color)
                            self.instance_count += 1

            if hasattr(self, "complete_setup"):
                self.complete_setup()

            self.allocate_shape_instances()

        def _render_scene(self):
            gl = OpenGLRenderer.gl

            self._switch_context()

            start_instance_idx = 0

            for shape, (vao, _, _, tri_count, _) in self._shape_gl_buffers.items():
                num_instances = len(self._shape_instances[shape])

                # Get textures for this shape (if any)
                textures = None
                if shape < len(self._shapes) and len(self._shapes[shape]) >= 6:
                    textures = self._shapes[shape][5]

                gl.glBindVertexArray(vao)
                
                # Bind textures if available, otherwise use a default white texture
                slot = int(0)
                gl.glUseProgram(self._shape_shader.id)
                gl.glUniform1i(gl.glGetUniformLocation(self._shape_shader.id, str_buffer("numTextures")), 0)

                if textures:
                    diffuse_maps = textures[0]
                    normal_maps = textures[1]
                    specular_maps = textures[2]

                    if diffuse_maps and len(diffuse_maps) > 0:
                        for i, tex_id in enumerate(diffuse_maps):
                            self.bind_texture(tex_id, slot)
                            gl.glUniform1i(gl.glGetUniformLocation(self._shape_shader.id, str_buffer(f"diffuseMaps[{i}]")), slot)
                            slot += 1

                        gl.glUniform1i(gl.glGetUniformLocation(self._shape_shader.id, str_buffer("numTextures")), len(diffuse_maps))
                    
                    if normal_maps:
                        for i, tex_id in enumerate(normal_maps):
                            self.bind_texture(tex_id, slot)
                            gl.glUniform1i(gl.glGetUniformLocation(self._shape_shader.id, str_buffer(f"normalMaps[{i}]")), slot)
                            slot += 1

                    if specular_maps:
                        for i, tex_id in enumerate(specular_maps):
                            self.bind_texture(tex_id, slot)
                            gl.glUniform1i(gl.glGetUniformLocation(self._shape_shader.id, str_buffer(f"specularMaps[{i}]")), slot)
                            slot += 1

                start_color = self._shapes[shape][2]
                gl.glUniform3f(gl.glGetUniformLocation(self._shape_shader.id, str_buffer("startColor")), start_color[0], start_color[1], start_color[2])

                gl.glDrawElementsInstancedBaseInstance(
                    gl.GL_TRIANGLES, tri_count, gl.GL_UNSIGNED_INT, None, num_instances, start_instance_idx
                )

                # Unbind textures after rendering
                if slot > 0:
                    for i in range(0, slot):
                        self.unbind_texture(i)

                start_instance_idx += num_instances

            if self.draw_axis:
                self._axis_instancer.render()

            for instancer in self._shape_instancers.values():
                instancer.render()

            gl.glBindVertexArray(0)
            self._render_post()

        def _render_post(self):
            gl = self.gl
            if not getattr(self, 'enable_postprocess', True):
                return
            
            # Create a temporary texture to hold the original color buffer
            if not hasattr(self, '_temp_color_texture'):
                self._temp_color_texture = gl.GLuint()
                gl.glGenTextures(1, self._temp_color_texture)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._temp_color_texture)
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D, 0, gl.GL_RGB,
                    self.screen_width, self.screen_height, 0,
                    gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None
                )
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            
            # Create a temporary depth texture to hold the current depth buffer
            if not hasattr(self, '_temp_depth_texture'):
                self._temp_depth_texture = gl.GLuint()
                gl.glGenTextures(1, self._temp_depth_texture)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self._temp_depth_texture)
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT24,
                    self.screen_width, self.screen_height, 0,
                    gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None
                )
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
            
            # Copy current color buffer to temporary texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._temp_color_texture)
            gl.glCopyTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, 0, 0, self.screen_width, self.screen_height, 0)
            
            # Copy current depth buffer to temporary texture
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._temp_depth_texture)
            gl.glCopyTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, 0, 0, self.screen_width, self.screen_height, 0)
            
            # Clear the current framebuffer
            gl.glClear(gl.GL_COLOR_BUFFER_BIT)
            
            # Render post-processed result back to the FBO
            gl.glUseProgram(self._postprocess_shader.id)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._temp_color_texture)
            gl.glUniform1i(self._loc_postprocess_color, 0)
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self._temp_depth_texture)
            gl.glUniform1i(self._loc_postprocess_depth, 1)

            inv_proj = np.linalg.inv(self._projection_matrix.reshape(4, 4)).astype(np.float32)
            gl.glUniformMatrix4fv(self._loc_postprocess_invproj, 1, gl.GL_FALSE, inv_proj.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            gl.glUniform2f(self._loc_postprocess_screensize, float(self.screen_width), float(self.screen_height))

            gl.glBindVertexArray(self._frame_vao)
            gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)

            # Clean up state
            gl.glActiveTexture(gl.GL_TEXTURE1)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glUseProgram(0)


        def render(self, state: newton.State):
            """
            Updates the renderer with the given simulation state.

            Args:
                state (newton.State): The simulation state to render.
            """
            if self.skip_rendering:
                return

            if self.model.particle_count:
                particle_q = state.particle_q.numpy()

                # render particles
                if self.show_particles:
                    self.render_points(
                        "particles", particle_q, radius=self.model.particle_radius.numpy(), colors=(0.8, 0.3, 0.2)
                    )

                # render tris
                if self.model.tri_count:
                    self.render_mesh(
                        "surface",
                        particle_q,
                        self.model.tri_indices.numpy().flatten(),
                        colors=(0.75, 0.25, 0.0),
                    )

                # render springs
                if self.model.spring_count:
                    self.render_line_list(
                        "springs", particle_q, self.model.spring_indices.numpy().flatten(), (0.25, 0.5, 0.25), 0.02
                    )

            # render muscles
            if self.model.muscle_count:
                body_q = state.body_q.numpy()

                muscle_start = self.model.muscle_start.numpy()
                muscle_links = self.model.muscle_bodies.numpy()
                muscle_points = self.model.muscle_points.numpy()
                muscle_activation = self.model.muscle_activation.numpy()

                # for s in self.skeletons:

                #     # for mesh, link in s.mesh_map.items():

                #     #     if link != -1:
                #     #         X_sc = wp.transform_expand(self.state.body_X_sc[link].tolist())

                #     #         #self.renderer.add_mesh(mesh, "../assets/snu/OBJ/" + mesh + ".usd", X_sc, 1.0, self.render_time)
                #     #         self.renderer.add_mesh(mesh, "../assets/snu/OBJ/" + mesh + ".usd", X_sc, 1.0, self.render_time)

                for m in range(self.model.muscle_count):
                    start = int(muscle_start[m])
                    end = int(muscle_start[m + 1])

                    points = []

                    for w in range(start, end):
                        link = muscle_links[w]
                        point = muscle_points[w]

                        X_sc = wp.transform_expand(body_q[link][0])

                        points.append(wp.transform_point(X_sc, point).tolist())

                    self.render_line_strip(
                        name=f"muscle_{m}", vertices=points, radius=0.0075, color=(muscle_activation[m], 0.2, 0.5)
                    )

            # update bodies
            if self.model.body_count:
                self.update_body_transforms(state.body_q)

        def render_contacts(
            self,
            state: newton.State,
            contacts: newton.Contacts,
            contact_point_radius: float = 1e-3,
        ):
            """
            Render contact points between rigid bodies.

            Args:
                state (newton.State): The simulation state.
                contacts (newton.Contacts): The contacts to render.
                contact_point_radius (float, optional): The radius of the contact points.
            """
            import warp as wp
            if self._contact_points0 is None or len(self._contact_points0) < contacts.soft_contact_max:
                self._contact_points0 = wp.array(
                    np.zeros((contacts.soft_contact_max, 3)), dtype=wp.vec3, device=self.model.device
                )
                self._contact_points1 = wp.array(
                    np.zeros((contacts.soft_contact_max, 3)), dtype=wp.vec3, device=self.model.device
                )
           

            try:
                soft_count = int(contacts.soft_contact_max)
                # These should be arrays of length soft_count
                soft_points = contacts.soft_contact_particle.numpy()[:soft_count]
                soft_forces = contacts.soft_contact_normal.numpy()[:soft_count]
            except Exception as e:
                print(f"Could not access softbody contact arrays: {e}")
                return

            particles_pos_cpu = state.particle_q.numpy()

            # Draw an arrow for each softbody contact
            for i in range(soft_count):
                id = soft_points[i]
                if id == -1:
                    break

                p = particles_pos_cpu[id]
                f = soft_forces[i]
                # Only draw if force is nonzero
                if np.linalg.norm(f) > 1e-8:
                    # Scale the force vector for visualization
                    arrow_scale = 0.1 * self.scaling  # You may want to adjust this factor
                    force_vec = np.array(f) * arrow_scale
                    start = np.array(p)
                    end = start + force_vec

                    # Compute arrow direction as quaternion
                    direction = force_vec / (np.linalg.norm(force_vec) + 1e-12)

                    # Default arrow points along +Y; compute rotation quaternion
                    import warp as wp
                    q = wp.quat_between_vectors(wp.vec3(0.0, 1.0, 0.0), wp.vec3(*direction))

                    # Arrow length and thickness
                    arrow_length = np.linalg.norm(force_vec)
                    base_radius = 0.01 * self.scaling
                    cap_radius = 0.02 * self.scaling
                    cap_height = 0.04 * self.scaling

                    # Use orange color for force arrows
                    color = (1.0, 0.6, 0.1)
                    self.render_arrow(
                        name=f"soft_contact_force_{i}",
                        pos=tuple(start),
                        rot=(float(q.x), float(q.y), float(q.z), float(q.w)),
                        base_radius=base_radius,
                        base_height=arrow_length - cap_height if arrow_length > cap_height else arrow_length * 0.7,
                        cap_radius=cap_radius,
                        cap_height=cap_height,
                        color=color,
                        visible=True,
                    )
        
        def bind_texture(self, texture_id: int, unit: int = 0):
            """Bind texture to specified texture unit"""
            assert(hasattr(self, '_texture_manager')), "Texture manager not initialized"
            if hasattr(self, '_texture_manager'):
                self._texture_manager.bind_texture(texture_id, unit)

        def unbind_texture(self, unit: int = 0):
            """Unbind texture from specified texture unit"""
            if hasattr(self, '_texture_manager'):
                self._texture_manager.unbind_texture(unit)

        def delete_texture(self, texture_id: int):
            """Delete a texture"""
            if hasattr(self, '_texture_manager'):
                self._texture_manager.delete_texture(texture_id)

        def _get_default_color(self, index):
            """Get default color without importing tab10_color_map if possible"""
            colors = [
                (0.12, 0.47, 0.71),  # blue
                (1.0, 0.5, 0.05),    # orange  
                (0.17, 0.63, 0.17),  # green
                (0.84, 0.15, 0.16),  # red
                (0.58, 0.4, 0.74),   # purple
            ]
            return colors[index % len(colors)]

        def _get_new_color(self):
            return tab10_color_map(self.instance_count)


    return SimRenderer


class SurgSimRendererUsd(CreateSurgSimRenderer(renderer=UsdRenderer)):
    """
    USD renderer for Newton Physics simulations.

    This renderer exports simulation data to USD (Universal Scene Description)
    format, which can be visualized in Omniverse or other USD-compatible viewers.

    Args:
        model (newton.Model): The Newton physics model to render.
        path (str): Output path for the USD file.
        scaling (float, optional): Scaling factor for the rendered objects.
            Defaults to 1.0.
        fps (int, optional): Frames per second for the animation. Defaults to 60.
        up_axis (newton.AxisType, optional): Up axis for the scene. If None,
            uses model's up axis.
        show_rigid_contact_points (bool, optional): Whether to show contact
            points. Defaults to False.
        contact_points_radius (float, optional): Radius of contact point
            spheres. Defaults to 1e-3.
        show_joints (bool, optional): Whether to show joint visualizations.
            Defaults to False.
        **render_kwargs: Additional arguments passed to the underlying
            UsdRenderer.

    Example:
        .. code-block:: python

            import newton

            model = newton.Model()  # your model setup
            renderer = newton.utils.SimRendererUsd(model, "output.usd", scaling=2.0)
            # In your simulation loop:
            renderer.begin_frame(time)
            renderer.render(state)
            renderer.end_frame()
            renderer.save()  # Save the USD file
    """

    pass


class SurgSimRendererOpenGL(CreateSurgSimRenderer(renderer=OpenGLRenderer)):
    """
    Real-time OpenGL renderer for Newton Physics simulations.

    This renderer provides real-time visualization of physics simulations using
    OpenGL, with interactive camera controls and various rendering options.

    Args:
        model (newton.Model): The Newton physics model to render.
        path (str): Window title for the OpenGL window.
        scaling (float, optional): Scaling factor for the rendered objects.
            Defaults to 1.0.
        fps (int, optional): Target frames per second. Defaults to 60.
        up_axis (newton.AxisType, optional): Up axis for the scene. If None,
            uses model's up axis.
        show_rigid_contact_points (bool, optional): Whether to show contact
            points. Defaults to False.
        contact_points_radius (float, optional): Radius of contact point
            spheres. Defaults to 1e-3.
        show_joints (bool, optional): Whether to show joint visualizations.
            Defaults to False.
        **render_kwargs: Additional arguments passed to the underlying
            OpenGLRenderer.

    Example:
        .. code-block:: python

            import newton

            model = newton.Model()  # your model setup
            renderer = newton.utils.SimRendererOpenGL(model, "Newton Simulator")
            # In your simulation loop:
            renderer.begin_frame(time)
            renderer.render(state)
            renderer.end_frame()

    Note:
        Keyboard shortcuts available during rendering:

        - W, A, S, D (or arrow keys) + mouse: FPS-style camera movement
        - X: Toggle wireframe rendering
        - B: Toggle backface culling
        - C: Toggle coordinate system axes
        - G: Toggle ground grid
        - T: Toggle depth rendering
        - I: Toggle info text
        - SPACE: Pause/continue simulation
        - TAB: Skip rendering (background simulation)
    """

    pass


SurgSimRenderer = SurgSimRendererOpenGL
