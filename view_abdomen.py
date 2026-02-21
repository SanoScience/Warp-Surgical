"""Minimal ovrtx viewer that loads abdomen.usdc and displays it in a pyglet window."""

from __future__ import annotations

import os
import math
import numpy as np
import pyglet

# Allow ovrtx to coexist with pxr (usd-core)
os.environ.setdefault("OVRTX_SKIP_USD_CHECK", "1")

import ovrtx
from ovrtx import Renderer, RendererConfig
from ovrtx.math import Matrix4d

import ovrtx._src.bindings as _ovrtx_bindings

if _ovrtx_bindings.OVRTX_LIBRARY_PATH_HINT is None:
    _candidate = os.path.join(
        os.path.dirname(os.path.dirname(ovrtx.__file__)),
        "examples", "c", "_deps", "ovrtx-src", "bin",
    )
    if os.path.isfile(os.path.join(_candidate, "libovrtx-dynamic.so")):
        _ovrtx_bindings.OVRTX_LIBRARY_PATH_HINT = _candidate

WIDTH, HEIGHT = 1280, 720
USD_FILE = os.path.join(os.path.dirname(__file__), "abdomen.usdc")
RENDER_PRODUCT = "/Render/OmniverseKit/HydraTextures/ViewportTexture0"

# Camera state
cam_pos = [0.0, -300.0, 150.0]  # pulled back along -Y (upAxis=Z scene)
cam_yaw = 90.0    # degrees, looking along +Y
cam_pitch = -10.0
cam_speed = 100.0
mouse_sensitivity = 0.15
keys_held: set[int] = set()


def yaw_pitch_to_front(yaw_deg, pitch_deg):
    yr = math.radians(yaw_deg)
    pr = math.radians(pitch_deg)
    return [math.cos(pr) * math.cos(yr), math.cos(pr) * math.sin(yr), math.sin(pr)]


cam_front = yaw_pitch_to_front(cam_yaw, cam_pitch)
cam_up = [0.0, 0.0, 1.0]  # Z-up to match the USD scene


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    fwd = target - eye
    fwd /= max(np.linalg.norm(fwd), 1e-10)
    right = np.cross(fwd, up)
    right /= max(np.linalg.norm(right), 1e-10)
    new_up = np.cross(right, fwd)
    return np.array([
        [right[0], right[1], right[2], 0.0],
        [new_up[0], new_up[1], new_up[2], 0.0],
        [-fwd[0], -fwd[1], -fwd[2], 0.0],
        [eye[0], eye[1], eye[2], 1.0],
    ], dtype=np.float64)


def write_camera(renderer):
    target = [cam_pos[i] + cam_front[i] for i in range(3)]
    mat = look_at(cam_pos, target, cam_up)
    m = Matrix4d()
    for i in range(4):
        m[i] = [mat[i, 0], mat[i, 1], mat[i, 2], mat[i, 3]]
    renderer.write_attribute(
        prim_paths=["/World/Camera"],
        attribute_name="omni:fabric:localMatrix",
        tensor=m.to_dltensor(),
        semantic="transform_4x4",
    )


def main():
    global cam_pos, cam_front, cam_yaw, cam_pitch

    window = pyglet.window.Window(width=WIDTH, height=HEIGHT, caption="abdomen.usdc", resizable=False)

    @window.event
    def on_key_press(symbol, modifiers):
        keys_held.add(symbol)

    @window.event
    def on_key_release(symbol, modifiers):
        keys_held.discard(symbol)

    @window.event
    def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
        global cam_yaw, cam_pitch, cam_front
        if buttons & pyglet.window.mouse.RIGHT:
            cam_yaw += dx * mouse_sensitivity
            cam_pitch += dy * mouse_sensitivity
            cam_pitch = max(-89.0, min(89.0, cam_pitch))
            cam_front = yaw_pitch_to_front(cam_yaw, cam_pitch)
            write_camera(renderer)

    @window.event
    def on_draw():
        pass

    # Create renderer and set up scene
    renderer = Renderer(config=RendererConfig(sync_mode=True))

    # Base layer: camera, light, render product
    base_usda = f"""#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
    metersPerUnit = 1
)

def Xform "World" {{
    def Camera "Camera" {{
        float focalLength = 18.14
        float horizontalAperture = 20.955
        float verticalAperture = 15.29
        float2 clippingRange = (1, 10000)
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def SphereLight "Light" {{
        float inputs:intensity = 500000
        float radius = 5
        color3f inputs:color = (1, 1, 1)
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,-300,200,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
    }}

    def DomeLight "DomeLight" {{
        float inputs:intensity = 500
    }}
}}

def "Render" {{
    def "OmniverseKit" {{
        def "HydraTextures" {{
            def RenderProduct "ViewportTexture0" {{
                rel camera = </World/Camera>
                rel orderedVars = [</Render/Vars/LdrColor>]
                uniform int2 resolution = ({WIDTH}, {HEIGHT})
            }}
        }}
    }}

    def RenderSettings "OmniverseGlobalRenderSettings" {{
        rel products = <{RENDER_PRODUCT}>
    }}

    def "Vars" {{
        def RenderVar "LdrColor" {{
            uniform string sourceName = "LdrColor"
        }}
    }}
}}
"""
    renderer.add_usd_layer(base_usda)

    # Load abdomen.usdc under /World/Abdomen
    abdomen_path = os.path.abspath(USD_FILE)
    renderer.add_usd(abdomen_path, path_prefix="/World/Abdomen")

    write_camera(renderer)

    dt = 1.0 / 60.0

    while not window.has_exit:
        window.dispatch_events()

        # WASD + QE movement
        moved = False
        speed = cam_speed * dt
        front = np.array(cam_front)
        right = np.cross(front, cam_up)
        rlen = np.linalg.norm(right)
        if rlen > 1e-8:
            right /= rlen

        from pyglet.window import key
        if key.W in keys_held:
            cam_pos = [cam_pos[i] + front[i] * speed for i in range(3)]; moved = True
        if key.S in keys_held:
            cam_pos = [cam_pos[i] - front[i] * speed for i in range(3)]; moved = True
        if key.D in keys_held:
            cam_pos = [cam_pos[i] + right[i] * speed for i in range(3)]; moved = True
        if key.A in keys_held:
            cam_pos = [cam_pos[i] - right[i] * speed for i in range(3)]; moved = True
        if key.E in keys_held:
            cam_pos[2] += speed; moved = True
        if key.Q in keys_held:
            cam_pos[2] -= speed; moved = True

        if moved:
            write_camera(renderer)

        # Render
        products = renderer.step(render_products={RENDER_PRODUCT}, delta_time=dt)

        pixels = None
        for _, product in products.items():
            for frame in product.frames:
                with frame.render_vars["LdrColor"].map(device="cpu") as var:
                    pixels = var.tensor.numpy().copy()

        if pixels is not None:
            window.switch_to()
            window.clear()
            image = pyglet.image.ImageData(WIDTH, HEIGHT, "RGBA", pixels.tobytes(), pitch=-WIDTH * 4)
            image.blit(0, 0)
            window.flip()
        else:
            window.flip()


if __name__ == "__main__":
    main()
