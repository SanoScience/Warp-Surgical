###########################################################################
# OpenGL renderer example
#
# Demonstrates how to set up tiled rendering and retrieves the pixels from
# OpenGLRenderer as a Warp array while keeping all memory on the GPU.
#
###########################################################################

import numpy as np

import warp as wp
import warp.render
from pyOpenHaptics.hd_device import HapticDevice
import pyOpenHaptics.hd as hd
import time
from dataclasses import dataclass, field
from pyOpenHaptics.hd_callback import hd_callback

@dataclass
class DeviceState:
    button: bool = False
    position: list = field(default_factory=list)
    joints: list = field(default_factory=list)
    gimbals: list = field(default_factory=list)
    force: list = field(default_factory=list)

@hd_callback
def state_callback():
    global device_state
    transform = hd.get_transform()
    joints = hd.get_joints()
    gimbals = hd.get_gimbals()
    device_state.position = [transform[3][0], -transform[3][1], transform[3][2]]
    device_state.joints = [joints[0], joints[1], joints[2]]
    device_state.gimbals = [gimbals[0], gimbals[1], gimbals[2]]
    hd.set_force(device_state.force)
    button = hd.get_buttons()
    device_state.button = True if button==1 else False


class Example:
    def __init__(self, ):
       
        self.renderer = wp.render.OpenGLRenderer(vsync=True)
        
        
        #self.renderer.render_ground()

    def render_mesh_from_warp_buffers(self):
        # Initialize vertices for a triangle (on CPU first, then copy)
        vertices_cpu = np.array([
            [0.0, 1.0, 0.0],   # top vertex
            [-1.0, -1.0, 0.0], # bottom left
            [1.0, -1.0, 0.0]   # bottom right
        ], dtype=np.float32)
        
        # Initialize indices
        indices_cpu = np.array([0, 1, 2], dtype=np.int32)
        
        # Create GPU arrays
        vertices_gpu = wp.array(vertices_cpu, dtype=wp.vec3, device="cuda:0")
        indices_gpu = wp.array(indices_cpu, dtype=wp.int32, device="cuda:0")
        
        # # Get data back from GPU (since renderer needs CPU data)
        # vertices_from_gpu = wp.to_numpy(vertices_gpu)
        # indices_from_gpu = wp.to_numpy(indices_gpu)
        
        # Render the mesh using the correct parameter names
        self.renderer.render_mesh(
            name="custom_mesh",
            points=vertices_cpu,  # Use 'points' instead of 'vertices'
            indices=indices_cpu,
            pos=[0.0, 0.0, 0.0],      # Use 'pos' instead of 'position'
            rot=[0.0, 0.0, 0.0, 1.0], # Use 'rot' instead of 'rotation'
            scale=1.0,
            smooth_shading=True
        )

    def render(self):
        time = self.renderer.clock_time
        self.renderer.begin_frame(time)


        self.renderer.render_sphere(
            "sphere",
            [dev_x * 0.01, -dev_y * 0.01, dev_z * 0.01],
            [0.0, 0.0, 0.0, 1.0],
            0.1,
        )

        
        self.renderer.render_cylinder(
            "cylinder",
            [3.2, 1.0, np.sin(time + 0.5)],
            np.array(wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), wp.sin(time + 0.5))),
            radius=0.5,
            half_height=0.8,
        )
        # self.renderer.render_cone(
        #     "cone",
        #     [-1.2, 1.0, 0.0],
        #     np.array(wp.quat_from_axis_angle(wp.vec3(0.707, 0.707, 0.0), time)),
        #     radius=0.5,
        #     half_height=0.8,
        # )
    
        self.render_mesh_from_warp_buffers()

        self.renderer.end_frame()

def wall_feedback():
    device_state.force = [0, 0, 0]



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")

    args = parser.parse_known_args()[0]

    device_state = DeviceState()
    device = HapticDevice(device_name="Default Device", callback=state_callback)
    time.sleep(0.2)

    with wp.ScopedDevice(args.device):
        example = Example()
        
        pre_dev_x, pre_dev_y = device_state.position[0], device_state.position[1]

        while example.renderer.is_running():
            dev_x, dev_y, dev_z = device_state.position[0], device_state.position[1], device_state.position[2]
            example.render()
            wall_feedback()
            pre_dev_x, pre_dev_y, pre_dev_z = dev_x, dev_y, dev_z

        example.renderer.clear()

        device.close()
