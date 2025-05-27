import math
import os
import time

import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
from warp.sim.render import SimRendererOpenGL
from pxr import Usd, UsdGeom

from pyOpenHaptics.hd_device import HapticDevice
import pyOpenHaptics.hd as hd
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
    device_state.position = [transform[3][0], transform[3][1], transform[3][2]]
    device_state.joints = [joints[0], joints[1], joints[2]]
    device_state.gimbals = [gimbals[0], gimbals[1], gimbals[2]]
    hd.set_force(device_state.force)
    button = hd.get_buttons()
    device_state.button = True if button==1 else False

@wp.kernel
def set_body_position(body_q: wp.array(dtype=wp.transformf), 
                      body_qd: wp.array(dtype=wp.spatial_vectorf),
                      body_id: int, posParameter: wp.array(dtype=wp.vec3)):
    t = body_q[body_id]
    pos = posParameter[0]

    # Set translation part (first 3 elements) while preserving rotation (last 4 elements)
    body_q[body_id] = wp.transform(pos * 0.01, wp.quat(t[3], t[4], t[5], t[6]))
    #body_qd[body_id] = wp.spatial_vector() # reset velocity to zero




def load_mesh_and_build_model(builder: wp.sim.ModelBuilder, vertical_offset=0.0):
    positions = []
    indices = []
    edges = []

    with open('meshes/liver.vertices', 'r') as f:
        lines = f.readlines()
        for line in lines:
            pos = [float(x) for x in line.split()]
            pos[1] += vertical_offset
            positions.append(pos)

    with open('meshes/liver.indices', 'r') as f:
        lines = f.readlines()
        for line in lines:
            indices.extend([int(x) for x in line.split()])

    with open('meshes/liver.edges', 'r') as f:
        lines = f.readlines()
        for line in lines:
            edges.extend([int(x) for x in line.split()])

    mass_total = 10.0
    mass = mass_total / len(positions)

    for i in range(0, len(positions)):
        position = wp.vec3(positions[i])
        position[1] += vertical_offset
        builder.add_particle(position, wp.vec3(0,0,0), mass=mass, radius=0.05)

    for i in range(0, len(edges), 2):
        builder.add_spring(edges[i], edges[i + 1], 1.0e3, 0.0, 0)

    for i in range(0, len(indices), 4):
        builder.add_tetrahedron(indices[i+0], indices[i + 1], indices[i + 2], indices[i + 3])

class WarpSim:
    def __init__(self, stage_path="output.usd", num_frames=300, use_opengl=True):
        self.sim_substeps = 64
        self.num_frames = num_frames
        fps = 60

        self.frame_dt = 1.0 / fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = wp.sim.ModelBuilder()

        # Import the mesh
        load_mesh_and_build_model(builder, vertical_offset=-5.5)

        # Add haptic device collision
        self.haptic_body_id = builder.add_body(origin=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
            m=0.0,  # Zero mass makes it kinematic
            armature=0.0)

        builder.add_shape_sphere(self.haptic_body_id, 
            has_ground_collision=False, 
            has_shape_collision=True, 
            radius=0.2, 
            pos=wp.vec3(0.0, 0.0, 0.0),
            density=100)

        self.model = builder.finalize()
        self.model.ground = True
        #self.model.gravity[1] = 0.0

        self.integrator = wp.sim.XPBDIntegrator(iterations=10)

        self.dev_pos_buffer = wp.array([0.0, 0.0, 0.0], dtype=wp.vec3, device=wp.get_device())

        self.rest = self.model.state()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.use_opengl = use_opengl
        
        if self.use_opengl:
            self.renderer = SimRendererOpenGL(self.model, "test window", scaling=1.0)
        elif stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=20.0)
        else:
            self.renderer = None
            
        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.state_1.clear_forces()

            wp.launch(
                set_body_position,
                dim=1,
                inputs=[self.state_0.body_q, self.state_0.body_qd, self.haptic_body_id, self.dev_pos_buffer],
                device=self.state_0.body_q.device,
            )
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
        
            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self):
        with wp.ScopedTimer("step"):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            if self.use_opengl:
                self.renderer.begin_frame()
                self.renderer.render(self.state_0)
                self.renderer.end_frame()
            else:
                self.renderer.begin_frame(self.sim_time)
                self.renderer.render(self.state_0)
                self.renderer.end_frame()

    def update_haptic_position(self, new_pos: wp.vec3):
        """Update the haptic position parameter for the CUDA graph"""
        wp.copy(self.dev_pos_buffer, wp.array([new_pos], dtype=wp.vec3, device=wp.get_device()))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="output.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--usd", action="store_true", help="Render to USD instead of OpenGL.")

    args = parser.parse_known_args()[0]
    
    # Haptic device setup
    device_state = DeviceState()
    device = HapticDevice(device_name="Default Device", callback=state_callback)
    device_state.force = [0.0, 0.0, 0.0]
    haptic_scale = 2.5
    time.sleep(0.2)

    with wp.ScopedDevice(args.device):
        sim = WarpSim(stage_path=args.stage_path, num_frames=args.num_frames, use_opengl=not(args.usd))
        pre_dev_pos = wp.vec3(device_state.position[0], device_state.position[1], device_state.position[2]) * haptic_scale

        if args.usd:
            while sim.renderer.is_running():
                sim.step()
                sim.render()
        else:
            for _ in range(args.num_frames):
                dev_pos = wp.vec3(device_state.position[0], device_state.position[1], device_state.position[2]) * haptic_scale
                sim.update_haptic_position(dev_pos)

                sim.step()
                sim.render()

                pre_dev_pos = dev_pos

            if sim.renderer:
                sim.renderer.save()
