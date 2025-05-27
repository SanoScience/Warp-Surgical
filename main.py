import math
import os

import numpy as np
import warp as wp
import warp.sim
import warp.sim.render
from warp.sim.render import SimRendererOpenGL
from pxr import Usd, UsdGeom


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

    for i in range(0, len(positions)):
        position = wp.vec3(positions[i])
        position[1] += vertical_offset
        builder.add_particle(position, wp.vec3(0,0,0), 1)

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
        load_mesh_and_build_model(builder, vertical_offset=-5.0)

        self.model = builder.finalize()
        self.model.ground = True
        #self.model.gravity[1] = 0.0

        self.integrator = wp.sim.XPBDIntegrator(iterations=10)

        self.rest = self.model.state()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.volume = wp.zeros(1, dtype=wp.float32)

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

    with wp.ScopedDevice(args.device):
        sim = WarpSim(stage_path=args.stage_path, num_frames=args.num_frames, use_opengl=not(args.usd))

        if args.usd:
            while sim.renderer.is_running():
                sim.step()
                sim.render()
        else:
            for _ in range(args.num_frames):
                sim.step()
                sim.render()

            if sim.renderer:
                sim.renderer.save()
