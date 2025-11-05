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

###########################################################################
# Example Basic Soft Body
#
# Shows how to create a simple deformable body loaded from a mesh.
# This example loads a mesh from a USD file and creates a soft body
# simulation using tetrahedral FEM elements.
#
# Command: python -m newton.examples basic_soft_body
#
###########################################################################

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples


class Example:
    def __init__(self, viewer):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 16
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer

        # Create scene
        scene = newton.ModelBuilder()

        # Material properties for the soft body
        young_mod = 5.0e4  # Young's modulus (stiffness)
        poisson_ratio = 0.3  # Poisson's ratio
        k_mu = 0.5 * young_mod / (1.0 + poisson_ratio)
        k_lambda = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
        
        density = 100.0  # Density of the soft body
        k_damp = 0.1  # Damping coefficient

        # Load mesh from USD file (bunny in this example)
        # You can replace "bunny.usd" with "bear.usd" or any other tetrahedral mesh
        usd_stage = Usd.Stage.Open(newton.examples.get_asset("bunny.usd"))
        usd_geom = UsdGeom.Mesh(usd_stage.GetPrimAtPath("/root/bunny"))

        mesh_points = np.array(usd_geom.GetPointsAttr().Get())
        mesh_indices = np.array(usd_geom.GetFaceVertexIndicesAttr().Get())

        # For this example, we'll create a simple tetrahedral mesh
        # In a real scenario, you would use a meshing library to convert
        # triangle mesh to tetrahedral mesh, or load a pre-generated tet mesh
        
        # Simple approach: create a coarse tetrahedral grid for demonstration
        # This is a placeholder - for actual bear/bunny models, you'd need proper
        # tetrahedral mesh generation (e.g., using TetGen, gmsh, etc.)
        vertices, tet_indices = create_simple_tet_mesh()
        
        # Add the soft body mesh to the scene
        scene.add_soft_mesh(
            pos=wp.vec3(0.0, 0.0, 2.0),  # Position in world space
            rot=wp.quat_identity(),  # Rotation
            scale=0.5,  # Scale factor
            vel=wp.vec3(0.0, 0.0, 0.0),  # Initial velocity
            vertices=vertices,
            indices=tet_indices,
            density=density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
            tri_ke=1e3,  # Triangle elastic stiffness
            tri_ka=1e3,  # Triangle area stiffness
            tri_kd=1e-1,  # Triangle damping
        )

        # Add ground plane for collision
        scene.add_ground_plane()

        # Finalize the model
        self.model = scene.finalize()
        
        # Set soft contact parameters
        self.model.soft_contact_ke = 1.0e3
        self.model.soft_contact_kd = 1.0e1
        self.model.soft_contact_mu = 0.5

        # Create solver for soft body simulation
        self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # Allocate simulation states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0)

        # Setup viewer
        self.viewer.set_model(self.model)

        # Capture simulation for CUDA graphs (if available)
        self.capture()

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model (e.g., gravity is automatic)
            self.viewer.apply_forces(self.state_0)

            # Compute collisions
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=0.01)
            
            # Step the solver
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)

            # Swap states
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def test(self):
        pass

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()


def create_simple_tet_mesh():
    """
    Creates a simple tetrahedral mesh for demonstration.
    
    For loading actual models like bear or bunny as soft bodies, you would need:
    1. A tetrahedral mesh file (not just surface triangles)
    2. Or use a meshing library to convert surface mesh to volumetric tet mesh
       (e.g., TetGen, gmsh, or similar tools)
    
    Returns:
        vertices: List of wp.vec3 vertex positions
        indices: List of tetrahedral indices (4 vertices per tet)
    """
    # Create a simple cube made of tetrahedra as a demonstration
    vertices = [
        wp.vec3(0.0, 0.0, 0.0),
        wp.vec3(1.0, 0.0, 0.0),
        wp.vec3(1.0, 1.0, 0.0),
        wp.vec3(0.0, 1.0, 0.0),
        wp.vec3(0.0, 0.0, 1.0),
        wp.vec3(1.0, 0.0, 1.0),
        wp.vec3(1.0, 1.0, 1.0),
        wp.vec3(0.0, 1.0, 1.0),
    ]
    
    # Cube can be divided into 5 or 6 tetrahedra
    # Here's a simple 5-tet decomposition
    indices = [
        0, 1, 2, 5,  # tet 0
        0, 2, 3, 7,  # tet 1
        0, 5, 2, 7,  # tet 2
        5, 2, 7, 6,  # tet 3
        0, 5, 7, 4,  # tet 4
    ]
    
    return vertices, indices


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()

    # Create example
    example = Example(viewer)

    # Run example
    newton.examples.run(example)


