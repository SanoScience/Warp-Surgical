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
# Example Soft Body from USD
#
# Shows how to create a deformable body loaded from a USD file containing
# tetrahedral mesh data. This is similar to how you would load a bear or
# other complex soft body model.
#
# Command: python -m newton.examples basic_soft_body_from_usd
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.utils.import_usd import parse_usd


class Example:
    def __init__(self, viewer, asset_name='/home/mnaskret/sano/assets/liver/liverDeformable.usd'):
        # setup simulation parameters first
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 32  # Increased from 16 for better stability with stiff materials
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.viewer = viewer
        self.asset_name = asset_name

        # Create scene
        scene = newton.ModelBuilder()

        # Try to load deformable body from USD
        try:
            # Use the built-in parse_usd function to load soft bodies
            xform = wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())
            parse_info = parse_usd(
                scene, 
                asset_name,
                xform=xform, 
                verbose=True,
                load_non_physics_prims=False  # Speed up loading by skipping visual shapes
            )
            
            # Check if any soft meshes were loaded
            if len(parse_info["path_soft_mesh_map"]) > 0:
                print(f"Successfully loaded {len(parse_info['path_soft_mesh_map'])} soft mesh(es) from {asset_name}")
                
                # IMPORTANT: Override material parameters for better volume preservation
                # The default k_mu and k_lambda from USD might be too low
                # For soft organs like liver, we need stiffer materials to prevent flattening
                num_loaded_tets = len(scene.tet_materials)
                print(f"Loaded {num_loaded_tets} tetrahedra with default material parameters")
                
                # Override with stiffer material parameters
                # For liver: young_mod ~3kPa in reality, but we need higher for stability
                young_mod = 5.0e6  # Increased for better shape recovery
                poisson_ratio = 0.48  # Very close to incompressible (0.5 = fully incompressible)
                k_mu_override = 0.5 * young_mod / (1.0 + poisson_ratio)
                k_lambda_override = young_mod * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio))
                
                # Very low damping to allow elastic recovery
                k_damp_override = 0.01  # Low damping for better recovery
                
                print(f"Overriding material: k_mu={k_mu_override:.2e}, k_lambda={k_lambda_override:.2e}, k_damp={k_damp_override}")
                
                # Update all loaded tetrahedra with new material parameters
                for i in range(len(scene.tet_materials)):
                    scene.tet_materials[i] = (k_mu_override, k_lambda_override, k_damp_override)
                
        except Exception as e:
            print(f"Could not load from {asset_name}: {e}")
        
        # Add ground plane for collision
        scene.add_ground_plane()

        # Finalize the model
        self.model = scene.finalize()
        
        # Set soft contact parameters - increase stiffness to prevent penetration
        self.model.soft_contact_ke = 1.0e6  # Increased from 1.0e4 for stiffer contact
        self.model.soft_contact_kd = 1.0e3  # Increased damping for stability
        self.model.soft_contact_mu = 0.5
        self.model.soft_contact_restitution = 0.0  # Reduce bounce

        # Create solver for soft body simulation
        # 
        # OPTION 1 (Current): XPBD with VERY low relaxation for volume preservation
        # - soft_body_relaxation: Lower = stiffer (default is 0.9)
        # - iterations: More iterations = better constraint satisfaction (default is 2)
        # For nearly incompressible materials like liver, use very low relaxation
        self.solver = newton.solvers.SolverXPBD(
            self.model,
            iterations=self.sim_substeps,  # Good for stiff constraints
            soft_body_relaxation=0.0001,  # Much lower for full shape recovery (was 0.01)
        )
        
        # OPTION 2 (Alternative): SemiImplicit solver uses FEM directly
        # May preserve volume better in some cases. Uncomment to try:
        # self.solver = newton.solvers.SolverSemiImplicit(self.model)

        # Allocate simulation states
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.collide(self.state_0, soft_contact_margin=0.01)

        # Setup viewer
        self.viewer.set_model(self.model)

        # Capture simulation for CUDA graphs (if available)
        # self.capture()
        self.graph = None

    def capture(self):
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
        else:
            self.graph = None

    def simulate(self):
        for substep in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model (e.g., gravity is automatic)
            self.viewer.apply_forces(self.state_0)

            # Compute collisions
            self.contacts = self.model.collide(self.state_0, soft_contact_margin=0.01)
            
            # Step the solver
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            
            # # Print bounding box of the mesh
            # particles = self.state_0.particle_q.numpy()
            # if len(particles) > 0:
            #     min_coords = particles.min(axis=0)
            #     max_coords = particles.max(axis=0)
            #     print(f"Mesh bounding box: min={min_coords}, max={max_coords}, size={max_coords - min_coords}")

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


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--asset",
        default='/home/mnaskret/sano/assets/liver/liverDeformable.usd',
        help="Name of the USD asset to load (e.g., 'liverDeformable.usd', 'bear.usd')",
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Create example
    example = Example(viewer, asset_name=args.asset)

    # Run example
    newton.examples.run(example)


