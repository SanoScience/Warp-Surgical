# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from hmac import new

import newton
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
import numpy as np
import warp as wp
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim._impl.newton_manager import NewtonManager

from ..asset_base import AssetBase
from .deformable_object_data import DeformableObjectData

if TYPE_CHECKING:
    from .deformable_object_cfg import DeformableObjectCfg

class DeformableObject(AssetBase):
    """A deformable object asset class.

    Deformable objects are assets that can be deformed in the simulation. They are typically used for
    soft bodies, such as stuffed animals and food items.

    Unlike rigid object assets, deformable objects have a more complex structure and require additional
    handling for simulation. The simulation of deformable objects follows a finite element approach, where
    the object is discretized into a mesh of nodes and elements. The nodes are connected by elements, which
    define the material properties of the object. The nodes can be moved and deformed, and the elements
    respond to these changes.

    The state of a deformable object comprises of its nodal positions and velocities, and not the object's root
    position and orientation. The nodal positions and velocities are in the simulation frame.

    Soft bodies can be `partially kinematic`_, where some nodes are driven by kinematic targets, and the rest are
    simulated. The kinematic targets are the desired positions of the nodes, and the simulation drives the nodes
    towards these targets. This is useful for partial control of the object, such as moving a stuffed animal's
    head while the rest of the body is simulated.
    
    .. note::
        **Physics Vertices vs Visual Points**: In Newton-based deformable bodies, there can be a distinction between
        physics **vertices** (used by the Newton solver for simulation) and visual **points** (used for USD mesh rendering).
        Newton typically uses more physics vertices for accurate simulation, while USD uses fewer visual points for efficient
        rendering. The class automatically handles downsampling from physics vertices to visual points via the USD attribute
        `mapping:pointToVertex`, which contains an array of (point_idx, vertex_idx) tuples mapping each visual point to its
        corresponding physics vertex. If this attribute is not present, a 1:1 identity mapping is assumed (physics vertices = 
        visual points).

    .. attention::
        This class is experimental and subject to change due to changes on the underlying PhysX API on which
        it depends. We will try to maintain backward compatibility as much as possible but some changes may be
        necessary.

    .. _partially kinematic: https://nvidia-omniverse.github.io/PhysX/physx/5.4.1/docs/SoftBodies.html#kinematic-soft-bodies
    """

    cfg: DeformableObjectCfg
    """Configuration instance for the deformable object."""

    def __init__(self, cfg: DeformableObjectCfg):
        """Initialize the deformable object.

        Args:
            cfg: A configuration instance.
        """
        # Initialize variables that will be used in the callback
        self._newton_body_ids = []  # For rigid body tracking (not used for deformables)
        self._newton_shape_indices = []  # Shape indices for deformable meshes
        self._mesh_prim_paths = []
        self._mesh_geoms = []
        
        # Mapping between Newton physics vertices and USD visual points
        self._point_to_vertex_mappings = []  # List of mappings, one per mesh instance
        self._vertex_to_point_mappings = []  # List of mappings, one per mesh instance
        
        super().__init__(cfg)
        
        # Register callback to get body IDs after Newton initializes
        NewtonManager.add_on_start_callback(self._get_newton_body_ids)
        
    """
    Properties
    """

    @property
    def data(self) -> DeformableObjectData:
        return self._data

    @property
    def num_instances(self) -> int:
        return self._num_instances

    @property
    def num_bodies(self) -> int:
        """Number of bodies in the asset.

        This is always 1 since each object is a single deformable body.
        """
        return 1

    @property
    def max_vertices_per_body(self) -> int:
        """The maximum number of visual points (mesh vertices) per deformable body."""
        return self._max_vertices_per_body

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None):
        # use ellipses object to skip initial indices.
        _env_ids: Sequence[int] | slice
        if env_ids is None:
            _env_ids = slice(None)
        else:
            _env_ids = env_ids
        # reset nodal state to default
        if self._data.nodal_pos_w is not None and self._data.default_nodal_state_w is not None:
            self._data.nodal_pos_w[_env_ids] = self._data.default_nodal_state_w[_env_ids, :, :3].clone()  # type: ignore
            self._data.nodal_vel_w[_env_ids] = self._data.default_nodal_state_w[_env_ids, :, 3:].clone()  # type: ignore

    def write_data_to_sim(self):
        """Write mesh vertex positions to USD for visualization."""
        # Sync physics state from Newton to USD (happens in update() as well)
        self._sync_physics_to_usd()

    def update(self, dt: float):
        self._data.update(dt)
        # Sync deformable mesh vertex positions from Newton to USD
        self._sync_physics_to_usd()

    """
    Operations - Write to simulation.
    """

    def write_nodal_state_to_sim(self, nodal_state: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the nodal state over selected environment indices into the simulation.

        The nodal state comprises of the nodal positions and velocities. Since these are nodes, the velocity only has
        a translational component. All the quantities are in the simulation frame.

        Args:
            nodal_state: Nodal state in simulation frame.
                Shape is (len(env_ids), max_vertices_per_body, 6).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # set into simulation
        self.write_nodal_pos_to_sim(nodal_state[..., :3], env_ids=env_ids)
        self.write_nodal_velocity_to_sim(nodal_state[..., 3:], env_ids=env_ids)

    def write_nodal_pos_to_sim(self, nodal_pos: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the nodal positions over selected environment indices into the simulation.

        The nodal position comprises of individual nodal positions (visual points) of the mesh 
        for the deformable body. The positions are in the simulation frame.

        Args:
            nodal_pos: Nodal positions (visual points) in simulation frame.
                Shape is (len(env_ids), max_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        _env_ids: Sequence[int] | slice
        if env_ids is None:
            _env_ids = slice(None)
        else:
            _env_ids = env_ids
        # set into internal buffers
        if self._data.nodal_pos_w is not None:
            self._data.nodal_pos_w[_env_ids] = nodal_pos.clone()  # type: ignore

    def write_nodal_velocity_to_sim(self, nodal_vel: torch.Tensor, env_ids: Sequence[int] | None = None):
        """Set the nodal velocity over selected environment indices into the simulation.

        The nodal velocity comprises of individual nodal velocities of the simulation mesh for the deformable
        body. Since these are nodes, the velocity only has a translational component. The velocities are in the
        simulation frame.

        Args:
            nodal_vel: Nodal velocities in simulation frame.
                Shape is (len(env_ids), max_vertices_per_body, 3).
            env_ids: Environment indices. If None, then all indices are used.
        """
        # resolve all indices
        _env_ids: Sequence[int] | slice
        if env_ids is None:
            _env_ids = slice(None)
        else:
            _env_ids = env_ids
        # set into internal buffers
        if self._data.nodal_vel_w is not None:
            self._data.nodal_vel_w[_env_ids] = nodal_vel.clone()  # type: ignore

    """
    Operations - Helper.
    """

    def transform_nodal_pos(
        self, nodal_pos: torch.tensor, pos: torch.Tensor | None = None, quat: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Transform the nodal positions based on the pose transformation.

        This function computes the transformation of the nodal positions based on the pose transformation.
        It multiplies the nodal positions with the rotation matrix of the pose and adds the translation.
        Internally, it calls the :meth:`isaaclab.utils.math.transform_points` function.

        Args:
            nodal_pos: The nodal positions (visual points) in the simulation frame. 
                Shape is (N, max_vertices_per_body, 3).
            pos: The position transformation. Shape is (N, 3).
                Defaults to None, in which case the position is assumed to be zero.
            quat: The orientation transformation as quaternion (w, x, y, z). Shape is (N, 4).
                Defaults to None, in which case the orientation is assumed to be identity.

        Returns:
            The transformed nodal positions (visual points). Shape is (N, max_vertices_per_body, 3).
        """
        # offset the nodal positions to center them around the origin
        mean_nodal_pos = nodal_pos.mean(dim=1, keepdim=True)
        nodal_pos = nodal_pos - mean_nodal_pos
        # transform the nodal positions based on the pose around the origin
        return math_utils.transform_points(nodal_pos, pos, quat) + mean_nodal_pos

    """
    Internal helper.
    """

    def _initialize_impl(self):
        # obtain the first prim in the regex expression (all others are assumed to be a copy of this)
        template_prim = sim_utils.find_first_matching_prim(self.cfg.prim_path)
        if template_prim is None:
            raise RuntimeError(f"Failed to find prim for expression: '{self.cfg.prim_path}'.")
        template_prim_path = template_prim.GetPath().pathString

        # find mesh prims (look for UsdGeom.Mesh prims)
        stage = get_current_stage()
        mesh_prims = sim_utils.get_all_matching_child_prims(
            template_prim_path,
            predicate=lambda prim: prim.IsA(UsdGeom.Mesh),  # type: ignore
        )
        
        if len(mesh_prims) == 0:
            raise RuntimeError(
                f"Failed to find a mesh when resolving '{self.cfg.prim_path}'."
                " Please ensure that the prim contains a UsdGeom.Mesh."
            )
        
        # Use the first mesh found
        mesh_prim = mesh_prims[0]
        mesh_prim_path = mesh_prim.GetPath().pathString
        
        # Note: _mesh_prim_paths, _mesh_geoms, _newton_body_ids already initialized in __init__()
        # Clear them here in case _initialize_impl is called multiple times
        self._mesh_prim_paths.clear()
        self._mesh_geoms.clear()
        self._point_to_vertex_mappings.clear()
        self._vertex_to_point_mappings.clear()
        # Don't clear _newton_body_ids as it was populated by the callback!
        
        # Find all mesh instances across environments
        all_matching_prims = sim_utils.find_matching_prims(self.cfg.prim_path)
        self._num_instances = len(all_matching_prims)
        
        for prim in all_matching_prims:
            # Get the prim path as a string
            prim_path_str = prim.GetPath().pathString
            # Find mesh in this instance
            instance_mesh_prims = sim_utils.get_all_matching_child_prims(
                prim_path_str,  # Now passing a string, not a Usd.Prim
                predicate=lambda prim: prim.IsA(UsdGeom.Mesh),  # type: ignore
            )
            if len(instance_mesh_prims) > 0:
                mesh_path = instance_mesh_prims[0].GetPath().pathString
                self._mesh_prim_paths.append(mesh_path)
                mesh_prim = stage.GetPrimAtPath(mesh_path)
                self._mesh_geoms.append(UsdGeom.Mesh(mesh_prim))  # type: ignore
                
                # Read vertex/point mapping attributes if they exist
                # These mappings handle the difference between physics vertices (Newton) and visual points (USD)
                point_to_vertex_attr = mesh_prim.GetAttribute("mapping:pointToVertex")
                vertex_to_point_attr = mesh_prim.GetAttribute("mapping:vertexToPoint")
                
                if point_to_vertex_attr and point_to_vertex_attr.Get() is not None:
                    point_to_vertex = np.array(point_to_vertex_attr.Get(), dtype=np.int32)
                    # This should be a 2D array of (point_idx, vertex_idx) tuples
                    # Reshape to ensure it's (N, 2)
                    if point_to_vertex.ndim == 1:
                        point_to_vertex = point_to_vertex.reshape(-1, 2)
                    self._point_to_vertex_mappings.append(point_to_vertex)
                    omni.log.info(f"Loaded point-to-vertex mapping for {mesh_path}: shape={point_to_vertex.shape}, {len(point_to_vertex)} entries")
                else:
                    # No mapping, assume 1:1 (identity mapping)
                    self._point_to_vertex_mappings.append(None)
                    omni.log.info(f"No point-to-vertex mapping found for {mesh_path}, using identity mapping")
                
                if vertex_to_point_attr and vertex_to_point_attr.Get() is not None:
                    vertex_to_point = np.array(vertex_to_point_attr.Get(), dtype=np.int32)
                    # This should be a 2D array of (vertex_idx, point_idx) tuples
                    # Reshape to ensure it's (N, 2)
                    if vertex_to_point.ndim == 1:
                        vertex_to_point = vertex_to_point.reshape(-1, 2)
                    self._vertex_to_point_mappings.append(vertex_to_point)
                    omni.log.info(f"Loaded vertex-to-point mapping for {mesh_path}: shape={vertex_to_point.shape}, {len(vertex_to_point)} entries")
                else:
                    # No mapping, assume 1:1 (identity mapping)
                    self._vertex_to_point_mappings.append(None)
                    omni.log.info(f"No vertex-to-point mapping found for {mesh_path}, using identity mapping")

        # Get mesh vertex count and topology from the first mesh
        mesh_geom = UsdGeom.Mesh(mesh_prim)  # type: ignore
        points_attr = mesh_geom.GetPointsAttr()
        points = points_attr.Get()
        
        # Use the number of visual points from USD for our buffer size
        self._max_vertices_per_body = len(points)
        omni.log.info(f"Using {self._max_vertices_per_body} visual points from USD mesh")
        
        # Store mesh topology for Newton builder
        face_vertex_counts = mesh_geom.GetFaceVertexCountsAttr().Get()
        face_vertex_indices = mesh_geom.GetFaceVertexIndicesAttr().Get()
        self._mesh_faces = []
        idx = 0
        for count in face_vertex_counts:
            if count == 3:  # Triangle
                self._mesh_faces.append([
                    int(face_vertex_indices[idx]),
                    int(face_vertex_indices[idx + 1]),
                    int(face_vertex_indices[idx + 2])
                ])
            idx += count

        # log information about the deformable body
        omni.log.info(f"Deformable body initialized at: {self.cfg.prim_path}")
        omni.log.info(f"  Instances: {self.num_instances}, Vertices: {self._max_vertices_per_body}, Faces: {len(self._mesh_faces)}")
        
        # container for data access
        self._data = DeformableObjectData(
            num_instances=self._num_instances, 
            max_vertices_per_body=self._max_vertices_per_body, 
            device=self.device
        )

        # create buffers
        self._create_buffers()
        
        # update the deformable body data
        self.update(0.0)

        # Initialize debug visualization handle
        if self._debug_vis_handle is None:
            # set initial state of debug visualization
            self.set_debug_vis(self.cfg.debug_vis)

    def _create_buffers(self):
        """Create buffers for storing data."""
        # constants
        self._ALL_INDICES = torch.arange(self.num_instances, dtype=torch.long, device=self.device)

        # Read initial mesh positions from USD
        stage = get_current_stage()
        nodal_positions = torch.zeros(
            (self.num_instances, self._max_vertices_per_body, 3), 
            dtype=torch.float32, 
            device=self.device
        )
        
        for i, mesh_geom in enumerate(self._mesh_geoms):
            points = mesh_geom.GetPointsAttr().Get()
            if points is not None:
                # Convert USD points to numpy array
                # These are visual points used for rendering - store them directly
                points_array = np.array(points, dtype=np.float32)
                nodal_positions[i] = torch.tensor(points_array, dtype=torch.float32, device=self.device)

        # default state
        # we use the initial nodal positions at spawn time as the default state
        nodal_velocities = torch.zeros_like(nodal_positions)
        self._data.default_nodal_state_w = torch.cat((nodal_positions, nodal_velocities), dim=-1)

        # Initialize current state
        self._data.nodal_pos_w = nodal_positions.clone()
        self._data.nodal_vel_w = nodal_velocities.clone()
    
    def _rotation_matrix_to_quaternion(self, R):
        """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
        trace = np.trace(R)
        if trace > 0:
            s = 0.5 / np.sqrt(trace + 1.0)
            w = 0.25 / s
            x = (R[2, 1] - R[1, 2]) * s
            y = (R[0, 2] - R[2, 0]) * s
            z = (R[1, 0] - R[0, 1]) * s
        else:
            if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
                w = (R[2, 1] - R[1, 2]) / s
                x = 0.25 * s
                y = (R[0, 1] + R[1, 0]) / s
                z = (R[0, 2] + R[2, 0]) / s
            elif R[1, 1] > R[2, 2]:
                s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
                w = (R[0, 2] - R[2, 0]) / s
                x = (R[0, 1] + R[1, 0]) / s
                y = 0.25 * s
                z = (R[1, 2] + R[2, 1]) / s
            else:
                s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
                w = (R[1, 0] - R[0, 1]) / s
                x = (R[0, 2] + R[2, 0]) / s
                y = (R[1, 2] + R[2, 1]) / s
                z = 0.25 * s
        return (w, x, y, z)
    
    def _get_newton_body_ids(self):
        """Query Newton soft body indices for our deformable prims.
        
        This callback runs after Newton has finalized the model.
        We use the soft_mesh_key attribute to directly map USD paths to soft body indices.
        """
        # Get the Newton model
        model = NewtonManager.get_model()
        if model is None:
            omni.log.warn("Cannot get soft body IDs: Newton model is None")
            return
        
        # Check if soft_mesh_key exists (requires Newton library modification)
        if not hasattr(model, 'soft_mesh_key'):
            omni.log.error("Newton model does not have 'soft_mesh_key' attribute. Please update Newton library.")
            return
        
        if not hasattr(model, 'soft_mesh_particle_range'):
            omni.log.error("Newton model does not have 'soft_mesh_particle_range' attribute. Please update Newton library.")
            return
        
        # Clear existing shape indices
        self._newton_shape_indices.clear()
        
        # Match our mesh prims with Newton's soft_mesh_key
        for mesh_idx, mesh_prim_path in enumerate(self._mesh_prim_paths):
            found = False
            
            # Try exact match first
            for newton_idx, soft_mesh_path in enumerate(model.soft_mesh_key):
                if mesh_prim_path == soft_mesh_path:
                    self._newton_shape_indices.append(newton_idx)
                    particle_range = model.soft_mesh_particle_range[newton_idx]
                    num_particles = particle_range[1] - particle_range[0]
                    omni.log.info(
                        f"Matched deformable mesh: {mesh_prim_path} -> "
                        f"Newton index {newton_idx} ({num_particles} particles)"
                    )
                    found = True
                    break
            
            # If exact match failed, try partial matching
            if not found:
                for newton_idx, soft_mesh_path in enumerate(model.soft_mesh_key):
                    if mesh_prim_path.startswith(soft_mesh_path + "/") or soft_mesh_path.startswith(mesh_prim_path + "/"):
                        self._newton_shape_indices.append(newton_idx)
                        particle_range = model.soft_mesh_particle_range[newton_idx]
                        num_particles = particle_range[1] - particle_range[0]
                        omni.log.info(
                            f"Matched deformable mesh (partial): {mesh_prim_path} <-> {soft_mesh_path} -> "
                            f"Newton index {newton_idx} ({num_particles} particles)"
                        )
                        found = True
                        break
            
            # Try matching just the mesh name
            if not found:
                mesh_name = mesh_prim_path.split("/")[-1]
                for newton_idx, soft_mesh_path in enumerate(model.soft_mesh_key):
                    newton_mesh_name = soft_mesh_path.split("/")[-1]
                    if mesh_name == newton_mesh_name:
                        self._newton_shape_indices.append(newton_idx)
                        particle_range = model.soft_mesh_particle_range[newton_idx]
                        num_particles = particle_range[1] - particle_range[0]
                        omni.log.info(
                            f"Matched deformable mesh (by name): {mesh_prim_path} <-> {soft_mesh_path} -> "
                            f"Newton index {newton_idx} ({num_particles} particles)"
                        )
                        found = True
                        break
            
            if not found:
                omni.log.warn(f"Deformable mesh not found in Newton model: {mesh_prim_path}")
        
        if len(self._newton_shape_indices) == 0:
            omni.log.warn(f"No deformable meshes matched for {self.cfg.prim_path}")
        else:
            omni.log.info(f"Successfully matched {len(self._newton_shape_indices)} deformable meshes")
    
    def _sync_physics_to_usd(self):
        """Sync deformable mesh vertex positions from Newton physics to USD.
        
        Important: Newton uses physics **vertices** while USD uses visual **points**.
        These may differ if there's a vertex-to-point mapping.
        
        Newton transforms the original USD vertices by: p = wp.quat_rotate(rot, v * scale) + pos
        We need to reverse this transformation: v = wp.quat_rotate(wp.quat_inverse(rot), p - pos) / scale
        """
        if not self._newton_shape_indices:
            return
        
        # Get current state and model from Newton
        state = NewtonManager.get_state_0()
        model = NewtonManager.get_model()
        if state is None or model is None:
            return
        
        # Check if particle positions are available
        if not hasattr(state, 'particle_q'):
            return
        
        # Check if soft mesh transform data is available
        has_transforms = (hasattr(model, 'soft_mesh_X') and 
                         hasattr(model, 'soft_mesh_q') and 
                         hasattr(model, 'soft_mesh_scale'))
        
        # Convert particle positions from warp to numpy
        # particle_q is a flat array of all soft body particle positions (x, y, z)
        # These are PHYSICS VERTICES from Newton solver (already transformed)
        particle_positions = wp.to_torch(state.particle_q).cpu().numpy()

        # Update each deformable mesh instance
        if self._data.nodal_pos_w is not None:
            for mesh_idx, newton_idx in enumerate(self._newton_shape_indices):
                # Get the particle range for this deformable body
                particle_range = model.soft_mesh_particle_range[newton_idx]
                start_idx = particle_range[0]
                end_idx = particle_range[1]
                # Extract Newton particle positions (physics vertices - transformed)
                newton_particle_positions = particle_positions[start_idx:end_idx]
                
                # NOTE: For deformable meshes, all transforms are pre-applied to vertices,
                # so Newton uses identity transform. No inverse transformation needed!
                # The Newton particle positions are already in the correct USD world space.
                
                # Convert Newton physics vertices to visual points
                if self._point_to_vertex_mappings[mesh_idx] is not None:
                    # Downsample from many Newton physics vertices to fewer visual points
                    # point_to_vertex is (N, 2) array where each row is (point_idx, vertex_idx)
                    point_to_vertex = self._point_to_vertex_mappings[mesh_idx]
                    # Extract the vertex indices (second column)
                    vertex_indices = point_to_vertex[:, 1]
                    # Use these indices to select the corresponding Newton particles
                    visual_points = newton_particle_positions[vertex_indices]
                else:
                    # No mapping: Newton particles are 1:1 with visual points
                    visual_points = newton_particle_positions
                
                # Update internal buffer with visual points
                self._data.nodal_pos_w[mesh_idx] = torch.from_numpy(visual_points).to(self.device)
                
                # Convert to USD format (list of Gf.Vec3f)
                usd_points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in visual_points]  # type: ignore
                
                # Update the mesh geometry in USD
                try:
                    self._mesh_geoms[mesh_idx].GetPointsAttr().Set(usd_points)
                except Exception as e:
                    omni.log.error(f"Failed to update mesh vertices for {self._mesh_prim_paths[mesh_idx]}: {e}")
    
    def _update_mesh_visualization(self):
        """Update USD mesh vertices for visualization.
        
        Note: nodal_pos_w always stores visual points that can be directly used for USD rendering.
        """
        if self._data.nodal_pos_w is None:
            return
        # Update each mesh instance
        for i, mesh_geom in enumerate(self._mesh_geoms):
            # Get visual point positions from internal buffer
            visual_points = self._data.nodal_pos_w[i].cpu().numpy()
            
            # Convert to USD format (list of Gf.Vec3f)
            usd_points = [Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in visual_points]  # type: ignore
            # Update the mesh
            mesh_geom.GetPointsAttr().Set(usd_points)

    """
    Internal simulation callbacks.
    """

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            if not hasattr(self, "target_visualizer"):
                self.target_visualizer = VisualizationMarkers(self.cfg.visualizer_cfg)
            # set their visibility to true
            self.target_visualizer.set_visibility(True)
        else:
            if hasattr(self, "target_visualizer"):
                self.target_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Visualize some vertices (e.g., first 10 vertices of first instance)
        if self._data.nodal_pos_w is not None and self.num_instances > 0:
            num_vis = min(10, self._max_vertices_per_body)
            if self.data.nodal_pos_w is not None:  # Additional check for mypy
                positions = self.data.nodal_pos_w[0, :num_vis, :].reshape(-1, 3)
        # show target visualizer
        self.target_visualizer.visualize(positions)

    def _invalidate_initialize_callback(self, event):
        """Invalidates the scene elements."""
        # call parent
        super()._invalidate_initialize_callback(event)