# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch

import isaaclab.utils.math as math_utils


class DeformableObjectData:
    """Data container for a deformable object.

    This class contains the data for a deformable object in the simulation. The data includes the nodal states of
    the deformable body in the object. The data is stored in the simulation world frame unless otherwise specified.

    For Newton-based deformable bodies, we store the **visual point** positions and velocities, which are used
    for USD mesh rendering. These are synced from Newton's physics vertices (which may be more numerous) using
    a downsampling mapping if one exists.
    """

    def __init__(self, num_instances: int, max_vertices_per_body: int, device: str):
        """Initializes the deformable object data.

        Args:
            num_instances: Number of deformable object instances.
            max_vertices_per_body: Maximum number of vertices per deformable body.
            device: The device used for processing.
        """
        # Set the parameters
        self.device = device
        self._num_instances = num_instances
        self._max_vertices_per_body = max_vertices_per_body

        # Set initial time stamp
        self._sim_timestamp = 0.0

        # Initialize state buffers (not lazy, directly accessible)
        # Note: These store VISUAL POINTS for USD rendering
        self.nodal_pos_w: torch.Tensor | None = None
        """Nodal positions (visual points) in simulation world frame. Shape is (num_instances, max_vertices_per_body, 3)."""
        
        self.nodal_vel_w: torch.Tensor | None = None
        """Nodal velocities (visual points) in simulation world frame. Shape is (num_instances, max_vertices_per_body, 3)."""

    def update(self, dt: float):
        """Updates the data for the deformable object.

        Args:
            dt: The time step for the update. This must be a positive value.
        """
        # update the simulation timestamp
        self._sim_timestamp += dt

    ##
    # Defaults.
    ##

    default_nodal_state_w: torch.Tensor | None = None
    """Default nodal state (visual points) ``[nodal_pos, nodal_vel]`` in simulation world frame.
    Shape is (num_instances, max_vertices_per_body, 6).
    """

    ##
    # Properties.
    ##

    @property
    def nodal_state_w(self):
        """Nodal state (visual points) ``[nodal_pos, nodal_vel]`` in simulation world frame.
        Shape is (num_instances, max_vertices_per_body, 6).
        """
        if self.nodal_pos_w is not None and self.nodal_vel_w is not None:
            return torch.cat((self.nodal_pos_w, self.nodal_vel_w), dim=-1)
        return None

    ##
    # Derived properties.
    ##

    @property
    def root_pos_w(self) -> torch.Tensor | None:
        """Root position from nodal positions of the mesh for the deformable bodies in simulation world frame.
        Shape is (num_instances, 3).

        This quantity is computed as the mean of the nodal positions.
        """
        if self.nodal_pos_w is not None:
            return self.nodal_pos_w.mean(dim=1)
        return None

    @property
    def root_vel_w(self) -> torch.Tensor | None:
        """Root velocity from vertex velocities for the deformable bodies in simulation world frame.
        Shape is (num_instances, 3).

        This quantity is computed as the mean of the nodal velocities.
        """
        if self.nodal_vel_w is not None:
            return self.nodal_vel_w.mean(dim=1)
        return None