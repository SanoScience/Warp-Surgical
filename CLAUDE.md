# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses `uv` for dependency management. Common commands:

- **Install dependencies**: `uv sync` 
- **Run main simulation**: `python main.py` (interactive OpenGL mode)
- **Run simulation with USD output**: `python main.py --usd --stage_path output.usd --num_frames 300`
- **Run specific haptic tests**: `python legacy/ws-openhaptics_test.py`

## Architecture Overview

This is a surgical simulation framework built on Warp (NVIDIA's Python physics simulation toolkit) with haptic device integration.

### Core Components

- **main.py**: Entry point that orchestrates the simulation loop, handling both real-time interactive mode and offline USD rendering
- **warp_simulation.py (WarpSim)**: Main simulation class that manages the physics world, mesh loading, and rendering pipeline
- **PBDSolver.py**: Custom Position-Based Dynamics solver extending Newton's XPBDSolver with surgical-specific constraints
- **haptic_device.py (HapticController)**: OpenHaptics integration for real-time haptic feedback and input
- **mesh_loader.py**: Tetrahedral mesh loading system that parses custom mesh formats (vertices, tetrahedra, triangles, connectors)

### Simulation Pipeline

1. **Mesh Loading**: Custom tetrahedral meshes are loaded from the `meshes/` directory (liver, gallbladder, fat tissues)
2. **Physics Setup**: Newton physics models are created with particles, tetrahedra, and constraint connections
3. **Haptic Integration**: Real-time haptic device position updates influence simulation through sphere collisions
4. **Rendering**: Dual rendering pipeline supports both real-time OpenGL visualization and offline USD output

### Key Algorithms

- **Position-Based Dynamics**: Volume preservation constraints for soft body simulation
- **Tetrahedra Surface Extraction**: Dynamic surface mesh generation from active tetrahedra for rendering
- **Haptic Collision**: Sphere-based collision detection between haptic device and soft tissues
- **Connector Constraints**: Custom constraint system linking particles to triangle surfaces for tissue connections

### Dependencies

- **warp-lang**: NVIDIA's GPU-accelerated physics simulation framework (local editable dependency)
- **newton-physics**: Extended physics solver framework (local editable dependency)
- **pyOpenHaptics**: Interface to haptic devices for force feedback
- **pyglet**: OpenGL rendering and window management
- **usd-core**: Universal Scene Description for offline rendering
- **trimesh**: 3D mesh processing library

### File Structure

- `meshes/`: Contains organ mesh data (vertices, tetrahedra, triangles, connectors) and centrelines
- `textures/`: Organ texture files with different damage states (base, blood, coagulation, damage)
- `assets/`: Robot models (Franka, MIRA, STAR, dVRK) and surgical instruments
- `haptics/`: Haptic device interaction examples and utilities
- `legacy/`: Legacy code and test files
- `simulation_kernels.py`: Custom Warp kernels for surgical simulation constraints
- `collision_kernels.py`: Specialized collision detection kernels
- `render_*.py`: Multiple rendering backends (OpenGL, SurgSim-style, custom)

### Specialized Modules

- **centrelines.py**: Vascular structure simulation with bleeding and clamping
- **grasping.py**: Tissue grasping and manipulation mechanics
- **heating.py**: Electrocautery simulation with heat conduction and coagulation
- **stretching.py**: Tissue stretching and tearing mechanics
- **robot_simulation.py**: Robotic arm integration and inverse kinematics
- **robot_ik.py**: Advanced inverse kinematics solver for surgical robots
- **surface_reconstruction.py**: Dynamic surface mesh generation from volume data

### Simulation Modes

- **Interactive Mode**: Real-time simulation with haptic device input and OpenGL visualization
- **Offline Mode**: Batch processing with USD file output for high-quality rendering pipelines