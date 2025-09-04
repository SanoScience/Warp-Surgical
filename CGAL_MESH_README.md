# CGAL Tetrahedral Mesh Generation for WarpSim

This implementation provides functionality to generate tetrahedral meshes from segmented images using CGAL, following the approach in CGAL's `mesh_3D_weighted_image.cpp` example.

## Overview

The implementation consists of three main modules:

1. **`cgal_mesh_generator.py`** - Core CGAL mesh generation functionality
2. **`cgal_to_warp.py`** - Format conversion from CGAL to WarpSim-compatible format
3. **`cgal_mesh_demo.py`** - Demonstration script with command-line interface

## Installation

### Dependencies

First, install the CGAL Python bindings. The implementation supports both official CGAL bindings and pygalmesh as a fallback:

```bash
# Option 1: Official CGAL Python bindings (recommended, 2025 release)
pip install cgal

# Option 2: Alternative pygalmesh (fallback)
pip install pygalmesh

# Additional dependencies (should be automatically installed)
pip install numpy pillow scipy
```

### Update Project Dependencies

The required dependencies have been added to `pyproject.toml`:

```bash
uv sync
```

## Usage

### Command Line Interface

The `cgal_mesh_demo.py` script provides a complete command-line interface for mesh generation:

#### Generate Mesh from Segmented Image
```bash
python cgal_mesh_demo.py --input path/to/segmented.inr --output liver_cgal --visualize
```

#### Create Test Volume and Generate Mesh
```bash
python cgal_mesh_demo.py --create-test-volume --output test_mesh --visualize
```

#### Create Simple Test Mesh (no CGAL required)
```bash
python cgal_mesh_demo.py --simple-mesh --output simple --visualize
```

### Mesh Quality Parameters

Control mesh generation quality with these parameters:

```bash
python cgal_mesh_demo.py --create-test-volume --output test_mesh \
    --facet-angle 30.0 \
    --facet-size 6.0 \
    --facet-distance 0.5 \
    --cell-radius-edge-ratio 3.0 \
    --cell-size 8.0 \
    --visualize
```

### Advanced Options

```bash
# Custom smoothing parameter
python cgal_mesh_demo.py --input image.inr --output result --sigma 2.0

# Adjust domain error tolerance
python cgal_mesh_demo.py --input image.inr --output result --relative-error-bound 1e-8

# Verbose output
python cgal_mesh_demo.py --create-test-volume --output test --verbose

# Visualization without haptic device
python cgal_mesh_demo.py --simple-mesh --output simple --visualize --no-haptic
```

## Programmatic Usage

### Basic Mesh Generation

```python
from cgal_mesh_generator import CgalMeshGenerator
from cgal_to_warp import CgalToWarpConverter

# Create generator
generator = CgalMeshGenerator()

# Load segmented image
generator.load_segmented_image("path/to/segmented.inr")

# Generate label weights for smooth boundaries
generator.generate_label_weights()

# Create mesh domain
generator.create_mesh_domain()

# Set mesh quality criteria
criteria = {
    'facet_angle': 30.0,
    'facet_size': 6.0,
    'cell_size': 8.0
}
generator.set_mesh_criteria(criteria)

# Generate mesh
generator.generate_mesh()

# Convert to WarpSim format
converter = CgalToWarpConverter()
converter.convert_cgal_mesh(generator.generated_mesh, "meshes/output", "model")
```

### Create Test Data

```python
from cgal_mesh_generator import create_sample_segmented_image
from cgal_to_warp import create_simple_test_mesh

# Create sample segmented volume
create_sample_segmented_image("test_volume.npy", size=32)

# Create simple test mesh without CGAL
create_simple_test_mesh("meshes/simple_test", "model")
```

## Output Format

Generated meshes are saved in WarpSim-compatible format:

- `model.vertices` - Vertex coordinates (x, y, z per line)
- `model.tetras` - Tetrahedron indices (4 vertex indices per line)
- `model.tris` - Surface triangle indices (3 vertex indices per line)
- `model.edges` - Edge connectivity (2 vertex indices per line)
- `model.uvs` - UV texture coordinates (u, v per line)

## Integration with WarpSim

### Using Generated Meshes

1. Generate mesh using the demo script
2. The mesh files are saved in `meshes/[output_name]/`
3. Modify `mesh_loader.py` to load your custom mesh, or use the generated demo script

### Custom Demo Script

When you generate a mesh with `--visualize`, a custom demo script is created:

```bash
python cgal_demo_[mesh_name].py
```

This script loads your generated mesh in WarpSim with full physics simulation and haptic support.

## Mesh Quality Parameters

Understanding the mesh quality parameters from CGAL:

- **`facet_angle`** (default: 30°) - Minimum angle in surface triangles
- **`facet_size`** (default: 6.0) - Maximum size of surface triangles
- **`facet_distance`** (default: 0.5) - Maximum distance from facet to domain boundary
- **`cell_radius_edge_ratio`** (default: 3.0) - Maximum radius-to-shortest-edge ratio for tetrahedra
- **`cell_size`** (default: 8.0) - Maximum size of tetrahedra

Smaller values create higher quality but denser meshes.

## Supported Image Formats

The implementation supports:
- **INR format** (native CGAL format, recommended)
- **Numpy arrays** (`.npy` files)
- **DICOM** (through PIL/Pillow)
- **Other formats** supported by PIL

For best results with CGAL, use INR format segmented images where different tissue types are labeled with integer values (0=background, 1=tissue1, 2=tissue2, etc.).

## Troubleshooting

### CGAL Not Available
If CGAL bindings are not installed, the demo falls back to simple mesh generation:
```bash
python cgal_mesh_demo.py --simple-mesh --output test
```

### Missing Dependencies
```bash
pip install cgal numpy pillow scipy
# or
pip install pygalmesh numpy pillow scipy
```

### Memory Issues with Large Images
For large segmented images, adjust mesh parameters to create coarser meshes:
```bash
python cgal_mesh_demo.py --input large_image.inr --output result \
    --cell-size 16.0 --facet-size 12.0
```

### Haptic Device Issues
Run without haptic support:
```bash
python cgal_mesh_demo.py --simple-mesh --output test --visualize --no-haptic
```

## Examples

### Example 1: Liver Segmentation
```bash
# Generate liver mesh from segmented CT data
python cgal_mesh_demo.py --input liver_segmented.inr --output liver_physics \
    --facet-angle 25 --cell-size 6.0 --visualize
```

### Example 2: Multi-organ Segmentation
```bash
# Create mesh from multi-organ segmentation with smooth boundaries
python cgal_mesh_demo.py --input multi_organ.inr --output organs \
    --sigma 1.5 --facet-distance 0.3 --visualize
```

### Example 3: High-Quality Small Mesh
```bash
# Generate high-quality mesh for detailed simulation
python cgal_mesh_demo.py --input small_organ.inr --output high_quality \
    --facet-angle 35 --cell-size 4.0 --cell-radius-edge-ratio 2.0 --visualize
```

## Architecture

The implementation follows the CGAL `mesh_3D_weighted_image.cpp` example:

1. **Image Loading** - Load segmented 3D images
2. **Weight Generation** - Create smoothed label weights using `generate_label_weights()`
3. **Domain Creation** - Create labeled mesh domain with error bounds
4. **Mesh Criteria** - Set quality constraints for facets and cells
5. **Mesh Generation** - Generate tetrahedral mesh using `make_mesh_3()`
6. **Format Conversion** - Convert to WarpSim format for physics simulation

This provides high-quality tetrahedral meshes suitable for surgical simulation with proper boundary representation and multi-material support.