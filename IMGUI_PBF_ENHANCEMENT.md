# ImGui Enhanced PBF Simulation

## Overview

The original `example_pbf_opengl.py` has been enhanced with a comprehensive ImGui interface for real-time parameter control and debugging. This provides an intuitive way to experiment with PBF simulation parameters and understand their effects.

## New Features

### 🎛️ **Real-time Parameter Control**
- **Timestep Slider**: Adjust simulation timestep from 1/240s to 1/30s
- **Constraint Iterations**: Control solver accuracy (1-12 iterations)
- **Damping Control**: Fine-tune velocity damping (0.5-0.99)
- **Gravity Adjustment**: Modify gravitational force (-20 to -1 m/s²)
- **Particle Properties**: Adjust particle radius and smoothing radius
- **Rest Density**: Control target fluid density (500-2000 kg/m³)
- **Domain Bounds**: Modify simulation boundaries with 3D vector controls

### 🔧 **Quick Preset Buttons**
- **Conservative**: Stable settings for reliable simulation
- **Fast**: Higher performance settings for real-time interaction  
- **High Quality**: Maximum accuracy settings for best visual results

### 📊 **Debug Information Display**
- **Particle Count**: Current number of active particles
- **Simulation Status**: Running/Paused state
- **Average Velocity**: Real-time velocity monitoring
- **Maximum Velocity**: Explosion detection aid
- **Particle Spread**: Spatial distribution analysis

### 📈 **Performance Monitoring**
- **Frame Rate**: Real-time FPS display
- **Frame Time**: Millisecond timing for optimization

### 🎯 **Simulation Control**
- **Pause/Resume**: Interactive simulation control
- **Reset Particles**: Restart with new initial conditions
- **Reset Parameters**: Restore original settings

## Technical Implementation

### ImGui Integration
```python
class PBFImGuiManager(ImGuiManager):
    def __init__(self, renderer, simulation):
        super().__init__(renderer)
        self.sim = simulation
        # UI setup and parameter storage
        
    def draw_ui(self):
        # Comprehensive parameter control interface
        # Real-time debug information
        # Performance monitoring
```

### Enhanced Renderer
```python
class OpenGLRenderer:
    def __init__(self, simulation, use_imgui=True):
        # Initialize Warp renderer for ImGui support
        self.warp_renderer = warp.render.OpenGLRenderer(vsync=False)
        self.imgui_manager = PBFImGuiManager(self.warp_renderer, simulation)
        
    def render(self):
        # Integrated 3D scene + ImGui rendering
        self.warp_renderer.begin_frame(time.time())
        # ... 3D rendering ...
        self.warp_renderer.end_frame()  # Renders ImGui
```

## Usage

### Running the Enhanced Simulation
```bash
python example_pbf_opengl.py
```

### Requirements
- **Core**: `warp-lang`, `numpy`, `PyOpenGL`
- **ImGui**: `warp` with render support (automatically detected)

### Fallback Behavior
If ImGui is not available, the simulation automatically falls back to:
- Basic keyboard controls (Space, R, Escape)
- Simple text overlay for status
- Same core PBF functionality

## Parameter Guide

### Stability Parameters
- **Timestep**: Lower values = more stable (try 1/150s for stability)
- **Iterations**: Higher values = better convergence (6-8 recommended)
- **Damping**: Higher values = more stable (0.95-0.97 recommended)

### Quality Parameters  
- **Smoothing Radius**: Controls particle interaction range
- **Rest Density**: Target fluid density (1000 kg/m³ = water)
- **Constraint Epsilon**: Solver regularization (higher = more stable)

### Performance Parameters
- **Grid Dimensions**: Spatial hash resolution (64³ default)
- **Particle Count**: Affects both quality and performance

## Troubleshooting

### ImGui Not Available
```
ImGui not available - using basic controls only
To enable ImGui: ensure warp.render module is available
```
**Solution**: ImGui requires the full Warp installation with render support.

### Parameter Instability
**Symptoms**: Particles exploding or erratic behavior
**Solutions**: 
- Use "Conservative" preset
- Reduce timestep
- Increase damping
- Reduce gravity

### Performance Issues
**Symptoms**: Low FPS, high frame times
**Solutions**:
- Use "Fast" preset
- Reduce particle count
- Decrease constraint iterations

## Key Improvements Over Original

1. **Interactive Parameter Tuning**: No need to restart simulation to change parameters
2. **Real-time Debugging**: Monitor simulation health and performance
3. **User-Friendly Interface**: Visual sliders instead of code modifications
4. **Stability Presets**: Quick access to proven parameter combinations
5. **Graceful Degradation**: Works with or without ImGui support

## Files Modified

- **`example_pbf_opengl.py`**: Enhanced with ImGui integration
- **`test_imgui_integration.py`**: Test suite for ImGui functionality
- **`IMGUI_PBF_ENHANCEMENT.md`**: This documentation

## Future Enhancements

Potential additions:
- **Particle Visualization Options**: Color coding by velocity, density, etc.
- **Recording/Playback**: Save and replay simulation states
- **Advanced Presets**: Scene-specific parameter sets
- **Performance Profiling**: Detailed timing breakdown
- **Parameter Curves**: Time-based parameter animation