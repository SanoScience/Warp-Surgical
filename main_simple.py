import argparse
import os
from pathlib import Path
import warp as wp

#from haptic_device import HapticController
from warp_simulation_simple import WarpSim

# Optional ImGui overlay using Warp's ImGuiManager (best-integrated path)
try:
    from warp.render.imgui_manager import ImGuiManager as _WarpImGuiManager
    _IMGUI_AVAILABLE = True
except Exception:
    _IMGUI_AVAILABLE = False

def parse_arguments():
    """Parse command line arguments."""
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
    
    # Mesh generation arguments
    parser.add_argument("--generate-meshes", action="store_true", 
                       help="Generate tetrahedral meshes from source images automatically")
    parser.add_argument("--mesh-source", type=str, 
                       default="G:/warp/warp-surgical/liver-sliced.inr",
                       help="Source image path for mesh generation")
    parser.add_argument("--mesh-quality", choices=["fast", "low", "balanced", "high", "very_high", "ultra"], 
                       default="balanced", help="Mesh quality preset")
    parser.add_argument("--cell-size", type=float, 
                       help="Override cell size parameter for mesh generation")
    parser.add_argument("--force-regenerate", action="store_true", 
                       help="Force mesh regeneration even if mesh files exist")
    parser.add_argument("--multi-label", action="store_true", 
                       help="Generate separate meshes for each segmentation class")
    parser.add_argument("--detect-labels", action="store_true", 
                       help="Detect and display available labels in the source image")
    
    parser.add_argument("--warp-mesh-path", type=lambda x: None if x == "None" else str(x),
                       default=None, help="Optional path to a Warp mesh folder (.mesh or WarpSim folder)")

    # Mesh cache options
    parser.add_argument("--rebuild-mesh-cache", action="store_true", help="Rebuild mesh caches before loading")
    parser.add_argument("--no-mesh-cache", action="store_true", help="Disable mesh cache for this run")

    # Viewer/loader integration options
    parser.add_argument("--reconstruct-surface", dest="reconstruct_surface", action="store_true",
                       help="Reconstruct surface from tets (with consistent winding)")
    parser.add_argument("--no-reconstruct-surface", dest="reconstruct_surface", action="store_false",
                       help="Do not reconstruct surface; use provided triangles if present")
    parser.set_defaults(reconstruct_surface=True)

    parser.add_argument("--mesh-load-mode", choices=["single", "split_subdomains"], default="single",
                       help="Load a single mesh (colored by subdomain IDs) or split into per-subdomain meshes")
    parser.add_argument("--flip-winding", action="store_true",
                       help="Force flip triangle winding after reconstruction (debug/fallback)")
    parser.add_argument("--coloring", choices=["subdomain", "spatial", "none"], default="subdomain",
                       help="Vertex coloring mode for the surface mesh")

    return parser.parse_known_args()[0]

def detect_and_display_labels(args):
    """Detect and display available labels in the source image."""
    print("=" * 60)
    print("Detecting labels in source image...")
    
    try:
        from mesh_generation_simple import SimpleMeshGenerator
        
        generator = SimpleMeshGenerator()
        if not generator.warp_cgal_available:
            print("- warp-cgal not available for label detection")
            return False
        
        source_path = getattr(args, 'mesh_source', 'G:/warp/warp-surgical/liver-sliced.inr')
        labels = generator.detect_labels(source_path)
        
        if labels:
            print(f"+ Found {len(labels)} labels: {labels}")
            print("+ You can use --multi-label to generate separate meshes for each label")
            return True
        else:
            print("- No labels detected in the image")
            return False
            
    except Exception as e:
        print(f"- Error during label detection: {e}")
        return False

def run_simulation(args):
    """Run the main simulation loop."""
    mesh_path = getattr(args, "warp_mesh_path", None)

    if args.generate_meshes:
        from mesh_generation_simple import ensure_meshes_ready

        print("Checking/generating tetrahedral meshes...")
        generated_mesh_path = ensure_meshes_ready(args)
        if generated_mesh_path:
            mesh_path = generated_mesh_path
            print("Mesh generation/validation complete!")
        else:
            print("Failed to ensure meshes are ready. Attempting to continue with existing meshes...")

    if mesh_path:
        resolved_mesh_path = Path(mesh_path)
        if resolved_mesh_path.exists():
            print(f"Using Warp mesh assets from: {resolved_mesh_path}")
            mesh_path = str(resolved_mesh_path)
        else:
            print(f"Warning: Warp mesh path not found: {resolved_mesh_path}. Falling back to bundled meshes.")
            mesh_path = None

    # Configure warp-cgal viewer bridge via environment
    os.environ['WARP_CGAL_RECONSTRUCT_SURFACE'] = '1' if args.reconstruct_surface else '0'
    os.environ['WARP_CGAL_SPLIT_SUBDOMAINS'] = '1' if args.mesh_load_mode == 'split_subdomains' else '0'
    if args.flip_winding:
        os.environ['WARP_CGAL_FORCE_FLIP_WINDING'] = '1'
    # Mesh cache behavior
    if getattr(args, "no_mesh_cache", False):
        os.environ['MESH_CACHE_DISABLE'] = '1'
    if getattr(args, "rebuild_mesh_cache", False):
        os.environ['MESH_CACHE_REBUILD'] = '1'

    # Initialize haptic controller
    #haptic_controller = HapticController(scale=1.0)

    # Initialize simulation
    sim = WarpSim(
        stage_path=args.stage_path,
        num_frames=args.num_frames,
        use_opengl=not args.usd,
        mesh_path=mesh_path,
        coloring_mode=args.coloring,
    )

    # Helper: setup an ImGui overlay that draws inside the renderer window via Warp ImGuiManager
    if _IMGUI_AVAILABLE and not args.usd:
        try:
            setup_imgui_overlay(sim)
        except Exception:
            pass

    if args.usd:
        # Offline rendering mode
        for _ in range(args.num_frames):
            # Update haptic position
            #haptic_pos = haptic_controller.get_scaled_position()
            #sim.update_haptic_position(haptic_pos)

            sim.step()
            sim.render()
    else:
        # Real-time interactive mode
        while sim.is_running():
            # Get current haptic device position
            #haptic_pos = haptic_controller.get_scaled_position()
            #sim.update_haptic_position(haptic_pos)

            # Advance simulation
            sim.step()
            sim.render()

    # Save results
    #sim.save()

def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Handle label detection mode
    if args.detect_labels:
        detect_and_display_labels(args)
        return
    
    with wp.ScopedDevice(args.device):
        run_simulation(args)

if __name__ == "__main__":
    main()


# ---------------- ImGui helpers (overlay) ----------------
class _UIState:
    def __init__(self, sim: WarpSim):
        # Read initial parameters from simulation
        self.substeps = int(getattr(sim, 'sim_substeps', 16))
        self.iterations = int(getattr(getattr(sim, 'integrator', None), 'iterations', 5))
        # Distance constraints
        self.dist_stiffness = 1.0
        self.dist_damping = 0.2
        try:
            if hasattr(sim.model, 'spring_stiffness') and sim.model.spring_stiffness is not None and len(sim.model.spring_stiffness) > 0:
                self.dist_stiffness = float(sim.model.spring_stiffness.numpy()[0])
        except Exception:
            pass
        try:
            if hasattr(sim.model, 'spring_damping') and sim.model.spring_damping is not None and len(sim.model.spring_damping) > 0:
                self.dist_damping = float(sim.model.spring_damping.numpy()[0])
        except Exception:
            pass

        # Volume constraints
        self.vol_enabled = bool(getattr(getattr(sim, 'integrator', None), 'volCnstrs', True))
        self.vol_stiffness = float(getattr(getattr(sim, 'integrator', None), 'vol_stiffness', 0.1))


def setup_imgui_overlay(sim: WarpSim):
    if not hasattr(sim, 'renderer') or sim.renderer is None:
        return
    if not hasattr(sim.renderer, 'render_2d_callbacks'):
        return

    state = _UIState(sim)

    class _SimImGuiManager(_WarpImGuiManager):
        def __init__(self, renderer, sim_ref, state_ref):
            super().__init__(renderer)
            self.sim = sim_ref
            self.state = state_ref
            # pick nice defaults
            self.window_pos = (10, 10)
            self.window_size = (360, 260)

        def draw_ui(self):
            if not self.is_available:
                return

            imgui = self.imgui
            imgui.set_next_window_size(self.window_size[0], self.window_size[1], imgui.ONCE)
            imgui.set_next_window_position(self.window_pos[0], self.window_pos[1], imgui.ONCE)

            imgui.begin("Simulation Controls")

            changed = False
            # Substeps
            ok, val = imgui.slider_int("Substeps", int(self.state.substeps), 1, 64)
            if ok and val != self.state.substeps:
                self.state.substeps = int(val)
                self.sim.set_substeps(self.state.substeps)
                changed = True

            # Iterations
            ok, ival = imgui.slider_int("Iterations", int(self.state.iterations), 1, 50)
            if ok and ival != self.state.iterations:
                self.state.iterations = int(ival)
                self.sim.set_iterations(self.state.iterations)
                changed = True

            imgui.separator()
            imgui.text("Distance Constraints")
            ok, ds = imgui.slider_float("Stiffness", float(self.state.dist_stiffness), 0.0, 10.0)
            if ok and abs(ds - self.state.dist_stiffness) > 1e-6:
                self.state.dist_stiffness = float(ds)
                self.sim.set_distance_stiffness(self.state.dist_stiffness)
                changed = True

            ok, dd = imgui.slider_float("Damping", float(self.state.dist_damping), 0.0, 5.0)
            if ok and abs(dd - self.state.dist_damping) > 1e-6:
                self.state.dist_damping = float(dd)
                self.sim.set_distance_damping(self.state.dist_damping)
                changed = True

            imgui.separator()
            imgui.text("Volume Constraints")
            ok, ve = imgui.checkbox("Enabled", bool(self.state.vol_enabled))
            if ok and ve != self.state.vol_enabled:
                self.state.vol_enabled = bool(ve)
                self.sim.set_volume_enabled(self.state.vol_enabled)
                changed = True

            ok, vs = imgui.slider_float("Vol. Stiffness", float(self.state.vol_stiffness), 0.0, 5.0)
            if ok and abs(vs - self.state.vol_stiffness) > 1e-6:
                self.state.vol_stiffness = float(vs)
                self.sim.set_volume_stiffness(self.state.vol_stiffness)
                changed = True

            if changed:
                imgui.text_colored("Updated", 0.6, 0.9, 0.7)

            imgui.end()

    mgr = _SimImGuiManager(sim.renderer, sim, state)
    if getattr(mgr, 'is_available', False):
        # Wire into renderer’s 2D overlay callbacks
        sim.renderer.render_2d_callbacks.append(mgr.render_frame)
        # Keep a reference alive for the session
        sim._imgui_manager = mgr


# No standalone draw function needed; overlay is driven inside sim.render()
