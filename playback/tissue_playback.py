import os
import glob
import argparse
import numpy as np
import warp as wp
from warp.render import OpenGLRenderer


class TissueData:
    """Stores mesh data for a single tissue type."""
    def __init__(self, name):
        self.name = name
        self.vertices = None
        self.triangles = None
        self.color = self._get_tissue_color(name)

    @staticmethod
    def _get_tissue_color(tissue_name):
        colors = {
            'Liver': (0.6, 0.2, 0.1),      
            'Fat': (0.9, 0.8, 0.4),        
            'Gallbladder': (0.3, 0.5, 0.2) 
        }
        return colors.get(tissue_name, (0.7, 0.7, 0.7))


class InstrumentData:
    """Stores data for a single laparoscopic instrument."""
    def __init__(self, position, rotation, entry_point):
        self.position = position      # (x, y, z)
        self.rotation = rotation      # (x, y, z, w) quaternion
        self.entry_point = entry_point  # (x, y, z)


class DataPlayback:

    def __init__(self, data_dir, fps=30):
        self.data_dir = data_dir
        self.fps = fps
        self.tissue_types = ['Liver', 'Fat', 'Gallbladder']

        # Find all frames
        self.frames = self._find_frames()
        if not self.frames:
            raise ValueError(f"No frame data found in {data_dir}")

        print(f"Found {len(self.frames)} frames in {data_dir}")

        # Initialize Warp
        wp.init()
        wp.set_module_options({"enable_backward": False})

        # Create renderer
        self.renderer = OpenGLRenderer(
            title="Tissue Data Playback",
            scaling=1.0,
            fps=fps,
            near_plane=0.01,
            far_plane=100.0
        )

        self.renderer._camera_pos = [0.0, 0.5, -2.0]

        self.current_frame = 0
        self.paused = False

    def _find_frames(self):
        """Find all available frame numbers in the data directory."""
        frame_files = glob.glob(os.path.join(self.data_dir, 'frame_*_verts_*.dat'))

        if not frame_files:
            return []

        # Extract frame numbers
        frame_numbers = set()
        for filepath in frame_files:
            filename = os.path.basename(filepath)
            # Parse frame_XXXXXX_verts_TissueType.dat
            parts = filename.split('_')
            if len(parts) >= 2:
                try:
                    frame_num = int(parts[1])
                    frame_numbers.add(frame_num)
                except ValueError:
                    continue

        return sorted(list(frame_numbers))

    def _load_vertices(self, frame_num, tissue_type):
        """Load vertex data from frame_XXXXXX_verts_TissueType.dat"""
        filename = f"frame_{frame_num:06d}_verts_{tissue_type}.dat"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            # First line: vertex count
            num_verts = int(f.readline().strip())

            # Read vertices (3 floats per line: X Y Z)
            vertices = []
            for _ in range(num_verts):
                line = f.readline().strip().replace(",", ".")
                if line:
                    coords = [float(x) for x in line.split()]
                    if len(coords) == 3:
                        vertices.append(coords)

            return np.array(vertices, dtype=np.float32)

    def _load_triangles(self, frame_num, tissue_type):
        """Load triangle data from frame_XXXXXX_tris_TissueType.dat"""
        filename = f"frame_{frame_num:06d}_tris_{tissue_type}.dat"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, 'r') as f:
            # First line: triangle count
            num_tris = int(f.readline().strip())

            # Read triangles (3 integers per line: vertex indices)
            # Filter out invalid triangles where all 3 are set to 0
            triangles = []
            for _ in range(num_tris):
                line = f.readline().strip()
                if line:
                    indices = [int(x) for x in line.split()]
                    if len(indices) == 3 and not(indices[0] == indices[1] == indices[2] == 0):
                        triangles.append(indices)

            return np.array(triangles, dtype=np.int32) if triangles else np.array([], dtype=np.int32).reshape(0, 3)

    def _load_instruments(self, frame_num):
        """Load instrument data from frame_XXXXXX_instruments.dat"""
        filename = f"frame_{frame_num:06d}_instruments.dat"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            return []

        instruments = []
        with open(filepath, 'r') as f:
            # First line: instrument count
            num_instruments = int(f.readline().strip())

            # Read each instrument (10 floats per line: pos xyz, rot xyzw, entry xyz)
            for _ in range(num_instruments):
                line = f.readline().strip().replace(",", ".")
                if line:
                    values = [float(x) for x in line.split()]
                    if len(values) == 10:
                        position = tuple(values[0:3])
                        rotation = tuple(values[3:7])
                        entry_point = tuple(values[7:10])
                        instruments.append(InstrumentData(position, rotation, entry_point))

        return instruments

    def load_frame(self, frame_num):
        """Load all tissue data and instruments for a specific frame."""
        tissues = []

        for tissue_type in self.tissue_types:
            tissue = TissueData(tissue_type)
            tissue.vertices = self._load_vertices(frame_num, tissue_type)
            tissue.triangles = self._load_triangles(frame_num, tissue_type)

            # Only add if both vertices and triangles exist
            if tissue.vertices is not None and tissue.triangles is not None:
                if len(tissue.vertices) > 0 and len(tissue.triangles) > 0:
                    tissues.append(tissue)

        # Load instruments
        instruments = self._load_instruments(frame_num)
        if instruments:
            print(f"  Instruments: {len(instruments)}")
            for i, instr in enumerate(instruments):
                print(f"    Instr {i}: pos={instr.position}, rot={instr.rotation}, entry={instr.entry_point}")

        return tissues, instruments

    def render_frame(self, tissues, instruments):
        """Render all tissues and instruments for the current frame."""
        self.renderer.begin_frame()

        # Work around the deregister_shape bug by collecting shape_ids first
        tissue_names = [f"tissue_{i}_mesh" for i in range(len(tissues))]

        # Also track instrument names from previous frame (assume max 10 instruments)
        instrument_names = []
        for i in range(10):
            instrument_names.extend([
                f"instrument_{i}_tip",
                f"instrument_{i}_entry",
                f"instrument_{i}_shaft"
            ])

        all_names = tissue_names + instrument_names
        shapes_to_remove = []

        if hasattr(self.renderer, '_instances'):
            # First pass: collect all shape_ids and remove instances
            for mesh_name in all_names:
                if mesh_name in self.renderer._instances:
                    inst = self.renderer._instances[mesh_name]
                    shape_id = inst[2] if len(inst) > 2 else None
                    if shape_id is not None:
                        shapes_to_remove.append(shape_id)
                    # Remove the instance
                    self.renderer.remove_shape_instance(mesh_name)

            # Second pass: deregister shapes in reverse order
            for shape_id in sorted(shapes_to_remove, reverse=True):
                self.renderer.deregister_shape(shape_id)

        for i, tissue in enumerate(tissues):
            mesh_name = f"tissue_{i}_mesh"

            self.renderer.render_mesh(
                name=mesh_name,
                points=tissue.vertices,
                indices=tissue.triangles.flatten(),
                colors=tissue.color,
                update_topology=True,
                visible=True
            )

        
        # Render instruments
        for i, instr in enumerate(instruments):
            # Render instrument tip as a capsule to show orientation (blue)
            self.renderer.render_capsule(
                name=f"instrument_{i}_tip",
                pos=instr.position,
                rot=instr.rotation,
                radius=0.008,
                half_height=0.025,  # Capsule length
                color=(0.2, 0.4, 0.9)  # Blue
            )

            # Render entry point (red sphere)
            self.renderer.render_sphere(
                name=f"instrument_{i}_entry",
                pos=instr.entry_point,
                rot=(0.0, 0.0, 0.0, 1.0),
                radius=0.015,
                color=(0.9, 0.2, 0.2)  # Red
            )

            # Render line from entry point to instrument tip (shaft)
            self.renderer.render_line_strip(
                name=f"instrument_{i}_shaft",
                vertices=[instr.entry_point, instr.position],
                color=(0.7, 0.7, 0.8),  # Light gray
                radius=0.003
            )
            
        
        self.renderer.end_frame()

    def run(self):
        """Main playback loop."""
        print(f"\nStarting playback...")
        print(f"Controls:")
        print(f"  Space: Pause/Resume")
        print(f"  Left/Right Arrow: Previous/Next frame (when paused)")
        print(f"  R: Reset to first frame")
        print(f"  ESC: Exit")

        frame_idx = 0

        while self.renderer.is_running():
            if not self.paused:
                # Auto-advance to next frame
                frame_num = self.frames[frame_idx]

                print(f"\nFrame {frame_idx}/{len(self.frames)-1} (frame number: {frame_num})")
                tissues, instruments = self.load_frame(frame_num)
                self.render_frame(tissues, instruments)

                # Loop back to start
                frame_idx = (frame_idx + 1) % len(self.frames)
            else:
                # Just re-render current frame when paused
                frame_num = self.frames[frame_idx]
                tissues, instruments = self.load_frame(frame_num)
                self.render_frame(tissues, instruments)

        print("\nPlayback stopped.")


def main():
    parser = argparse.ArgumentParser(description='Playback exported tissue data')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='exported_data',
        help='Directory containing the exported .dat files '
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Playback frame rate (default: 30)'
    )

    args = parser.parse_args()

    # Validate data directory
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        print(f"Please create it and place your .dat files there, or specify a different directory with --data_dir")
        return

    try:
        playback = DataPlayback(args.data_dir, fps=args.fps)
        playback.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
