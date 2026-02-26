from __future__ import annotations

import glob
import os
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


@dataclass
class TissueData:
    """Stores mesh data for a single tissue type."""

    name: str
    vertices: np.ndarray | None = None
    triangles: np.ndarray | None = None
    color: tuple[float, float, float] = (0.7, 0.7, 0.7)

    def __post_init__(self):
        self.color = get_tissue_color(self.name)


@dataclass
class InstrumentData:
    """Stores data for a single laparoscopic instrument."""

    position: np.ndarray
    rotation: np.ndarray
    entry_point: np.ndarray


def get_tissue_color(tissue_name: str) -> tuple[float, float, float]:
    colors = {
        "Liver": (0.6, 0.2, 0.1),
        "Fat": (0.9, 0.8, 0.4),
        "Gallbladder": (0.3, 0.5, 0.2),
    }
    return colors.get(tissue_name, (0.7, 0.7, 0.7))


class DataLoader:
    def __init__(self, data_dir: str, tissue_types: list[str], scene_rot: R):
        self.data_dir = data_dir
        self.tissue_types = list(tissue_types)
        self.scene_rot = scene_rot

    def find_frames(self) -> list[int]:
        """Find all available frame numbers in the data directory."""
        frame_files = glob.glob(os.path.join(self.data_dir, "frame_*_verts_*.dat"))
        if not frame_files:
            return []

        frame_numbers = set()
        for filepath in frame_files:
            filename = os.path.basename(filepath)
            parts = filename.split("_")
            if len(parts) >= 2:
                try:
                    frame_num = int(parts[1])
                    frame_numbers.add(frame_num)
                except ValueError:
                    continue

        return sorted(list(frame_numbers))

    def load_vertices(self, frame_num: int, tissue_type: str) -> np.ndarray | None:
        """Load vertex data from frame_XXXXXX_verts_TissueType.dat"""
        filename = f"frame_{frame_num:06d}_verts_{tissue_type}.dat"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, "r") as f:
            num_verts = int(f.readline().strip())
            vertices = []
            for _ in range(num_verts):
                line = f.readline().strip().replace(",", ".")
                if line:
                    coords = [float(x) for x in line.split()]
                    if len(coords) == 3:
                        vertices.append(coords)

        vertices_np = np.array(vertices, dtype=np.float32)
        if vertices_np.size:
            vertices_np = self.scene_rot.apply(vertices_np).astype(np.float32)
        return vertices_np

    def load_triangles(self, frame_num: int, tissue_type: str) -> np.ndarray | None:
        """Load triangle data from frame_XXXXXX_tris_TissueType.dat"""
        filename = f"frame_{frame_num:06d}_tris_{tissue_type}.dat"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            return None

        with open(filepath, "r") as f:
            num_tris = int(f.readline().strip())
            triangles = []
            for _ in range(num_tris):
                line = f.readline().strip()
                if line:
                    indices = [int(x) for x in line.split()]
                    if len(indices) == 3 and not (indices[0] == indices[1] == indices[2] == 0):
                        triangles.append(indices)

        if triangles:
            return np.array(triangles, dtype=np.int32)
        return np.array([], dtype=np.int32).reshape(0, 3)

    def load_instruments(self, frame_num: int) -> list[InstrumentData]:
        """Load instrument data from frame_XXXXXX_instruments.dat"""
        filename = f"frame_{frame_num:06d}_instruments.dat"
        filepath = os.path.join(self.data_dir, filename)

        if not os.path.exists(filepath):
            return []

        instruments = []
        with open(filepath, "r") as f:
            num_instruments = int(f.readline().strip())
            for _ in range(num_instruments):
                line = f.readline().strip().replace(",", ".")
                if line:
                    values = [float(x) for x in line.split()]
                    if len(values) == 10:
                        position = self.scene_rot.apply(values[0:3]).astype(np.float32)
                        quat = np.array(values[3:7], dtype=np.float32)
                        quat_norm = float(np.linalg.norm(quat))
                        if quat_norm < 1.0e-8:
                            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
                        else:
                            quat = quat / quat_norm
                        rotation = (self.scene_rot * R.from_quat(quat)).as_quat().astype(np.float32)
                        entry_point = self.scene_rot.apply(values[7:10]).astype(np.float32)
                        instruments.append(InstrumentData(position, rotation, entry_point))

        return instruments

    def load_frame(self, frame_num: int) -> tuple[list[TissueData], list[InstrumentData]]:
        """Load all tissue data and instruments for a specific frame."""
        tissues = []

        for tissue_type in self.tissue_types:
            tissue = TissueData(tissue_type)
            tissue.vertices = self.load_vertices(frame_num, tissue_type)
            tissue.triangles = self.load_triangles(frame_num, tissue_type)

            if tissue.vertices is not None and tissue.triangles is not None:
                if len(tissue.vertices) > 0 and len(tissue.triangles) > 0:
                    tissues.append(tissue)

        instruments = self.load_instruments(frame_num)
        return tissues, instruments
