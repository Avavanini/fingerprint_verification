"""
descriptor.py — Minutiae descriptor computation and template serialization.

Computes a local descriptor for each minutia (based on the neighborhood
structure in the ridge map) and provides serialization to JSON and NumPy formats.

Functions:
    compute_descriptors(skeleton, minutiae) -> list of descriptors
    create_template(minutiae, descriptors, metadata) -> dict
    save_template(template, path, format) -> None
    load_template(path) -> dict
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


Minutia = Tuple[int, int, str, float]


def compute_descriptors(skeleton: np.ndarray,
                        minutiae: List[Minutia],
                        radius: int = 30,
                        num_sectors: int = 8) -> List[np.ndarray]:
    """
    Compute a local descriptor for each minutia based on the ridge structure
    in a circular neighborhood.

    The descriptor divides a circular region around each minutia into sectors
    and computes the ridge pixel density in each sector, forming a histogram.

    Args:
        skeleton: Thinned binary image (0 and 255, uint8).
        minutiae: List of (x, y, type, orientation) tuples.
        radius: Radius of the circular neighborhood.
        num_sectors: Number of angular sectors for the descriptor.

    Returns:
        List of descriptor vectors (numpy arrays), one per minutia.
    """
    h, w = skeleton.shape
    binary = (skeleton > 0).astype(np.float64)
    descriptors = []

    for x, y, mtype, orientation in minutiae:
        descriptor = np.zeros(num_sectors, dtype=np.float64)
        sector_counts = np.zeros(num_sectors, dtype=np.float64)

        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                ny, nx = y + dy, x + dx

                # Check bounds
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue

                # Check within circle
                dist = np.sqrt(dx**2 + dy**2)
                if dist > radius or dist < 2:
                    continue

                # Compute angle relative to minutia orientation
                angle = np.arctan2(dy, dx) - orientation
                if angle < 0:
                    angle += 2 * np.pi

                # Assign to sector
                sector = int(angle / (2 * np.pi) * num_sectors) % num_sectors
                sector_counts[sector] += 1.0
                descriptor[sector] += binary[ny, nx]

        # Normalize: ridge density per sector
        for s in range(num_sectors):
            if sector_counts[s] > 0:
                descriptor[s] = descriptor[s] / sector_counts[s]

        descriptors.append(descriptor)

    return descriptors


def create_template(minutiae: List[Minutia],
                    descriptors: List[np.ndarray],
                    metadata: Optional[Dict[str, Any]] = None) -> dict:
    """
    Create a fingerprint template containing minutiae and their descriptors.

    Args:
        minutiae: List of (x, y, type, orientation) tuples.
        descriptors: List of descriptor vectors.
        metadata: Optional dict with image info (subject_id, finger, etc.).

    Returns:
        Template dictionary with keys:
            - 'minutiae': list of dicts with x, y, type, orientation
            - 'descriptors': list of descriptor arrays
            - 'metadata': metadata dict
            - 'num_minutiae': count
    """
    minutiae_dicts = []
    for (x, y, mtype, orientation), desc in zip(minutiae, descriptors):
        minutiae_dicts.append({
            'x': int(x),
            'y': int(y),
            'type': mtype,
            'orientation': float(orientation),
            'descriptor': desc.tolist(),
        })

    template = {
        'num_minutiae': len(minutiae),
        'minutiae': minutiae_dicts,
        'metadata': metadata or {},
    }

    return template


def save_template(template: dict, path: str, fmt: str = "json") -> None:
    """
    Save a fingerprint template to disk.

    Args:
        template: Template dictionary from create_template().
        path: Output file path.
        fmt: Format — "json" for human-readable, "npy" for compact NumPy.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        with open(p, 'w') as f:
            json.dump(template, f, indent=2)
    elif fmt == "npy":
        # For NumPy format, save minutiae array + descriptors matrix
        minutiae_arr = np.array(
            [(m['x'], m['y'], 0 if m['type'] == 'ending' else 1, m['orientation'])
             for m in template['minutiae']],
            dtype=np.float64
        )
        descriptors_arr = np.array(
            [m['descriptor'] for m in template['minutiae']],
            dtype=np.float64
        )
        np.savez(p,
                 minutiae=minutiae_arr,
                 descriptors=descriptors_arr,
                 metadata=json.dumps(template.get('metadata', {})))
    else:
        raise ValueError(f"Unknown format: {fmt}")


def load_template(path: str) -> dict:
    """
    Load a fingerprint template from disk.

    Args:
        path: Path to template file (.json or .npz).

    Returns:
        Template dictionary.
    """
    p = Path(path)

    if p.suffix == '.json':
        with open(p, 'r') as f:
            return json.load(f)
    elif p.suffix == '.npz':
        data = np.load(p, allow_pickle=True)
        minutiae_arr = data['minutiae']
        descriptors_arr = data['descriptors']
        metadata = json.loads(str(data['metadata']))

        minutiae = []
        for i, row in enumerate(minutiae_arr):
            minutiae.append({
                'x': int(row[0]),
                'y': int(row[1]),
                'type': 'ending' if row[2] == 0 else 'bifurcation',
                'orientation': float(row[3]),
                'descriptor': descriptors_arr[i].tolist(),
            })

        return {
            'num_minutiae': len(minutiae),
            'minutiae': minutiae,
            'metadata': metadata,
        }
    else:
        raise ValueError(f"Unknown file format: {p.suffix}")
