"""
minutiae.py — Minutiae detection using Crossing Number algorithm.

Functions:
    detect_minutiae(skeleton) -> list of minutiae
"""

import numpy as np
from typing import List, Tuple


def detect_minutiae(skeleton: np.ndarray, border_margin: int = 10,
                    min_distance: int = 10) -> List[Tuple[int, int, str, float]]:
    """
    Detect minutiae (ridge endings and bifurcations) using the Crossing Number method.
    
    The Crossing Number (CN) for a pixel p is defined as:
        CN(p) = 0.5 * sum(|p[i] - p[i+1]|) for i in 0..7 (8-connected neighbors)
    
    - CN = 1 → Ridge ending
    - CN = 3 → Ridge bifurcation
    
    Args:
        skeleton: Thinned binary image (0 and 255).
        border_margin: Pixels from border to ignore (spurious minutiae filter).
        min_distance: Minimum distance between valid minutiae.
        
    Returns:
        List of (x, y, type, orientation) tuples.
        type is "ending" or "bifurcation".
    """
    h, w = skeleton.shape
    binary = (skeleton > 0).astype(np.uint8)
    minutiae = []
    
    # 8-connected neighbor offsets (clockwise from top-left)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
    
    for i in range(border_margin, h - border_margin):
        for j in range(border_margin, w - border_margin):
            if binary[i, j] == 0:
                continue
            
            # Get 8-connected neighbors
            neighbors = [binary[i + di, j + dj] for di, dj in offsets]
            
            # Compute Crossing Number
            cn = 0
            for k in range(8):
                cn += abs(int(neighbors[k]) - int(neighbors[(k + 1) % 8]))
            cn = cn // 2
            
            if cn == 1:
                minutiae.append((j, i, "ending", 0.0))
            elif cn == 3:
                minutiae.append((j, i, "bifurcation", 0.0))
    
    # Filter minutiae that are too close together
    if min_distance > 0:
        minutiae = _filter_close_minutiae(minutiae, min_distance)
    
    return minutiae


def _filter_close_minutiae(minutiae: list, min_distance: int) -> list:
    """Remove minutiae that are closer than min_distance to each other."""
    if not minutiae:
        return minutiae
    
    filtered = [minutiae[0]]
    for m in minutiae[1:]:
        too_close = False
        for f in filtered:
            dist = np.sqrt((m[0] - f[0])**2 + (m[1] - f[1])**2)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append(m)
    
    return filtered
