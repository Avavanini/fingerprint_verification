"""
minutiae.py — Minutiae detection using Crossing Number algorithm.

Detects ridge endings (CN=1) and bifurcations (CN=3) on thinned fingerprint
skeleton images. Includes border filtering, proximity filtering, and local
ridge orientation estimation.

Functions:
    detect_minutiae(skeleton) -> list of minutiae
    compute_orientations(skeleton, minutiae) -> updated minutiae list
    visualize_minutiae(image, minutiae) -> annotated image
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


# Type alias for a minutia: (x, y, type_str, orientation_radians)
Minutia = Tuple[int, int, str, float]


def detect_minutiae(skeleton: np.ndarray, border_margin: int = 15,
                    min_distance: int = 10,
                    mask: Optional[np.ndarray] = None) -> List[Minutia]:
    """
    Detect minutiae (ridge endings and bifurcations) using the Crossing Number method.

    The Crossing Number (CN) for a ridge pixel p is defined as:
        CN(p) = 0.5 * sum(|p[i] - p[i+1]|) for i in 0..7 (8-connected neighbors)

    - CN = 1 → Ridge ending
    - CN = 3 → Ridge bifurcation

    After detection, spurious minutiae near borders and mask boundaries are
    removed, and local ridge orientation is computed for each remaining minutia.

    Args:
        skeleton: Thinned binary image (0 and 255, uint8).
        border_margin: Pixels from image border to ignore.
        min_distance: Minimum distance between valid minutiae.
        mask: Optional foreground mask (255=foreground). Minutiae outside mask
              or within border_margin of mask boundary are removed.

    Returns:
        List of (x, y, type, orientation) tuples.
        - x, y: pixel coordinates
        - type: "ending" or "bifurcation"
        - orientation: angle in radians [0, 2*pi)
    """
    h, w = skeleton.shape
    binary = (skeleton > 0).astype(np.uint8)
    minutiae = []

    # 8-connected neighbor offsets (clockwise from top-left)
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]

    for i in range(border_margin, h - border_margin):
        for j in range(border_margin, w - border_margin):
            if binary[i, j] == 0:
                continue

            # Skip if outside foreground mask
            if mask is not None and mask[i, j] == 0:
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

    # Filter minutiae near mask boundaries (if mask provided)
    if mask is not None:
        minutiae = _filter_near_mask_boundary(minutiae, mask, margin=border_margin)

    # Filter minutiae that are too close together
    if min_distance > 0:
        minutiae = _filter_close_minutiae(minutiae, min_distance)

    # Compute orientations
    minutiae = compute_orientations(skeleton, minutiae)

    return minutiae


def compute_orientations(skeleton: np.ndarray,
                         minutiae: List[Minutia],
                         window_size: int = 15) -> List[Minutia]:
    """
    Compute local ridge orientation for each minutia using gradient analysis.

    For ridge endings, orientation points along the ridge direction (away
    from the ending). For bifurcations, orientation is based on the dominant
    gradient in a local window.

    Args:
        skeleton: Thinned binary image (0 and 255).
        minutiae: List of (x, y, type, orientation) tuples.
        window_size: Size of the local window for gradient computation.

    Returns:
        Updated minutiae list with computed orientations.
    """
    h, w = skeleton.shape
    binary = (skeleton > 0).astype(np.uint8)
    half = window_size // 2

    # 8-connected neighbor offsets
    offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1),
               (1, 1), (1, 0), (1, -1), (0, -1)]

    updated = []
    for x, y, mtype, _ in minutiae:
        if mtype == "ending":
            # For ridge endings: find the single neighbor and compute direction
            neighbors_on = []
            for di, dj in offsets:
                ni, nj = y + di, x + dj
                if 0 <= ni < h and 0 <= nj < w and binary[ni, nj] == 1:
                    neighbors_on.append((nj, ni))

            if neighbors_on:
                # Orientation points away from the ridge (opposite of neighbor)
                nx, ny = neighbors_on[0]
                angle = np.arctan2(y - ny, x - nx)
                if angle < 0:
                    angle += 2 * np.pi
            else:
                angle = 0.0
        else:
            # For bifurcations: use gradient-based orientation in local window
            y1 = max(0, y - half)
            y2 = min(h, y + half + 1)
            x1 = max(0, x - half)
            x2 = min(w, x + half + 1)
            patch = skeleton[y1:y2, x1:x2].astype(np.float64)

            # Compute gradients
            gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
            gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)

            # Dominant orientation from averaged gradient
            angle = np.arctan2(np.sum(gy), np.sum(gx))
            if angle < 0:
                angle += 2 * np.pi

        updated.append((x, y, mtype, float(angle)))

    return updated


def _filter_close_minutiae(minutiae: List[Minutia],
                           min_distance: int) -> List[Minutia]:
    """Remove minutiae that are closer than min_distance to each other.

    When two minutiae are too close, the bifurcation is kept over the ending
    (bifurcations are generally more reliable).
    """
    if not minutiae:
        return minutiae

    # Sort: bifurcations first (they're more reliable)
    minutiae_sorted = sorted(minutiae, key=lambda m: (m[2] == "ending", m[0], m[1]))

    filtered = [minutiae_sorted[0]]
    for m in minutiae_sorted[1:]:
        too_close = False
        for f in filtered:
            dist = np.sqrt((m[0] - f[0])**2 + (m[1] - f[1])**2)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            filtered.append(m)

    return filtered


def _filter_near_mask_boundary(minutiae: List[Minutia],
                               mask: np.ndarray,
                               margin: int = 10) -> List[Minutia]:
    """Remove minutiae that are near the boundary of the foreground mask.

    Minutiae near the mask edge are often artifacts caused by the
    segmentation boundary cutting through ridges.
    """
    if mask is None:
        return minutiae

    # Erode the mask to create a safe zone
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (margin * 2, margin * 2))
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    return [(x, y, t, o) for x, y, t, o in minutiae
            if eroded_mask[y, x] == 255]


def visualize_minutiae(image: np.ndarray, minutiae: List[Minutia],
                       show_orientation: bool = True,
                       arrow_length: int = 15) -> np.ndarray:
    """
    Draw detected minutiae on a fingerprint image.

    Ridge endings are drawn as red circles, bifurcations as blue circles.
    Orientation arrows are optionally drawn.

    Args:
        image: Grayscale or BGR fingerprint image.
        minutiae: List of (x, y, type, orientation) tuples.
        show_orientation: If True, draw orientation arrows.
        arrow_length: Length of orientation arrows in pixels.

    Returns:
        BGR annotated image.
    """
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    for x, y, mtype, orientation in minutiae:
        if mtype == "ending":
            color = (0, 0, 255)    # Red
            radius = 4
        else:
            color = (255, 0, 0)    # Blue
            radius = 4

        cv2.circle(vis, (x, y), radius, color, 1, lineType=cv2.LINE_AA)

        if show_orientation:
            dx = int(arrow_length * np.cos(orientation))
            dy = int(arrow_length * np.sin(orientation))
            cv2.arrowedLine(vis, (x, y), (x + dx, y + dy),
                            color, 1, tipLength=0.3, line_type=cv2.LINE_AA)

    return vis
