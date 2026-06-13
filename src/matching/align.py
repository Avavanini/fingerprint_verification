"""
align.py — Fingerprint alignment for minutiae matching.

Implements a heuristic alignment based on the most similar local ridge descriptor.
"""

import numpy as np
from typing import Dict, Tuple, List


def align_templates(probe: dict, gallery: dict) -> Tuple[List[dict], float, float, float]:
    """
    Align the probe template to the gallery template using a descriptor-based heuristic.
    
    1. Finds the pair of minutiae (one from probe, one from gallery) that have
       the most similar local ridge density descriptor.
    2. Computes the spatial transformation (rotation and translation) required to
       align the probe's reference minutia with the gallery's reference minutia.
    3. Applies this transformation to all minutiae in the probe.
    
    Args:
        probe: The probe template (must contain 'minutiae' list with 'descriptor').
        gallery: The gallery template.
        
    Returns:
        Tuple containing:
            - aligned_probe_minutiae: List of minutiae dicts with transformed coordinates/orientation.
            - delta_x: Translation in x.
            - delta_y: Translation in y.
            - delta_theta: Rotation angle applied (in radians).
    """
    if not probe['minutiae'] or not gallery['minutiae']:
        return [], 0.0, 0.0, 0.0

    probe_mins = probe['minutiae']
    gallery_mins = gallery['minutiae']
    
    best_dist = float('inf')
    best_pair = None
    
    # 1. Find the best matching descriptor pair (reference pair)
    for p_min in probe_mins:
        p_desc = np.array(p_min.get('descriptor', []))
        if len(p_desc) == 0:
            continue
            
        for g_min in gallery_mins:
            # We only match endings with endings, bifurcations with bifurcations
            if p_min['type'] != g_min['type']:
                continue
                
            g_desc = np.array(g_min.get('descriptor', []))
            if len(g_desc) == 0:
                continue
                
            # Euclidean distance between descriptors
            dist = np.linalg.norm(p_desc - g_desc)
            if dist < best_dist:
                best_dist = dist
                best_pair = (p_min, g_min)
                
    if best_pair is None:
        # Fallback to no alignment if no suitable pair found
        return probe_mins, 0.0, 0.0, 0.0
        
    ref_p, ref_g = best_pair
    
    # 2. Compute transformation
    # Delta theta: angle required to rotate probe to match gallery orientation
    delta_theta = ref_g['orientation'] - ref_p['orientation']
    
    # Normalize delta_theta to [-pi, pi]
    delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
    
    cos_theta = np.cos(delta_theta)
    sin_theta = np.sin(delta_theta)
    
    # After rotation, where does the probe reference point end up?
    rx = ref_p['x'] * cos_theta - ref_p['y'] * sin_theta
    ry = ref_p['x'] * sin_theta + ref_p['y'] * cos_theta
    
    # Translation to match the gallery reference point
    delta_x = ref_g['x'] - rx
    delta_y = ref_g['y'] - ry
    
    # 3. Apply transformation to all probe minutiae
    aligned_minutiae = []
    for p_min in probe_mins:
        # Rotate
        nx = p_min['x'] * cos_theta - p_min['y'] * sin_theta
        ny = p_min['x'] * sin_theta + p_min['y'] * cos_theta
        
        # Translate
        nx += delta_x
        ny += delta_y
        
        # New orientation
        n_ori = (p_min['orientation'] + delta_theta) % (2 * np.pi)
        
        aligned_m = {
            'x': float(nx),
            'y': float(ny),
            'type': p_min['type'],
            'orientation': float(n_ori),
            'descriptor': p_min.get('descriptor', [])
        }
        aligned_minutiae.append(aligned_m)
        
    return aligned_minutiae, float(delta_x), float(delta_y), float(delta_theta)
