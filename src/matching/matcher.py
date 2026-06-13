"""
matcher.py — Classical minutiae pairing algorithm.

Pairs aligned probe minutiae with gallery minutiae based on spatial and angular tolerances.
"""

import numpy as np
from typing import List, Tuple


def match_minutiae(aligned_probe: List[dict],
                   gallery: dict,
                   spatial_tol: float = 15.0,
                   angular_tol: float = 30.0 * np.pi / 180.0) -> List[Tuple[dict, dict]]:
    """
    Find corresponding minutiae pairs between the aligned probe and the gallery.
    
    A pair is valid if:
    1. They are of the same type (ending or bifurcation).
    2. Spatial distance <= spatial_tol.
    3. Angular difference <= angular_tol.
    
    Greedy matching: we pair each probe minutia with the closest gallery minutia
    that hasn't been matched yet.
    
    Args:
        aligned_probe: List of aligned probe minutiae (from align_templates).
        gallery: The gallery template.
        spatial_tol: Maximum allowed pixel distance between matched minutiae.
        angular_tol: Maximum allowed angular difference in radians.
        
    Returns:
        List of tuples: (probe_minutia, gallery_minutia)
    """
    if not aligned_probe or not gallery.get('minutiae'):
        return []
        
    gallery_mins = gallery['minutiae']
    matched_pairs = []
    
    # Keep track of which gallery minutiae have already been paired
    matched_gallery_indices = set()
    
    for p_min in aligned_probe:
        best_dist = spatial_tol
        best_g_idx = -1
        
        for g_idx, g_min in enumerate(gallery_mins):
            if g_idx in matched_gallery_indices:
                continue
                
            if p_min['type'] != g_min['type']:
                continue
                
            # Compute spatial distance
            dist = np.sqrt((p_min['x'] - g_min['x'])**2 + (p_min['y'] - g_min['y'])**2)
            if dist > spatial_tol:
                continue
                
            # Compute angular difference (handle 2*pi wrapping)
            angle_diff = abs(p_min['orientation'] - g_min['orientation'])
            angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
            
            if angle_diff > angular_tol:
                continue
                
            if dist < best_dist:
                best_dist = dist
                best_g_idx = g_idx
                
        if best_g_idx != -1:
            matched_pairs.append((p_min, gallery_mins[best_g_idx]))
            matched_gallery_indices.add(best_g_idx)
            
    return matched_pairs
