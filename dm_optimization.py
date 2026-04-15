
"""
Created on Fri Feb 06 10:01:02 2026

@author: lmazzolo

Shared utilities for TIPTOP DM optimization.
HARMONI - MORFEO compatible.

"""

from __future__ import annotations

# ---- DM optimization according to wide/narrow field ----
def build_opt_lists(radius, fov):
    """
    Build the DM optimization zenith/weight arrays.

    Parameters
    ----------
    radius_arcsec:
        Optimization radius (arcsec).
    fov_arcsec:
        Optimized FoV diameter (arcsec).

    Returns
    -------
    (optimization_zenith, optimization_weight)
        Two lists to be written to the .ini file.
    """
    # Reference MORFEO configuration
    fovMORFEO = 80 #radius arcsec
    OptWeight_MORFEO = [10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    rap = fov/fovMORFEO
    OptZen = [0] + [radius] * 8 + [fov] * 8
    OptWeight = [w * rap for w in OptWeight_MORFEO]
    return (OptZen, OptWeight)