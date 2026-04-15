"""
Created on Wed Feb 04 10:58:33 2026

@author: lmazzolo

Shared utilities for TIPTOP atmosphere profiles.
HARMONI - MORFEO compatible.

"""

from __future__ import annotations

from typing import List, Tuple, Optional

import numpy as np

# ELT case
def atmo_select(qno_txt: str, isoangle_desired: Optional[float] = None) -> Tuple[float, List[float], List[float]]:
    """Return (seeing_arcsec, cn2_weights, wind_speed) for a named profile.

    Parameters
    ----------
    qno_txt:
        Atmosphere profile name: "Q1", "Q2", "Q3", "Q4" or "Median".
    isoangle_desired:
        Optional desired isoplanatic angle [arcsec]. When provided, the function
        tries to reweight the Cn2 profile accordingly using TIPTOP helper
        functions (``iso_angle`` and ``cn2_from_iso``) if they are available.

    Returns
    -------
    seeing_arcsec:
        Seeing at 500 nm in arcsec.
    cn2_weights:
        List of Cn2 layer weights (sums to 1).
    wind_speed:
        List of wind speeds per layer [m/s].
    """

    qno_txt = str(qno_txt)
    qno_map = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3, "Median": 4}
    if qno_txt not in qno_map:
        raise ValueError("Invalid atmosphere profile. Expected one of: 'Q1','Q2','Q3','Q4','Median'.")

    qno = qno_map[qno_txt]

    # --- Seeing (ELT) ---
    r0 = [0.234, 0.178, 0.139, 0.097, 0.157]
    seeing = 0.9759 * 0.5 / (r0[qno] * 4.84814)

    # --- Cn2 weights (per layer) ---
    k_wind = [0.925, 0.968, 1.079, 1.370, 1.052]
    cn2 = [
        [22.6, 11.2, 10.1, 6.4, 4.15, 4.15, 4.15, 4.15, 3.1, 2.26,
         1.13, 2.21, 1.33, 0.88, 1.47, 1.77, 0.59, 2.06, 1.92, 1.03,
         2.3, 3.75, 2.76, 1.43, 0.89, 0.58, 0.36, 0.31, 0.27, 0.2,
         0.16, 0.09, 0.12, 0.07, 0.06],
        [25.1, 11.6, 9.57, 5.84, 3.7, 3.7, 3.7, 3.7, 3.25, 3.47,
         1.74, 3, 1.8, 1.2, 1.3, 1.56, 0.52, 1.82, 1.7, 0.91,
         1.87, 3.03, 2.23, 1.15, 0.72, 0.47, 0.3, 0.25, 0.22, 0.16,
         0.13, 0.07, 0.11, 0.06, 0.05],
        [25.5, 11.9, 9.32, 5.57, 4.5, 4.5, 4.5, 4.5, 4.19, 4.04,
         2.02, 3.04, 1.82, 1.21, 0.86, 1.03, 0.34, 1.2, 1.11, 0.6,
         1.43, 2.31, 1.7, 0.88, 0.55, 0.36, 0.22, 0.19, 0.17, 0.12,
         0.1, 0.06, 0.08, 0.04, 0.04],
        [23.6, 13.1, 9.81, 5.77, 6.58, 6.58, 6.58, 6.58, 5.4, 3.2,
         1.6, 2.18, 1.31, 0.87, 0.37, 0.45, 0.15, 0.52, 0.49, 0.26,
         0.8, 1.29, 0.95, 0.49, 0.31, 0.2, 0.12, 0.1, 0.09, 0.07,
         0.06, 0.03, 0.05, 0.02, 0.02],
        [24.2, 12.0, 9.68, 5.90, 4.73, 4.73, 4.73, 4.73, 3.99, 3.24,
         1.62, 2.60, 1.56, 1.04, 1.00, 1.20, 0.40, 1.40, 1.30, 0.70,
         1.60, 2.59, 1.90, 0.99, 0.62, 0.40, 0.25, 0.22, 0.19, 0.14,
         0.11, 0.06, 0.09, 0.05, 0.04],
    ]

    cn2w = np.array(cn2[qno], dtype=float)
    cn2w *= 1.0 / cn2w.sum()

    cn2_heights = [
        30.0, 90.0, 150.0, 200.0, 245.0, 300.0, 390.0, 600.0, 1130.0, 1880.0,
        2630.0, 3500.0, 4500.0, 5500.0, 6500.0, 7500.0, 8500.0, 9500.0,
        10500.0, 11500.0, 12500.0, 13500.0, 14500.0, 15500.0, 16500.0,
        17500.0, 18500.0, 19500.0, 20500.0, 21500.0, 22500.0, 23500.0,
        24500.0, 25500.0, 26500.0,
    ]

    if isoangle_desired is not None:
        # Requires TIPTOP helper functions.
        try:
            from tiptop.tiptop import iso_angle, cn2_from_iso  # type: ignore

            isoangle = iso_angle(cn2w.tolist(), cn2_heights, seeing)
            cn2w_new = cn2_from_iso(float(isoangle_desired), cn2w.tolist(), cn2_heights, seeing)
            cn2w = np.array(cn2w_new, dtype=float)
            cn2w *= 1.0 / cn2w.sum()
            isoangle_new = iso_angle(cn2w.tolist(), cn2_heights, seeing)

            print(f"[INFO] Isoplanatic angle: initial={isoangle} arcsec -> desired={isoangle_desired} arcsec -> new={isoangle_new} arcsec")
        except Exception as e:
            raise RuntimeError(
                "isoangle_desired was provided but TIPTOP iso-angle helpers were not available."
            ) from e

    # --- Wind speeds (per layer) ---
    wind_speed = (
        np.array(
            [
                5.5, 5.5, 5.1, 5.5, 5.6, 5.7, 5.8, 6.0, 6.5, 7.0,
                7.5, 8.5, 9.5, 11.5, 17.5, 23.0, 26.0, 29.0, 32.0, 27.0,
                22.0, 14.5, 9.5, 6.3, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                8.5, 9.0, 9.5, 10.0, 10.0,
            ],
            dtype=float,
        )
        * k_wind[qno]
    ).tolist()

    return float(seeing), cn2w.tolist(), wind_speed
