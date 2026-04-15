
# -*- coding: utf-8 -*-
"""
Core helper utilities for the TIPTOP HARMONI UI/backend.

This module intentionally does NOT run any simulation by itself.
It only provides:
- magnitude -> photons conversion
- magnitude -> LO frame-rate law
- INI helper utilities
- barycenter / blur helpers
- default blur configuration

It is designed to be imported safely by the Dash backend without any
side effects or accidental TIPTOP execution.
"""
print("[STARTUP] Utils_HRM_MCAO import started", flush=True)

import ast
from configparser import ConfigParser

import numpy as np

from atmosphere import atmo_select
from dm_optimization import build_opt_lists

print("[STARTUP] Utils_HRM_MCAO imports done", flush=True)

# ----------------------------------------------------------------------
# Photons / frame-rate helpers
# ----------------------------------------------------------------------
def photons_from_mag(sensor_kind: str, mag: float, framerate_hz: float) -> float:
    """
    Convert a guide-star magnitude into photons per frame.

    Parameters
    ----------
    sensor_kind : str
        Supported values:
        - "LO"
        - "FOCUS"
    mag : float
        Guide-star magnitude.
    framerate_hz : float
        Sensor frame rate in Hz.

    Returns
    -------
    float
        Number of photons per frame.
    """
    sensor_kind = sensor_kind.upper()

    if sensor_kind == "LO":
        mag0flux = 1.69e12   # (J + H) MORFEO-equivalent zero point
        n_sa = 1.0
    elif sensor_kind == "FOCUS":
        mag0flux = 6.09e12   # (R + I) MORFEO-equivalent zero point
        n_sa = 50.0
    else:
        raise ValueError("sensor_kind must be 'LO' or 'FOCUS'")

    return float(mag0flux) * (10.0 ** (-float(mag) / 2.5)) / n_sa / float(framerate_hz)


def framerate_from_mag_lo(mag: float) -> float:
    """
    Empirical law for the LO frame rate as a function of magnitude.

    Parameters
    ----------
    mag : float
        LO guide-star magnitude.

    Returns
    -------
    float
        LO frame rate in Hz.
    """
    m = float(mag)
    return 400.0 / (1.0 + np.exp((m - 18.0) * 2.5)) + 100.0


# ----------------------------------------------------------------------
# INI helpers
# ----------------------------------------------------------------------
def _ensure_section(parser: ConfigParser, sec: str) -> None:
    """Ensure that an INI section exists."""
    if not parser.has_section(sec):
        parser.add_section(sec)


def _set_list(parser: ConfigParser, section: str, option: str, values) -> None:
    """Write a list of floats into an INI option."""
    _ensure_section(parser, section)
    parser.set(section, option, str(list(map(float, values))))


def _repeat_first_ini_value(parser: ConfigParser, section: str, key: str, n: int) -> None:
    """
    Read an INI value as a Python literal list, take the first element,
    and repeat it `n` times.
    """
    if not parser.has_section(section):
        return
    if not parser.has_option(section, key):
        return

    try:
        raw = parser.get(section, key)
        values = ast.literal_eval(raw)
        if not isinstance(values, (list, tuple)) or len(values) == 0:
            return
        parser.set(section, key, str([values[0]] * n))
    except Exception as exc:
        print(f"[WARN] Could not adapt {section}.{key}: {exc}")


def _read_ini_float_or_list_first(parser: ConfigParser, section: str, key: str):
    """
    Read an INI value that may be either:
    - a scalar string such as "12.3"
    - a list string such as "[12.3]"
    """
    if (not parser.has_section(section)) or (not parser.has_option(section, key)):
        return None, None

    raw = parser.get(section, key).strip()
    if not raw:
        return None, None

    try:
        value = ast.literal_eval(raw)
        if isinstance(value, (list, tuple)) and len(value) > 0:
            return float(value[0]), "list"
    except Exception:
        pass

    try:
        return float(raw), "scalar"
    except Exception:
        return None, None


def _write_ini_preserve_kind(parser: ConfigParser, section: str, key: str, value: float, kind: str):
    """Write an INI value while preserving scalar/list representation."""
    _ensure_section(parser, section)
    if kind == "list":
        parser.set(section, key, str([float(value)]))
    else:
        parser.set(section, key, str(float(value)))


# ----------------------------------------------------------------------
# Generic utilities
# ----------------------------------------------------------------------
def _extract_scalar_metric(val) -> float:
    """
    Convert a TIPTOP metric into a scalar float.
    """
    v = val

    for _ in range(3):
        if isinstance(v, (list, tuple)):
            if len(v) == 0:
                return float("nan")
            v = v[0]

    if hasattr(v, "get"):
        try:
            v = v.get()
        except Exception:
            return float("nan")

    try:
        arr = np.asarray(v, dtype=float)
    except Exception:
        return float("nan")

    if arr.size == 0 or not np.isfinite(arr).any():
        return float("nan")

    return float(arr.ravel()[0])


def _expand_to_list(x, n, name="parameter"):
    """
    Normalize a parameter to a list of length `n`.
    """
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) != n:
            raise ValueError(f"{name} must have length {n}, got {len(x)}")
        return [float(v) for v in x]
    return [float(x)] * n


# ----------------------------------------------------------------------
# Barycenter / blur helpers
# ----------------------------------------------------------------------
def compute_barycenter_from_polar(zenith_list, azimuth_list):
    """
    Compute the geometric barycenter of an asterism in the field plane.

    Assumptions
    -----------
    - zenith  -> polar radius in arcsec
    - azimuth -> polar angle in degrees
    """
    zen = np.asarray(zenith_list, dtype=float)
    az_deg = np.asarray(azimuth_list, dtype=float)
    az_rad = np.deg2rad(az_deg)

    x = zen * np.cos(az_rad)
    y = zen * np.sin(az_rad)

    x_bar = float(np.mean(x))
    y_bar = float(np.mean(y))
    rho_bar = float(np.hypot(x_bar, y_bar))

    return {
        "x_arcsec": x_bar,
        "y_arcsec": y_bar,
        "radius_arcsec": rho_bar,
    }


def evaluate_blur_model(model_cfg, bary_radius_arcsec):
    """
    Evaluate a blur model from the barycenter radius.

    Supported laws
    --------------
    - "polynomial":
        a * r^2 + b * r + c
    - "poly_plus_linear":
        a * r^2 + b * r + c + d * r
    """
    law = str(model_cfg.get("law", "polynomial")).strip().lower()
    r = float(bary_radius_arcsec)

    a = float(model_cfg.get("poly_a_mas_per_arcsec2", 0.0))
    b = float(model_cfg.get("poly_b_mas_per_arcsec", 0.0))
    c = float(model_cfg.get("poly_c_mas", 0.0))
    d = float(model_cfg.get("poly_d_mas_per_arcsec", 0.0))

    if law == "polynomial":
        blur_mas = a * r**2 + b * r + c
    elif law == "poly_plus_linear":
        blur_mas = a * r**2 + b * r + c + d * r
    else:
        raise ValueError(f"Unknown blur law: {law}")

    return float(max(0.0, blur_mas))


def compute_blur_mas(blur_cfg, bary_radius_arcsec, blur_model_key=None):
    """
    Compute blur in mas from a blur configuration and barycenter radius.
    """
    if not blur_cfg:
        return None

    if not bool(blur_cfg.get("enabled", False)):
        return None

    models = blur_cfg.get("models", {})
    if not isinstance(models, dict) or not models:
        return None

    if blur_model_key is None:
        blur_model_key = sorted(models.keys())[0]

    if str(blur_model_key) not in models:
        raise KeyError(f"Blur model '{blur_model_key}' not found")

    model_cfg = models[str(blur_model_key)]
    return evaluate_blur_model(model_cfg, bary_radius_arcsec)


def add_blur_to_ini_jitter(parser: ConfigParser, blur_mas):
    """
    Add blur in quadrature to telescope.jitter_FWHM.
    """
    if blur_mas is None:
        return None

    if not np.isfinite(float(blur_mas)):
        return None

    sec = "telescope"
    key = "jitter_FWHM"

    old, kind = _read_ini_float_or_list_first(parser, sec, key)
    if old is None:
        old = 0.0
        kind = "scalar"

    new = float(np.sqrt(old**2 + float(blur_mas)**2))
    _write_ini_preserve_kind(parser, sec, key, new, kind)

    return {
        "jitter_old_mas": float(old),
        "blur_mas": float(blur_mas),
        "jitter_new_mas": float(new),
    }


# ----------------------------------------------------------------------
# Blur configuration families
# ----------------------------------------------------------------------

# 2 s detector
blur_2s_detec_cfg = {
    "enabled": True,
    "models": {
        "1": {
            "law": "poly_plus_linear",
            "poly_a_mas_per_arcsec2": -0.00009,
            "poly_b_mas_per_arcsec": 0.062,
            "poly_c_mas": 3.6034,
            "poly_d_mas_per_arcsec": 0.1,
        },
        "2": {
            "law": "polynomial",
            "poly_a_mas_per_arcsec2": -0.00009,
            "poly_b_mas_per_arcsec": 0.062,
            "poly_c_mas": 3.6034,
        },
        "3": {
            "law": "polynomial",
            "poly_a_mas_per_arcsec2": 0.0005,
            "poly_b_mas_per_arcsec": +0.00007,
            "poly_c_mas": 3.7351,
        },
    },
}


# 900 s post AO
blur_post_ao_cfg = {
    "enabled": True,
    "models": {
        "1": {
            "law": "poly_plus_linear",
            "poly_a_mas_per_arcsec2": -0.00006,
            "poly_b_mas_per_arcsec": 0.0596,
            "poly_c_mas": 1.9794,
            "poly_d_mas_per_arcsec": 0.1,
        },
        "2": {
            "law": "polynomial",
            "poly_a_mas_per_arcsec2": -0.00006,
            "poly_b_mas_per_arcsec": 0.0596,
            "poly_c_mas": 1.9794,
        },
        "3": {
            "law": "polynomial",
            "poly_a_mas_per_arcsec2": 0.0005,
            "poly_b_mas_per_arcsec": -0.0015,
            "poly_c_mas": 2.1359,
        },
    },
}

# 900 s at detector
blur_900s_detector_cfg = {
    "enabled": True,
    "models": {
        "1": {
            "law": "poly_plus_linear",
            "poly_a_mas_per_arcsec2": -0.00006,
            "poly_b_mas_per_arcsec": 0.0596,
            "poly_c_mas": 3.9974,
            "poly_d_mas_per_arcsec": 0.1,
        },
        "2": {
            "law": "polynomial",
            "poly_a_mas_per_arcsec2": -0.00006,
            "poly_b_mas_per_arcsec": 0.0596,
            "poly_c_mas": 3.9974,
        },
        "3": {
            "law": "polynomial",
            "poly_a_mas_per_arcsec2": 0.0005,
            "poly_b_mas_per_arcsec": -0.0015,
            "poly_c_mas": 4.154,
        },
    },
}

# Single OB
blur_single_ob_cfg = {
    "enabled": True,
    "models": {
        "1": {
            "law": "poly_plus_linear",
            "poly_a_mas_per_arcsec2": 0.0001,
            "poly_b_mas_per_arcsec": 0.0388,
            "poly_c_mas": 5.6491,
            "poly_d_mas_per_arcsec": 0.1,
        },
        "2": {
            "law": "polynomial",
            "poly_a_mas_per_arcsec2": 0.0001,
            "poly_b_mas_per_arcsec": 0.0388,
            "poly_c_mas": 5.6491,
        },
        "3": {
            "law": "polynomial",
            "poly_a_mas_per_arcsec2": 0.0004,
            "poly_b_mas_per_arcsec": -0.0038,
            "poly_c_mas": 5.799,
        },
    },
}

# Multi OB
blur_multi_ob_cfg = {
    "enabled": True,
    "models": {
        "1": {
            "law": "poly_plus_linear",
            "poly_a_mas_per_arcsec2": 0.0001,
            "poly_b_mas_per_arcsec": 0.0388,
            "poly_c_mas": 6.3484,
            "poly_d_mas_per_arcsec": 0.1,
        },
        "2": {
            "law": "polynomial",
            "poly_a_mas_per_arcsec2": 0.0001,
            "poly_b_mas_per_arcsec": 0.0388,
            "poly_c_mas": 6.3484,
        },
        "3": {
            "law": "polynomial",
            "poly_a_mas_per_arcsec2": 0.0004,
            "poly_b_mas_per_arcsec": -0.0038,
            "poly_c_mas": 6.4983,
        },
    },
}

# Single OB Coarse
blur_single_ob_coarse_cfg = {
    "enabled": True,
    "models": {
        "1": { 
            "law": "poly_plus_linear", 
            "poly_a_mas_per_arcsec2": 0.0001, 
            "poly_b_mas_per_arcsec": 0.0388, 
            "poly_c_mas": 42.405,
            "poly_d_mas_per_arcsec": 0.1
                },
        "2": {
            "law": "polynomial", 
            "poly_a_mas_per_arcsec2": 0.0001, 
            "poly_b_mas_per_arcsec": 0.0388,
            "poly_c_mas": 42.405
            },
        "3": { 
            "law": "polynomial", 
            "poly_a_mas_per_arcsec2": 0.0004, 
            "poly_b_mas_per_arcsec": -0.0038,
            "poly_c_mas": 42.554
            },
    },
}

BLUR_FAMILIES = {
    "detector_2s": {
        "label": "2s @ detector",
        "config": blur_2s_detec_cfg,
    },
    "post_ao_900s": {
        "label": "900s post AO",
        "config": blur_post_ao_cfg,
    },
    "detector_900s": {
        "label": "900s @ detector",
        "config": blur_900s_detector_cfg,
    },
    "single_ob": {
        "label": "Single-OB (4x900s)",
        "config": blur_single_ob_cfg,
    },
    "multi_ob": {
        "label": "Multi-OB",
        "config": blur_multi_ob_cfg,
    },
    "single_ob_coarse": {
        "label": "Single-OB #Coarse",
        "config": blur_single_ob_coarse_cfg,
    },
}