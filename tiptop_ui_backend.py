
# -*- coding: utf-8 -*-
"""
Backend helpers for the Dash UI.

Important behavior
------------------
- `build_ini_only(...)` only writes a temporary INI file in a temporary workspace.
  It never calls TIPTOP.
- `run_simulation_from_ini(...)` is the only function that calls TIPTOP.
- Files are intended to be packaged for immediate download and then deleted.
"""
print("[STARTUP] tiptop_ui_backend import started", flush=True)

import glob
import os
import tempfile
from configparser import ConfigParser

from Utils_HRM_MCAO import (
    photons_from_mag,
    framerate_from_mag_lo,
    _ensure_section,
    _set_list,
    _repeat_first_ini_value,
    _extract_scalar_metric,
    _expand_to_list,
    compute_barycenter_from_polar,
    compute_blur_mas,
    add_blur_to_ini_jitter,
    atmo_select,
    build_opt_lists,
    BLUR_FAMILIES,
)

print("[STARTUP] tiptop_ui_backend imports done", flush=True)

PRESETS = {
    "best": {
        "label": "Best",
        "qno_txt": "Q1",
        "ngs_Zenith": [40, 40],
        "ngs_Azimuth": [60, -60],
        "mag_lo": [16, 20],
        "mag_focus": [18, 21.5],
    },
    "median": {
        "label": "Median",
        "qno_txt": "Median",
        "ngs_Zenith": [60, 60],
        "ngs_Azimuth": [49, -49],
        "mag_lo": [17.5, 20],
        "mag_focus": [19.5, 21.5],
    },
    "worst": {
        "label": "Worst",
        "qno_txt": "Q4",
        "ngs_Zenith": [70, 70],
        "ngs_Azimuth": [31, -31],
        "mag_lo": [17.5, 20],
        "mag_focus": [19.5, 21.5],
    },
}

def get_blur_family_options():
    return [
        {"value": key, "label": value["label"]}
        for key, value in BLUR_FAMILIES.items()
    ]

def _normalize_energy_mode(energy_mode: str) -> str:
    mode = str(energy_mode).strip().lower()
    if mode not in {"ensquared", "encircled"}:
        raise ValueError("energy_mode must be 'ensquared' or 'encircled'")
    return mode


def build_ini_only(
    preset_name="median",
    src_Zenith=[0],
    src_Azimuth=[0],
    wvl_sci=2.2,
    zenith_angle=30,
    energy_mode="ensquared",
    ee_radius_mas=50.0,
    use_blur=True,
    blur_mode="family", # "family" or "custom"
    blur_family_key="post_ao_900s",
    custom_blur_mas=None,
    ini_basename="HARMONI_MCAO",
    params_dir="./",
    output_dir=None,
    optimization_radius=10.0,
    optimization_fov=80.0,
    scale_freq_focus=0.2,
    mmse_rec_lo=True,
    # path_apodizer=None,
    path_pupil=None,
):
    """
    Build a temporary INI file for one simulation without running TIPTOP.
    """
    preset_key = str(preset_name).lower().strip()
    if preset_key not in PRESETS:
        raise KeyError(f"Unknown preset: {preset_name}")

    cfg = PRESETS[preset_key]
    preset_label = cfg["label"]

    qno_txt = cfg["qno_txt"]
    ngs_Zenith = cfg["ngs_Zenith"]
    ngs_Azimuth = cfg["ngs_Azimuth"]
    mag_lo = cfg["mag_lo"]
    mag_focus = cfg["mag_focus"]

    energy_mode = _normalize_energy_mode(energy_mode)
    ee_radius_mas = float(ee_radius_mas)

    if ee_radius_mas <= 0:
        raise ValueError("ee_radius_mas must be > 0")

    if not params_dir:
        raise ValueError("params_dir must be provided.")

    if not output_dir:
        output_dir = tempfile.mkdtemp(prefix=f"tiptop_{preset_key}_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    if len(ngs_Zenith) != len(ngs_Azimuth):
        raise ValueError("ngs_Zenith and ngs_Azimuth must have the same length.")

    n_ngs = len(ngs_Zenith)
    if n_ngs == 0:
        raise ValueError("At least one guide star is required.")

    mag_lo_list = _expand_to_list(mag_lo, n_ngs, name="mag_lo")
    mag_focus_list = _expand_to_list(mag_focus, n_ngs, name="mag_focus")

    fr_lo_list = [framerate_from_mag_lo(m) for m in mag_lo_list]
    fr_focus_list = [fr * float(scale_freq_focus) for fr in fr_lo_list]

    wvl_m = float(wvl_sci) * 1e-6

    photons_lo = [
        photons_from_mag("LO", mag, fr)
        for mag, fr in zip(mag_lo_list, fr_lo_list)
    ]
    photons_focus = [
        photons_from_mag("FOCUS", mag, fr)
        for mag, fr in zip(mag_focus_list, fr_focus_list)
    ]

    optimization_zenith, optimization_weight = build_opt_lists(
            optimization_radius,
            optimization_fov,
        )

    auto_blur_model_key = str(n_ngs)
    
    bary = compute_barycenter_from_polar(ngs_Zenith, ngs_Azimuth)

    selected_blur_label = "Disabled"
    blur_mas = None

    if bool(use_blur):
        blur_mode = str(blur_mode).strip().lower()

        if blur_mode == "family":
            if blur_family_key not in BLUR_FAMILIES:
                raise KeyError(f"Unknown blur family: {blur_family_key}")

            selected_blur_cfg = BLUR_FAMILIES[blur_family_key]["config"]
            selected_blur_label = BLUR_FAMILIES[blur_family_key]["label"]

            blur_mas = compute_blur_mas(
                blur_cfg=selected_blur_cfg,
                bary_radius_arcsec=bary["radius_arcsec"],
                blur_model_key=auto_blur_model_key,
            )

        elif blur_mode == "custom":
            if custom_blur_mas is None:
                raise ValueError("custom_blur_mas must be provided when blur_mode='custom'.")

            blur_mas = float(custom_blur_mas)
            if blur_mas < 0:
                raise ValueError("custom_blur_mas must be >= 0.")

            selected_blur_label = "Custom blur"

        else:
            raise ValueError("blur_mode must be 'family' or 'custom'")

    template_ini = os.path.join(params_dir, f"{ini_basename}.ini")
    if not os.path.exists(template_ini):
        raise FileNotFoundError(f"Template INI not found: {template_ini}")

    parser = ConfigParser()
    parser.optionxform = str
    loaded_files = parser.read(template_ini)
    if not loaded_files:
        raise RuntimeError(f"ConfigParser could not read template INI: {template_ini}")

    ini_name = f"{ini_basename}_{preset_label}"
    ini_file = os.path.join(output_dir, f"{ini_name}.ini")

    seeing, cn2w, wind_speed = atmo_select(qno_txt)

    _ensure_section(parser, "atmosphere")
    parser.set("atmosphere", "Seeing", str(float(seeing)))
    parser.set("atmosphere", "Cn2Weights", str(list(map(float, cn2w))))
    parser.set("atmosphere", "WindSpeed", str(list(map(float, wind_speed))))

    _set_list(parser, "sources_science", "Zenith", src_Zenith)
    _set_list(parser, "sources_science", "Azimuth", src_Azimuth)
    parser.set("sources_science", "Wavelength", str([float(wvl_m)]))

    if wvl_m > 1000e-9:
        _ensure_section(parser, "sensor_science")
        parser.set("sensor_science", "PixelScale", str(2.0))
    if wvl_m > 2000e-9:
        _ensure_section(parser, "sensor_science")
        parser.set("sensor_science", "PixelScale", str(4.0))

    _ensure_section(parser, "telescope")
    parser.set("telescope", "ZenithAngle", str(float(zenith_angle)))

    # if path_apodizer is not None:
    #     parser.set("telescope", "PathApodizer", repr(str(path_apodizer)))
    if path_pupil is not None:
        parser.set("telescope", "PathPupil", repr(str(path_pupil)))

    jitter_info = add_blur_to_ini_jitter(parser, blur_mas)

    _ensure_section(parser, "DM")
    parser.set("DM", "OptimizationZenith", str(optimization_zenith))
    parser.set("DM", "OptimizationWeight", str(optimization_weight))

    _ensure_section(parser, "RTC")
    parser.set("RTC", "MMSE_Rec_LO", str(bool(mmse_rec_lo)))

    _set_list(parser, "sources_LO", "Zenith", ngs_Zenith)
    _set_list(parser, "sources_LO", "Azimuth", ngs_Azimuth)
    _set_list(parser, "sensor_LO", "NumberPhotons", photons_lo)
    _set_list(parser, "RTC", "SensorFrameRate_LO", fr_lo_list)

    _set_list(parser, "sensor_Focus", "NumberPhotons", photons_focus)
    _set_list(parser, "RTC", "SensorFrameRate_Focus", fr_focus_list)

    _repeat_first_ini_value(parser, "sensor_LO", "NumberLenslets", n_ngs)
    _repeat_first_ini_value(parser, "sensor_Focus", "NumberLenslets", n_ngs)

    with open(ini_file, "w", encoding="utf-8") as configfile:
        parser.write(configfile)

    return {
        "preset": preset_key,
        "preset_label": preset_label,
        "qno_txt": qno_txt,
        "ngs_Zenith": ngs_Zenith,
        "ngs_Azimuth": ngs_Azimuth,
        "mag_lo": mag_lo_list,
        "mag_focus": mag_focus_list,
        "src_Zenith": src_Zenith,
        "src_Azimuth": src_Azimuth,
        "wvl_sci": float(wvl_sci),
        "wvl_m": float(wvl_m),
        "zenith_angle": float(zenith_angle),
        "energy_mode": energy_mode,
        "ee_radius_mas": ee_radius_mas,
        "ensquared_energy": (energy_mode == "ensquared"),
        "fr_lo": fr_lo_list,
        "fr_focus": fr_focus_list,
        "photons_lo": photons_lo,
        "photons_focus": photons_focus,
        "blur_enabled": bool(use_blur),
        "blur_mode": blur_mode if bool(use_blur) else "disabled",
        "custom_blur_mas": None if custom_blur_mas is None else float(custom_blur_mas),
        "blur_family_key": blur_family_key,
        "blur_family_label": selected_blur_label,
        "blur_model_key": auto_blur_model_key,
        "barycenter": bary,
        "blur_mas": blur_mas,
        "jitter_info": jitter_info,
        "ini_file": ini_file,
        "ini_name": ini_name,
        "output_dir": output_dir,
        # "path_apodizer": path_apodizer,
        "path_pupil": path_pupil,
    }

def compute_strehl_like_maoppy(psf, pupil, samp, threshold=None):
            import numpy as np
            from tiptop.tiptop import  FourierUtils

            otf = np.fft.fftshift(FourierUtils.psf2otf(psf))
            otf = otf / otf.max()

            otf_dl = FourierUtils.telescopeOtf(pupil, samp)
            otf_dl = FourierUtils.interpolateSupport(otf_dl, otf.shape)

            if threshold is not None:
                mask = np.real(otf_dl) > threshold
                return np.real(otf[mask].sum() / otf_dl[mask].sum())

            return np.real(otf.sum() / otf_dl.sum())

def run_simulation_from_ini(
    ini_info,
    verbose=False,
    save_FITS=False,
    save_PSDs=False,
    sr_method="otf",
):
    """
    Run TIPTOP from a dictionary returned by `build_ini_only`.
    """
    from tiptop.tiptop import baseSimulation, FourierUtils
    import numpy as np

    rad2mas = 3600 * 180 * 1000 / np.pi

    sr_method = str(sr_method).strip().lower()
    if sr_method not in {"max", "otf"}:
        raise ValueError("sr_method must be 'max' or 'otf'")

    output_prefix = f"psf_{ini_info['preset_label']}"
    sim = baseSimulation(
        path=ini_info["output_dir"],
        parametersFile=ini_info["ini_name"],
        outputDir=ini_info["output_dir"],
        outputFile=output_prefix,
        verbose=verbose,
        getHoErrorBreakDown=True,
        ensquaredEnergy=bool(ini_info["ensquared_energy"]),
        eeRadiusInMas=float(ini_info["ee_radius_mas"]),
        savePSDs=save_PSDs,
    )

    sim.doOverallSimulation()
    sim.computeMetrics()

    # Recompute Strehl with the otf method
    if sr_method == "otf" and hasattr(sim, "results") and sim.results is not None:
        sr_values = []
        for i in range(sim.nWvl):
            if sim.nWvl > 1:
                results = sim.results[i]
            else:
                results = sim.results

            samp = sim.wvl[i] * rad2mas / (sim.psInMas * 2 * sim.tel_radius)
            sr_i = []
            for img in results:
                sr_i.append(
                    FourierUtils.getStrehl(
                        img.sampling,
                        sim.fao.ao.tel.pupil,
                        samp,
                        method='otf',
                        psfInOnePix=True,
                    )
                )

            if sim.nWvl > 1:
                sr_values.append(sr_i)
            else:
                sr_values = sr_i

        sim.sr = sr_values

    psf_data = None
    psf_pix_mas = None

    if hasattr(sim, "cubeResultsArray") and sim.cubeResultsArray is not None:
        try:
            psf_data = np.asarray(sim.cubeResultsArray[0])
        except Exception:
            psf_data = None

    if hasattr(sim, "psInMas"):
        try:
            psf_pix_mas = float(sim.psInMas)
        except Exception:
            psf_pix_mas = None

    fits_files = []
    if save_FITS:
        sim.saveResults()
        fits_files = sorted(glob.glob(os.path.join(ini_info["output_dir"], "*.fits*")))

    sr = _extract_scalar_metric(sim.sr)
    fwhm = _extract_scalar_metric(sim.fwhm)
    ee = _extract_scalar_metric(sim.ee)
    penalty = _extract_scalar_metric(sim.penalty)
    ho_res = _extract_scalar_metric(sim.HO_res)
    lo_res = _extract_scalar_metric(sim.LO_res)
    gf_res = _extract_scalar_metric(sim.GF_res)

    out = dict(ini_info)
    out.update(
        {
            "strehl": sr,
            "fwhm": fwhm,
            "ee": ee,
            "penalty": penalty,
            "HO_res": ho_res,
            "LO_res": lo_res,
            "GF_res": gf_res,
            "save_FITS": bool(save_FITS),
            "save_PSDs":bool(save_PSDs),
            "fits_files": fits_files,
            "simulation": sim,
            "psf_data": None if psf_data is None else np.asarray(psf_data).tolist(),
            "psf_pix_mas": psf_pix_mas,
        }
    )
    return out
