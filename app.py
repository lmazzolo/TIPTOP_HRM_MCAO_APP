
# -*- coding: utf-8 -*-
"""
Dash app for preparing or running a single TIPTOP simulation.

Behavior
--------
- "Generate INI" builds the INI file and immediately offers it as a download.
- "Run simulation" runs TIPTOP and can immediately offer FITS files as a download.
- Files are created in a temporary workspace and removed after packaging.
"""
print("[STARTUP] app.py import started", flush=True)

import base64
import os
import shutil
import traceback
import zipfile
from io import BytesIO

from dash import Dash, dcc, html, Input, Output, State, callback_context, no_update

import numpy as np
import plotly.graph_objects as go

# import tiptop
# print("TIPTOP version:", getattr(tiptop, "__version__", "unknown"))

from tiptop_ui_backend import (
    PRESETS,
    build_ini_only,
    run_simulation_from_ini,
    get_blur_family_options,
)

print("[STARTUP] app.py imports done", flush=True)

def parse_float_list(text, field_name):
    if text is None:
        raise ValueError(f"Field '{field_name}' is empty.")
    parts = [x.strip() for x in str(text).split(",") if x.strip()]
    if not parts:
        raise ValueError(f"Field '{field_name}' is empty.")
    try:
        return [float(x) for x in parts]
    except Exception as exc:
        raise ValueError(
            f"Field '{field_name}' must be a comma-separated list of numbers."
        ) from exc


def parse_bool(value):
    if isinstance(value, list):
        return len(value) > 0
    return bool(value)


def metric_card(title, value):
    try:
        shown = f"{float(value):.3f}"
    except Exception:
        shown = str(value)
    return html.Div(
        [
            html.Div(title, style={"fontWeight": "700", "fontSize": "0.9rem"}),
            html.Div(shown, style={"fontSize": "1.1rem"}),
        ],
        style={
            "border": "1px solid #ddd",
            "borderRadius": "12px",
            "padding": "12px",
            "backgroundColor": "#fafafa",
        },
    )


def nice_value(v, digits=3):
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    if isinstance(v, list):
        return ", ".join(nice_value(x, digits=digits) for x in v)
    return str(v)


def info_row(label, value, digits=3):
    return html.Div(
        [
            html.Div(
                label,
                style={"fontWeight": "600", "color": "#555", "minWidth": "220px"},
            ),
            html.Div(nice_value(value, digits=digits), style={"flex": "1"}),
        ],
        style={
            "display": "flex",
            "gap": "12px",
            "padding": "8px 0",
            "borderBottom": "1px solid #eee",
        },
    )


def section_card(title, children):
    return html.Div(
        [
            html.H4(title, style={"marginTop": "0", "marginBottom": "12px"}),
            html.Div(children),
        ],
        style={
            "backgroundColor": "white",
            "border": "1px solid #ddd",
            "borderRadius": "12px",
            "padding": "16px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
        },
    )


def make_text_download(path, filename=None):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return {
        "content": content,
        "filename": filename or os.path.basename(path),
        "type": "text/plain",
    }


def make_binary_download(path, filename=None, mime_type="application/octet-stream"):
    with open(path, "rb") as f:
        payload = f.read()
    return {
        "content": base64.b64encode(payload).decode("utf-8"),
        "filename": filename or os.path.basename(path),
        "type": mime_type,
        "base64": True,
    }


def make_fits_download(fits_files, base_name):
    if not fits_files:
        return no_update

    if len(fits_files) == 1:
        return make_binary_download(
            fits_files[0],
            filename=os.path.basename(fits_files[0]),
            mime_type="application/fits",
        )

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in fits_files:
            zf.write(fp, arcname=os.path.basename(fp))

    return {
        "content": base64.b64encode(zip_buffer.getvalue()).decode("utf-8"),
        "filename": f"{base_name}_fits.zip",
        "type": "application/zip",
        "base64": True,
    }


def cleanup_workspace(data):
    if not data:
        return
    output_dir = data.get("output_dir")
    if output_dir and os.path.isdir(output_dir):
        shutil.rmtree(output_dir, ignore_errors=True)


def energy_mode_label(mode):
    return "Ensquared energy" if str(mode).lower() == "ensquared" else "Encircled energy"

def jitter_value(data, key):
    jitter_info = data.get("jitter_info")
    if not jitter_info:
        return "N/A"
    return jitter_info.get(key, "N/A")

def make_psf_figure(psf, pix_mas, title="PSF", cmap="Spectral_r"):
    psf = np.asarray(psf, dtype=float)

    if psf.ndim != 2:
        raise ValueError("PSF must be a 2D array.")

    total_flux = np.sum(psf)
    if total_flux > 0:
        psf = psf / total_flux

    ny, nx = psf.shape
    x_axis = (np.arange(nx) - nx // 2) * pix_mas * 1e-3
    y_axis = (np.arange(ny) - ny // 2) * pix_mas * 1e-3

    psf_max = np.max(psf)
    if psf_max <= 0:
        z_plot = np.zeros_like(psf)
        zmin = -6
        zmax = 0
    else:
        vmin = psf_max * 1e-6
        z_plot = np.log10(np.clip(psf, vmin, None))
        zmin = np.log10(vmin)
        zmax = np.log10(psf_max)

    fig = go.Figure(
        data=go.Heatmap(
            x=x_axis,
            y=y_axis,
            z=z_plot,
            colorscale=cmap,
            zmin=zmin,
            zmax=zmax,
            colorbar=dict(title="log10(Intensity)"),
            hovertemplate=(
                "x: %{x:.4f} arcsec<br>"
                "y: %{y:.4f} arcsec<br>"
                "log10(I): %{z:.3f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        title=title,
        template="plotly_dark",
        xaxis_title="[arcsec]",
        yaxis_title="[arcsec]",
        margin=dict(l=50, r=30, t=50, b=50),
    )
    fig.update_yaxes(scaleanchor="x")
    return fig

def derived_inputs_display(result):
    return html.Div(
        [
            section_card(
                "Generated INI",
                [
                    info_row("INI name", result["ini_name"]),
                    info_row("FITS download requested", "Yes" if result.get("save_FITS", False) else "No"),
                    info_row("PSD output requested", "Yes" if result.get("save_PSDs", False) else "No"),
                    info_row("Energy metric", energy_mode_label(result["energy_mode"])),
                    info_row("Energy radius [mas]", result["ee_radius_mas"], digits=2),
                    info_row(
                        "Pupil stop",
                        "CPS4 Pupil Stop" if result.get("path_pupil") else "Full aperture",
                    ),
                ],
            ),
            section_card(
                "Computed sensor values",
                [
                    info_row("LO frame rates [Hz]", result["fr_lo"], digits=2),
                    info_row("Focus frame rates [Hz]", result["fr_focus"], digits=2),
                    info_row("LO photons / frame / subaperture", result["photons_lo"], digits=2),
                    info_row("Focus photons / frame / subaperture", result["photons_focus"], digits=2),
                ],
            ),
            section_card(
                "Barycenter",
                [
                    info_row("Barycenter X [arcsec]", result["barycenter"]["x_arcsec"], digits=3),
                    info_row("Barycenter Y [arcsec]", result["barycenter"]["y_arcsec"], digits=3),
                    info_row("Barycenter radius [arcsec]", result["barycenter"]["radius_arcsec"], digits=3),
                ],
            ),
            section_card(
                "Blur settings",
                [
                    info_row("Blur enabled", "Yes" if result.get("blur_enabled", False) else "No"),
                    info_row("Blur mode", result.get("blur_mode", "disabled")),
                    info_row("Blur family / source", result["blur_family_label"]),
                    *(
                        [info_row("Automatic blur sub-model", result["blur_model_key"])]
                        if result.get("blur_mode") == "family"
                        else []
                    ),
                    *(
                        [info_row("Custom blur input [mas]", result.get("custom_blur_mas"))]
                        if result.get("blur_mode") == "custom"
                        else []
                    ),
                    info_row("Added blur [mas]", result["blur_mas"], digits=3),
                    info_row("Initial jitter_FWHM [mas]", jitter_value(result, "jitter_old_mas"), digits=3),
                    info_row("Final jitter_FWHM [mas]", jitter_value(result, "jitter_new_mas"), digits=3),
                ],
            ),
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": "1fr 1fr",
            "gap": "16px",
        },
    )


def ini_display(ini_info):
    return html.Div(
        [
            html.H3("INI generated"),
            html.Div(
                [
                    section_card(
                        "Configuration",
                        [
                            info_row("Preset", ini_info["preset_label"]),
                            info_row(
                                "Pupil stop",
                                "CPS4 Pupil Stop" if ini_info.get("path_pupil") else "Full aperture",
                            ),
                            info_row("Atmosphere profile", ini_info["qno_txt"]),
                            info_row("Energy metric", energy_mode_label(ini_info["energy_mode"])),
                            info_row("Energy radius [mas]", ini_info["ee_radius_mas"], digits=2),
                            info_row("INI name", ini_info["ini_name"]),
                        ],
                    ),
                    section_card(
                        "Science target",
                        [
                            info_row("Source position", "On-axis"),
                            info_row("Science distance [arcsec]", ini_info["src_Zenith"]),
                            info_row("Science angle [deg]", ini_info["src_Azimuth"]),
                            info_row("Science wavelength [µm]", ini_info["wvl_sci"], digits=3),
                            info_row("Zenith angle [deg]", ini_info["zenith_angle"], digits=2),
                        ],
                    ),
                    section_card(
                        "Computed values",
                        [
                            info_row("LO frame rates [Hz]", ini_info["fr_lo"], digits=2),
                            info_row("Focus frame rates [Hz]", ini_info["fr_focus"], digits=2),
                            info_row("LO photons / frame", ini_info["photons_lo"], digits=2),
                            info_row("Focus photons / frame", ini_info["photons_focus"], digits=2),
                        ],
                    ),
                    section_card(
                        "Barycenter and blur",
                        [
                            info_row("Barycenter X [arcsec]", ini_info["barycenter"]["x_arcsec"], digits=3),
                            info_row("Barycenter Y [arcsec]", ini_info["barycenter"]["y_arcsec"], digits=3),
                            info_row("Barycenter radius [arcsec]", ini_info["barycenter"]["radius_arcsec"], digits=3),
                            info_row("Added blur [mas]", ini_info["blur_mas"], digits=3),
                            info_row("Initial jitter [mas]", jitter_value(ini_info, "jitter_old_mas"), digits=3),
                            info_row("Final jitter [mas]", jitter_value(ini_info, "jitter_new_mas"), digits=3),
                        ],
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 1fr",
                    "gap": "16px",
                },
            ),
        ]
    )

def psf_display(result):
    psf_data = result.get("psf_data")
    pix_mas = result.get("psf_pix_mas")

    if psf_data is None or pix_mas is None:
        return section_card(
            "PSF preview",
            [html.Div("No PSF data available for display.")],
        )

    psf = np.asarray(psf_data, dtype=float)

    sr = result.get("strehl")
    fwhm = result.get("fwhm")

    title = "AO PSF"
    if sr is not None and fwhm is not None:
        try:
            title = f"AO PSF (SR={100*float(sr):.1f}%, FWHM={float(fwhm):.1f} mas)"
        except Exception:
            pass

    psf_fig = make_psf_figure(psf, pix_mas=float(pix_mas), title=title)

    return html.Div(
        [
            section_card(
                "PSF preview",
                [
                    dcc.Graph(
                        figure=psf_fig,
                        style={"height": "520px"},
                        config={"displayModeBar": True},
                    )
                ],
            ),
        ]
    )

app = Dash(__name__)
print("[STARTUP] Dash app created", flush=True)
app.title = "TIPTOP HRM MCAO launcher"
server = app.server
print("[STARTUP] Flask server created", flush=True)
@server.route("/ping")
def ping():
    return "pong", 200
@server.route("/healthz")
def healthz():
    return "ok", 200

BOX = {
    "border": "1px solid #ddd",
    "borderRadius": "12px",
    "padding": "14px",
    "marginBottom": "16px",
    "backgroundColor": "white",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
}
LABEL = {"fontWeight": "600", "marginBottom": "4px", "display": "block"}
TEXT = {
    "width": "100%",
    "padding": "8px",
    "borderRadius": "8px",
    "border": "1px solid #ccc",
}

RUNNING_STATUS = html.Div(
    "Simulation is running - please wait...",
    style={
        "padding": "12px",
        "backgroundColor": "#fff3cd",
        "border": "1px solid #ffecb5",
        "borderRadius": "10px",
        "marginBottom": "16px",
        "color": "#664d03",
        "fontWeight": "600",
    },
)

DEFAULT_SRC_ZENITH = [0.0]
DEFAULT_SRC_AZIMUTH = [0.0]

BLUR_FAMILY_OPTIONS = get_blur_family_options()
print("[STARTUP] blur family options created", flush=True)

app.layout = html.Div(
    [
        html.H2("TIPTOP - HARMONI MCAO - .ini Generator & Simulation launcher"),
        html.P(
            "Choose a preset, edit the parameters, select the energy metric and radius, then generate the INI file or run the simulation."
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Preset", style=LABEL),
                        dcc.RadioItems(
                            id="preset",
                            options=[
                                {"label": "Best", "value": "best"},
                                {"label": "Median", "value": "median"},
                                {"label": "Worst", "value": "worst"},
                            ],
                            value="median",
                            inline=True,
                        ),
                    ],
                    style=BOX,
                ),
                html.Div(
                    [
                        html.Div("Science target", style={"fontWeight": "700", "marginBottom": "10px"}),
                        html.P(
                            "Source on-axis",
                            style={
                                "marginTop": "0",
                                "marginBottom": "12px",
                                "fontWeight": "600",
                                "color": "#084298",
                            },
                        ),
                        html.Label("Science wavelength [micron]", style=LABEL),
                        dcc.Input(id="wvl_sci", type="number", value=2.2, step=0.01, style=TEXT),
                        html.Br(), html.Br(),
                        html.Label("Zenith angle [deg]", style=LABEL),
                        dcc.Input(id="zenith_angle", type="number", value=30, step=0.1, style=TEXT),
                        html.Br(), html.Br(),
                        html.Label("Energy metric", style=LABEL),
                        dcc.RadioItems(
                            id="energy_mode",
                            options=[
                                {"label": "Ensquared energy", "value": "ensquared"},
                                {"label": "Encircled energy", "value": "encircled"},
                            ],
                            value="ensquared",
                            inline=True,
                        ),
                        html.Br(),
                        html.Label("Energy radius [mas]", style=LABEL),
                        dcc.Input(id="ee_radius_mas", type="number", value=50, step="any", min=0.1, style=TEXT),
                    ],
                    style=BOX,
                ),
                html.Div(
                    [
                        html.Div("Options", style={"fontWeight": "700", "marginBottom": "10px"}),

                        html.Label("Pupil mode", style=LABEL),
                        dcc.RadioItems(
                            id="apodizer_mode",
                            options=[
                                {"label": "Full aperture", "value": "full_aperture"},
                                {"label": "Undersized", "value": "undersized"},
                            ],
                            value="full_aperture",
                            inline=True,
                        ),
                        html.Br(),
                        html.Div(id="apodizer_warning"),
                        html.Div(id="pupil_figure_container"),
                        html.Br(),

                        dcc.Checklist(
                            id="use_blur",
                            options=[{"label": "Enable blur", "value": "yes"}],
                            value=["yes"],
                        ),
                        html.Br(),

                        html.Div(
                            id="blur_options_container",
                            children=[
                                html.Label("Blur mode", style=LABEL),
                                dcc.RadioItems(
                                    id="blur_mode",
                                    options=[
                                        {"label": "Predefined blur model", "value": "family"},
                                        {"label": "Custom blur [mas]", "value": "custom"},
                                    ],
                                    value="family",
                                    inline=True,
                                ),
                                html.Br(),

                                html.Div(
                                    id="blur_family_container",
                                    children=[
                                        html.Label("Blur model", style=LABEL),
                                        dcc.Dropdown(
                                            id="blur_family_key",
                                            options=BLUR_FAMILY_OPTIONS,
                                            value="post_ao_900s",
                                            clearable=False,
                                        ),
                                        html.Br(),
                                        html.Div(
                                            [
                                                html.Div(
                                                    "For predefined blur models, the sub-model is automatically selected based on the number of NGS (1, 2, or 3) in the asterism."
                                                ),
                                                html.Div(
                                                    "Options labeled #coarse correspond to blur models adapted to a 25 mas spaxel scale, while the other options correspond to models adapted to a 6 mas spaxel scale."
                                                ),
                                            ],
                                            style={"fontSize": "0.92rem", "color": "#555"},
                                        ),
                                    ],
                                ),

                                html.Div(
                                    id="custom_blur_container",
                                    children=[
                                        html.Br(),
                                        html.Label("Custom blur [mas]", style=LABEL),
                                        dcc.Input(
                                            id="custom_blur_mas",
                                            type="number",
                                            value=3.0,
                                            step="any",
                                            min=0.0,
                                            style=TEXT,
                                        ),
                                        html.Br(),
                                        html.Br(),
                                        html.Div(
                                            [
                                                "This value is added in quadrature to the ",
                                                html.Code("telescope.jitter_FWHM"),
                                                " parameter."
                                            ],
                                            style={"fontSize": "0.92rem", "color": "#555"},
                                        )
                                    ],
                                ),
                            ],
                        ),
                        html.Br(),

                        # html.Label("Strehl computation", style=LABEL),
                        # dcc.RadioItems(
                        #     id="sr_method",
                        #     options=[
                        #         {"label": "Max", "value": "max"},
                        #         {"label": "OTF", "value": "otf"},
                        #     ],
                        #     value="max",
                        #     inline=True,
                        # ),
                        # html.Br(),

                        dcc.Checklist(
                            id="save_fits",
                            options=[{"label": "Download FITS after simulation", "value": "yes"}],
                            value=[],
                        ),
                        html.Br(),

                        dcc.Checklist(
                            id="save_psds",
                            options=[{"label": "Include PSDs in FITS output", "value": "yes"}],
                            value=[],
                        ),
                        html.Br(),

                        html.Div(
                            "FITS download contains PSF outputs, and PSDs if requested.",
                            style={"fontSize": "0.92rem", "color": "#555"},
                        ),
                    ],
                    style=BOX,

                ),
                html.Div(
                    [
                        html.Div("Preset summary (read-only)", style={"fontWeight": "700", "marginBottom": "10px"}),
                        html.Div(id="preset_summary"),
                    ],
                    style=BOX,
                ),
                html.Div(
                    [
                        html.Button(
                            "Generate INI",
                            id="generate_ini_button",
                            n_clicks=0,
                            style={
                                "padding": "12px 18px",
                                "borderRadius": "10px",
                                "border": "none",
                                "backgroundColor": "#6c757d",
                                "color": "white",
                                "fontWeight": "700",
                                "cursor": "pointer",
                                "marginRight": "12px",
                            },
                        ),
                        html.Button(
                            "Run simulation",
                            id="run_button",
                            n_clicks=0,
                            style={
                                "padding": "12px 18px",
                                "borderRadius": "10px",
                                "border": "none",
                                "backgroundColor": "#2c7be5",
                                "color": "white",
                                "fontWeight": "700",
                                "cursor": "pointer",
                            },
                        ),
                        html.Div(
                            [
                                html.Span("⚠ ", style={"fontWeight": "bold"}),
                                "Simulations can take several minutes. Please wait for completion before launching another run."
                            ],
                            style={
                                "marginTop": "10px",
                                "fontSize": "0.9rem",
                                "color": "#555",
                                "backgroundColor": "#f7f7f7",
                                "padding": "8px 10px",
                                "borderRadius": "6px",
                                "border": "1px solid #ddd",
                            },
                        ),
                    ],
                    style={"marginBottom": "20px"},
                ),
            ],
            style={"maxWidth": "900px"},
        ),
        html.Div(id="status"),
        html.Div(id="results"),
        dcc.Download(id="download_ini"),
        dcc.Download(id="download_fits"),
    ],
    style={
        "maxWidth": "1150px",
        "margin": "0 auto",
        "padding": "24px",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f5f7fb",
    },
)
print("[STARTUP] app.layout created", flush=True)

@app.callback(
    Output("preset_summary", "children"),
    Input("preset", "value"),
)
def update_preset_summary(preset_name):
    p = PRESETS[preset_name]
    return section_card(
        "Preset summary",
        [
            info_row("Atmosphere profile", p["qno_txt"]),
            info_row("Guide-star distance(s) [arcsec]", p["ngs_Zenith"]),
            info_row("Guide-star angle(s) [deg]", p["ngs_Azimuth"]),
            info_row("NGS magnitudes (H)", p["mag_lo"]),
            # info_row("Focus magnitudes (R)", p["mag_focus"]),
        ],
    )

@app.callback(
    Output("apodizer_warning", "children"),
    Input("apodizer_mode", "value"),
)
def update_apodizer_warning(mode):
    if mode == "undersized":
        return html.Div(
            [
                "⚠ Undersized pupil selected. The INI file will NOT include this option. ",
                "It is applied only during the simulation because the corresponding FITS file cannot be distributed. ",
                "Please contact us if you need it."
            ],
            style={
                "marginTop": "8px",
                "fontSize": "0.9rem",
                "color": "#664d03",
                "backgroundColor": "#fff3cd",
                "padding": "8px 10px",
                "borderRadius": "6px",
                "border": "1px solid #ffecb5",
            },
        )
    return ""

@app.callback(
    Output("pupil_figure_container", "children"),
    Input("apodizer_mode", "value"),
)
def update_pupil_figure(mode):
    if mode != "undersized":
        return ""

    return html.Div(
        [
            html.Div(
                "Illustration of the effective pupil used in simulation.",
                style={"marginBottom": "8px", "fontSize": "0.95rem", "color": "#555"},
            ),
            html.Img(
                src="/assets/pupil_stop_schema.png",
                style={
                    "width": "100%",
                    "maxWidth": "1200px",
                    "border": "1px solid #ddd",
                    "borderRadius": "8px",
                },
            ),
        ],
        style={"marginTop": "10px"},
    )


@app.callback(
    Output("blur_options_container", "style"),
    Output("blur_family_container", "style"),
    Output("custom_blur_container", "style"),
    Input("use_blur", "value"),
    Input("blur_mode", "value"),
)
def toggle_blur_sections(use_blur, blur_mode):
    blur_enabled = parse_bool(use_blur)

    hidden = {"display": "none"}
    visible = {"display": "block"}

    if not blur_enabled:
        return hidden, hidden, hidden

    if blur_mode == "family":
        return visible, visible, hidden

    if blur_mode == "custom":
        return visible, hidden, visible

    return visible, hidden, hidden


@app.callback(
    Output("status", "children"),
    Output("results", "children"),
    Output("download_ini", "data"),
    Output("download_fits", "data"),
    Input("generate_ini_button", "n_clicks"),
    Input("run_button", "n_clicks"),
    State("preset", "value"),
    # State("src_zenith", "value"),
    # State("src_azimuth", "value"),
    State("wvl_sci", "value"),
    State("zenith_angle", "value"),
    State("energy_mode", "value"),
    State("ee_radius_mas", "value"),
    State("apodizer_mode", "value"),
    State("use_blur", "value"),
    State("blur_mode", "value"),
    State("blur_family_key", "value"),
    State("custom_blur_mas", "value"),
    # State("sr_method", "value"),
    State("save_fits", "value"),
    State("save_psds", "value"),
    prevent_initial_call=True,
    running=[
        (Output("status", "children"), RUNNING_STATUS, True),
        (Output("generate_ini_button", "disabled"), True, False),
        (Output("run_button", "disabled"), True, False),
    ],
)
def handle_actions(
    n_generate,
    n_run,
    preset,
    # src_zenith,
    # src_azimuth,
    wvl_sci,
    zenith_angle,
    energy_mode,
    ee_radius_mas,
    apodizer_mode,
    use_blur,
    blur_mode,
    blur_family_key,
    custom_blur_mas,
    # sr_method,
    save_fits,
    save_psds,
):
    ini_info = None
    try:
        triggered = callback_context.triggered[0]["prop_id"].split(".")[0]

        # path_apodizer = None
        # if apodizer_mode == "undersized":
        #     path_apodizer = 'undersized_pupil_stop_480_on_ELTgrid.fits'
        path_pupil = None
        if triggered == "run_button" and apodizer_mode == "undersized":
            path_pupil = "effective_pupil_480.fits" #tiptop ne gère pas correctement apodizer

        ini_info = build_ini_only(
            preset_name=preset,
            src_Zenith=DEFAULT_SRC_ZENITH,
            src_Azimuth=DEFAULT_SRC_AZIMUTH,
            # src_Zenith=parse_float_list(src_zenith, "src_Zenith"),
            # src_Azimuth=parse_float_list(src_azimuth, "src_Azimuth"),
            wvl_sci=float(wvl_sci),
            zenith_angle=float(zenith_angle),
            energy_mode=energy_mode,
            ee_radius_mas=float(ee_radius_mas),
            use_blur=parse_bool(use_blur),
            blur_mode=blur_mode,
            blur_family_key=blur_family_key,
            custom_blur_mas=custom_blur_mas,
            ini_basename="HARMONI_MCAO",
            params_dir="./",
            output_dir=None,
            optimization_radius=10.0,
            optimization_fov=80.0,
            scale_freq_focus=0.2,
            mmse_rec_lo=True,
            # path_apodizer= path_apodizer,
            path_pupil=path_pupil,
        )

        if triggered == "generate_ini_button":
            ini_download = make_text_download(
                ini_info["ini_file"],
                filename=f"{ini_info['ini_name']}.ini",
            )
            status = html.Div(
                "INI file generated successfully. The download should start automatically.",
                style={
                    "padding": "12px",
                    "backgroundColor": "#eef6ff",
                    "border": "1px solid #b6d4fe",
                    "borderRadius": "10px",
                    "marginBottom": "16px",
                    "color": "#084298",
                    "fontWeight": "600",
                },
            )
            results = ini_display(ini_info)
            cleanup_workspace(ini_info)
            return status, results, ini_download, no_update

        result = run_simulation_from_ini(
            ini_info,
            verbose=False,
            save_FITS=parse_bool(save_fits),
            save_PSDs=parse_bool(save_psds),
            # sr_method=sr_method,
        )

        fits_download = no_update
        fits_requested = parse_bool(save_fits)
        if fits_requested:
            fits_download = make_fits_download(result.get("fits_files", []), result["ini_name"])

        status_text = "Simulation completed successfully."
        if fits_requested:
            if fits_download is no_update:
                status_text = "Simulation completed successfully, but no FITS file was found to download."
            else:
                status_text = "Simulation completed successfully. The FITS download should start automatically."

        status = html.Div(
            status_text,
            style={
                "padding": "12px",
                "backgroundColor": "#e9f7ef",
                "border": "1px solid #b7e1c1",
                "borderRadius": "10px",
                "marginBottom": "16px",
                "color": "#1e6b35",
                "fontWeight": "600",
            },
        )

        results = html.Div(
            [
                html.H3("Simulation results"),
                html.Div(
                    [
                        metric_card("Strehl", result.get("strehl")),
                        metric_card("FWHM", result.get("fwhm")),
                        metric_card("EE", result.get("ee")),
                        metric_card("HO residual", result.get("HO_res")),
                        metric_card("LO residual", result.get("LO_res")),
                        metric_card("Focus residual", result.get("GF_res")),
                    ],
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(160px, 1fr))",
                        "gap": "12px",
                        "marginBottom": "16px",
                    },
                ),
                derived_inputs_display(result),
                psf_display(result),
            ]
        )

        cleanup_workspace(result)
        return status, results, no_update, fits_download

    except Exception as exc:
        cleanup_workspace(ini_info)
        err = html.Div(
            [
                html.Div(
                    f"Error: {exc}",
                    style={"fontWeight": "700", "marginBottom": "8px"},
                ),
                html.Pre(
                    traceback.format_exc(),
                    style={
                        "whiteSpace": "pre-wrap",
                        "backgroundColor": "#fff5f5",
                        "padding": "12px",
                        "borderRadius": "10px",
                        "overflowX": "auto",
                    },
                ),
            ],
            style={
                "padding": "12px",
                "backgroundColor": "#fff1f0",
                "border": "1px solid #ffccc7",
                "borderRadius": "10px",
                "color": "#a8071a",
            },
        )
        return err, html.Div(), no_update, no_update

print("[STARTUP] app.py fully loaded", flush=True)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)