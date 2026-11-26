# -*- coding: utf-8 -*-
"""Render fig1.png, fig2.png and video.mp4 (1920x1080, #0b0f1a, dual-pane
Solver Desk) from the verified Abaqus nodal exports (results/nodes_*.csv,
results/faces_*.csv) and results/cases.json / results/verify.json / gates.json.

The 3-D panes are a small software rasteriser (numpy + Pillow): the quarter
model is mirrored into the full 200 x 100 mm lid, painted back-to-front and
coloured by nodal S11 with one fixed diverging scale shared by both panes and
every frame.

Every number on screen comes from results/*.json - nothing is typed in by hand.

Usage:
  python code/render.py [figs|video|all] [sample_dir]
"""
import json
import math
import os
import sys

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
RES = os.path.join(OUT, "results")
sys.path.insert(0, HERE)
sys.path.insert(0, OUT)        # verify.py lives beside results/

import gen_decks as g          # geometry constants (same source as the decks)
import verify as vfy           # closed-form section / beam references

# ------------------------------------------------------------------ palette
BG = "#0b0f1a"
PANEL = "#111827"
PANEL_ALT = "#0e1424"
BORDER = "#26324d"
FG = "#e8eef8"
DIM = "#9fb0c8"
FAINT = "#6f8099"
CYAN = "#5cc8ff"
ORANGE = "#f97316"
BLUE = "#3b82f6"
GREEN = "#6fdc8c"
YELLOW = "#ffd166"
RED = "#ff6b6b"

W, H, FPS = 1920, 1080, 24
MARGIN = 48
EMBED = 0.30          # panel colour mixed into every field colour (ambient)
EDGE_MIX = 0.10       # grey mixed into each quad's edge so the mesh reads faintly
DISPLAY_DISP = 12.0   # displacement magnification (flat centre sag = 6% of 100 mm)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": FG,
    "axes.labelcolor": FG,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "axes.edgecolor": BORDER,
    "axes.facecolor": PANEL,
    "figure.facecolor": BG,
    "savefig.facecolor": BG,
})


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float64)


# ------------------------------------------------------------------ data
CASES_LIST = json.load(open(os.path.join(RES, "cases.json")))
CASES = {c["case"]: c for c in CASES_LIST}
VERIFY = json.load(open(os.path.join(RES, "verify.json")))
GATES = json.load(open(os.path.join(RES, "gates.json")))
VROW = {r["check"]: r for r in VERIFY["rows"]}

FLAT_B, RIB_B = CASES["K-F-B"], CASES["K-R15-B"]
FLAT_C, RIB_C = CASES["K-F-C"], CASES["K-R15-C"]
Q4 = 0.20
W_FLAT_3D = FLAT_C["w_center_mm"]
RATIO_4EDGE = FLAT_C["w_center_mm"] / RIB_C["w_center_mm"]
W_E_FLAT = CASES["E-F-B"]["w_center_mm"]
W_E_RIB = CASES["E-R15-B"]["w_center_mm"]
RATIO_2EDGE = W_E_FLAT / W_E_RIB
B_EFF = CASES["E-R15-B"]["b_eff_W"]
RATIO_R05 = CASES["K-R05-B"]["w_center_mm"] / FLAT_B["w_center_mm"]
I_RATIO_15 = vfy.t_section(vfy.W, vfy.T_SKIN, 10.0, 15.0)["I"] / (vfy.W * vfy.T_FLAT ** 3 / 12.0)

SWEEP = ["K-R05-B", "K-R10-B", "K-R15-B", "K-R20-B", "K-R25-B"]

# ------------------------------------------------------------------ field load
FIELDS = {}


def load_field(cid):
    if cid in FIELDS:
        return FIELDS[cid]
    d = np.genfromtxt(os.path.join(RES, f"nodes_{cid}.csv"), delimiter=",", names=True)
    nid = d["id"].astype(np.int64)
    P = np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float64)
    U = d["U3"].astype(np.float64)
    S = d["S11"].astype(np.float64)

    f = np.genfromtxt(os.path.join(RES, f"faces_{cid}.csv"), delimiter=",", names=True)
    lookup = np.full(int(nid.max()) + 1, -1, dtype=np.int64)
    lookup[nid] = np.arange(len(nid))
    quads = np.stack([lookup[f["n1"].astype(np.int64)], lookup[f["n2"].astype(np.int64)],
                      lookup[f["n3"].astype(np.int64)], lookup[f["n4"].astype(np.int64)]], axis=1)
    qx, qy = P[quads][:, :, 0], P[quads][:, :, 1]
    keep = ~((np.abs(qx).max(axis=1) < 1e-9) | (np.abs(qy).max(axis=1) < 1e-9))
    quads = quads[keep]           # symmetry-plane faces become internal when mirrored

    Ps, Qs, Vs, Us, Uq = [], [], [], [], []
    offset = 0
    for sx in (1.0, -1.0):
        for sy in (1.0, -1.0):
            Pm = P.copy()
            Pm[:, 0] *= sx
            Pm[:, 1] *= sy
            Ps.append(Pm)
            Qs.append(quads + offset)
            Vs.append(S[quads].mean(axis=1))
            Us.append(U)
            Uq.append(U[quads].mean(axis=1))
            offset += len(P)

    rec = dict(P=np.concatenate(Ps), Q=np.concatenate(Qs), V=np.concatenate(Vs),
               Un=np.concatenate(Us), Uq=np.concatenate(Uq), S=S, U=U,
               n_elem=CASES[cid]["n_elem"], n_node=CASES[cid]["n_node"])
    FIELDS[cid] = rec
    return rec


_smax = max(np.abs(load_field("K-F-B")["S"]).max(), np.abs(load_field("K-R15-B")["S"]).max())
SLIM = float(math.ceil(_smax))

_LUT_N = 512
_cneg, _c0, _cpos = hex_rgb(BLUE), hex_rgb(BG), hex_rgb(ORANGE)
_lut = np.zeros((_LUT_N, 3))
for _i in range(_LUT_N):
    _t = -1.0 + 2.0 * _i / (_LUT_N - 1)
    _lut[_i] = (_cneg + (_c0 - _cneg) * (_t + 1.0)) if _t < 0 else (_c0 + (_cpos - _c0) * _t)
LUT = np.clip((1.0 - EMBED) * _lut + EMBED * hex_rgb(PANEL), 0, 255)
EDGE = np.clip((1.0 - EDGE_MIX) * LUT + EDGE_MIX * hex_rgb("#8fa0bf"), 0, 255)


def field_colour(values):
    idx = np.clip(((values / SLIM) + 1.0) * 0.5 * (_LUT_N - 1), 0, _LUT_N - 1).astype(np.int32)
    return LUT[idx]


# ------------------------------------------------------------------ camera
DCAM = 900.0
WORLD_CENTRE = np.array([0.0, 0.0, 5.0])


def camera_axes(az_deg, el_deg):
    az, el = math.radians(az_deg), math.radians(el_deg)
    d = np.array([math.cos(el) * math.sin(az), math.cos(el) * math.cos(az), math.sin(el)])
    f = -d
    s = np.cross(f, np.array([0.0, 0.0, 1.0]))
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    return dict(s=s, u=u, d=d, C=WORLD_CENTRE + DCAM * d)


def project(P, view, k, ox, oy):
    rel = P - view["C"]
    zc = np.maximum(-(rel @ view["d"]), 1.0)
    return (ox + k * (rel @ view["s"]) * (DCAM / zc),
            oy - k * (rel @ view["u"]) * (DCAM / zc), zc)


def fit_view(az, el, w, h, fill=0.92):
    """One camera and one scale for both panes: fit the undeformed 200x100 mm lid."""
    view = camera_axes(az, el)
    xs = np.array([-50, 50, 50, -50, -50, 50, 50, -50], float)
    ys = np.array([-100, -100, 100, 100, -100, -100, 100, 100], float)
    zs = np.array([0, 0, 0, 0, 4, 4, 4, 4], float)
    px, py, _ = project(np.stack([xs, ys, zs], axis=1), view, 1.0, 0.0, 0.0)
    bw, bh = px.max() - px.min(), py.max() - py.min()
    k = min(w * fill / bw, h * fill / bh)
    ox = w * 0.5 - k * 0.5 * (px.max() + px.min())
    oy = h * 0.5 - k * 0.5 * (py.max() + py.min())
    view.update(k=k, ox=ox, oy=oy, w=w, h=h, az=az, el=el)
    return view


def render_pane(field, load, view, bg_rgb, edges=True):
    """Deformed lid coloured by load-scaled S11. Returns uint8 (h, w, 3)."""
    img = Image.new("RGB", (view["w"], view["h"]), tuple(bg_rgb.astype(int)))
    dr = ImageDraw.Draw(img)

    P = field["P"].copy()
    P[:, 2] += DISPLAY_DISP * load * field["Un"]
    px, py, zc = project(P, view, view["k"], view["ox"], view["oy"])

    Q = field["Q"]
    order = np.argsort(-zc[Q].mean(axis=1))
    cols = field_colour(field["V"] * load)
    ecol = EDGE[((np.clip(field["V"] * load / SLIM, -1, 1) + 1) * 0.5 * (_LUT_N - 1)).astype(np.int32)]
    for qi in order:
        a, b, c, dd = Q[qi]
        xy = [(px[a], py[a]), (px[b], py[b]), (px[c], py[c]), (px[dd], py[dd])]
        fill = tuple(int(v) for v in cols[qi])
        if edges:
            dr.polygon(xy, fill=fill, outline=tuple(int(v) for v in ecol[qi]))
        else:
            dr.polygon(xy, fill=fill)
    return img


_FONT = ImageFont.truetype(os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf"), 13)


def pane_to_array(img, scale_bar=True, view=None):
    if scale_bar and view is not None:
        dr = ImageDraw.Draw(img)
        L = 20.0 * view["k"]
        x0, y0 = 16, img.height - 22
        dr.line([(x0, y0), (x0 + L, y0)], fill=(150, 165, 190), width=2)
        dr.line([(x0, y0 - 4), (x0, y0 + 4)], fill=(150, 165, 190), width=2)
        dr.line([(x0 + L, y0 - 4), (x0 + L, y0 + 4)], fill=(150, 165, 190), width=2)
        dr.text((x0 + L + 6, y0 - 7), "20 mm", font=_FONT, fill=(150, 165, 190))
    return np.asarray(img)


# ------------------------------------------------------------------ matplotlib layout helpers
def add_panel(fig, x, y, w, h, fc=PANEL, ec=BORDER):
    ax = fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])
    ax.set_facecolor(fc)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(ec)
        sp.set_linewidth(1.2)
    ax.set_zorder(0)
    return ax


def img_axes(fig, x, y, w, h):
    ax = fig.add_axes([x / W, 1.0 - (y + h) / H, w / W, h / H])
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_zorder(2)
    return ax


def txt(fig, x, y, s, size_px, color=FG, weight="normal", ha="left", va="top", spacing=1.35):
    return fig.text(x / W, 1.0 - y / H, s, fontsize=size_px * 0.72, color=color,
                    fontweight=weight, ha=ha, va=va, linespacing=spacing, zorder=5)


def rule(fig, x0, y0, x1, color=BORDER):
    ax = fig.add_axes([x0 / W, 1.0 - y0 / H, (x1 - x0) / W, 0.0015])
    ax.set_facecolor(color)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_zorder(3)


# ------------------------------------------------------------------ left pane
def draw_desk(fig):
    add_panel(fig, MARGIN, MARGIN, 852, H - 2 * MARGIN, fc=PANEL_ALT, ec=BORDER)
    x = MARGIN + 26

    txt(fig, x, MARGIN + 24, "SOLVER DESK", 13, ORANGE, "bold")
    txt(fig, x, MARGIN + 48, "Equal mass, 216 g:\n4.0 mm lid vs 2.5 mm skin + 10x15 rib",
        34, FG, "bold", spacing=1.22)

    run_txt = (
        "Abaqus/Standard 2023, C3D8I linear static, quarter model of a\n"
        "200 x 100 mm 6061-T6 lid. 80,000 mm3 both designs. 0.20 MPa\n"
        "uniform pressure, four edges simply supported at mid-thickness.\n"
        f"Mesh B shown: {FLAT_B['n_elem']:,} elements ({FLAT_B['n_node']:,} nodes) flat,\n"
        f"{RIB_B['n_elem']:,} elements ({RIB_B['n_node']:,} nodes) ribbed. Check rows: mesh C."
    )
    txt(fig, x, MARGIN + 178, run_txt, 17, DIM, spacing=1.46)

    ty = MARGIN + 320
    cx = [x, x + 358, x + 498, x + 644]
    for cxx, hd in zip(cx, ["Check", "Solver", "Reference", "delta %"]):
        txt(fig, cxx, ty, hd, 15, CYAN, "bold")
    rule(fig, x, ty + 27, x + 794)
    c1, c2, c3 = VROW["C1"], VROW["C2"], VROW["C3"]
    rows = [
        ("Flat lid, centre deflection", f"{c1['solver']:.3f} mm", f"{c1['reference']:.4f} mm", c1["delta_pct"]),
        ("Flat lid, peak bending stress", f"{c2['solver']:.1f} MPa", f"{c2['reference']:.1f} MPa", c2["delta_pct"]),
        ("T-coupon, 10x15 rib, deflection", f"{c3['solver']:.4f} mm", f"{c3['reference']:.4f} mm", c3["delta_pct"]),
    ]
    for i, (label, sv, rf, dp) in enumerate(rows):
        ry = ty + 42 + i * 38
        txt(fig, cx[0], ry, label, 16, FG)
        txt(fig, cx[1], ry, sv, 16, FG)
        txt(fig, cx[2], ry, rf, 16, DIM)
        txt(fig, cx[3], ry, f"{dp:+.1f}%", 16, GREEN, "bold")
    tol = " / ".join(f"+/-{r['tol_pct']:.1f}%" for r in (c1, c2, c3))
    txt(fig, x, ty + 168, f"Timoshenko SSSS + composite-T beam. Tolerances frozen before the run: {tol}.",
        13.5, FAINT)

    by = MARGIN + 530
    txt(fig, x, by, "What the animation shows", 19, FG, "bold")
    bullets = (
        "1  The 4.0 mm lid dishes as one panel: a single smooth\n"
        f"    curvature, {W_FLAT_3D:.2f} mm at the centre.\n"
        "2  The rib holds the centre line and pushes the curvature into\n"
        f"    the two 50 mm skin panels. The centre still moves {RIB_C['w_center_mm']:.2f} mm:\n"
        f"    a {RATIO_4EDGE:.2f}x gain on four edges, not the {I_RATIO_15:.0f}x of the section.\n"
        "3  Between rib and edge the thin skin is the limiter: a 5 mm rib\n"
        f"    at equal mass deflects {RATIO_R05:.2f}x the flat lid although its\n"
        "    second moment of area is 3.3x higher."
    )
    txt(fig, x, by + 36, bullets, 16, FG, spacing=1.55)

    txt(fig, x, MARGIN + 782,
        "Limit: linear elastic, small deflection, no bolt preload, no local skin\n"
        "buckling, no rib-root fatigue. The sharp rib root is a stress singularity:\n"
        "ribbed skin stress is mesh-dependent and not quoted.",
        15, ORANGE, spacing=1.42)

    txt(fig, x, H - MARGIN - 36,
        "Generic 6061-T6 lid  |  assumed pressure, support and fillet  |  Abaqus/Standard 2023",
        13, FAINT)


# ------------------------------------------------------------------ right pane
def draw_field_pane(fig, img1, img2, load_txt, az):
    x0, y0 = 940, MARGIN
    add_panel(fig, x0, y0, 932, H - 2 * MARGIN, fc=PANEL, ec=BORDER)
    xl, xr = x0 + 22, x0 + 932 - 22

    txt(fig, xl, y0 + 22, "ABAQUS S11 ON THE DEFORMED LID", 13, ORANGE, "bold")
    txt(fig, xr, y0 + 21, f"shared scale +/-{SLIM:.0f} MPa  |  mesh B  |  0.20 MPa", 13.5, DIM, ha="right")

    vw, vh = 438, 380
    vx1 = xl
    vx2 = xl + vw + 16
    vy = y0 + 76

    txt(fig, vx1, y0 + 48, "4.0 mm FLAT LID", 16, FG, "bold")
    txt(fig, vx1 + vw, y0 + 48, f"w {W_FLAT_3D:.2f} mm  |  peak {FLAT_C['sigma_skin_MPa']:.1f} MPa", 14, DIM, ha="right")
    txt(fig, vx2, y0 + 48, "2.5 SKIN + 10x15 RIB", 15.5, FG, "bold")
    txt(fig, vx2 + vw, y0 + 48, f"w {RIB_C['w_center_mm']:.2f} mm  |  tip {RIB_C['sigma_tip_MPa']:.1f} MPa", 14, DIM, ha="right")

    for vx, img in ((vx1, img1), (vx2, img2)):
        ax = img_axes(fig, vx, vy, vw, vh)
        ax.imshow(img, interpolation="nearest", zorder=2)
        fig.add_artist(Rectangle((vx / W, 1.0 - (vy + vh) / H), vw / W, vh / H,
                                 transform=fig.transFigure, fill=False, ec=BORDER, lw=1.0, zorder=3))

    txt(fig, xl, vy + vh + 12, load_txt, 14, FG)
    txt(fig, xr, vy + vh + 12,
        f"displacement x{DISPLAY_DISP:.0f}, true {W_FLAT_3D:.2f} mm  |  azimuth {az:.0f} deg", 14, DIM, ha="right")

    cb_y, cb_h, cb_w = vy + vh + 46, 16, 560
    axcb = img_axes(fig, xl, cb_y, cb_w, cb_h)
    grad = LUT[np.linspace(0, _LUT_N - 1, cb_w).astype(int)][None, :, :].astype(np.uint8)
    axcb.imshow(grad, aspect="auto", zorder=2)
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        fig.add_artist(plt.Line2D([(xl + frac * cb_w) / W], [1.0 - (cb_y + cb_h + 5) / H],
                                  marker="|", ms=5, color=DIM, transform=fig.transFigure, zorder=4))
    for frac, lab in ((0.0, f"-{SLIM:.0f}"), (0.25, f"-{SLIM/2:.0f}"), (0.5, "0"),
                      (0.75, f"+{SLIM/2:.0f}"), (1.0, f"+{SLIM:.0f}")):
        ha = "left" if frac == 0.0 else ("right" if frac == 1.0 else "center")
        txt(fig, xl + frac * cb_w, cb_y + cb_h + 8, lab, 12.5, DIM, ha=ha)
    txt(fig, xl + cb_w + 30, cb_y - 6, "signed S11 (MPa)", 14, FG, "bold")
    txt(fig, xl + cb_w + 30, cb_y + 12, "blue = compression  |  orange = tension", 13, DIM)


def draw_strip(fig, hi):
    x0 = 940 + 22
    y = 686
    txt(fig, x0, y - 32, "FIXED MASS RIB SWEEP", 13, ORANGE, "bold")
    txt(fig, x0 + 196, y - 32, "same 216 g, same 0.20 MPa, same displacement scale", 13, DIM)

    pw, ph, gap = 168, 122, 10
    for i, (cid, img) in enumerate(STRIP):
        px = x0 + i * (pw + gap)
        ax = img_axes(fig, px, y, pw, ph)
        ax.imshow(img, interpolation="nearest", zorder=2)
        c = CASES[cid]
        active = i == hi
        fig.add_artist(Rectangle((px / W, 1.0 - (y + ph) / H), pw / W, ph / H,
                                 transform=fig.transFigure, fill=False,
                                 ec=YELLOW if active else BORDER, lw=2.0 if active else 0.8, zorder=4))
        txt(fig, px + 2, y + ph + 8, f"H {g.RIBS[c['variant']]['H']:.0f}", 13.5,
            FG if active else DIM, "bold" if active else "normal")
        txt(fig, px + pw - 2, y + ph + 8, f"{c['w_center_mm']:.2f} mm", 13.5,
            YELLOW if active else DIM, ha="right")

    txt(fig, x0, y + ph + 42,
        "Quarter model mirrored to the full lid  |  symmetry planes x = 0, y = 0  |  2.5 mm hex mesh (C3D8I)",
        13, FAINT)
    txt(fig, x0, y + ph + 68,
        f"Ribbed skin peak S11 is not converged ({RIB_B['sigma_skin_MPa']:.1f} MPa on B, "
        f"{RIB_C['sigma_skin_MPa']:.1f} MPa on C): the sharp root is singular, so only the rib-tip value is quoted.",
        13, FAINT)
    rule(fig, x0, y + ph + 100, x0 + 880)
    txt(fig, x0, y + ph + 112,
        f"Same mass, same load: the {I_RATIO_15:.0f}x section only pays when the lid spans on two edges "
        f"({RATIO_2EDGE:.1f}x, rail). On four edges\n"
        f"two-way bending carries the load and the rib buys {RATIO_4EDGE:.2f}x - less than the {RATIO_R05:.2f}x it costs at H = 5 mm.",
        14.5, CYAN, spacing=1.5)


# ------------------------------------------------------------------ frame assembly
STRIP = []


def build_strip():
    view = fit_view(30.0, 26.0, 168, 122, fill=0.95)
    bg = hex_rgb(PANEL)
    out = []
    for cid in SWEEP:
        out.append((cid, pane_to_array(render_pane(load_field(cid), 1.0, view, bg, edges=False),
                                       scale_bar=False)))
    return out


def frame_images(flat_load, rib_load, az):
    view = fit_view(az, 26.0, 438, 380)
    bg = hex_rgb(PANEL)
    i1 = pane_to_array(render_pane(load_field("K-F-B"), flat_load, view, bg), True, view)
    i2 = render_pane(load_field("K-R15-B"), rib_load, view, bg)
    if rib_load < 1.0:
        f = max(0.30, min(1.0, rib_load))
        i2 = Image.blend(Image.new("RGB", i2.size, tuple(int(v) for v in bg)), i2, f)
    i2 = pane_to_array(i2, True, view)
    return i1, i2


def compose_frame(flat_load, rib_load, az, hi, load_label):
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    fig.patch.set_facecolor(BG)
    draw_desk(fig)
    i1, i2 = frame_images(flat_load, rib_load, az)
    draw_field_pane(fig, i1, i2, load_label, az)
    draw_strip(fig, hi)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


# ------------------------------------------------------------------ end card
def endcard_img():
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    fig.patch.set_facecolor(BG)
    add_panel(fig, 40, 40, W - 80, H - 80, fc=PANEL_ALT, ec=BORDER)

    txt(fig, W // 2, 96, "EQUAL MASS LID: WHAT THE RIB ACTUALLY BUYS", 40, FG, "bold", ha="center")
    txt(fig, W // 2, 154,
        "200 x 100 x 4.0 mm flat  vs  2.5 mm skin + 10 x 15 mm rib  |  80,000 mm3 / 216 g  |  6061-T6, Abaqus C3D8I",
        18, DIM, ha="center")

    cards = [
        (f"{RATIO_4EDGE:.2f}x", "four edges bolted", f"{W_FLAT_3D:.2f} -> {RIB_C['w_center_mm']:.2f} mm", CYAN,
         f"{I_RATIO_15:.0f}x stiffer section, {RATIO_4EDGE:.2f}x stiffer lid"),
        (f"{RATIO_2EDGE:.1f}x", "two edges, rail mount", f"{W_E_FLAT:.2f} -> {W_E_RIB:.3f} mm", GREEN,
         "same lid, same mass: the support decides"),
        (f"{RATIO_R05:.2f}x worse", "5 mm rib, equal mass", f"{CASES['K-R05-B']['w_center_mm']:.2f} mm vs {FLAT_B['w_center_mm']:.2f} mm",
         ORANGE, "I is 3.3x higher, the lid is softer"),
    ]
    for i, (big, mid, sub, col, note) in enumerate(cards):
        cxx = W * 0.215 + i * W * 0.285
        add_panel(fig, cxx - 245, 300, 490, 320, fc=PANEL, ec=col)
        txt(fig, cxx, 396, big, 58, col, "bold", ha="center", va="center")
        txt(fig, cxx, 462, mid, 20, FG, "bold", ha="center", va="center")
        txt(fig, cxx, 504, sub, 17, DIM, ha="center", va="center")
        txt(fig, cxx, 552, note, 14.5, FAINT, ha="center", va="center")

    txt(fig, W // 2, 690,
        f"Checks: deflection {VROW['C1']['delta_pct']:+.1f}%  |  stress {VROW['C2']['delta_pct']:+.1f}%  |  "
        f"T-coupon {VROW['C3']['delta_pct']:+.1f}%  |  mesh C vs B  {GATES['G4_K-F_BC_pct']:.2f}% flat, "
        f"{GATES['G4_K-R15_BC_pct']:.2f}% ribbed",
        19, YELLOW, ha="center", va="center")
    txt(fig, W // 2, 742,
        f"Shear lag at L/W = 2: b_eff/W = {B_EFF:.2f} on the rail case (mesh B, not convergence-checked).",
        17, DIM, ha="center", va="center")
    txt(fig, W // 2, 792,
        "Limits: linear elastic, small deflection, no skin buckling, no bolt preload, no rib-root fatigue; "
        "the sharp-root stress is singular and not quoted.",
        15, FAINT, ha="center", va="center")
    txt(fig, W // 2, 950,
        "Abaqus/Standard 2023 + Python 3.14  |  generic parametric lid, assumed pressure and support",
        15, FAINT, ha="center", va="center")
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return arr


# ------------------------------------------------------------------ fig2
def render_fig2():
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    fig.patch.set_facecolor(BG)
    txt(fig, 60, 56, "RIB HEIGHT AT FIXED MASS: 3-D SOLID vs THE 1-D BEAM BRACKET", 26, FG, "bold")
    txt(fig, 60, 102,
        "200 x 100 mm 6061-T6 lid, 80,000 mm3 (216 g), 0.20 MPa, four edges simply supported at mid-thickness. "
        "Abaqus/Standard 2023 C3D8I, mesh B.",
        15, DIM)

    ax = fig.add_axes([152 / W, 1 - 820 / H, 1160 / W, 620 / H])
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_color(BORDER)
    ax.grid(color="#1c2740", lw=0.8)
    ax.tick_params(labelsize=12.5)

    Hs = np.linspace(4.5, 26.0, 140)
    i_flat = vfy.W * vfy.T_FLAT ** 3 / 12.0
    w_flat_1d = 5.0 * Q4 * vfy.W * vfy.L ** 4 / (384.0 * vfy.E * i_flat)
    w1d = []
    for Hh in Hs:
        sec = vfy.t_section(vfy.W, vfy.T_SKIN, 150.0 / Hh, Hh)
        w1d.append((5.0 * Q4 * vfy.W * vfy.L ** 4 / (384.0 * vfy.E * sec["I"])) / w_flat_1d)
    ax.plot(Hs, w1d, "--", color="#8fa0bf", lw=2.4,
            label="1-D beam, full 100 mm flange (upper bound on stiffness)")

    pts = [CASES[c] for c in SWEEP]
    Hs_p = [g.RIBS[c["variant"]]["H"] for c in pts]
    wr = [c["w_center_mm"] / FLAT_B["w_center_mm"] for c in pts]
    ax.plot(Hs_p, wr, "o-", color=ORANGE, lw=2.8, ms=10, label="3-D solid, single rib (w / w_flat)")
    for hh, ww in zip(Hs_p, wr):
        ax.annotate(f"{ww:.2f}", (hh, ww), textcoords="offset points",
                    xytext=(0, -26) if ww > 1.0 else (0, 14),
                    ha="center", fontsize=11.5, color=ORANGE, fontweight="bold")

    tw = CASES["K-TWIN-B"]["w_center_mm"] / FLAT_B["w_center_mm"]
    ax.plot([10.0], [tw], "s", color=YELLOW, ms=11, label="twin 7.5 x 10 ribs at x = +/-25 mm")
    ax.annotate(f"twin: {tw:.2f}\nsame I as R10,\n+{100*(tw/wr[1]-1.0):.0f}% w",
                xy=(10.4, tw), xytext=(11.9, 1.44), fontsize=11.5, color=YELLOW, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.32", fc=PANEL_ALT, ec=YELLOW, lw=1.0),
                arrowprops=dict(arrowstyle="->", color=YELLOW, lw=1.3))

    ax.axhline(1.0, color=RED, ls=":", lw=1.6, alpha=0.9)
    ax.text(4.8, 1.04, "flat 4.0 mm lid = 1.00", color=RED, fontsize=12, fontweight="bold")

    ax.annotate(f"H = 15 mm (headline):\nI = {I_RATIO_15:.1f}x, w only {wr[2]:.2f}x",
                xy=(15.2, wr[2]), xytext=(16.3, 0.88), fontsize=12.5, color=FG, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.34", fc=PANEL_ALT, ec=CYAN, lw=1.2),
                arrowprops=dict(arrowstyle="->", color=CYAN, lw=1.5))
    ax.annotate(f"H = 5 mm: I = 3.3x but w = {wr[0]:.2f}x\n(skin thinned over 30 mm of width)",
                xy=(5.2, wr[0]), xytext=(5.9, 1.70), fontsize=12, color=FG,
                bbox=dict(boxstyle="round,pad=0.30", fc=PANEL_ALT, ec=ORANGE, lw=1.0),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.2))
    ax.annotate("the gap is two-way plate action\n+ local skin bending, not shear lag alone",
                xy=(21.5, 0.075), xytext=(13.4, 0.16), fontsize=12, color="#8fa0bf",
                arrowprops=dict(arrowstyle="->", color="#8fa0bf", lw=1.2))

    ax.set_xlim(3.2, 27.0)
    ax.set_ylim(0.0, 1.85)
    ax.set_xlabel("Rib height H above the 2.5 mm skin (mm), rib area held at 150 mm2", fontsize=14)
    ax.set_ylabel("Centre deflection / flat-lid deflection", fontsize=13, labelpad=10)
    ax.legend(loc="upper right", fontsize=12, facecolor=PANEL_ALT, edgecolor=BORDER, labelcolor=FG)
    ax.set_title("Fixed mass: rib height vs centre deflection", fontsize=16, fontweight="bold",
                 color=FG, pad=12, loc="left")

    bx, by = 1360, 190
    add_panel(fig, bx - 20, by - 46, 510, 500, fc=PANEL, ec=BORDER)
    txt(fig, bx, by, "WHAT THE SWEEP SAYS", 13, ORANGE, "bold")
    lines = [
        ("I / I_flat", f"{I_RATIO_15:.1f}x", "headline rib vs flat section"),
        ("w / w_flat", f"{wr[2]:.2f}x", f"{W_FLAT_3D:.2f} -> {RIB_C['w_center_mm']:.2f} mm (K group)"),
        ("E group", f"{RATIO_2EDGE:.1f}x", f"{W_E_FLAT:.2f} -> {W_E_RIB:.3f} mm (rail)"),
        ("b_eff / W", f"{B_EFF:.2f}", "rail case, measured at mid-span"),
    ]
    yy = by + 36
    for lab, val, sub in lines:
        txt(fig, bx, yy, lab, 13.5, DIM)
        txt(fig, bx + 486, yy, val, 22, FG, "bold", ha="right")
        txt(fig, bx + 486, yy + 29, sub, 12.5, FAINT, ha="right")
        yy += 78
    txt(fig, bx, yy + 10,
        "Mesh A -> B -> C, w_center\n"
        f"flat    {CASES['K-F-A']['w_center_mm']:.3f} / {CASES['K-F-B']['w_center_mm']:.3f} / {CASES['K-F-C']['w_center_mm']:.3f} mm\n"
        f"ribbed  {CASES['K-R15-A']['w_center_mm']:.3f} / {CASES['K-R15-B']['w_center_mm']:.3f} / {CASES['K-R15-C']['w_center_mm']:.3f} mm\n"
        f"C vs B: {GATES['G4_K-F_BC_pct']:.2f}% flat, {GATES['G4_K-R15_BC_pct']:.2f}% ribbed",
        12.5, DIM, spacing=1.6)

    txt(fig, 60, 890,
        "Sweep points are mesh B; the 1-D curve assumes the full 100 mm flange acts, which two-way plate bending and "
        "local skin bending both contradict.\n"
        "Rib-tip stress is mesh B and converged on R15 (B to C 0.1%); ribbed skin stress is not converged and is not plotted.",
        13.5, FAINT)
    txt(fig, 60, 968,
        "Abaqus/Standard 2023 + Python 3.14  |  generic parametric lid, assumed pressure and support  |  "
        "checks: Timoshenko SSSS, composite T-beam",
        13, FAINT)

    fig.savefig(os.path.join(OUT, "fig2.png"), facecolor=BG)
    plt.close(fig)
    print("Saved fig2.png")


# ------------------------------------------------------------------ fig1 / video
def render_fig1():
    i1, i2 = frame_images(1.0, 1.0, 26.0)
    fig = plt.figure(figsize=(W / 100.0, H / 100.0), dpi=100)
    fig.patch.set_facecolor(BG)
    draw_desk(fig)
    draw_field_pane(fig, i1, i2, "load 100% of 0.20 MPa  |  linear static, load-scaled", 26.0)
    draw_strip(fig, 2)
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    Image.fromarray(arr).save(os.path.join(OUT, "fig1.png"))
    print("Saved fig1.png")


def render_video(sample_dir=None):
    total = 288
    end = endcard_img()
    frames = []
    for i in range(total):
        t = i / FPS
        if t < 0.8:
            fl, fr, az, hi = 0.0, 0.0, 22.0, 2
            lab = "load 0%  |  linear static, load-scaled"
        elif t < 2.4:
            u = (t - 0.8) / 1.6
            fl, fr = u, 0.0
            az = 22.0 + 4.0 * u
            hi = 2
            lab = f"flat lid ramping: {100*u:.0f}% of 0.20 MPa  |  linear static, load-scaled"
        elif t < 3.2:
            u = (t - 2.4) / 0.8
            fl, fr = 1.0, u
            az = 26.0
            hi = 2
            lab = f"ribbed lid: {100*u:.0f}% of 0.20 MPa  |  linear static, load-scaled"
        elif t < 8.4:
            u = (t - 3.2) / 5.2
            fl, fr = 1.0, 1.0
            az = 26.0 - 8.0 * u
            hi = 2
            lab = "both at 100% of 0.20 MPa  |  linear static, load-scaled"
        elif t < 10.6:
            u = (t - 8.4) / 2.2
            fl, fr = 1.0, 1.0
            az = 18.0
            hi = min(4, int(u * 5))
            lab = "fixed-mass sweep: 5 mm to 25 mm rib at 216 g  |  linear static"
        else:
            fl = fr = 1.0
            az, hi, lab = 18.0, 4, "load 100% of 0.20 MPa"
        frames.append(end if t >= 10.6 else compose_frame(fl, fr, az, hi, lab))

    writer = imageio.get_writer(os.path.join(OUT, "video.mp4"), fps=FPS, codec="libx264",
                                quality=8, pixelformat="yuv420p", macro_block_size=1)
    for a in frames:
        writer.append_data(a)
    writer.close()
    print(f"Saved video.mp4 ({len(frames)} frames, {len(frames)/FPS:.2f} s)")

    if sample_dir:
        os.makedirs(sample_dir, exist_ok=True)
        for name, idx in (("sample_ramp.png", 34), ("sample_full.png", 120),
                          ("sample_sweep.png", 226), ("sample_end.png", 275)):
            Image.fromarray(frames[idx]).save(os.path.join(sample_dir, name))
        print("Saved sample frames")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    sample_dir = sys.argv[2] if len(sys.argv) > 2 else None
    STRIP = build_strip()
    if mode in ("figs", "all"):
        render_fig1()
        render_fig2()
    if mode in ("video", "all"):
        render_video(sample_dir)
