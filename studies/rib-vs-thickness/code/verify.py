#!/usr/bin/env python3
"""
rib-vs-thickness - closed-form reference spec and checker.

Stage 2 deliverable: this file contains NO solver output. It computes every
reference value the check table will compare against, and it defines the
3 frozen tolerances. Once Stage 3 has run, `--measure` reads
`results/measurements.json`, prints the `Check | Solver | Reference | delta %`
table, writes `results/verify.json`, and exits non-zero if any row is outside
its frozen tolerance.

Run:
    python verify.py              # references + mass audit only (no solver numbers)
    python verify.py --measure    # same, plus the 3-row check table from results/

Units: mm, N, MPa (= N/mm^2), tonne-free mass in grams.
No third-party imports. Python 3.9+.

References (as verified in TOPIC_BRIEF.md section 3 on 2026-11-19):
  [T]  Timoshenko plate tables, NASA TR 32-1528 (Butterworth & Yasui 1971),
       Table 5, pp. 10-11: w_max = alpha*q*a^4/D, M_max = beta*q*a^2, nu = 0.3.
  [M4] MIT OCW 2.080J Lecture 4: D = E*t^3/(12*(1-nu^2)); I = b*h^3/12;
       sigma = M*z/I.
  [M5] MIT OCW 2.080J Lecture 5 Eq. 5.27: w_max = 5*q*L^4/(384*E*I) for a
       simply supported beam under uniform load.
  [M7] MIT OCW 2.080J Lecture 7 Eq. 7.67-7.68: composite T-section neutral
       axis and parallel-axis I; Sec. 7.9: shear-lag effective breadth.
  [N]  NASA NTRS 20030032265 (6061-T6): E = 68.9 GPa, nu = 0.33,
       rho = 2.70e3 kg/m^3, sigma_y = 276 MPa.
"""

import json
import math
import os
import sys

# ---------------------------------------------------------------- constants
E = 68900.0          # MPa       6061-T6 elastic modulus [N]
NU = 0.33            # -         Poisson ratio [N]
RHO = 2.70e-3        # g/mm^3    2.70e3 kg/m^3 [N]
SY = 276.0           # MPa       tensile yield [N]
G = 9810.0           # mm/s^2    gravity, for mass-in-grams sanity only

# ---------------------------------------------------------------- geometry
L = 200.0            # mm   plate length (y), span direction of the rib
W = 100.0            # mm   plate width (x), short side a of the plate tables
T_FLAT = 4.0         # mm   monolithic plate thickness
T_SKIN = 2.5         # mm   thinned skin of every ribbed variant
V_TARGET = 80000.0   # mm^3 fixed material volume (all variants must hit this)

# rib variants at fixed volume: 200 mm * t_r * H = 20000 mm^3 spread over ribs
#   t_r * H * n_rib = 150 mm^2
RIBS = {
    "R15":  dict(n=1, t_r=10.0, H=15.0),   # brief's headline rib
    "R05":  dict(n=1, t_r=30.0, H=5.0),
    "R10":  dict(n=1, t_r=15.0, H=10.0),
    "R20":  dict(n=1, t_r=7.5,  H=20.0),
    "R25":  dict(n=1, t_r=6.0,  H=25.0),
    "TWIN": dict(n=2, t_r=7.5,  H=10.0),   # ribs on x = +/-25 mm
}
TWIN_X = 25.0        # mm   centreline offset of each twin rib
RFIL = 3.0           # mm   root fillet radius, fillet variant only

# loads, one per test group
Q_4EDGE = 0.20       # MPa  uniform pressure, four-edge simply supported group
Q_2EDGE = 0.01       # MPa  uniform pressure, two-edge (rail) group
Q_COUPON = 0.05      # MPa  uniform pressure, 40 mm T-beam check coupon

# Timoshenko Table 5, b/a = 2.0, four edges simply supported, nu = 0.3
ALPHA_SSSS = 0.01013   # w_max = alpha * q * a^4 / D
BETA_SSSS = 0.1017     # M_max = beta * q * a^2   (moment spanning side a)

# check ids -> frozen tolerances (percent). Frozen 2026-11-19, before any run.
CHECKS = [
    dict(id="C1", case="K-F-C", quantity="w_center_mm",
         name="Flat 4-edge SS centre deflection",
         ref="Timoshenko SSSS, alpha*q*a^4/D (TR 32-1528 Table 5)",
         tol_pct=2.5),
    dict(id="C2", case="K-F-C", quantity="sig_max_MPa",
         name="Flat 4-edge SS peak bending stress, centre bottom",
         ref="Timoshenko moment factor, 6*beta*q*a^2/t^2 (Table 5)",
         tol_pct=4.0),
    dict(id="C3", case="C3-C", quantity="w_center_mm",
         name="Ribbed 40 mm T-coupon centre deflection, 2-edge SS",
         ref="Euler-Bernoulli composite T, 5*q*W*L^4/(384*E*I_T)",
         tol_pct=8.0),
]


# ---------------------------------------------------------------- mechanics
def plate_D(t, e=E, nu=NU):
    """Plate flexural rigidity, N*mm. [M4] Eq. 4.60."""
    return e * t ** 3 / (12.0 * (1.0 - nu * nu))


def t_section(width, t_s, t_r, H, n=1, x_off=0.0):
    """Composite skin + n ribs, neutral axis from the flat face.

    Returns dict with A (mm^2), ybar (mm), I (mm^4, about the x-axis so that
    bending is about x and stress is S11 in a plate spanning y).
    x_off is the distance of a rib centreline from the section mid-plane; it
    does not change I about x (parallel-axis term is in x, not z).
    """
    a1 = width * t_s
    a2 = n * t_r * H
    a = a1 + a2
    y1 = t_s / 2.0
    y2 = t_s + H / 2.0
    ybar = (a1 * y1 + a2 * y2) / a
    i = (width * t_s ** 3 / 12.0 + a1 * (ybar - y1) ** 2
         + n * t_r * H ** 3 / 12.0 + a2 * (ybar - y2) ** 2)
    c_top = (t_s + H) - ybar
    return dict(A=a, ybar=ybar, I=i, c_top=c_top,
                S_top=i / c_top, S_bot=i / ybar,
                t_eq=(12.0 * i / width) ** (1.0 / 3.0),
                mass_g=a * L * RHO)


def fillet_area_centroid(t_s, r):
    """One root fillet: area r^2 - pi*r^2/4, centroid height above t_s."""
    a = r * r - math.pi * r * r / 4.0
    z = ((r * r * (r / 2.0)
          - (math.pi * r * r / 4.0) * (4.0 * r / (3.0 * math.pi))) / a) + t_s
    i_own = 0.0076 * r ** 4      # numeric fit, r=3 mm -> 0.62 mm^4
    return a, z, i_own


def rf_section(width, t_s, t_r, r, h):
    """Fillet variant: skin + rib + two 3 mm roots, volume already corrected."""
    a_sk = width * t_s
    a_rib = t_r * h
    a_f, z_f, i_own = fillet_area_centroid(t_s, r)
    a = a_sk + a_rib + 2.0 * a_f
    z_rib = t_s + h / 2.0
    ybar = (a_sk * t_s / 2.0 + a_rib * z_rib + 2.0 * a_f * z_f) / a
    i = (width * t_s ** 3 / 12.0 + a_sk * (ybar - t_s / 2.0) ** 2
         + t_r * h ** 3 / 12.0 + a_rib * (ybar - z_rib) ** 2
         + 2.0 * (i_own + a_f * (ybar - z_f) ** 2))
    c_top = (t_s + h) - ybar
    return dict(A=a, ybar=ybar, I=i, c_top=c_top, S_top=i / c_top,
                S_bot=i / ybar, t_eq=(12.0 * i / width) ** (1.0 / 3.0),
                mass_g=a * L * RHO, H=h)


def check_volume():
    """Every variant must carry exactly V_TARGET. Returns audit rows."""
    rows = []
    v_flat = W * T_FLAT * L
    rows.append(("F", v_flat, v_flat - V_TARGET))
    for name, r in RIBS.items():
        v = (W * T_SKIN + r["n"] * r["t_r"] * r["H"]) * L
        rows.append((name, v, v - V_TARGET))
    # fillet variant: two 3 mm root fillets per rib add (r^2 - pi*r^2/4) each;
    # H is solved down so the total stays at V_TARGET. (Context only, not a
    # headline case: it exists to make the rib-root stress convergeable.)
    add = 2.0 * (RFIL ** 2 - math.pi * RFIL ** 2 / 4.0)
    h_fil = (V_TARGET / L - W * T_SKIN - add) / RIBS["R15"]["t_r"]
    v = (W * T_SKIN + RIBS["R15"]["t_r"] * h_fil + add) * L
    rows.append(("RFIL", v, v - V_TARGET))
    return rows, h_fil


def ref_flat_plate():
    """Checks C1 and C2: flat 200x100x4, four edges SS, uniform pressure."""
    a = W                      # short side, spans the load
    d = plate_D(T_FLAT)
    w = ALPHA_SSSS * Q_4EDGE * a ** 4 / d
    m = BETA_SSSS * Q_4EDGE * a ** 2          # N*mm per mm of width
    sig = 6.0 * m / T_FLAT ** 2
    return dict(D=d, w=w, M=m, sigma=sig, util=sig / SY)


def ref_t_coupon(width):
    """Check C3: ribbed coupon, two opposite edges SS, uniform pressure.

    Beam idealisation: the whole coupon width acts with the rib, no shear lag.
    At L/W = 200/40 = 5 the 1-D idealisation is the best available closed form,
    and the residual gap to the solid is reported rather than assumed away.
    """
    sec = t_section(width, T_SKIN, RIBS["R15"]["t_r"], RIBS["R15"]["H"])
    w = 5.0 * Q_COUPON * width * L ** 4 / (384.0 * E * sec["I"])
    m = Q_COUPON * width * L ** 2 / 8.0
    return dict(sec=sec, w=w, M=m,
                sig_top=m * sec["c_top"] / sec["I"],
                sig_bot=m * sec["ybar"] / sec["I"])


def ref_beam_2edge():
    """Context only: 1-D beam value for the 200x100 two-edge (rail) group."""
    i_flat = W * T_FLAT ** 3 / 12.0
    w_flat = 5.0 * Q_2EDGE * W * L ** 4 / (384.0 * E * i_flat)
    sec = t_section(W, T_SKIN, RIBS["R15"]["t_r"], RIBS["R15"]["H"])
    w_rib = 5.0 * Q_2EDGE * W * L ** 4 / (384.0 * E * sec["I"])
    return dict(w_flat=w_flat, w_rib=w_rib, i_flat=i_flat,
                ratio=w_flat / w_rib)


def shear_lag_ref(b_eff_over_b):
    """Axial stress drop implied by an effective width, for context only.

    1-D section with a flange of width b_eff instead of the full width; the
    same bending moment then produces a larger extreme-fibre stress.
    """
    sec_full = t_section(W, T_SKIN, RIBS["R15"]["t_r"], RIBS["R15"]["H"])
    sec_eff = t_section(W * b_eff_over_b, T_SKIN,
                        RIBS["R15"]["t_r"], RIBS["R15"]["H"])
    return dict(I_full=sec_full["I"], I_eff=sec_eff["I"],
                I_ratio=sec_eff["I"] / sec_full["I"])


# ---------------------------------------------------------------- reporting
def reference_table():
    flat = ref_flat_plate()
    coup = ref_t_coupon(40.0)
    ref = {
        "C1": flat["w"],
        "C2": flat["sigma"],
        "C3": coup["w"],
    }
    return flat, coup, ref


def print_header():
    print("=" * 78)
    print("rib-vs-thickness  |  closed-form reference spec (no solver output)")
    print("=" * 78)
    print(f"6061-T6: E={E/1000:.1f} GPa  nu={NU}  rho=2700 kg/m^3  "
          f"Sy={SY:.0f} MPa")
    print(f"plate {L:.0f} x {W:.0f} mm, fixed volume {V_TARGET:.0f} mm^3")
    print()


def print_sections():
    print("Section audit (neutral axis from the flat face, I about x)")
    hdr = f"{'variant':7s} {'A mm2':>7s} {'ybar':>7s} {'I mm4':>9s} " \
          f"{'c_top':>7s} {'S_top':>8s} {'S_bot':>8s} {'t_eq':>6s} " \
          f"{'I/If':>6s} {'mass g':>7s}"
    print(hdr)
    print("-" * len(hdr))
    i_flat = W * T_FLAT ** 3 / 12.0
    print(f"{'F':7s} {W*T_FLAT:7.1f} {T_FLAT/2:7.3f} {i_flat:9.1f} "
          f"{T_FLAT/2:7.3f} {i_flat/(T_FLAT/2):8.1f} {i_flat/(T_FLAT/2):8.1f} "
          f"{T_FLAT:6.2f} {1.0:6.2f} {W*T_FLAT*L*RHO:7.1f}")
    for name, r in RIBS.items():
        s = t_section(W, T_SKIN, r["t_r"], r["H"], r["n"])
        print(f"{name:7s} {s['A']:7.1f} {s['ybar']:7.3f} {s['I']:9.1f} "
              f"{s['c_top']:7.3f} {s['S_top']:8.1f} {s['S_bot']:8.1f} "
              f"{s['t_eq']:6.2f} {s['I']/i_flat:6.2f} {s['mass_g']:7.1f}")
    _, h_fil = check_volume()
    f = rf_section(W, T_SKIN, RIBS["R15"]["t_r"], RFIL, h_fil)
    print(f"{'RFIL':7s} {f['A']:7.1f} {f['ybar']:7.3f} {f['I']:9.1f} "
          f"{f['c_top']:7.3f} {f['S_top']:8.1f} {f['S_bot']:8.1f} "
          f"{f['t_eq']:6.2f} {f['I']/i_flat:6.2f} {f['mass_g']:7.1f}")
    print(f"{'':7s} (RFIL: 3.0 mm root fillets, H = {h_fil:.3f} mm, "
          f"volume held at 80,000 mm^3)")
    print(f"{'':7s} (TWIN and R10 share I: parallel-axis I depends on rib area")
    print(f"{'':7s}  and height, not on where the rib sits across the width)")
    print()


def print_references():
    flat, coup, ref = reference_table()
    print("Reference values (what the check table compares against)")
    print("-" * 78)
    print(f"C1  flat centre deflection, 4-edge SS, q={Q_4EDGE:.2f} MPa")
    print(f"    D = E t^3/(12(1-nu^2)) = {flat['D']:.0f} N*mm")
    print(f"    w = {ALPHA_SSSS} * q * a^4 / D = {flat['w']:.5f} mm")
    print(f"C2  flat peak bending stress at the centre, bottom face")
    print(f"    M = {BETA_SSSS} * q * a^2 = {flat['M']:.2f} N*mm/mm")
    print(f"    sigma = 6M/t^2 = {flat['sigma']:.3f} MPa  "
          f"({flat['util']*100:.1f}% of Sy)")
    print(f"C3  ribbed 40 mm coupon, 2-edge SS, q={Q_COUPON:.2f} MPa")
    print(f"    I_T = {coup['sec']['I']:.1f} mm^4, ybar = "
          f"{coup['sec']['ybar']:.3f} mm, c_top = {coup['sec']['c_top']:.3f} mm")
    print(f"    w = 5 q W L^4/(384 E I_T) = {coup['w']:.5f} mm")
    print(f"    sigma_top = {coup['sig_top']:.3f} MPa (compression), "
          f"sigma_bot = {coup['sig_bot']:.3f} MPa")
    print()
    print("Frozen tolerances (set 2026-11-19, before any solver number existed)")
    print("-" * 78)
    for c in CHECKS:
        print(f"{c['id']}  +/-{c['tol_pct']:.1f}%  {c['name']}")
        print(f"     ref: {c['ref']}")
    print()
    print("Context values (reported in RESULTS/fig2, never gated)")
    print("-" * 78)
    b = ref_beam_2edge()
    print(f"2-edge rail group, 1-D beam: flat w = {b['w_flat']:.4f} mm, "
          f"ribbed w = {b['w_rib']:.5f} mm, ratio {b['ratio']:.2f}x")
    print(f"   (I ratio for the headline rib: "
          f"{t_section(W, T_SKIN, 10.0, 15.0)['I']/(W*T_FLAT**3/12):.2f}x)")
    for frac in (1.0, 0.9, 0.8, 0.7, 0.6):
        s = shear_lag_ref(frac)
        print(f"   flange width {frac*W:5.1f} mm -> I_T {s['I_eff']:8.1f} mm^4 "
              f"({s['I_ratio']*100:5.1f}% of full-width T)")
    print()
    print("   The 1-D T-beam is an upper bound on stiffness for the 200x100")
    print("   four-edge case: two-way plate action plus shear lag both make")
    print("   the solid softer. That gap is a result, reported as b_eff/W,")
    print("   not a pass/fail row.")
    print()


def print_mass_audit():
    rows, h_fil = check_volume()
    print("Mass audit (fixed 80,000 mm^3 at 2.70e-3 g/mm^3)")
    print("-" * 78)
    ok = True
    for name, v, err in rows:
        flag = "OK" if abs(err) < 1e-6 else "MISMATCH"
        ok = ok and abs(err) < 1e-6
        print(f"{name:7s} volume {v:9.3f} mm^3   mass {v*RHO:7.2f} g   {flag}")
    print(f"RFIL variant height (volume-corrected): H = {h_fil:.3f} mm "
          f"with r = {RFIL:.1f} mm root fillets")
    print(f"all variants at 216 g: {'yes' if ok else 'NO - fix the generator'}")
    print()
    return ok


# ---------------------------------------------------------------- measurement
def load_measurements(path="results/measurements.json"):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def print_check_table(meas):
    """meas: {"F-C": {"w_center_mm": .., "sig_max_MPa": ..}, "C3-C": {...}}"""
    flat, coup, ref = reference_table()
    rows = []
    ok = True
    for c in CHECKS:
        case = meas.get(c["case"], {})
        val = case.get(c["quantity"])
        if val is None:
            rows.append(dict(check=c["id"], name=c["name"], case=c["case"],
                             solver=None, reference=ref[c["id"]],
                             delta_pct=None, tol_pct=c["tol_pct"],
                             status="MISSING"))
            ok = False
            continue
        refv = ref[c["id"]]
        d = (val - refv) / refv * 100.0
        good = abs(d) <= c["tol_pct"]
        ok = ok and good
        rows.append(dict(check=c["id"], name=c["name"], case=c["case"],
                         solver=val, reference=refv, delta_pct=d,
                         tol_pct=c["tol_pct"],
                         status="PASS" if good else "FAIL"))

    print("Check table (frozen tolerances)")
    print("-" * 78)
    print(f"{'#':2s} {'Check':44s} {'Solver':>9s} {'Ref':>9s} "
          f"{'d %':>7s} {'tol':>5s} {'':4s}")
    for r in rows:
        s = "-" if r["solver"] is None else f"{r['solver']:.4f}"
        d = "-" if r["delta_pct"] is None else f"{r['delta_pct']:+.2f}"
        print(f"{r['check']:2s} {r['name'][:44]:44s} {s:>9s} "
              f"{r['reference']:.4f} {d:>7s} {r['tol_pct']:4.1f}% "
              f"{r['status']:4s}")
    print()
    out = dict(generated_by="verify.py", study="rib-vs-thickness",
               tolerances_frozen="2026-11-19",
               all_checks_pass=ok and all(r["status"] == "PASS" for r in rows),
               rows=rows)
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "verify.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote results/verify.json  all_checks_pass="
          f"{out['all_checks_pass']}")
    return out["all_checks_pass"]


def main(argv):
    print_header()
    print_sections()
    print_references()
    mass_ok = print_mass_audit()
    if "--measure" not in argv:
        print("No solver results read (Stage 2 mode). Table rows are empty by")
        print("design; run Stage 3, then: python verify.py --measure")
        return 0 if mass_ok else 2
    meas = load_measurements()
    if meas is None:
        print("results/measurements.json not found - nothing to check.")
        return 3
    passed = print_check_table(meas)
    return 0 if (passed and mass_ok) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
