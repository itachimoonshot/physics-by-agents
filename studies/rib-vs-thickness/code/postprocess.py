# -*- coding: utf-8 -*-
"""Post-process extracted CSVs for rib-vs-thickness into cases.json, measurements.json,
and evaluate all PLAN.md Section 9 global pass gates.
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(STUDY_DIR, "results")

sys.path.insert(0, HERE)
import gen_decks as g


def load_nodes(cid):
    path = os.path.join(RESULTS_DIR, f"nodes_{cid}.csv")
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "id": int(r["id"]),
                "x": float(r["x"]),
                "y": float(r["y"]),
                "z": float(r["z"]),
                "U1": float(r["U1"]),
                "U2": float(r["U2"]),
                "U3": float(r["U3"]),
                "S11": float(r["S11"]),
                "S22": float(r["S22"]),
                "Mises": float(r["Mises"]),
            })
    return rows


def load_summary(cid):
    with open(os.path.join(RESULTS_DIR, f"summary_{cid}.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def compute_b_eff_W(nodes):
    """Effective breadth ratio at mid-span y=0 (PLAN.md 8.3) using longitudinal S22."""
    tol = 1e-4
    bot = {}
    top = {}
    for nd in nodes:
        if abs(nd["y"]) < tol:
            xr = round(nd["x"], 4)
            if abs(nd["z"]) < tol:
                bot[xr] = nd["S22"]
            elif abs(nd["z"] - g.T_SKIN) < tol:
                top[xr] = nd["S22"]
    xs = sorted(set(bot.keys()) & set(top.keys()))
    assert xs and abs(xs[0]) < tol and abs(xs[-1] - 50.0) < tol, f"Bad skin line xs={xs[:3]}..{xs[-3:]}"
    sig_mem = [0.5 * (top[x] + bot[x]) for x in xs]
    integral = 0.0
    for i in range(len(xs) - 1):
        integral += 0.5 * (sig_mem[i] + sig_mem[i + 1]) * (xs[i + 1] - xs[i])
    peak = max(abs(s) for s in sig_mem)
    b_eff = integral / peak
    return b_eff / 50.0


def analyze_case(cid, cfg):
    nodes = load_nodes(cid)
    summ = load_summary(cid)
    tol = 1e-4

    # Center deflection at (x=0, y=0, max z at x=0,y=0)
    origin_nodes = [nd for nd in nodes if abs(nd["x"]) < tol and abs(nd["y"]) < tol]
    center_top = max(origin_nodes, key=lambda nd: nd["z"])
    w_center_mm = abs(center_top["U3"])
    w_max_mm = max(abs(nd["U3"]) for nd in nodes)

    # Skin bottom z=0
    skin_bot = [nd for nd in nodes if abs(nd["z"]) < tol]
    if cfg["variant"] == "F":
        if cfg["group"] == "K":
            near_c = [nd for nd in skin_bot if math.hypot(nd["x"], nd["y"]) <= 5.0 + tol]
            sigma_skin_MPa = max(nd["S11"] for nd in near_c)
        else:
            sigma_skin_MPa = max(nd["S22"] for nd in skin_bot)
    else:
        # Report max S11 on bottom face (skin bending across panels) and also record S22
        sigma_skin_MPa = max(nd["S11"] for nd in skin_bot)

    # Rib tip (max z over the whole solid, for ribbed variants)
    if cfg["variant"] == "F":
        sigma_tip_MPa = None
    else:
        z_tip = max(nd["z"] for nd in nodes)
        tip_nodes = [nd for nd in nodes if abs(nd["z"] - z_tip) < tol]
        # Rib spans y, so longitudinal bending stress at rib tip is S22 (compression < 0)
        tip_s22 = min(tip_nodes, key=lambda nd: nd["S22"])
        sigma_tip_MPa = tip_s22["S22"]

    # Rib root stress (RFIL only): nodes along fillet arc 5.0 <= x <= 8.0, 2.5 <= z <= 5.5
    if cfg["variant"] == "RFIL":
        root_nodes = [
            nd for nd in nodes
            if (5.0 - tol <= nd["x"] <= 8.0 + tol and 2.5 - tol <= nd["z"] <= 5.5 + tol
                and abs(math.hypot(8.0 - nd["x"], 5.5 - nd["z"]) - g.RFIL_R) < 0.05)
        ]
        peak_root_s11 = max(root_nodes, key=lambda nd: abs(nd["S11"]))
        peak_root_s22 = max(root_nodes, key=lambda nd: abs(nd["S22"]))
        peak_root_mises = max(root_nodes, key=lambda nd: nd["Mises"])
        sigma_root_MPa = peak_root_s11["S11"]
        extra_root = {
            "sigma_root_S22_MPa": round(peak_root_s22["S22"], 4),
            "sigma_root_Mises_MPa": round(peak_root_mises["Mises"], 4),
        }
    else:
        sigma_root_MPa = None
        extra_root = {}

    b_eff_W = compute_b_eff_W(nodes) if cid == "E-R15-B" else None

    rec = {
        "case": cid,
        "group": cfg["group"],
        "variant": cfg["variant"],
        "mesh": cfg["mesh"],
        "n_elem": summ["n_elem"],
        "n_node": summ["n_node"],
        "q_MPa": cfg["q_MPa"],
        "w_center_mm": round(w_center_mm, 6),
        "w_max_mm": round(w_max_mm, 6),
        "sigma_skin_MPa": round(sigma_skin_MPa, 4),
        "sigma_skin_S22_MPa": round(max(nd["S22"] for nd in skin_bot), 4),
        "sigma_tip_MPa": round(sigma_tip_MPa, 4) if sigma_tip_MPa is not None else None,
        "sigma_root_MPa": round(sigma_root_MPa, 4) if sigma_root_MPa is not None else None,
        "RF3_sum_N": round(summ["RF3_sum_N"], 6),
        "V_mm3": round(summ["V_mm3"], 3),
        "b_eff_W": round(b_eff_W, 5) if b_eff_W is not None else None,
        "wall_s": summ.get("wall_s", 0.0),
    }
    rec.update(extra_root)
    return rec


def main():
    cases = []
    by_id = {}
    for cid, cfg in g.CASES.items():
        rec = analyze_case(cid, cfg)
        cases.append(rec)
        by_id[cid] = rec

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "cases.json"), "w", encoding="utf-8") as fh:
        json.dump(cases, fh, indent=2)

    measurements = {
        "K-F-C": {
            "w_center_mm": by_id["K-F-C"]["w_center_mm"],
            "sig_max_MPa": by_id["K-F-C"]["sigma_skin_MPa"],
        },
        "C3-C": {
            "w_center_mm": by_id["C3-C"]["w_center_mm"],
        },
    }
    with open(os.path.join(RESULTS_DIR, "measurements.json"), "w", encoding="utf-8") as fh:
        json.dump(measurements, fh, indent=2)

    # Check global pass gates (PLAN.md Section 9)
    gates = {}
    # Gate 2: V_mm3 within 0.5% of 80,000 for 100mm cases (C3 is 40mm width: (40*2.5+150)*200 = 50,000 mm^3)
    v_errs = []
    for c in cases:
        v_target = 80000.0 if g.CASES[c["case"]]["width"] == 100.0 else 50000.0
        err_pct = abs(c["V_mm3"] - v_target) / v_target * 100.0
        if err_pct > 0.5:
            v_errs.append((c["case"], c["V_mm3"], v_target, err_pct))
    gates["G2_volume_ok"] = len(v_errs) == 0

    # Gate 3: |abs(RF3_sum) - q*A_loaded| / (q*A_loaded) <= 1e-6
    rf_errs = []
    for c in cases:
        cfg = g.CASES[c["case"]]
        a_q = (cfg["width"] / 2.0) * (g.L / 2.0)
        f_exp = cfg["q_MPa"] * a_q
        rel = abs(abs(c["RF3_sum_N"]) - f_exp) / f_exp
        if rel > 1e-6:
            rf_errs.append((c["case"], c["RF3_sum_N"], f_exp, rel))
    gates["G3_reaction_ok"] = len(rf_errs) == 0

    # Gate 4: Mesh convergence on K-F and K-R15
    for prefix in ("K-F", "K-R15"):
        wA = by_id[f"{prefix}-A"]["w_center_mm"]
        wB = by_id[f"{prefix}-B"]["w_center_mm"]
        wC = by_id[f"{prefix}-C"]["w_center_mm"]
        dBC = abs(wC - wB) / wC * 100.0
        dAB = abs(wB - wA) / wC * 100.0
        gates[f"G4_{prefix}_BC_pct"] = round(dBC, 4)
        gates[f"G4_{prefix}_AB_pct"] = round(dAB, 4)
        gates[f"G4_{prefix}_ok"] = (dBC <= 1.5 and dAB <= 5.0)

    # Gate 5: sigma_root on RFIL converged <= 5.0%
    sB = abs(by_id["K-RFIL-B"]["sigma_root_MPa"])
    sC = abs(by_id["K-RFIL-C"]["sigma_root_MPa"])
    d_root = abs(sC - sB) / sC * 100.0
    gates["G5_RFIL_S11_pct"] = round(d_root, 4)
    gates["G5_RFIL_ok"] = d_root <= 5.0

    # Gate 6: sweep monotone in H (R05 -> R10 -> R15 -> R20 -> R25)
    sweep_ids = ["K-R05-B", "K-R10-B", "K-R15-B", "K-R20-B", "K-R25-B"]
    ws = [by_id[k]["w_center_mm"] for k in sweep_ids]
    gates["G6_sweep_w_center_mm"] = dict(zip(sweep_ids, ws))
    gates["G6_monotone_ok"] = all(ws[i] >= ws[i + 1] for i in range(len(ws) - 1))

    with open(os.path.join(RESULTS_DIR, "gates.json"), "w", encoding="utf-8") as fh:
        json.dump(gates, fh, indent=2)

    print(json.dumps(gates, indent=2))
    if v_errs or rf_errs:
        print("VOLUME ERRORS:", v_errs)
        print("RF3 ERRORS:", rf_errs)
        sys.exit(1)


if __name__ == "__main__":
    main()
