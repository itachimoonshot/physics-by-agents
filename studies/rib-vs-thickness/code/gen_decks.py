# -*- coding: utf-8 -*-
"""Parametric Abaqus C3D8I input-deck generator for study rib-vs-thickness (PLAN.md).

Pure Python 3.14 + numpy. Builds an all-quad 2D cross-section in (x, z) and sweeps
along y in equal layers to form an all-hex C3D8I quarter model (0 <= x <= W/2,
0 <= y <= L/2, z >= 0).

Units: mm, N, MPa, tonne, s.

Usage:
  python code/gen_decks.py --list
  python code/gen_decks.py runs [CASE ...]
"""
import math
import os
import sys
import numpy as np

# ------------------------------------------------------------------ constants (PLAN.md)
E = 68900.0
NU = 0.33
RHO_TONNE = 2.70e-9   # tonne/mm^3 (2.70e-3 g/mm^3)

L = 200.0
W_FULL = 100.0
W_COUPON = 40.0
T_FLAT = 4.0
T_SKIN = 2.5
RFIL_R = 3.0

RIBS = {
    "R05":  dict(n=1, t_r=30.0, H=5.0),
    "R10":  dict(n=1, t_r=15.0, H=10.0),
    "R15":  dict(n=1, t_r=10.0, H=15.0),
    "R20":  dict(n=1, t_r=7.5,  H=20.0),
    "R25":  dict(n=1, t_r=6.0,  H=25.0),
    "TWIN": dict(n=2, t_r=7.5,  H=10.0, x_off=25.0),
}

LEVELS = {
    "A": 5.00,
    "B": 2.50,
    "C": 1.25,
}

# RFIL volume-corrected rib height (matches verify.py)
_FIL_ADD = 2.0 * (RFIL_R ** 2 - math.pi * RFIL_R ** 2 / 4.0)
H_RFIL = (80000.0 / L - W_FULL * T_SKIN - _FIL_ADD) / RIBS["R15"]["t_r"]


def _case(cid, group, variant, level, width, q_MPa, four_edge):
    return dict(
        case=cid, group=group, variant=variant, mesh=level,
        width=width, q_MPa=q_MPa, four_edge=four_edge,
    )


CASES = {c["case"]: c for c in [
    # K group: 4-edge SS, W=100, q=0.20 MPa
    _case("K-F-A",    "K", "F",    "A", W_FULL, 0.20, True),
    _case("K-F-B",    "K", "F",    "B", W_FULL, 0.20, True),
    _case("K-F-C",    "K", "F",    "C", W_FULL, 0.20, True),
    _case("K-R15-A",  "K", "R15",  "A", W_FULL, 0.20, True),
    _case("K-R15-B",  "K", "R15",  "B", W_FULL, 0.20, True),
    _case("K-R15-C",  "K", "R15",  "C", W_FULL, 0.20, True),
    _case("K-R05-B",  "K", "R05",  "B", W_FULL, 0.20, True),
    _case("K-R10-B",  "K", "R10",  "B", W_FULL, 0.20, True),
    _case("K-R20-B",  "K", "R20",  "B", W_FULL, 0.20, True),
    _case("K-R25-B",  "K", "R25",  "B", W_FULL, 0.20, True),
    _case("K-TWIN-B", "K", "TWIN", "B", W_FULL, 0.20, True),
    _case("K-RFIL-B", "K", "RFIL", "B", W_FULL, 0.20, True),
    _case("K-RFIL-C", "K", "RFIL", "C", W_FULL, 0.20, True),
    # E group: 2-edge SS, W=100, q=0.01 MPa
    _case("E-F-B",    "E", "F",    "B", W_FULL, 0.01, False),
    _case("E-R15-B",  "E", "R15",  "B", W_FULL, 0.01, False),
    _case("E-TWIN-B", "E", "TWIN", "B", W_FULL, 0.01, False),
    # C3 group: 2-edge SS coupon, W=40, R15 rib, q=0.05 MPa
    _case("C3-A",     "C3", "R15", "A", W_COUPON, 0.05, False),
    _case("C3-B",     "C3", "R15", "B", W_COUPON, 0.05, False),
    _case("C3-C",     "C3", "R15", "C", W_COUPON, 0.05, False),
]}


# ------------------------------------------------------------------ 2D cross-section builder
class SectionMesh2D:
    """Collects conformal 2D quad blocks in (x, z) with node deduplication."""

    def __init__(self):
        self.pts = []
        self._map = {}
        self.quads = []  # (a, b, c, d) CCW in (x, z)

    def node(self, x, z):
        key = (round(float(x), 8), round(float(z), 8))
        idx = self._map.get(key)
        if idx is None:
            idx = len(self.pts)
            self.pts.append((float(x), float(z)))
            self._map[key] = idx
        return idx

    def add_rect(self, xs, zs):
        nx, nz = len(xs) - 1, len(zs) - 1
        grid = np.zeros((nx + 1, nz + 1), dtype=int)
        for i, x in enumerate(xs):
            for k, z in enumerate(zs):
                grid[i, k] = self.node(x, z)
        for i in range(nx):
            for k in range(nz):
                self.quads.append((
                    int(grid[i, k]),
                    int(grid[i + 1, k]),
                    int(grid[i + 1, k + 1]),
                    int(grid[i, k + 1]),
                ))

    def add_tfi(self, nu, nv, bottom_fn, top_fn, left_fn, right_fn):
        """Transfinite interpolation on [0,1]^2; edges must agree at corners."""
        p00 = np.array(bottom_fn(0.0))
        p10 = np.array(bottom_fn(1.0))
        p11 = np.array(top_fn(1.0))
        p01 = np.array(top_fn(0.0))
        grid = np.zeros((nu + 1, nv + 1), dtype=int)
        for i in range(nu + 1):
            u = i / nu
            bu = np.array(bottom_fn(u))
            tu = np.array(top_fn(u))
            for j in range(nv + 1):
                v = j / nv
                lv = np.array(left_fn(v))
                rv = np.array(right_fn(v))
                p = ((1 - v) * bu + v * tu + (1 - u) * lv + u * rv
                     - ((1 - u) * (1 - v) * p00 + u * (1 - v) * p10
                        + u * v * p11 + (1 - u) * v * p01))
                grid[i, j] = self.node(p[0], p[1])
        for i in range(nu):
            for j in range(nv):
                self.quads.append((
                    int(grid[i, j]),
                    int(grid[i + 1, j]),
                    int(grid[i + 1, j + 1]),
                    int(grid[i, j + 1]),
                ))


def _seg_linspace(x0, x1, seed, min_n=2):
    n = max(min_n, int(round((x1 - x0) / seed)))
    return np.linspace(x0, x1, n + 1)


def build_section(variant, level, width):
    seed = LEVELS[level]
    w_half = width / 2.0
    m = SectionMesh2D()

    if variant == "F":
        xs = _seg_linspace(0.0, w_half, seed, min_n=4)
        zs = np.linspace(0.0, T_FLAT, 5)  # 4 layers -> exact z=2.0 mid-thickness line
        m.add_rect(xs, zs)
        z_mid_support = T_FLAT / 2.0
        return m, z_mid_support

    z_mid_support = T_SKIN / 2.0
    zs_skin = np.array([0.0, T_SKIN / 2.0, T_SKIN])  # 2 layers -> exact z=1.25

    if variant == "TWIN":
        r = RIBS["TWIN"]
        x1 = r["x_off"] - r["t_r"] / 2.0   # 21.25
        x2 = r["x_off"] + r["t_r"] / 2.0   # 28.75
        xs1 = _seg_linspace(0.0, x1, seed, min_n=2)
        xs2 = _seg_linspace(x1, x2, seed, min_n=3)
        xs3 = _seg_linspace(x2, w_half, seed, min_n=2)
        xs_all = np.concatenate([xs1[:-1], xs2[:-1], xs3])
        m.add_rect(xs_all, zs_skin)
        nz_rib = 4 if level != "C" else 8
        zs_rib = np.linspace(T_SKIN, T_SKIN + r["H"], nz_rib + 1)
        m.add_rect(xs2, zs_rib)
        return m, z_mid_support

    if variant == "RFIL":
        x_rib = RIBS["R15"]["t_r"] / 2.0   # 5.0
        r_fil = RFIL_R                     # 3.0
        x_fil_end = x_rib + r_fil          # 8.0
        z_fil_top = T_SKIN + r_fil         # 5.5
        z_top = T_SKIN + H_RFIL

        k = 2 if level == "B" else 4       # 4 or 8 elements along the 90-deg fillet arc
        nx_rib = max(2, int(round(x_rib / seed)))
        xs_rib = np.linspace(0.0, x_rib, nx_rib + 1)
        x_mid_fil = x_rib + 0.30 * r_fil   # 5.9
        z_mid_fil = T_SKIN + 0.30 * r_fil  # 3.4
        xs_fil = np.concatenate([
            np.linspace(x_rib, x_mid_fil, k + 1)[:-1],
            np.linspace(x_mid_fil, x_fil_end, k + 1),
        ])
        xs_flange = _seg_linspace(x_fil_end, w_half, seed, min_n=4)
        xs_skin = np.concatenate([xs_rib[:-1], xs_fil[:-1], xs_flange])
        m.add_rect(xs_skin, zs_skin)

        # Lower rib alongside fillet [0, 5.0] x [2.5, 5.5]
        zs_rib_low = np.concatenate([
            np.linspace(T_SKIN, z_mid_fil, k + 1)[:-1],
            np.linspace(z_mid_fil, z_fil_top, k + 1),
        ])
        m.add_rect(xs_rib, zs_rib_low)

        # Upper rib [0, 5.0] x [5.5, z_top]
        nz_up = max(4, int(round((H_RFIL - r_fil) / seed)))
        zs_rib_up = np.linspace(z_fil_top, z_top, nz_up + 1)
        m.add_rect(xs_rib, zs_rib_up)

        # 3-block conformal fillet Y-mesh (strictly convex corners)
        p0 = (x_rib, T_SKIN)
        p1 = (x_mid_fil, T_SKIN)
        p2 = (x_fil_end, T_SKIN)
        th_mid = math.pi / 4.0
        p3 = (x_fil_end - r_fil * math.cos(th_mid), z_fil_top - r_fil * math.sin(th_mid))
        p4 = (x_rib, z_fil_top)
        p5 = (x_rib, z_mid_fil)
        c_pt = (x_rib + 0.2167 * r_fil, T_SKIN + 0.2167 * r_fil)  # (5.65, 3.15)

        def lerp(a, b):
            return lambda t: (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))

        def arc(t0, t1):
            def fn(t):
                th = t0 + t * (t1 - t0)
                return (x_fil_end - r_fil * math.cos(th),
                        z_fil_top - r_fil * math.sin(th))
            return fn

        # Block 1: p0 -> p1 -> c_pt -> p5
        m.add_tfi(k, k, lerp(p0, p1), lerp(p5, c_pt), lerp(p0, p5), lerp(p1, c_pt))
        # Block 2: p1 -> p2 -> p3 -> c_pt (right edge is lower arc pi/2 -> pi/4)
        m.add_tfi(k, k, lerp(p1, p2), lerp(c_pt, p3), lerp(p1, c_pt), arc(math.pi / 2.0, th_mid))
        # Block 3: p5 -> c_pt -> p3 -> p4 (top edge is upper arc 0 -> pi/4)
        m.add_tfi(k, k, lerp(p5, c_pt), arc(0.0, th_mid), lerp(p5, p4), lerp(c_pt, p3))
        return m, z_mid_support

    # Single central rib R05..R25
    r = RIBS[variant]
    x_rib = r["t_r"] / 2.0
    nx_rib = max(2, int(round(x_rib / seed)))
    nx_flange = max(2, int(round((w_half - x_rib) / seed)))
    xs_rib = np.linspace(0.0, x_rib, nx_rib + 1)
    xs_flange = _seg_linspace(x_rib, w_half, seed, min_n=2)
    xs_skin = np.concatenate([xs_rib[:-1], xs_flange])
    m.add_rect(xs_skin, zs_skin)

    nz_rib = 4 if level != "C" else 8
    zs_rib = np.linspace(T_SKIN, T_SKIN + r["H"], nz_rib + 1)
    m.add_rect(xs_rib, zs_rib)
    return m, z_mid_support


def section_area(pts, quads):
    area = 0.0
    for a, b, c, d in quads:
        xa, za = pts[a]
        xb, zb = pts[b]
        xc, zc = pts[c]
        xd, zd = pts[d]
        s = 0.5 * ((xa * zb - xb * za) + (xb * zc - xc * zb)
                   + (xc * zd - xd * zc) + (xd * za - xa * zd))
        assert s > 0.0, f"Non-CCW or degenerate 2D quad: {(a, b, c, d)} area={s}"
        area += s
    return area


def top_load_edges(pts, quads, w_half):
    """Return list of (quad_idx, face_tag) for exposed upper/rib faces."""
    edge_counts = {}
    face_tags = ("S1", "S4", "S2", "S6")
    for qi, q in enumerate(quads):
        edges = ((q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0]))
        for ei, (u, v) in enumerate(edges):
            key = (min(u, v), max(u, v))
            edge_counts.setdefault(key, []).append((qi, face_tags[ei], u, v))

    loaded = []
    tol = 1e-6
    for key, owners in edge_counts.items():
        if len(owners) != 1:
            continue
        qi, ftag, u, v = owners[0]
        x1, z1 = pts[u]
        x2, z2 = pts[v]
        # Exclude bottom z=0, symmetry x=0, outer edge x=W/2
        if abs(z1) < tol and abs(z2) < tol:
            continue
        if abs(x1) < tol and abs(x2) < tol:
            continue
        if abs(x1 - w_half) < tol and abs(x2 - w_half) < tol:
            continue
        loaded.append((qi, ftag))
    return loaded


def write_ids(fh, ids):
    for s in range(0, len(ids), 16):
        fh.write(", ".join(str(v) for v in ids[s:s + 16]) + "\n")


def write_deck(case_cfg, out_dir):
    cid = case_cfg["case"]
    width = case_cfg["width"]
    w_half = width / 2.0
    y_half = L / 2.0
    level = case_cfg["mesh"]
    seed = LEVELS[level]

    sec, z_mid = build_section(case_cfg["variant"], level, width)
    area_2d = section_area(sec.pts, sec.quads)
    loaded_2d = top_load_edges(sec.pts, sec.quads, w_half)

    ny = int(round(y_half / seed))
    ys = np.linspace(0.0, y_half, ny + 1)

    n2d = len(sec.pts)
    q2d = len(sec.quads)

    os.makedirs(out_dir, exist_ok=True)
    inp_path = os.path.join(out_dir, f"{cid}.inp")

    symx, symy, supx, supy = [], [], [], []
    tol = 1e-6

    with open(inp_path, "w", encoding="utf-8") as fh:
        fh.write("*Heading\n")
        fh.write(f"** {cid}: rib-vs-thickness, generic 6061-T6 lid, quarter model\n")
        fh.write("*Preprint, echo=NO, model=NO, history=NO, contact=NO\n")
        fh.write("*Node\n")
        for j, y in enumerate(ys):
            base = j * n2d
            for idx, (x, z) in enumerate(sec.pts):
                nid = base + idx + 1
                fh.write(f"{nid}, {x:.8f}, {y:.8f}, {z:.8f}\n")
                if abs(x) < tol:
                    symx.append(nid)
                if abs(y) < tol:
                    symy.append(nid)
                if abs(x - w_half) < tol and abs(z - z_mid) < tol:
                    supx.append(nid)
                if abs(y - y_half) < tol and abs(z - z_mid) < tol:
                    supy.append(nid)

        fh.write("*Element, type=C3D8I, elset=SOLID\n")
        eid = 0
        for j in range(ny):
            b0 = j * n2d
            b1 = (j + 1) * n2d
            for a, b, c, d in sec.quads:
                eid += 1
                n1, n2, n3, n4 = b0 + a + 1, b0 + b + 1, b1 + b + 1, b1 + a + 1
                n5, n6, n7, n8 = b0 + d + 1, b0 + c + 1, b1 + c + 1, b1 + d + 1
                fh.write(f"{eid}, {n1}, {n2}, {n3}, {n4}, {n5}, {n6}, {n7}, {n8}\n")

        fh.write("*Nset, nset=SYMX\n"); write_ids(fh, symx)
        fh.write("*Nset, nset=SYMY\n"); write_ids(fh, symy)
        fh.write("*Nset, nset=SUPX\n"); write_ids(fh, supx)
        fh.write("*Nset, nset=SUPY\n"); write_ids(fh, supy)

        # Element sets per face tag for *Surface / *Dsload
        by_face = {}
        for qi, ftag in loaded_2d:
            eids_for_q = [j * q2d + qi + 1 for j in range(ny)]
            by_face.setdefault(ftag, []).extend(eids_for_q)

        for ftag, elist in sorted(by_face.items()):
            fh.write(f"*Elset, elset=EL_LOAD_{ftag}\n")
            write_ids(fh, elist)

        fh.write("*Surface, type=ELEMENT, name=TOPFACE\n")
        for ftag in sorted(by_face.keys()):
            fh.write(f"EL_LOAD_{ftag}, {ftag}\n")

        fh.write("*Solid Section, elset=SOLID, material=AL6061T6\n,\n")
        fh.write("*Material, name=AL6061T6\n")
        fh.write("*Elastic\n")
        fh.write(f"{E:.1f}, {NU:.4f}\n")
        fh.write("*Density\n")
        fh.write(f"{RHO_TONNE:.3e}\n")
        fh.write("*Step, name=LOAD, nlgeom=NO\n")
        fh.write("*Static\n")
        fh.write("1., 1., 1e-05, 1.\n")
        fh.write("*Boundary\n")
        fh.write("SYMX, 1, 1\n")
        fh.write("SYMY, 2, 2\n")
        if case_cfg["four_edge"]:
            fh.write("SUPX, 3, 3\n")
        fh.write("SUPY, 3, 3\n")
        fh.write("*Dsload\n")
        fh.write(f"TOPFACE, P, {case_cfg['q_MPa']:.6f}\n")
        fh.write("*Output, field, frequency=1\n")
        fh.write("*Node Output\nU, RF\n")
        fh.write("*Element Output\nS, E\n")
        fh.write("*End Step\n")

    n_nodes = n2d * (ny + 1)
    n_elems = q2d * ny
    vol_full = 4.0 * area_2d * y_half
    return n_nodes, n_elems, vol_full


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or args[0] == "--list":
        print(f"{'Case':10s} {'Grp':4s} {'Var':5s} {'Lv':3s} {'W':>5s} {'q':>6s} {'4E':>4s}")
        for c in CASES.values():
            print(f"{c['case']:10s} {c['group']:4s} {c['variant']:5s} {c['mesh']:3s} "
                  f"{c['width']:5.1f} {c['q_MPa']:6.3f} {str(c['four_edge']):>4s}")
        sys.exit(0)
    outroot = args[0]
    targets = args[1:] if len(args) > 1 else list(CASES.keys())
    for cid in targets:
        nn, ne, vf = write_deck(CASES[cid], os.path.join(out_dir := os.path.join(outroot, cid)))
        print(f"{cid:10s} nodes={nn:6d} elems={ne:6d} V_full={vf:9.2f} mm^3")
