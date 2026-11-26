# -*- coding: utf-8 -*-
"""Abaqus ODB extractor for rib-vs-thickness (run via: abaqus python code/extract.py <job> <results_dir>).

Writes into <results_dir>:
  nodes_<job>.csv  (id,x,y,z,U1,U2,U3,S11,S22,Mises)
  faces_<job>.csv  (n1,n2,n3,n4 - external quad faces of the deformed solid)
  summary_<job>.json (per-case summary metrics for postprocess.py)
"""
from odbAccess import openOdb
from abaqusConstants import ELEMENT_NODAL
import json
import os
import sys


# Local face indices (0-based into 8-node C3D8I connectivity)
HEX_FACES = (
    (0, 3, 2, 1),  # S1 (-z)
    (4, 5, 6, 7),  # S2 (+z)
    (0, 1, 5, 4),  # S3 (-y)
    (1, 2, 6, 5),  # S4 (+x)
    (2, 3, 7, 6),  # S5 (+y)
    (3, 0, 4, 7),  # S6 (-x)
)


def hex_volume(coords_list):
    """Exact volume of a linear hexahedron by decomposing into 6 tetrahedra."""
    p = coords_list
    def tet(a, b, c, d):
        ax, ay, az = p[a][0] - p[d][0], p[a][1] - p[d][1], p[a][2] - p[d][2]
        bx, by, bz = p[b][0] - p[d][0], p[b][1] - p[d][1], p[b][2] - p[d][2]
        cx, cy, cz = p[c][0] - p[d][0], p[c][1] - p[d][1], p[c][2] - p[d][2]
        det = (ax * (by * cz - bz * cy)
               - ay * (bx * cz - bz * cx)
               + az * (bx * cy - by * cx))
        return abs(det) / 6.0

    # 5-tet decomposition of a hexahedron (0..7)
    return (tet(0, 1, 3, 4) + tet(1, 2, 3, 6) + tet(1, 4, 5, 6)
            + tet(3, 4, 6, 7) + tet(1, 3, 4, 6))


def extract_case(job, results_dir):
    odb = openOdb(job + ".odb", readOnly=True)
    inst = odb.rootAssembly.instances[list(odb.rootAssembly.instances.keys())[0]]
    fr = odb.steps["LOAD"].frames[-1]

    coords = {}
    for nd in inst.nodes:
        c = nd.coordinates
        coords[nd.label] = (float(c[0]), float(c[1]), float(c[2]))

    # Compute mesh volume and external boundary faces
    face_map = {}
    v_quarter = 0.0
    n_elem = len(inst.elements)
    for el in inst.elements:
        conn = [int(n) for n in el.connectivity]
        v_quarter += hex_volume([coords[nid] for nid in conn])
        for f in HEX_FACES:
            quad = (conn[f[0]], conn[f[1]], conn[f[2]], conn[f[3]])
            key = tuple(sorted(quad))
            if key in face_map:
                del face_map[key]
            else:
                face_map[key] = quad

    # Nodal displacements
    U = {}
    for v in fr.fieldOutputs["U"].values:
        if v.instance is not None:
            d = v.data
            U[v.nodeLabel] = (float(d[0]), float(d[1]), float(d[2]))

    # Reaction force RF3 (summed over all nodes; unconstrained nodes have RF=0)
    rf3_sum = 0.0
    for v in fr.fieldOutputs["RF"].values:
        if v.instance is not None:
            rf3_sum += float(v.data[2])

    # Element-nodal stress S averaged onto nodes
    s_acc = {}
    s_field = fr.fieldOutputs["S"].getSubset(position=ELEMENT_NODAL)
    for v in s_field.values:
        d = v.data
        rec = s_acc.get(v.nodeLabel)
        if rec is None:
            s_acc[v.nodeLabel] = [float(d[0]), float(d[1]), float(v.mises), 1]
        else:
            rec[0] += float(d[0])
            rec[1] += float(d[1])
            rec[2] += float(v.mises)
            rec[3] += 1

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    ids = sorted(coords.keys())
    nodes_csv = os.path.join(results_dir, "nodes_%s.csv" % job)
    with open(nodes_csv, "w") as fh:
        fh.write("id,x,y,z,U1,U2,U3,S11,S22,Mises\n")
        for nid in ids:
            x, y, z = coords[nid]
            u1, u2, u3 = U[nid]
            rec = s_acc[nid]
            n = float(rec[3])
            s11, s22, mis = rec[0] / n, rec[1] / n, rec[2] / n
            fh.write("%d,%.6f,%.6f,%.6f,%.8e,%.8e,%.8e,%.6f,%.6f,%.6f\n"
                     % (nid, x, y, z, u1, u2, u3, s11, s22, mis))

    faces_csv = os.path.join(results_dir, "faces_%s.csv" % job)
    with open(faces_csv, "w") as fh:
        fh.write("n1,n2,n3,n4\n")
        for quad in face_map.values():
            fh.write("%d,%d,%d,%d\n" % quad)

    summary = {
        "case": job,
        "n_node": len(ids),
        "n_elem": n_elem,
        "RF3_sum_N": rf3_sum,
        "V_mm3": 4.0 * v_quarter,
    }
    sum_path = os.path.join(results_dir, "summary_%s.json" % job)
    with open(sum_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print("EXTRACTED %s: nodes=%d elems=%d faces=%d RF3=%.4f V_full=%.2f"
          % (job, len(ids), n_elem, len(face_map), rf3_sum, 4.0 * v_quarter))
    odb.close()


if __name__ == "__main__":
    job = sys.argv[1]
    res_dir = sys.argv[2]
    extract_case(job, res_dir)
