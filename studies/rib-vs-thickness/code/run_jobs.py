# -*- coding: utf-8 -*-
"""Generate, solve, and extract all 19 Abaqus cases for rib-vs-thickness.

Usage:
  python code/run_jobs.py [--cpus 8] [--parallel 1] [CASE ...]
"""
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.dirname(HERE)
RUNS_DIR = os.path.join(STUDY_DIR, "runs")
RESULTS_DIR = os.path.join(STUDY_DIR, "results")
ABQ = os.environ.get("ABAQUS_CMD", "abq2023")   # abaqus command on PATH, or set ABAQUS_CMD

sys.path.insert(0, HERE)
import gen_decks as g


def sh(args, cwd, log_path):
    cmd = ["cmd", "/c", ABQ] + args if os.name == "nt" else [ABQ] + args
    with open(log_path, "a", encoding="utf-8") as fh:
        return subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT).returncode


def is_done(case_dir, cid):
    sta = os.path.join(case_dir, f"{cid}.sta")
    nodes_csv = os.path.join(RESULTS_DIR, f"nodes_{cid}.csv")
    sum_json = os.path.join(RESULTS_DIR, f"summary_{cid}.json")
    if not (os.path.exists(sta) and os.path.exists(nodes_csv) and os.path.exists(sum_json)):
        return False
    return "COMPLETED SUCCESSFULLY" in open(sta, errors="ignore").read()


def run_one(cid, cpus):
    case_dir = os.path.join(RUNS_DIR, cid)
    os.makedirs(case_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    if is_done(case_dir, cid):
        return cid, "already_done", 0.0

    t0 = time.time()
    cfg = g.CASES[cid]
    nn, ne, vol_full = g.write_deck(cfg, case_dir)
    log_path = os.path.join(case_dir, f"run_{cid}.log")
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write(f"case {cid}: nodes={nn}, elems={ne}, V_full={vol_full:.2f} mm^3\n")

    rc = sh([f"job={cid}", f"input={cid}.inp", f"cpus={cpus}", "interactive", "ask_delete=OFF"],
            case_dir, log_path)
    sta = os.path.join(case_dir, f"{cid}.sta")
    ok = os.path.exists(sta) and "COMPLETED SUCCESSFULLY" in open(sta, errors="ignore").read()
    wall_s = time.time() - t0
    if ok:
        rc_ex = sh(["python", os.path.join(HERE, "extract.py"), cid, RESULTS_DIR], case_dir, log_path)
        sum_json = os.path.join(RESULTS_DIR, f"summary_{cid}.json")
        if rc_ex == 0 and os.path.exists(sum_json):
            import json
            with open(sum_json, "r", encoding="utf-8") as fh:
                s = json.load(fh)
            s["wall_s"] = round(wall_s, 3)
            with open(sum_json, "w", encoding="utf-8") as fh:
                json.dump(s, fh, indent=2)
    return cid, ("ok" if is_done(case_dir, cid) else f"FAILED rc={rc}"), wall_s


if __name__ == "__main__":
    args = sys.argv[1:]
    cpus, par = 8, 1
    if "--cpus" in args:
        i = args.index("--cpus")
        cpus = int(args[i + 1])
        del args[i:i + 2]
    if "--parallel" in args:
        i = args.index("--parallel")
        par = int(args[i + 1])
        del args[i:i + 2]
    targets = args if args else list(g.CASES.keys())
    with ThreadPoolExecutor(max_workers=par) as ex:
        for cid, status, dt in ex.map(lambda c: run_one(c, cpus), targets):
            print(f"{cid:10s} {status:14s} {dt:7.2f} s", flush=True)
