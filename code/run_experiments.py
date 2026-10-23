"""Reproduce the twelve TH and TH-curvature cases in the report."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", nargs="+", type=int, default=[2, 4, 8])
    parser.add_argument("--geometry", nargs="+", type=int, choices=[2, 3], default=[2, 3])
    parser.add_argument("--modes", nargs="+", choices=["TH", "TH-curvature"], default=["TH", "TH-curvature"])
    parser.add_argument("--S-max", type=float, default=1e4)
    parser.add_argument("--quadrature", type=int, default=16)
    parser.add_argument("--error-quadrature", type=int, default=20)
    parser.add_argument("--label", default="reproduced")
    args = parser.parse_args()
    folder = DATA_ROOT / args.label
    folder.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    rows, failures = [], []
    for geometry in args.geometry:
        output = folder / f"geometry_{geometry}"
        output.mkdir(exist_ok=True)
        for n in args.n:
            for mode in args.modes:
                start = time.perf_counter()
                command = [sys.executable, "-u", str(ROOT / "eikonal_th_curvature.py"),
                    "--n", str(n), "--disc", mode, "--geometry-degree", str(geometry),
                    "--protocol", "exact", "--S-max", str(args.S_max),
                    "--quadrature-degree", str(args.quadrature),
                    "--error-quadrature-degree", str(args.error_quadrature), "--out", str(output)]
                log = output / f"mobius_{mode}_exact_n{n}.log"
                with log.open("w", encoding="utf-8") as stream:
                    run = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
                path = output / f"mobius_{mode}_exact_n{n}.json"
                if run.returncode != 0 or not path.exists():
                    failure = dict(geometry_degree=geometry, n=n, mode=mode,
                                   exit_code=run.returncode, log=str(log))
                    failures.append(failure)
                    print(json.dumps(failure), flush=True)
                    continue
                data = json.loads(path.read_text())
                row = dict(geometry_degree=geometry, n=n, mode=mode,
                    **{k: data["summary"][k] for k in ["failed", "cells", "dofs", "hmax", "solve_time"]},
                    **{k: data["history"][-1][k] for k in
                       ["S", "residual", "residual_audit", "e_u_L2", "e_grad_L2", "e_gradu_L2", "Z_normal_L2",
                        "curvature_L2", "curvature_vector_error_L2"]},
                    wall_seconds=time.perf_counter()-start,
                    raw_json=str(path.relative_to(DATA_ROOT.parent)).replace("\\", "/"))
                rows.append(row)
                if row["failed"] or row["S"] != args.S_max:
                    failures.append(dict(geometry_degree=geometry, n=n, mode=mode,
                                         failed=row["failed"], S=row["S"]))
                print(json.dumps({k: row[k] for k in ["geometry_degree", "n", "mode", "failed", "S", "e_u_L2", "Z_normal_L2"]}), flush=True)
                (folder / "summary.json").write_text(json.dumps(rows, indent=2)+"\n")
                with (folder / "summary.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
    import dolfinx, basix, ufl
    from petsc4py import PETSc
    record = dict(driver_sha256=hashlib.sha256((ROOT / "eikonal_th_curvature.py").read_bytes()).hexdigest(),
                  configuration=vars(args), python=sys.version, dolfinx=dolfinx.__version__,
                  basix=basix.__version__, ufl=ufl.__version__, PETSc=PETSc.Sys.getVersion(),
                  mpi_processes=1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1,
                  failures=failures)
    (folder / "provenance.json").write_text(json.dumps(record, indent=2)+"\n")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
