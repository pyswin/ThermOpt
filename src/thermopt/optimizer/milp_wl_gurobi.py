from __future__ import annotations

"""
Gurobi backend for the Phase-1 MILP init (see milp_wl.py).

scipy's bundled HiGHS wrapper exposes no incumbent callback, so it is
impossible to observe the position of chiplets *during* a single solve —
only the final result after the time limit. Gurobi's callback API does
expose incumbents (MIPSOL) as they are found, which lets us record a
snapshot of chiplet positions roughly once per `snapshot_interval` seconds
of wall-clock solve time, using the most recent incumbent whenever the
solver goes a while without improving.

The MILP formulation itself (variables, constraints, objective) is NOT
duplicated here — it is built once by `milp_wl.build_init_problem` and
translated generically into a Gurobi model, so a change to the shared
formulation is automatically picked up by both backends.
"""

from dataclasses import dataclass

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout
from thermopt.optimizer.milp_wl import MILPInitResult, MILPProblem, build_init_problem, decode_init_solution


@dataclass(frozen=True)
class Snapshot:
    t: float
    layout: Layout


def initialize_with_snapshots(
    case: FloorplanCase,
    config: dict,
) -> tuple[MILPInitResult, list[Snapshot]]:
    """
    Same Phase-1 MILP init as milp_wl.initialize(), solved with Gurobi so that
    chiplet positions can be snapshotted every `snapshot_interval` seconds of
    solve time (default 1.0s), in addition to the final result.

    Extra config keys
    ------------------
    snapshot_interval  float  seconds between saved position snapshots (default 1.0)
    seed               int    Gurobi MIP seed (default 42)
    threads             int    Gurobi Threads param (default: solver default)
    """
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError(
            "milp_wl_gurobi requires gurobipy; run under an environment with a Gurobi license "
            "(e.g. the atplace_py39 conda env)."
        ) from exc

    time_limit = float(config.get("milp_time_limit", 120.0))
    mip_rel_gap = float(config.get("mip_rel_gap", 0.001))
    verbose = bool(config.get("verbose", False))
    snapshot_interval = float(config.get("snapshot_interval", 1.0))
    seed = int(config.get("seed", 42))

    problem = build_init_problem(case, config)
    model = _build_gurobi_model(problem, seed=seed)
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_rel_gap
    model.Params.OutputFlag = 1 if verbose else 0
    if "threads" in config:
        model.Params.Threads = int(config["threads"])

    snapshots: list[Snapshot] = []
    state = {"last_saved_t": -1e9, "last_vals": None}

    def _record(t: float, vals: np.ndarray) -> None:
        layout = decode_init_solution(case, problem, vals)
        snapshots.append(Snapshot(t=float(t), layout=layout))
        state["last_saved_t"] = t

    def callback(model: "gp.Model", where: int) -> None:
        if where == GRB.Callback.MIPSOL:
            t = model.cbGet(GRB.Callback.RUNTIME)
            vals = np.array(model.cbGetSolution(model._vars))
            state["last_vals"] = vals
            if t - state["last_saved_t"] >= snapshot_interval:
                _record(t, vals)
        elif where == GRB.Callback.MIP:
            t = model.cbGet(GRB.Callback.RUNTIME)
            if state["last_vals"] is not None and t - state["last_saved_t"] >= snapshot_interval:
                _record(t, state["last_vals"])

    model.optimize(callback)

    has_solution = model.SolCount > 0
    if not has_solution:
        from thermopt.optimizer.milp_wl import _grid_fallback

        return (
            MILPInitResult(
                layout=_grid_fallback(case),
                solver_success=False,
                solver_message=f"gurobi status={model.Status}, no feasible solution found",
                solver_objective=float("inf"),
            ),
            snapshots,
        )

    final_vals = np.array([v.X for v in model._vars])
    layout = decode_init_solution(case, problem, final_vals)
    # Always keep a snapshot of the true final incumbent, even if it landed inside
    # the last `snapshot_interval` window.
    solve_time = model.Runtime
    if not snapshots or snapshots[-1].t < solve_time - 1e-6:
        snapshots.append(Snapshot(t=float(solve_time), layout=layout))

    solver_success = model.Status == GRB.OPTIMAL
    result = MILPInitResult(
        layout=layout,
        solver_success=bool(solver_success),
        solver_message=f"gurobi status={model.Status}",
        solver_objective=float(model.ObjVal),
    )
    return result, snapshots


def _build_gurobi_model(problem: MILPProblem, seed: int) -> "gp.Model":
    import gurobipy as gp
    from gurobipy import GRB

    model = gp.Model("milp_wl_init")
    model.Params.Seed = seed

    variables = []
    for i in range(problem.num_vars):
        vtype = GRB.BINARY if problem.integrality[i] else GRB.CONTINUOUS
        lb = 0.0 if problem.integrality[i] else float(problem.lb[i])
        ub = 1.0 if problem.integrality[i] else float(problem.ub[i])
        variables.append(model.addVar(lb=lb, ub=ub, vtype=vtype, obj=float(problem.obj[i])))
    model.update()
    model._vars = variables
    model.setAttr(GRB.Attr.ModelSense, GRB.MINIMIZE)

    for row, lo, hi in zip(problem.rows, problem.lows, problem.highs):
        expr = gp.LinExpr([(coeff, variables[col]) for col, coeff in row.items()])
        finite_lo = lo > -np.inf
        finite_hi = hi < np.inf
        if finite_lo and finite_hi:
            if lo == hi:
                model.addLConstr(expr, GRB.EQUAL, lo)
            else:
                model.addRange(expr, lo, hi)
        elif finite_hi:
            model.addLConstr(expr, GRB.LESS_EQUAL, hi)
        elif finite_lo:
            model.addLConstr(expr, GRB.GREATER_EQUAL, lo)
    model.update()
    return model
