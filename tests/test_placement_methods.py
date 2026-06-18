from thermopt.layout.geometry import hpwl, total_overlap_penalty
from thermopt.layout.objects import Chiplet, FloorplanCase, Layout, Net, Placement
from thermopt.objective.cost import Objective
from thermopt.optimizer import atmplace, atplace, nesterov


def test_atplace_finds_legal_two_block_layout() -> None:
    case = FloorplanCase(
        chiplets=(Chiplet("A", 4, 4, 1), Chiplet("B", 4, 4, 1)),
        nets=(Net("N0", ("A", "B"), ((2, 0), (-2, 0))),),
        outline_width=10,
        outline_height=6,
    )
    layout = Layout((Placement("A", 0, 0), Placement("B", 6, 0)))
    objective = Objective(
        case,
        {"grid_size": [8, 8], "ambient": 25.0, "scale": 0.1, "sigma_factor": 1.0},
        {"alpha": 1.0, "beta": 0.0, "gamma": 10.0, "delta": 10.0, "thermal_mode": "topk"},
        layout,
    )

    result = atplace.optimize(case, layout, objective, {"milp_time_limit": 5, "refine_steps": 5, "legal_perturb_iterations": 10}, seed=1)

    assert total_overlap_penalty(case, result.best_layout) == 0
    assert hpwl(case, result.best_layout) == 0
    assert [phase["phase"] for phase in result.phases] == ["initial", "clump_milp", "analytical_refine", "legal_perturb"]


def test_atmplace_finds_legal_two_block_layout() -> None:
    case = FloorplanCase(
        chiplets=(Chiplet("A", 4, 4, 1), Chiplet("B", 4, 4, 1)),
        nets=(Net("N0", ("A", "B"), ((2, 0), (-2, 0))),),
        outline_width=10,
        outline_height=6,
    )
    layout = Layout((Placement("A", 0, 0), Placement("B", 6, 0)))
    objective = Objective(
        case,
        {"grid_size": [8, 8], "ambient": 25.0, "scale": 0.1, "sigma_factor": 1.0},
        {"alpha": 1.0, "beta": 0.0, "gamma": 10.0, "delta": 10.0, "thermal_mode": "topk"},
        layout,
    )

    result = atmplace.optimize(
        case,
        layout,
        objective,
        {"milp_time_limit": 5, "cgd_steps": 5, "legalize_iterations": 10},
        seed=1,
    )

    assert total_overlap_penalty(case, result.best_layout) == 0
    assert hpwl(case, result.best_layout) == 0
    assert [phase["phase"] for phase in result.phases] == ["initial", "milp_seed", "orientation_cgd", "legalization"]


def test_nesterov_finds_legal_two_block_layout() -> None:
    case = FloorplanCase(
        chiplets=(Chiplet("A", 4, 4, 1), Chiplet("B", 4, 4, 1)),
        nets=(Net("N0", ("A", "B"), ((2, 0), (-2, 0))),),
        outline_width=10,
        outline_height=6,
    )
    layout = Layout((Placement("A", 0, 0), Placement("B", 6, 0)))
    objective = Objective(
        case,
        {"grid_size": [8, 8], "ambient": 25.0, "scale": 0.1, "sigma_factor": 1.0},
        {"alpha": 1.0, "beta": 0.0, "gamma": 10.0, "delta": 10.0, "thermal_mode": "topk"},
        layout,
    )

    result = nesterov.optimize(
        case,
        layout,
        objective,
        {"steps": 20, "learning_rate": 0.04, "legalize_candidates": 8, "report_every": 5},
        seed=1,
    )

    assert total_overlap_penalty(case, result.best_layout) == 0
    assert hpwl(case, result.best_layout) == 0
    assert [phase["phase"] for phase in result.phases] == ["initial", "nesterov_global", "legalization"]
