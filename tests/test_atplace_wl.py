from thermopt.layout.geometry import hpwl, total_overlap_penalty
from thermopt.layout.objects import Chiplet, FloorplanCase, Layout, Net, Placement
from thermopt.objective.cost import Objective
from thermopt.optimizer.atplace_wl import optimize


def test_atplace_wl_finds_legal_two_block_layout() -> None:
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

    result = optimize(case, layout, objective, {"milp_time_limit": 5, "refine_steps": 5, "legal_perturb_iterations": 10}, seed=1)

    assert total_overlap_penalty(case, result.best_layout) == 0
    assert hpwl(case, result.best_layout) == 0
    assert [phase["phase"] for phase in result.phases] == ["initial", "clump_milp", "analytical_refine", "legal_perturb"]
