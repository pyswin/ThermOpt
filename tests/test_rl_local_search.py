from thermopt.data.case_generator import generate_random_case, random_initial_layout
from thermopt.objective.cost import Objective
from thermopt.optimizer.rl_local_search import optimize


def test_rl_local_search_runs_and_returns_result():
    case_cfg = {
        "num_chiplets": 6,
        "outline_width": 60.0,
        "outline_height": 50.0,
        "min_size": 8.0,
        "max_size": 15.0,
        "min_power": 1.0,
        "max_power": 5.0,
        "num_nets": 8,
        "net_min_degree": 2,
        "net_max_degree": 3,
    }
    case = generate_random_case(case_cfg, seed=42)
    layout = random_initial_layout(case, seed=42)
    thermal_cfg = {"backend": "heuristic", "grid_size": (32, 32), "ambient": 25.0, "scale": 0.05}
    obj_cfg = {"alpha": 1.0, "beta": 0.0, "gamma": 50.0, "delta": 80.0}
    objective = Objective(case, thermal_cfg, obj_cfg, layout)

    result = optimize(
        case,
        layout,
        objective,
        {
            "num_candidates": 3,
            "total_steps": 60,
            "epsilon_start": 0.5,
            "epsilon_end": 0.1,
            "hidden_dim": 16,
            "train_after": 10,
            "batch_size": 8,
            "report_every": 10,
        },
        seed=7,
    )

    assert result.attempted_moves == 60
    assert result.accepted_moves >= 0
    assert len(result.best_curve) > 1
    assert result.best_cost.total <= result.best_curve[0] + 1e-9
    assert result.baseline_wl > 0
    assert result.final_wl > 0
