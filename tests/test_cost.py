from thermopt.data.case_generator import generate_random_case, random_initial_layout
from thermopt.objective.cost import Objective
from helpers import DummyThermalBackend


def test_objective_returns_finite_cost() -> None:
    config = {
        "num_chiplets": 5,
        "outline_width": 50,
        "outline_height": 40,
        "min_size": 5,
        "max_size": 10,
        "min_power": 1,
        "max_power": 5,
        "num_nets": 4,
    }
    case = generate_random_case(config, seed=1)
    layout = random_initial_layout(case, seed=2)
    objective = Objective(
        case,
        {"grid_size": [30, 20], "ambient": 25, "scale": 1.0, "sigma_factor": 1.0},
        {"alpha": 1, "beta": 1, "gamma": 10, "delta": 10, "thermal_mode": "topk", "topk_percent": 0.05},
        layout,
        thermal_backend=DummyThermalBackend(),
    )
    result = objective(layout)
    assert result.total > 0
    assert result.metrics["tmax"] >= 25
