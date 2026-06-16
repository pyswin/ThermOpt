from thermopt.data.case_generator import generate_random_case, random_initial_layout
from thermopt.objective.cost import Objective
from thermopt.optimizer.genetic_algorithm import optimize


def test_genetic_algorithm_runs_and_tracks_best_curve() -> None:
    case = generate_random_case(
        {
            "num_chiplets": 6,
            "outline_width": 50,
            "outline_height": 40,
            "min_size": 5,
            "max_size": 10,
            "min_power": 1,
            "max_power": 5,
            "num_nets": 5,
        },
        seed=10,
    )
    layout = random_initial_layout(case, seed=11)
    objective = Objective(
        case,
        {"grid_size": [24, 18], "ambient": 25, "scale": 1.0, "sigma_factor": 1.0},
        {"alpha": 1, "beta": 1, "gamma": 20, "delta": 30, "thermal_mode": "topk", "topk_percent": 0.05},
        layout,
    )

    result = optimize(
        case,
        layout,
        objective,
        {"population_size": 8, "generations": 4, "elite_count": 2, "mutation_rate": 0.5, "move_scale": 4.0},
        seed=12,
    )

    assert result.population_size == 8
    assert result.generations == 4
    assert len(result.best_curve) == 5
    assert result.best_cost.total <= max(result.best_curve)
