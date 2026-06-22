from thermopt.data.case_generator import generate_random_case, random_initial_layout
from thermopt.objective.cost import Objective
from thermopt.optimizer.rl_policy import optimize
from helpers import DummyThermalBackend


def test_rl_policy_optimizer_trains_and_returns_best_layout() -> None:
    case = generate_random_case(
        {
            "num_chiplets": 5,
            "outline_width": 50,
            "outline_height": 40,
            "min_size": 5,
            "max_size": 10,
            "min_power": 1,
            "max_power": 5,
            "num_nets": 4,
        },
        seed=30,
    )
    layout = random_initial_layout(case, seed=31)
    objective = Objective(
        case,
        {"grid_size": [24, 18], "ambient": 25, "scale": 1.0, "sigma_factor": 1.0},
        {"alpha": 1, "beta": 1, "gamma": 20, "delta": 30, "thermal_mode": "topk", "topk_percent": 0.05},
        layout,
        thermal_backend=DummyThermalBackend(),
    )

    result = optimize(
        case,
        layout,
        objective,
        {
            "episodes": 3,
            "max_steps": 4,
            "rollout_steps": 4,
            "learning_rate": 0.02,
            "gamma": 0.95,
            "move_scale": 4.0,
        },
        seed=32,
    )

    assert result.training_episodes == 3
    assert result.rollout_steps == 4
    assert len(result.episode_returns) == 3
    assert result.best_cost.total <= max(result.best_curve)
