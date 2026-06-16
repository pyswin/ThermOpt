from thermopt.data.case_generator import generate_random_case, random_initial_layout
from thermopt.objective.cost import Objective
from thermopt.optimizer.rl_environment import ThermalFloorplanEnv


def test_rl_environment_reset_and_step() -> None:
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
        seed=20,
    )
    layout = random_initial_layout(case, seed=21)
    objective = Objective(
        case,
        {"grid_size": [24, 18], "ambient": 25, "scale": 1.0, "sigma_factor": 1.0},
        {"alpha": 1, "beta": 1, "gamma": 20, "delta": 30, "thermal_mode": "topk", "topk_percent": 0.05},
        layout,
    )
    env = ThermalFloorplanEnv(case, layout, objective, max_steps=2, move_scale=4.0, seed=22)

    observation = env.reset(seed=23)
    assert observation["step"] == 0
    assert len(observation["placements"]) == len(case.chiplets)

    first = env.step(0)
    assert first.done is False
    assert "cost" in first.info

    second = env.step(1)
    assert second.done is True
