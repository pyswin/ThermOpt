from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from thermopt.layout.objects import FloorplanCase, Layout
from thermopt.thermal.surrogate_input import (
    KELVIN_TO_CELSIUS,
    SURROGATE_NATIVE_GRID_SIZE,
    coordinate_grid,
    grid_size_from_config,
    layout_signature,
    rasterize_power_channel,
    resample_grid,
)


def _default_demo_root() -> Path:
    return Path(__file__).with_name("thermfm_t")


@dataclass
class ThermFMThermalBackend:
    case: FloorplanCase
    config: dict
    work_dir: Path | None = None
    name: str = "thermfm"
    predictor: Callable[[np.ndarray], np.ndarray] | None = None
    model_dir: Path | None = None
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _output_grid_size: tuple[int, int] = field(init=False, repr=False)
    _demo_root: Path = field(init=False, repr=False)
    _predictor_impl: Callable[[np.ndarray], np.ndarray] | None = field(default=None, init=False, repr=False)
    _input_phys: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._output_grid_size = grid_size_from_config(self.config)

        demo_root_cfg = self.config.get("thermfm_demo_root")
        model_dir_cfg = self.config.get("thermfm_model_dir")
        if demo_root_cfg is not None:
            self._demo_root = Path(demo_root_cfg).expanduser().resolve()
        elif model_dir_cfg is not None:
            self._demo_root = Path(model_dir_cfg).expanduser().resolve().parent
        else:
            self._demo_root = _default_demo_root().resolve()

        if self.model_dir is not None:
            self.model_dir = Path(self.model_dir).expanduser().resolve()
        elif model_dir_cfg is not None:
            self.model_dir = Path(model_dir_cfg).expanduser().resolve()
        else:
            self.model_dir = (self._demo_root / "model").resolve()

        self._input_phys = np.zeros((3, *SURROGATE_NATIVE_GRID_SIZE), dtype=np.float32)
        grid_x, grid_y = coordinate_grid(self.case, SURROGATE_NATIVE_GRID_SIZE)
        self._input_phys[1, :, :] = grid_x
        self._input_phys[2, :, :] = grid_y

    @property
    def runtime_mode(self) -> str:
        return self.name

    def _resolve_device(self, torch_module) -> str:
        requested = str(self.config.get("thermfm_device", self.config.get("device", "auto"))).lower()
        if requested == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("Therm-FM backend requested cuda, but torch.cuda.is_available() is false")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("thermfm_device must be one of: auto, cpu, cuda")
        return requested

    def _load_real_predictor(self) -> Callable[[np.ndarray], np.ndarray]:
        model_py = self._demo_root / "model.py"
        if not model_py.is_file():
            raise FileNotFoundError(f"missing Therm-FM demo model file: {model_py}")
        if self.model_dir is None or not self.model_dir.is_dir():
            raise FileNotFoundError(f"missing Therm-FM demo model directory: {self.model_dir}")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "Therm-FM backend requires optional dependencies: torch and transformers."
            ) from exc

        spec = importlib.util.spec_from_file_location("thermopt_thermfm_demo_model", model_py)
        if spec is None or spec.loader is None:
            raise FileNotFoundError(f"unable to import Therm-FM demo model from {model_py}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception as exc:  # pragma: no cover - import-time failures depend on optional deps
            raise RuntimeError(f"failed to load Therm-FM demo model from {model_py}: {exc}") from exc

        ScOT = getattr(module, "ScOT", None)
        if ScOT is None:
            raise AttributeError(f"Therm-FM demo model does not expose ScOT: {model_py}")

        stats_path = self.model_dir / "normalization_constants.json"
        if not stats_path.is_file():
            raise FileNotFoundError(f"missing Therm-FM normalization constants: {stats_path}")
        with stats_path.open("r", encoding="utf-8") as f:
            stats = json.load(f)

        mean_in = np.array(stats["input"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std_in = np.array(stats["input"]["std"], dtype=np.float32).reshape(-1, 1, 1)
        mean_out = np.array(stats["output"]["mean"], dtype=np.float32).reshape(-1, 1, 1)
        std_out = np.array(stats["output"]["std"], dtype=np.float32).reshape(-1, 1, 1)
        device = self._resolve_device(torch)
        model = ScOT.from_pretrained(str(self.model_dir)).to(device).eval()

        def predict(input_phys: np.ndarray) -> np.ndarray:
            input_phys = np.asarray(input_phys, dtype=np.float32)
            if input_phys.shape != (3, *SURROGATE_NATIVE_GRID_SIZE):
                raise ValueError(
                    f"Therm-FM input must have shape (3, 64, 64), got {input_phys.shape}"
                )
            input_norm = (input_phys - mean_in) / std_in
            x = torch.from_numpy(input_norm).unsqueeze(0).to(device)
            with torch.inference_mode():
                prediction_norm = model(pixel_values=x).output
            prediction_k = prediction_norm[0].detach().cpu().numpy() * std_out + mean_out
            prediction_k = np.asarray(prediction_k, dtype=np.float32)
            if prediction_k.ndim == 3 and prediction_k.shape[0] == 1:
                prediction_k = prediction_k[0]
            if prediction_k.ndim != 2:
                raise ValueError(
                    f"Therm-FM model returned unsupported prediction shape: {prediction_k.shape}"
                )
            return prediction_k - KELVIN_TO_CELSIUS

        return predict

    def _predict(self, input_phys: np.ndarray) -> np.ndarray:
        if self.predictor is not None:
            return np.asarray(self.predictor(input_phys), dtype=np.float32)
        if self._predictor_impl is None:
            self._predictor_impl = self._load_real_predictor()
        return np.asarray(self._predictor_impl(input_phys), dtype=np.float32)

    def _update_power_channel(self, layout: Layout) -> None:
        power = self._input_phys[0]
        power.fill(0.0)
        rasterize_power_channel(self.case, layout, self._input_phys[1], self._input_phys[2], out=power)

    def simulate(self, case: FloorplanCase, layout: Layout) -> np.ndarray:
        if case != self.case:
            raise ValueError("ThermFMThermalBackend is bound to a different case")

        layout_key = layout_signature(case, layout)
        if layout_key in self._cache:
            return np.array(self._cache[layout_key], copy=True)

        self._update_power_channel(layout)
        temperature = np.asarray(self._predict(self._input_phys), dtype=np.float32)
        if temperature.ndim == 3 and temperature.shape[0] == 1:
            temperature = temperature[0]
        if temperature.ndim != 2:
            raise ValueError(f"Therm-FM predictor must return a 2D temperature map, got {temperature.shape}")
        if temperature.shape != self._output_grid_size:
            temperature = resample_grid(temperature, self._output_grid_size).astype(np.float32)

        self._cache[layout_key] = np.array(temperature, copy=True)
        return np.array(temperature, copy=True)
