from __future__ import annotations

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
    return Path(__file__).with_name("ufno_demo")


@dataclass
class UFNOThermalBackend:
    case: FloorplanCase
    config: dict
    work_dir: Path | None = None
    name: str = "ufno"
    predictor: Callable[[np.ndarray], np.ndarray] | None = None
    model_path: Path | None = None
    _cache: dict[str, np.ndarray] = field(default_factory=dict, init=False, repr=False)
    _output_grid_size: tuple[int, int] = field(init=False, repr=False)
    _demo_root: Path = field(init=False, repr=False)
    _predictor_impl: Callable[[np.ndarray], np.ndarray] | None = field(default=None, init=False, repr=False)
    _input_phys: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._output_grid_size = grid_size_from_config(self.config)

        demo_root_cfg = self.config.get("ufno_demo_root")
        model_path_cfg = self.config.get("ufno_model_path")
        if demo_root_cfg is not None:
            self._demo_root = Path(demo_root_cfg).expanduser().resolve()
        elif model_path_cfg is not None:
            self._demo_root = Path(model_path_cfg).expanduser().resolve().parent
        else:
            self._demo_root = _default_demo_root().resolve()

        if self.model_path is not None:
            self.model_path = Path(self.model_path).expanduser().resolve()
        elif model_path_cfg is not None:
            self.model_path = Path(model_path_cfg).expanduser().resolve()
        else:
            self.model_path = (self._demo_root / "model.pt").resolve()

        grid_x, grid_y = coordinate_grid(self.case, SURROGATE_NATIVE_GRID_SIZE)
        self._input_phys = np.zeros((*SURROGATE_NATIVE_GRID_SIZE, 1, 3), dtype=np.float32)
        self._input_phys[:, :, 0, 1] = grid_x
        self._input_phys[:, :, 0, 2] = grid_y

    @property
    def runtime_mode(self) -> str:
        return self.name

    def _resolve_device(self, torch_module) -> str:
        requested = str(self.config.get("ufno_device", self.config.get("device", "auto"))).lower()
        if requested == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested == "cuda" and not torch_module.cuda.is_available():
            raise RuntimeError("U-FNO backend requested cuda, but torch.cuda.is_available() is false")
        if requested not in {"cpu", "cuda"}:
            raise ValueError("ufno_device must be one of: auto, cpu, cuda")
        return requested

    def _load_real_predictor(self) -> Callable[[np.ndarray], np.ndarray]:
        ufno_py = self._demo_root / "ufno.py"
        normalize_py = self._demo_root / "models" / "normalize.py"
        if not ufno_py.is_file():
            raise FileNotFoundError(f"missing U-FNO demo model file: {ufno_py}")
        if not self.model_path or not self.model_path.is_file():
            raise FileNotFoundError(f"missing U-FNO model bundle: {self.model_path}")

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("U-FNO backend requires the optional torch dependency.") from exc

        if str(self._demo_root) not in sys.path:
            sys.path.insert(0, str(self._demo_root))

        try:
            import ufno as _ufno_module  # noqa: F401
            from models.normalize import normalize as _normalize_class  # noqa: F401
        except Exception as exc:  # pragma: no cover - import-time failures depend on optional deps
            raise RuntimeError(f"failed to load U-FNO demo modules from {self._demo_root}: {exc}") from exc

        model_bundle = torch.load(self.model_path, map_location=self._resolve_device(torch))
        if not isinstance(model_bundle, (list, tuple)) or len(model_bundle) != 3:
            raise ValueError(f"unexpected U-FNO model bundle structure in {self.model_path}")
        x_normalizer, model, y_normalizer = model_bundle
        model.eval()

        def predict(input_phys: np.ndarray) -> np.ndarray:
            input_phys = np.asarray(input_phys, dtype=np.float32)
            if input_phys.shape != (*SURROGATE_NATIVE_GRID_SIZE, 1, 3):
                raise ValueError(
                    f"U-FNO input must have shape (64, 64, 1, 3), got {input_phys.shape}"
                )
            x = torch.from_numpy(input_phys).unsqueeze(0).to(next(model.parameters()).device)
            with torch.inference_mode():
                xn = x_normalizer.forward(x)
                out = model(xn)
                pred = y_normalizer.inverse(out)
            pred = pred[0, ..., 0].detach().cpu().numpy().astype(np.float32)
            return pred - KELVIN_TO_CELSIUS

        return predict

    def _predict(self, input_phys: np.ndarray) -> np.ndarray:
        if self.predictor is not None:
            return np.asarray(self.predictor(input_phys), dtype=np.float32)
        if self._predictor_impl is None:
            self._predictor_impl = self._load_real_predictor()
        return np.asarray(self._predictor_impl(input_phys), dtype=np.float32)

    def _update_power_channel(self, layout: Layout) -> None:
        power = self._input_phys[:, :, 0, 0]
        power.fill(0.0)
        rasterize_power_channel(self.case, layout, self._input_phys[:, :, 0, 1], self._input_phys[:, :, 0, 2], out=power)

    def simulate(self, case: FloorplanCase, layout: Layout) -> np.ndarray:
        if case != self.case:
            raise ValueError("UFNOThermalBackend is bound to a different case")

        layout_key = layout_signature(case, layout)
        if layout_key in self._cache:
            return np.array(self._cache[layout_key], copy=True)

        self._update_power_channel(layout)
        temperature = np.asarray(self._predict(self._input_phys), dtype=np.float32)
        if temperature.ndim == 3 and temperature.shape[0] == 1:
            temperature = temperature[0]
        if temperature.ndim != 2:
            raise ValueError(f"U-FNO predictor must return a 2D temperature map, got {temperature.shape}")
        if temperature.shape != self._output_grid_size:
            temperature = resample_grid(temperature, self._output_grid_size).astype(np.float32)
        temperature = np.ascontiguousarray(temperature.T, dtype=np.float32)  # model ij -> display xy

        self._cache[layout_key] = np.array(temperature, copy=True)
        return np.array(temperature, copy=True)
