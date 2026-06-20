from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import mkstemp
from typing import Any, Dict, Optional, Tuple


PARSER_KEYS = {
    "interposer_size",
    "fence_width",
    "fence_height",
    "num_bins_x",
    "num_bins_y",
    "num_grid_x",
    "num_grid_y",
    "reso_interposer",
}


def case_config_path(case_dir: str | Path, config_name: str = "reproduce.json") -> Path:
    return Path(case_dir).expanduser().resolve() / config_name


def load_case_config(case_cfg_path: str | Path) -> Dict[str, Any]:
    case_cfg_path = Path(case_cfg_path)
    with case_cfg_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"case config is not a json object: {case_cfg_path}")
    if data.get("schema_version", None) != 1:
        raise ValueError(f"Unsupported schema_version in {case_cfg_path}: {data.get('schema_version', None)}")
    if "defaults" in data and not isinstance(data["defaults"], dict):
        raise ValueError(f"Invalid 'defaults' in {case_cfg_path}: must be an object")
    for mode in ("wl", "thermal"):
        if mode not in data or not isinstance(data[mode], dict):
            raise ValueError(f"Missing or invalid '{mode}' in {case_cfg_path}")
    return data


def select_mode_config(case_cfg: Dict[str, Any], mode: str, case_cfg_path: str | Path) -> Dict[str, Any]:
    cfg = case_cfg.get(mode)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid mode config '{mode}' in {case_cfg_path}: must be an object")
    banned_keys = {"flip_opt", "dis_bet_chips"}
    found = sorted(k for k in banned_keys if k in cfg)
    if found:
        raise ValueError(f"Unsupported keys in {case_cfg_path} for mode '{mode}': {', '.join(found)}")
    return cfg


def split_config_for_parser(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    parser_cfg: Dict[str, Any] = {}
    runtime_cfg: Dict[str, Any] = {}
    for key, value in cfg.items():
        if key in PARSER_KEYS:
            parser_cfg[key] = value
        else:
            runtime_cfg[key] = value
    return parser_cfg, runtime_cfg


def write_temp_json(data: Dict[str, Any], prefix: str = "thermopt-") -> str:
    fd, path = mkstemp(prefix=prefix, suffix=".json")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


def load_case_overrides(
    case_dir: str | Path,
    mode: str = "thermal",
    config_name: str = "reproduce.json",
    allow_missing: bool = False,
) -> Tuple[Optional[Path], Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    cfg_path = case_config_path(case_dir, config_name)
    if not cfg_path.exists():
        if allow_missing:
            return None, {}, {}, {}, {}
        raise FileNotFoundError(f"case config not found: {cfg_path}")

    case_cfg = load_case_config(cfg_path)
    mode_cfg = select_mode_config(case_cfg, mode, cfg_path)
    parser_cfg, runtime_cfg = split_config_for_parser(mode_cfg)
    defaults = case_cfg.get("defaults", {}) or {}
    return cfg_path, case_cfg, parser_cfg, runtime_cfg, defaults


def get_case_outline(
    case_dir: str | Path,
    config_name: str = "reproduce.json",
    mode: str = "thermal",
) -> Optional[tuple[float, float]]:
    cfg_path = case_config_path(case_dir, config_name)
    if not cfg_path.exists():
        return None
    case_cfg = load_case_config(cfg_path)
    mode_cfg = select_mode_config(case_cfg, mode, cfg_path)
    size = mode_cfg.get("interposer_size")
    if not (isinstance(size, (list, tuple)) and len(size) == 2):
        return None
    return float(size[0]), float(size[1])
