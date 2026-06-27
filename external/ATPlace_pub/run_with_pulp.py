#!/usr/bin/env python3
"""Wrapper around ATPlace reproduce.py that forces ILPsolver=pulp."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/data/xinli/ThermOpt/external/ATPlace_pub")
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

import Params
from Chiplet import Chiplet
from Interposer import Passive_Interposer
from System import System_25D
from ATPLACE.PlaceFlow import placeflow_core
from utils.blocks_parser import parse_blocks
from utils.nets_parser import parse_nets
from utils.pl_parser import parse_pls


CASE_INTERPOSER_SIZE = {
    "Case1": [42000.0, 42000.0],
    "Case2": [32000.0, 32000.0],
    "Case3": [39000.0, 39000.0],
    "Case4": [37000.0, 37000.0],
    "Case5": [57000.0, 59000.0],
    "Case6": [49000.0, 53000.0],
    "Case7": [30000.0, 25000.0],
    "Case8": [26000.0, 23000.0],
    "Case9": [59000.0, 61000.0],
    "Case10": [47000.0, 47000.0],
}


def build_system(case_dir, case_name, params):
    options = {
        "filename_blocks": str(case_dir / f"{case_name}.blocks"),
        "filename_nets": str(case_dir / f"{case_name}.nets"),
        "filename_pl": str(case_dir / f"{case_name}.pl"),
    }
    modules, block_headers = parse_blocks(options)
    locations = parse_pls(options)
    nets, net_headers = parse_nets(options)
    num_chiplets = int(block_headers["Headers"]["NumHardRectilinearBlocks"])
    num_terminals = int(block_headers["Headers"]["NumTerminals"])
    system = System_25D(num_chiplets, num_terminals)
    interposer = Passive_Interposer()
    for module_name, module in modules["Modules"].items():
        if "rectangles" in module:
            chiplet = Chiplet(module_name)
            chiplet.set_chiplet_size(*module["rectangles"][0][-2:])
            chiplet.set_chiplet_loc(*module["rectangles"][0][:2])
            system.append_chiplet(module_name, chiplet)
        elif "terminal" in module:
            center = locations["Modules"][module_name]["center"]
            interposer.append_terminal(module_name, center)
            system.append_terminal(module_name, center)
    system.num_nets = int(net_headers["Headers"]["NumNets"])
    system.num_pins = int(net_headers["Headers"]["NumPins"]) - system.num_nodes + num_chiplets
    pin_id = 0
    for net_idx, net in enumerate(nets["Nets"]):
        system.net_id.append(net_idx)
        system.net_weights.append(1.0)
        system.net2pin_map.append([])
        for pin in net:
            node_id = system.node_name2id_map[pin[0]]
            if len(pin) < 3 or pin[2] is None:
                continue
            pin_offset_x = float(pin[1])
            pin_offset_y = float(pin[2])
            existing_pin = None
            for old_pin_id in system.node2pin_map[node_id]:
                if (system.pin_offset_x[old_pin_id] == pin_offset_x
                        and system.pin_offset_y[old_pin_id] == pin_offset_y):
                    existing_pin = old_pin_id
                    break
            if existing_pin is not None:
                system.net2pin_map[net_idx].append(existing_pin)
                system.pin2net_map[existing_pin].append(net_idx)
            else:
                system.net2pin_map[net_idx].append(pin_id)
                system.pin2net_map.append([net_idx])
                system.node2pin_map[node_id].append(pin_id)
                system.pin2node_map.append(node_id)
                system.pin_offset_x.append(pin_offset_x)
                system.pin_offset_y.append(pin_offset_y)
                pin_id += 1
    interposer.set_interposer_size(params.interposer_size)
    fence = [
        params.fence_width,
        interposer.width - params.fence_width,
        params.fence_height,
        interposer.height - params.fence_height,
    ]
    system.set_interposer_size(fence, interposer)
    system.set_bins(params)
    system.num_grid_x = params.num_grid_x
    system.num_grid_y = params.num_grid_y
    system.initialize()
    system.set_granularity(params.reso_interposer)
    import numpy as np
    system.area_cplt = (np.array(system.node_size_x) * np.array(system.node_size_y)).sum()
    system.powermap = np.zeros(num_chiplets)
    power_file = case_dir / f"{case_name}.power"
    if power_file.exists():
        with power_file.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) != 2:
                    continue
                name, power = parts
                if name in system.node_name2id_map:
                    system.powermap[system.node_names.index(name)] = float(power)
    return system


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--param-file", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    param_file = Path(args.param_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load params: prefer reproduce.json's "wl" section (matches paper settings)
    params = Params.Params(SRC / "params.json")
    reproduce_file = Path(args.case_dir) / "reproduce.json"
    if reproduce_file.exists():
        with open(reproduce_file) as f:
            repro = json.load(f)
        data = repro.get("wl", repro)  # use "wl" sub-dict if present
    else:
        with open(param_file) as f:
            data = json.load(f)
    params.fromJson(data)
    # Normalize stages (ensure "iteration" key is set)
    default_stage = (Params.Params(SRC / "params.json").floorplan_stages or [{}])[0]
    stages = data.get("floorplan_stages") or params.floorplan_stages or [default_stage]
    merged = []
    for stage in stages:
        item = dict(default_stage)
        item.update(stage)
        merged.append(item)
    params.floorplan_stages = merged
    params.interposer_size = data.get("interposer_size") or CASE_INTERPOSER_SIZE[args.case]
    params.fence_width = getattr(params, "fence_width", 0.0)
    params.fence_height = getattr(params, "fence_height", 0.0)
    params.result_dir = str(out_dir)
    params.thermal_dir = str(ROOT / "thermal") + os.sep
    # Use Gurobi (academic license)
    params.ILPsolver = "grb"
    params.thermal_solver = getattr(params, "thermal_solver", "hotspot")
    params.random_seed = 42

    print(f"ILPsolver={params.ILPsolver}, temp_aware_opt={getattr(params,'temp_aware_opt',False)}")
    system = build_system(case_dir, args.case, params)
    print(f"System built: {system.num_nodes} chiplets, {system.num_nets} nets")

    start = time.time()
    result = placeflow_core(params, system, None)
    elapsed = time.time() - start
    print(f"Placement done in {elapsed:.1f}s")
    print(f"Result: {result}")

    # Extract HPWL from result (dict, list/tuple, or attr)
    if isinstance(result, dict):
        hpwl = result['hpwl']
    elif isinstance(result, (list, tuple)):
        hpwl = result[0]
    elif hasattr(result, 'hpwl'):
        hpwl = result.hpwl
    else:
        hpwl = result
    summary = {
        "case": args.case,
        "hpwl": float(hpwl),
        "twl_m": float(hpwl) / 1e6,
        "runtime_s": elapsed,
    }
    # Save chip positions and sizes for FNO training
    if isinstance(result, dict) and "best_fp_pos" in result:
        import numpy as np
        pos_data = result["best_fp_pos"]
        size_x = result.get("best_fp_size", [None, None])[0]
        size_y = result.get("best_fp_size", [None, None])[1]
        # pos_data is list of lists of tensors [[x_tensor, rot_tensor]]
        if pos_data and len(pos_data[0]) >= 1:
            x_pos = pos_data[0][0]
            if hasattr(x_pos, 'detach'):
                x_pos = x_pos.detach().cpu().numpy()
            else:
                x_pos = np.array(x_pos)
            n = len(x_pos) // 2
            placement = {
                "x": x_pos[:n].tolist(),
                "y": x_pos[n:].tolist(),
                "w": size_x.tolist() if hasattr(size_x, 'tolist') else list(size_x),
                "h": size_y.tolist() if hasattr(size_y, 'tolist') else list(size_y),
            }
            summary["placement"] = placement
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "placement"}, indent=2))


if __name__ == "__main__":
    main()
