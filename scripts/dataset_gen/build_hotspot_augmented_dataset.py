#!/usr/bin/env python3
"""
build_hotspot_augmented_dataset.py — Case1-10 end-to-end HotSpot-augmented
thermal dataset.

Per case:
  Phase A/B: solve the Phase-1 MILP init with Gurobi, snapshot every ~1s of
  solve time, dedup consecutive-identical incumbents, pick 5 seed snapshots
  evenly from the *tail* `--seed_tail_fraction` (default last 20%) of the
  deduped trajectory — not the full 0-100% range — since downstream
  optimization starts from the MILP final solution, so perturbations should
  mostly explore layouts near it. Then generate perturbations around those
  seeds: each hop composes 1-2 operations from different families
  (swap / translate[=move_small or move_more] / rotate — see
  thermopt.optimizer.perturb.generate_one's max_families) and hops chain up
  to `--chain_depth` times from a seed before resetting, rather than always
  single-hopping from a fixed seed — a plain single-family, single-hop
  perturbation from only a handful of MILP seeds turned out to duplicate the
  same layout hundreds of times for small cases (little room to reach a new
  sequence-pair ordering in one step). Every candidate is deduped by exact
  layout fingerprint as it's generated, so the resulting pool has zero
  duplicates by construction, not by post-hoc filtering. Result is persisted
  to `CaseN/layout_pool.json` (up to `--target` unique legal candidates) so
  Phase C never has to redo the Gurobi solve or perturbation generation.

  Phase C: run every pool candidate through the real HotSpot binary and
  write json + augmented CSVs via ThermalDatasetGenerator.save_sample_format_json /
  save_sample_format_pointwise_augmented (same schema generate_thermal_dataset.py
  produces). HotSpot failures are rare (crash/timeout on a legal-but-pathological
  geometry) but are topped up with freshly generated (deduped-against-the-pool)
  replacements rather than shipping short.

Sample-selection policy:
  - Legality = overlap<=0 and outline<=0 AND a minimum pairwise clearance of
    `--min_gap` mm (default 0.05) between every pair of chiplets. This
    matters because the model trained on this dataset will be asked to
    predict layouts that respect a real minimum spacing at inference time;
    training on the sequence-pair packer's natural zero-gap output would be
    a train/inference distribution mismatch. min_gap is enforced *during*
    sequence-pair compaction (thermopt.optimizer.sequence_pair.decode_sequence_pair),
    not just filtered after the fact, so the legal-candidate yield stays high.
  - Intermediate MILP snapshots are deduped and filtered to the legality
    above, capped at `--intermediate_cap_fraction` (default 20% of target)
    so perturbations stay the dominant source — in practice min_gap>0 means
    MILP (which has no min_gap mechanism) contributes ~0, and the dataset
    ends up ~100% perturbation-sourced.
  - `--chain_depth` is tiered by chiplet count via `_chain_depth_for()` (small
    cases get deeper chains — they duplicate layouts far more readily); pass
    --chain_depth explicitly to override uniformly.
  - Use --layout_only to stop after Phase A/B (no HotSpot) and inspect
    layout_pool.json / layout_summary.json for diversity before committing
    to the full run.

Output layout:
    atplace/hotspot_augmented_dataset/<STAMP>/CaseN/
        layout_pool.json              <- Phase A/B result (candidates + stats)
        layout_summary.json           <- written only in --layout_only mode
        json/sample_XXXXXX.json
        augmented/sample_XXXXXX.csv   <- the deliverable
        case_summary.json
    atplace/hotspot_augmented_dataset/<STAMP>/run_summary.json

Resume: rerunning with the same --out_dir skips any case whose
case_summary.json already reports successful >= target. A case with an
existing layout_pool.json reuses it (skips re-solving MILP); HotSpot
simulation resumes from whichever pool indices don't already have an
augmented CSV, so no sample is ever re-simulated or overwritten.

Usage (needs gurobipy + a valid license; also used for the HotSpot phase
since that env already carries the full scientific stack):
    # 1. cheap sanity pass first — layouts only, no HotSpot:
    /data/xinli/anaconda3/envs/atplace_py39/bin/python \\
        scripts/build_hotspot_augmented_dataset.py --layout_only --out_dir atplace/hotspot_augmented_dataset/<STAMP>

    # 2. once the pool quality looks good, run the full thing in tmux:
    tmux new -s hotspot_dataset
    /data/xinli/anaconda3/envs/atplace_py39/bin/python \\
        scripts/build_hotspot_augmented_dataset.py --workers 64 --out_dir atplace/hotspot_augmented_dataset/<STAMP> \\
        2>&1 | tee atplace/hotspot_augmented_dataset/run.log
    # detach: Ctrl-b d   |   reattach: tmux attach -t hotspot_dataset
    # tail progress from outside the session:
    tail -f atplace/hotspot_augmented_dataset/run.log
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from thermopt.data.atplace import load_atplace_case
from thermopt.data.thermal_dataset import ThermalDatasetGenerator, ThermalSample
from thermopt.layout.geometry import min_pairwise_gap, total_outline_penalty, total_overlap_penalty
from thermopt.layout.objects import FloorplanCase, Layout, Placement
from thermopt.optimizer import perturb
from thermopt.optimizer.milp_wl_gurobi import Snapshot, initialize_with_snapshots

CASES_DIR = ROOT / "external/ATPlace_pub/cases"
DEFAULT_CASES = [f"Case{i}" for i in range(1, 11)]
TARGET_DEFAULT = 1000
N_SEEDS = 5
MILP_TIME_LIMIT_DEFAULT = 100.0
SNAPSHOT_INTERVAL = 1.0
SEED_DEFAULT = 42
PERTURB_BATCH = 300
MAX_PERTURB_ATTEMPTS_DEFAULT = 60_000
LEGAL_TOL = 1e-9
INTERMEDIATE_CAP_FRACTION_DEFAULT = 0.2
MIN_GAP_DEFAULT = 0.05
SEED_TAIL_FRACTION_DEFAULT = 0.2


def _chain_depth_for(n_chiplets: int) -> int:
    """Small cases have far fewer distinct legal sequence-pair orderings to
    reach, so they need much deeper perturbation chains to avoid duplicating
    layouts (see the duplicate-layout investigation). Thresholds are fit to
    match the per-case values that investigation found necessary (Case1-5
    <=12 chiplets -> 15, Case6-7 20-28 chiplets -> 8, Case8-10 >=36 -> 5)."""
    if n_chiplets <= 15:
        return 15
    if n_chiplets <= 30:
        return 8
    return 5

THERMAL_CONFIG = {
    "backend": "hotspot",
    "hotspot_binary": str(ROOT / "external/ATPlace_pub/thermal/hotspot"),
    "hotspot_required": True,
    "hotspot_allow_fallback": False,
    "grid_size": [64, 64],
    "ambient": 25.0,
    "scale": 0.05,
    "sigma_factor": 1.0,
    "hotspot_timeout_sec": 300,
}


# --------------------------------------------------------------- Phase A/B

def _layout_to_record(case: FloorplanCase, layout: Layout, source: str, extra: dict) -> dict:
    chiplets = []
    for p in layout.placements:
        chiplet = case.chiplet_by_id[p.chiplet_id]
        w, h = p.rotated_size(chiplet)
        chiplets.append(
            {
                "name": p.chiplet_id,
                "x_mm": round(p.x - 0.5 * w, 6),
                "y_mm": round(p.y - 0.5 * h, 6),
                "cx_mm": round(p.x, 6),
                "cy_mm": round(p.y, 6),
                "width_mm": round(w, 6),
                "height_mm": round(h, 6),
                "rotation": p.rotation,
                "power": chiplet.power,
            }
        )
    return {"source": source, "chiplets": chiplets, **extra}


def _layout_fingerprint(record: dict) -> str:
    key = sorted(
        (c["name"], round(c["cx_mm"], 3), round(c["cy_mm"], 3), int(c["rotation"]) % 360)
        for c in record["chiplets"]
    )
    return hashlib.md5(json.dumps(key).encode()).hexdigest()


def _record_to_layout(record: dict) -> Layout:
    placements = tuple(
        Placement(chiplet_id=c["name"], x=c["cx_mm"], y=c["cy_mm"], rotation=int(c["rotation"]) % 360)
        for c in record["chiplets"]
    )
    return Layout(placements)


def _layouts_equal(a: Layout, b: Layout, tol: float = 1e-9) -> bool:
    ba, bb = a.by_id, b.by_id
    return all(
        abs(ba[cid].x - bb[cid].x) < tol and abs(ba[cid].y - bb[cid].y) < tol and ba[cid].rotation == bb[cid].rotation
        for cid in ba
    )


def _dedup_snapshots(snapshots: list[Snapshot]) -> list[Snapshot]:
    out: list[Snapshot] = []
    prev: Layout | None = None
    for snap in snapshots:
        if prev is None or not _layouts_equal(prev, snap.layout):
            out.append(snap)
        prev = snap.layout
    return out


def _pick_seed_indices(n_unique: int, n_seeds: int, tail_fraction: float) -> list[int]:
    """Seeds are drawn evenly from the tail `tail_fraction` of the deduped MILP
    trajectory (biased toward the converged/near-converged end), not the full
    0-100% range — downstream optimization starts from the MILP final solution,
    so perturbations should mostly explore layouts near it rather than early,
    rough mid-solve states."""
    if n_unique <= 1:
        return [0] if n_unique else []
    if n_unique <= n_seeds:
        return list(range(n_unique))
    window_size = min(n_unique, max(n_seeds, int(round(tail_fraction * n_unique))))
    window_start = n_unique - window_size
    fracs = np.linspace(0.0, 1.0, n_seeds)
    return sorted(set(window_start + int(round(f * (window_size - 1))) for f in fracs))


def _cap_records_evenly(records: list[dict], cap: int) -> list[dict]:
    if cap <= 0 or len(records) <= cap:
        return records
    idx = sorted(set(int(round(f * (len(records) - 1))) for f in np.linspace(0.0, 1.0, cap)))
    return [records[i] for i in idx]


def _solve_and_extract(
    case: FloorplanCase, milp_time_limit: float, seed: int, n_seeds: int, min_gap: float, seed_tail_fraction: float
) -> tuple[list[dict], list[Layout], list[float], int, int]:
    config = {
        "milp_time_limit": milp_time_limit,
        "mip_rel_gap": 0.001,
        "snapshot_interval": SNAPSHOT_INTERVAL,
        "seed": seed,
        "verbose": False,
    }
    _result, snapshots = initialize_with_snapshots(case, config)
    unique = _dedup_snapshots(snapshots)

    intermediate_records = []
    for snap in unique:
        overlap = total_overlap_penalty(case, snap.layout)
        outline = total_outline_penalty(case, snap.layout)
        gap = min_pairwise_gap(case, snap.layout)
        if overlap <= LEGAL_TOL and outline <= LEGAL_TOL and gap >= min_gap - LEGAL_TOL:
            intermediate_records.append(_layout_to_record(case, snap.layout, "intermediate", {"t": snap.t}))

    seed_idx = _pick_seed_indices(len(unique), n_seeds, seed_tail_fraction)
    seed_layouts = [unique[i].layout for i in seed_idx]
    seed_ts = [unique[i].t for i in seed_idx]
    return intermediate_records, seed_layouts, seed_ts, len(snapshots), len(unique)


def _generate_perturb_batch(
    case: FloorplanCase,
    seed_layouts: list[Layout],
    rng: np.random.Generator,
    n: int,
    min_gap: float,
    chain_depth: int,
    seen: set[str],
) -> list[dict]:
    """n perturbation *attempts* (not n successes), organized into chains of up
    to `chain_depth` hops each: each hop continues from the previous hop's
    legal result (rather than always restarting at the seed), then resets
    back to the next round-robin seed once the chain hits chain_depth. This
    stays within chain_depth hops of a seed (so samples remain in its
    neighborhood) while still reaching far more distinct sequence-pair
    orderings than repeatedly single-hopping from a handful of fixed seeds —
    see the small-case duplicate-layout investigation for why that matters.

    `seen` is a set of layout fingerprints (mutated in place) shared across
    every call for a case: a hop whose legal result duplicates an
    already-recorded layout still advances the chain (it's a legitimate
    step), but is not appended to the returned records — this is the
    generation-time dedup that keeps the final dataset duplicate-free
    instead of padding it with repeats of the same handful of states."""
    chiplet_n = len(case.chiplets)
    small_k = (1, max(1, chiplet_n // 6))
    more_k = (max(2, chiplet_n // 6 + 1), max(2, chiplet_n // 2))
    records = []
    seed_i = 0
    current = seed_layouts[0]
    hops_since_reset = chain_depth  # force a reset on the first iteration
    for _ in range(n):
        if hops_since_reset >= chain_depth:
            current = seed_layouts[seed_i % len(seed_layouts)]
            seed_i += 1
            hops_since_reset = 0
        result = perturb.generate_one(case, current, rng, small_k, more_k, min_gap=min_gap, max_families=2)
        hops_since_reset += 1
        if result.legal:
            current = result.layout
            record = _layout_to_record(
                case,
                result.layout,
                "perturb",
                {"move_type": result.move_type, "touched": list(result.touched)},
            )
            fp = _layout_fingerprint(record)
            if fp not in seen:
                seen.add(fp)
                records.append(record)
    return records


# ------------------------------------------------------------------ Phase C

_WORKER_GEN: ThermalDatasetGenerator | None = None


def _worker_init(case_dir: str, work_dir_root: str) -> None:
    global _WORKER_GEN
    work_dir = Path(work_dir_root) / f"worker_{os.getpid()}"
    _WORKER_GEN = ThermalDatasetGenerator(
        Path(case_dir),
        dict(THERMAL_CONFIG),
        initial_layout="pl",
        randomize_position=False,
        randomize_power=False,
        randomize_rotation=False,
        work_dir=work_dir,
        seed=0,
    )


def _worker_simulate(item: tuple[int, dict]) -> tuple[int, dict, list | None, str | None]:
    sample_id, record = item
    gen = _WORKER_GEN
    assert gen is not None
    try:
        layout = _record_to_layout(record)
        temp_map = gen.backend.simulate(gen.case, layout)
        return sample_id, record, temp_map.tolist(), None
    except Exception as exc:
        return sample_id, record, None, f"{type(exc).__name__}: {exc}"


def _save_augmented_sample(
    gen: ThermalDatasetGenerator, case_out: Path, sample_id: int, record: dict, temp_map: np.ndarray
) -> None:
    configs = [
        {
            "name": c["name"],
            "x": c["x_mm"],
            "y": c["y_mm"],
            "width": c["width_mm"],
            "height": c["height_mm"],
            "rotation": c["rotation"],
            "power": c["power"],
        }
        for c in record["chiplets"]
    ]
    sample = ThermalSample(
        sample_id=sample_id,
        timestamp=datetime.now().isoformat(),
        variation_type=record.get("move_type", record["source"]),
        layout_mode="random",  # closest ThermalDatasetGenerator layout_mode alias for MILP+perturbation-derived layouts
        chiplet_configs=configs,
        temperature_map=temp_map,
        temperature_stats={
            "min": float(temp_map.min()),
            "max": float(temp_map.max()),
            "mean": float(temp_map.mean()),
            "std": float(temp_map.std()),
        },
    )
    json_path = case_out / "json" / f"sample_{sample_id:06d}.json"
    augmented_path = case_out / "augmented" / f"sample_{sample_id:06d}.csv"
    gen.save_sample_format_json(sample, json_path)

    payload = json.loads(json_path.read_text())
    payload["dataset_source"] = record["source"]
    payload["dataset_meta"] = {k: v for k, v in record.items() if k not in ("chiplets", "source")}
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # writes grid_x/grid_y/chiplet_id/chiplet_power/temperature + occupancy_mask/
    # edge_mask/coord_x_norm/coord_y_norm directly -- no separate pointwise+augment pass needed.
    gen.save_sample_format_pointwise_augmented(sample, augmented_path)


# --------------------------------------------------------------------- driver

def _existing_sample_ids(case_out: Path) -> list[int]:
    aug_dir = case_out / "augmented"
    if not aug_dir.is_dir():
        return []
    ids = []
    for path in aug_dir.glob("sample_*.csv"):
        try:
            ids.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return ids


def _build_layout_pool(
    case_name: str,
    case_out: Path,
    target: int,
    milp_time_limit: float,
    seed: int,
    intermediate_cap_fraction: float,
    max_perturb_attempts: int,
    min_gap: float,
    seed_tail_fraction: float,
    chain_depth: int,
) -> tuple[list[dict], dict]:
    """Phase A/B: build a deduped pool of up to `target` unique legal candidate
    layouts (intermediate MILP frames first, capped, then perturbations),
    persisted to layout_pool.json so Phase C (or a --layout_only inspection)
    never has to redo the Gurobi solve or perturbation generation."""
    pool_path = case_out / "layout_pool.json"
    if pool_path.exists():
        cached = json.loads(pool_path.read_text())
        print(f"[{case_name}] reusing existing layout_pool.json ({len(cached['pool'])} candidates)", flush=True)
        return cached["pool"], cached["stats"]

    case_dir = CASES_DIR / case_name
    ci = load_atplace_case(case_dir, {}, seed=seed)
    case = ci.case
    print(
        f"[{case_name}] n_chiplets={len(case.chiplets)}  solving MILP (t={milp_time_limit:.0f}s, "
        f"min_gap={min_gap}mm, chain_depth={chain_depth})...",
        flush=True,
    )

    t0 = time.time()
    intermediate_records, seed_layouts, seed_ts, n_raw, n_unique = _solve_and_extract(
        case, milp_time_limit, seed, N_SEEDS, min_gap, seed_tail_fraction
    )
    milp_t = time.time() - t0
    intermediate_cap = max(1, int(target * intermediate_cap_fraction))
    capped_intermediate = _cap_records_evenly(intermediate_records, intermediate_cap)
    print(
        f"[{case_name}] MILP done in {milp_t:.0f}s: {n_raw} raw snapshots -> {n_unique} unique -> "
        f"{len(intermediate_records)} legal (capped to {len(capped_intermediate)}), "
        f"seeds@t={[round(t, 1) for t in seed_ts]} (tail {seed_tail_fraction:.0%} of trajectory)",
        flush=True,
    )

    seen: set[str] = set()
    candidate_pool: list[dict] = []
    for rec in capped_intermediate:
        fp = _layout_fingerprint(rec)
        if fp not in seen:
            seen.add(fp)
            candidate_pool.append(rec)

    rng = np.random.default_rng(seed + 1)
    attempts = 0
    while len(candidate_pool) < target:
        if attempts >= max_perturb_attempts:
            print(
                f"[{case_name}] hit perturbation attempt cap ({max_perturb_attempts}) with only "
                f"{len(candidate_pool)}/{target} unique legal candidates; stopping short.",
                flush=True,
            )
            break
        batch_n = min(PERTURB_BATCH, max_perturb_attempts - attempts)
        new_records = _generate_perturb_batch(case, seed_layouts, rng, batch_n, min_gap, chain_depth, seen)
        attempts += batch_n
        candidate_pool.extend(new_records)
        print(f"[{case_name}] pool={len(candidate_pool)}/{target} unique  (attempts so far: {attempts})", flush=True)
    candidate_pool = candidate_pool[:target]

    stats = {
        "milp_solve_time_s": milp_t,
        "n_raw_snapshots": n_raw,
        "n_unique_snapshots": n_unique,
        "n_legal_intermediate_available": len(intermediate_records),
        "n_legal_intermediate_used": len(capped_intermediate),
        "perturb_attempts": attempts,
        "chain_depth": chain_depth,
        "min_gap_mm": min_gap,
        "seed_tail_fraction": seed_tail_fraction,
        "seed_ts": seed_ts,
        "pool_size": len(candidate_pool),
        "target": target,
        "source_counts": dict(Counter(r["source"] for r in candidate_pool)),
        "move_type_counts": dict(
            Counter(r["move_type"] for r in candidate_pool if r["source"] == "perturb")
        ),
    }
    pool_path.write_text(json.dumps({"pool": candidate_pool, "stats": stats}, indent=2))
    print(
        f"[{case_name}] layout pool built: {len(candidate_pool)}/{target} unique legal candidates "
        f"({attempts} perturb attempts), source_counts={stats['source_counts']}",
        flush=True,
    )
    return candidate_pool, stats


def run_case(
    case_name: str,
    out_root: Path,
    target: int,
    workers: int,
    milp_time_limit: float,
    seed: int,
    intermediate_cap_fraction: float,
    max_perturb_attempts: int,
    min_gap: float,
    seed_tail_fraction: float,
    chain_depth: int,
    layout_only: bool,
) -> dict:
    case_out = out_root / case_name
    for sub in ("json", "augmented"):
        (case_out / sub).mkdir(parents=True, exist_ok=True)

    summary_path = case_out / "case_summary.json"
    if not layout_only and summary_path.exists():
        existing = json.loads(summary_path.read_text())
        if existing.get("successful", 0) >= target:
            print(f"[{case_name}] already complete ({existing['successful']}/{target}), skipping", flush=True)
            return existing

    pool, stats = _build_layout_pool(
        case_name, case_out, target, milp_time_limit, seed, intermediate_cap_fraction,
        max_perturb_attempts, min_gap, seed_tail_fraction, chain_depth,
    )

    if layout_only:
        summary = {"case": case_name, "mode": "layout_only", **stats}
        (case_out / "layout_summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    case_dir = CASES_DIR / case_name
    existing_ids = set(_existing_sample_ids(case_out))
    successes = len(existing_ids)
    failures = 0
    source_counts: dict[str, int] = {"intermediate": 0, "perturb": 0}

    hotspot_pool = mp.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(str(case_dir), str(case_out / "_thermal_workdir")),
    )
    gen_for_save = ThermalDatasetGenerator(
        case_dir,
        dict(THERMAL_CONFIG),
        initial_layout="pl",
        randomize_position=False,
        randomize_power=False,
        randomize_rotation=False,
        work_dir=case_out / "_thermal_workdir_main",
        seed=0,
    )

    def _simulate_indexed(indexed_records: list[tuple[int, dict]]) -> None:
        nonlocal successes, failures
        batch_size = max(workers * 4, 64)
        for start in range(0, len(indexed_records), batch_size):
            batch = indexed_records[start : start + batch_size]
            for sample_id, record, temp_map, err in hotspot_pool.imap_unordered(_worker_simulate, batch):
                if err is not None or temp_map is None:
                    failures += 1
                    continue
                _save_augmented_sample(gen_for_save, case_out, sample_id, record, np.array(temp_map, dtype=float))
                source_counts[record["source"]] = source_counts.get(record["source"], 0) + 1
                successes += 1
                if successes % 50 == 0 or successes >= target:
                    print(f"[{case_name}] {successes}/{target} successful (failures so far: {failures})", flush=True)

    try:
        todo = [(i, pool[i]) for i in range(len(pool)) if i not in existing_ids]
        _simulate_indexed(todo)

        # Rare-failure top-up: empirically HotSpot failure rate has been 0 across
        # thousands of runs, but if a few sims genuinely fail, generate that many
        # more fresh (deduped-against-the-pool) candidates rather than shipping short.
        if successes < target and failures > 0:
            seen = {_layout_fingerprint(r) for r in pool}
            case_dir_case = load_atplace_case(case_dir, {}, seed=seed).case
            _intermediate, seed_layouts, _seed_ts, _n_raw, _n_unique = _solve_and_extract(
                case_dir_case, milp_time_limit, seed, N_SEEDS, min_gap, seed_tail_fraction
            )
            rng = np.random.default_rng(seed + len(pool) + 1)
            needed = target - successes
            print(f"[{case_name}] topping up {needed} samples to replace HotSpot failures...", flush=True)
            extra = _generate_perturb_batch(
                case_dir_case, seed_layouts, rng, max(needed * 3, PERTURB_BATCH), min_gap, chain_depth, seen
            )[:needed]
            pool.extend(extra)
            (case_out / "layout_pool.json").write_text(
                json.dumps({"pool": pool, "stats": stats}, indent=2)
            )
            start_idx = len(pool) - len(extra)
            _simulate_indexed([(start_idx + i, rec) for i, rec in enumerate(extra)])
    finally:
        # terminate (not close/join-to-completion): once the target is reached we don't
        # want to wait out the rest of an in-flight batch of HotSpot subprocess calls.
        hotspot_pool.terminate()
        hotspot_pool.join()
        shutil.rmtree(case_out / "_thermal_workdir", ignore_errors=True)
        shutil.rmtree(case_out / "_thermal_workdir_main", ignore_errors=True)

    summary = {
        "case": case_name,
        "target": target,
        "min_gap_mm": min_gap,
        "seed_tail_fraction": seed_tail_fraction,
        "chain_depth": chain_depth,
        "seed_ts": stats["seed_ts"],
        "successful": successes,
        "failed_hotspot_sims": failures,
        "source_counts": source_counts,
        "n_legal_intermediate_available": stats["n_legal_intermediate_available"],
        "n_legal_intermediate_used": stats["n_legal_intermediate_used"],
        "perturb_attempts": stats["perturb_attempts"],
        "milp_solve_time_s": stats["milp_solve_time_s"],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[{case_name}] DONE: {successes}/{target} successful, {failures} hotspot failures", flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a HotSpot-augmented thermal dataset for Case1-10.")
    parser.add_argument("--cases", nargs="+", default=DEFAULT_CASES)
    parser.add_argument("--target", type=int, default=TARGET_DEFAULT)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--milp_time_limit", type=float, default=MILP_TIME_LIMIT_DEFAULT)
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--intermediate_cap_fraction", type=float, default=INTERMEDIATE_CAP_FRACTION_DEFAULT)
    parser.add_argument("--max_perturb_attempts", type=int, default=MAX_PERTURB_ATTEMPTS_DEFAULT)
    parser.add_argument("--min_gap", type=float, default=MIN_GAP_DEFAULT, help="Minimum chiplet-to-chiplet clearance in mm")
    parser.add_argument(
        "--seed_tail_fraction",
        type=float,
        default=SEED_TAIL_FRACTION_DEFAULT,
        help="Perturbation seeds are drawn evenly from this fraction of the tail end of the deduped "
        "MILP trajectory (default 0.2 = last 20%%), since downstream optimization starts from the "
        "MILP final solution.",
    )
    parser.add_argument(
        "--chain_depth",
        type=int,
        default=None,
        help="Perturbations chain up to this many hops from each seed (continuing from the previous "
        "hop's legal result) before resetting back to the seed, instead of always single-hopping from "
        "a fixed seed. Default: chiplet-count-tiered value from _chain_depth_for() (small cases get a "
        "deeper chain since they duplicate layouts much more readily). Pass a value to override "
        "uniformly for every --cases.",
    )
    parser.add_argument(
        "--layout_only",
        action="store_true",
        help="Stop after building each case's deduped layout_pool.json (Phase A/B) — no HotSpot "
        "simulation, no augmented CSVs. Use to inspect candidate-pool quality/diversity before "
        "committing to the (much slower) full HotSpot run.",
    )
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    out_root = args.out_dir
    if out_root is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_root = ROOT / "atplace" / "hotspot_augmented_dataset" / stamp
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"Output root: {out_root}", flush=True)

    t_total = time.time()
    all_summaries: dict[str, dict] = {}
    for case_name in args.cases:
        if not (CASES_DIR / case_name).exists():
            print(f"[{case_name}] SKIP (no case dir)", flush=True)
            continue
        if args.chain_depth is not None:
            chain_depth = args.chain_depth
        else:
            ci = load_atplace_case(CASES_DIR / case_name, {}, seed=args.seed)
            chain_depth = _chain_depth_for(len(ci.case.chiplets))
        try:
            all_summaries[case_name] = run_case(
                case_name,
                out_root,
                args.target,
                args.workers,
                args.milp_time_limit,
                args.seed,
                args.intermediate_cap_fraction,
                args.max_perturb_attempts,
                args.min_gap,
                args.seed_tail_fraction,
                chain_depth,
                args.layout_only,
            )
        except Exception:
            print(f"[{case_name}] FAILED with exception:\n{traceback.format_exc()}", flush=True)
            all_summaries[case_name] = {"case": case_name, "error": traceback.format_exc()}

    total_t = time.time() - t_total
    (out_root / "run_summary.json").write_text(
        json.dumps({"cases": all_summaries, "target": args.target, "total_time_s": total_t}, indent=2)
    )
    print(f"\nAll done in {total_t / 60:.1f} min. Output: {out_root}", flush=True)


if __name__ == "__main__":
    main()
