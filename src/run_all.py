from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


BASE = Path(__file__).resolve().parents[1]


def build_dataset_map(data_dir: Path) -> Dict[str, Dict[str, str | None]]:
    return {
        "AAPL_filtered.csv": {"value_column": "adj_close", "date_column": "date"},
        "vix_daily.csv": {"value_column": "close", "date_column": "date"},
        "daily_sunspots_time_series_1850-01_2025-01.csv": {"value_column": "counts", "date_column": "date"},
        "Walmart_Sales.csv": {"value_column": "Weekly_Sales", "date_column": "Date"},
        "seattle-weather.csv": {"value_column": "temp_max", "date_column": "date"},
    }


def load_base_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_config(cfg: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def run_hpo(temp_cfg: Path, model_selector: str, study_name: str) -> None:
    cmd = [
        "python3",
        "-m",
        "src.hpo_optuna",
        "--config",
        str(temp_cfg),
        "--trials",
        "20",
        "--study",
        study_name,
        "--model",
        model_selector,
    ]
    subprocess.run(cmd, check=True)


def move_hpo_dir(results_dir: Path) -> Path:
    src_dir = results_dir / "hpo"
    dst_dir = results_dir / "HPO"
    # If already in desired place, return
    if dst_dir.exists():
        return dst_dir
    # Move if source exists; otherwise, just return the most plausible directory
    if src_dir.exists():
        dst_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_dir), str(dst_dir))
        return dst_dir
    # Fallback: return results_dir (caller will search for summaries)
    return results_dir


def find_latest_summary(base_dir: Path) -> Path:
    # Search common locations: HPO/, hpo/, or anywhere under results_dir
    candidates: List[Path] = []
    for sub in (base_dir / "HPO", base_dir / "hpo", base_dir):
        if sub.exists():
            candidates.extend(sub.glob("study-summary-*.json"))
    if not candidates:
        candidates = list(base_dir.rglob("study-summary-*.json"))
    if not candidates:
        raise FileNotFoundError(f"No HPO summary found under {base_dir}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def run_refit(temp_cfg: Path, summary_path: Path) -> None:
    cmd = [
        "python3",
        "-m",
        "src.refit",
        "--config",
        str(temp_cfg),
        "--study-summary",
        str(summary_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    base_cfg_path = BASE / "config.json"
    base_cfg = load_base_config(base_cfg_path)
    data_dir = BASE / "data"
    ds_map = build_dataset_map(data_dir)

    models: List[dict] = base_cfg.get("models", [])
    if not models:
        raise ValueError("No models defined in config.json -> models")

    for ds_file, meta in ds_map.items():
        ds_path = data_dir / ds_file
        if not ds_path.exists():
            print(f"[WARN] dataset missing: {ds_path}")
            continue
        dataset_tag = ds_path.stem

        for model_cfg in models:
            model_name = str(model_cfg.get("name") or model_cfg.get("type"))
            model_type = str(model_cfg.get("type"))

            # Derive per-run results dir: results/<model>/<dataset>/
            results_dir = BASE / "results" / model_name / dataset_tag

            # Build temp config
            cfg = dict(base_cfg)
            cfg["results_dir"] = str(results_dir)
            cfg["dataset"] = dict(cfg.get("dataset", {}))
            cfg["dataset"]["path"] = str(ds_path)
            cfg["dataset"]["value_column"] = meta["value_column"]
            cfg["dataset"]["date_column"] = meta["date_column"]

            temp_cfg_path = results_dir / "tmp" / f"config-{model_name}-{dataset_tag}.json"
            write_config(cfg, temp_cfg_path)

            print(f"\n=== HPO: model={model_name} dataset={dataset_tag} ===")
            study_name = f"hpo_{model_name}_{dataset_tag}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            run_hpo(temp_cfg_path, model_selector=model_name if model_cfg.get("name") else model_type, study_name=study_name)
            hpo_dir = move_hpo_dir(results_dir)
            try:
                summary = find_latest_summary(hpo_dir)
            except FileNotFoundError as e:
                print(f"[WARN] {e}; skipping refit for model={model_name} dataset={dataset_tag}")
                continue

            print(f"=== REFIT: model={model_name} dataset={dataset_tag} ===")
            run_refit(temp_cfg_path, summary)

    print("\nAll HPO and refit runs completed.")


if __name__ == "__main__":
    main()


