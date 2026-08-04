"""Utilities for summarizing N-k evaluation artifacts in analysis notebooks."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


_K_PATTERN = re.compile(r"_n-(\d+)")


def resolve_results_root(base_path: str | Path = "naht_results") -> Path:
    """Resolve a results root from either the repository or ``src`` directory."""
    base_path = Path(base_path)
    candidates = [
        base_path,
        Path("..") / base_path,
        Path("../..") / base_path,
        Path.cwd() / base_path,
        Path.cwd().parent / base_path,
        Path.cwd().parent.parent / base_path,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve results root from: {candidates}")


def _first_artifact(run_dir: Path, filename: str) -> Path | None:
    """Find a Sacred artifact without assuming the numeric run id is ``1``."""
    matches = sorted(run_dir.glob(f"*/{filename}"))
    return matches[0] if matches else None


def _first_metric(info_path: Path | None, key: str) -> float:
    if info_path is None:
        return np.nan
    with info_path.open() as stream:
        values = json.load(stream).get(key, [])
    if not values:
        return np.nan
    value = values[0]
    return float(value["value"] if isinstance(value, dict) else value)


def _collect_runs(
    evaluation: str,
    matchups: Iterable[tuple[str, Path]],
    n_total: int,
) -> pd.DataFrame:
    rows = []
    for target, matchup_dir in matchups:
        sacred_dir = matchup_dir / "sacred"
        if not sacred_dir.is_dir():
            continue
        for run_dir in sorted(path for path in sacred_dir.iterdir() if path.is_dir()):
            k_match = _K_PATTERN.search(run_dir.name)
            info_path = _first_artifact(run_dir, "info.json")
            config_path = _first_artifact(run_dir, "config.json")
            k = int(k_match.group(1)) if k_match else np.nan
            rows.append(
                {
                    "evaluation": evaluation,
                    "target": target.upper(),
                    "k": k,
                    "n_controlled_agents": n_total - k if not pd.isna(k) else np.nan,
                    "run_name": run_dir.name,
                    "info_found": info_path is not None,
                    "config_found": config_path is not None,
                    "test_return_mean": _first_metric(info_path, "test_return_mean"),
                    "test_classifier_accuracy": _first_metric(
                        info_path, "test_classifier_accuracy"
                    ),
                }
            )
    return pd.DataFrame(rows)


def collect_specialized_teammate_runs(
    results_root: Path, task: str = "mpe-pp", n_total: int = 3
) -> pd.DataFrame:
    """Collect exact-N-k runs for POAM policies specialized to one teammate type."""
    root = results_root / task / "specialized_poam_50ep_exact_nk_eval"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing specialized teammate evaluation folder: {root}")

    matchups = []
    for matchup_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = re.fullmatch(r"poam-vs-(.+)-vs-(.+)", matchup_dir.name)
        if match and match.group(1) == match.group(2):
            matchups.append((match.group(2), matchup_dir))
    return _collect_runs("Specialized POAM", matchups, n_total)


def collect_type_conditional_runs(
    results_root: Path, task: str = "mpe-pp", n_total: int = 3
) -> pd.DataFrame:
    """Collect LSTM type-conditional runs across policy and evaluation seeds."""
    root = results_root / task / "type_conditional_lstm_nk_eval"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing type-conditional evaluation folder: {root}")

    matchups = []
    for matchup_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = re.fullmatch(r"type_conditional-vs-(.+)", matchup_dir.name)
        if match:
            matchups.append((match.group(1), matchup_dir))
    return _collect_runs("Type conditional LSTM", matchups, n_total)


def collect_poam_all_types_runs(
    results_root: Path, task: str = "mpe-pp", n_total: int = 3
) -> pd.DataFrame:
    """Collect N-k runs for the single POAM policy trained across all types."""
    root = results_root / task / "in_distr_eval"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing all-types POAM evaluation folder: {root}")

    matchups = []
    for matchup_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = re.fullmatch(r"poam-pqvmq_open-vs-(.+)", matchup_dir.name)
        if match:
            matchups.append((match.group(1), matchup_dir))
    return _collect_runs("POAM all types", matchups, n_total)


def artifact_status(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize result-directory completeness without hiding missing artifacts."""
    if runs.empty:
        return pd.DataFrame(
            columns=["evaluation", "target", "k", "run_dirs", "info_json", "config_json"]
        )
    status = (
        runs.groupby(["evaluation", "target", "k"], dropna=False)
        .agg(
            run_dirs=("run_name", "size"),
            info_json=("info_found", "sum"),
            config_json=("config_found", "sum"),
        )
        .reset_index()
        .sort_values(["evaluation", "target", "k"])
    )
    status["info_missing"] = status["run_dirs"] - status["info_json"]
    status["config_missing"] = status["run_dirs"] - status["config_json"]
    return status


def metric_summary(runs: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Compute target/N-k means and normal-approximation 95% confidence intervals."""
    usable = runs.dropna(subset=[metric])
    if usable.empty:
        return pd.DataFrame(
            columns=[
                "evaluation",
                "target",
                "n_controlled_agents",
                "mean",
                "ci",
                "n_evals",
            ]
        )
    summary = (
        usable.groupby(["evaluation", "target", "n_controlled_agents"])[metric]
        .agg(["mean", "sem", "count"])
        .reset_index()
        .rename(columns={"count": "n_evals"})
    )
    summary["ci"] = 1.96 * summary.pop("sem").fillna(0.0)
    return summary.sort_values(["evaluation", "target", "n_controlled_agents"])


def performance_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize test returns."""
    return metric_summary(runs, "test_return_mean")


def classifier_accuracy_summary(runs: pd.DataFrame) -> pd.DataFrame:
    """Summarize type-classifier accuracy when that metric is available."""
    return metric_summary(runs, "test_classifier_accuracy")
