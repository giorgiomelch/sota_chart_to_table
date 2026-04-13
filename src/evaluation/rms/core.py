"""
Core RMS computation: Hungarian matching, precision/recall/F1, public API.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .distance import d_theta_scatter, entry_similarity, nl_tau
from .parser import _detect_chart_type, _extract_ranges, json_to_mappings
from .types import AxisRanges, Mapping, ScatterVal


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _drop_series_from_row(mappings: list[Mapping]) -> list[Mapping]:
    """Set row='' for all non-metadata mappings (ignores the series dimension)."""
    return [
        Mapping(row="" if m.row != "__meta__" else m.row, col=m.col, val=m.val)
        for m in mappings
    ]


def _rms_single(P: list[Mapping], T: list[Mapping], tau: float, theta: float,
                ranges: AxisRanges | None = None) -> dict:
    """
    Compute RMS scores for a single orientation of predicted vs ground-truth mappings.

    Returns dict with: precision, recall, f1, matched_sim, pairs.
    """
    N, M = len(P), len(T)

    if N == 0 and M == 0:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "matched_sim": 0.0, "pairs": []}
    if N == 0:
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "matched_sim": 0.0, "pairs": []}
    if M == 0:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "matched_sim": 0.0, "pairs": []}

    # If either side has a single series, the series name carries no information.
    # Normalise both so key matching uses only the categorical key.
    def _data_rows(ms: list[Mapping]) -> set[str]:
        return {m.row for m in ms if m.row != "__meta__"}

    if len(_data_rows(P)) <= 1 or len(_data_rows(T)) <= 1:
        P = _drop_series_from_row(P)
        T = _drop_series_from_row(T)

    # For degenerate single-series scatter (all rows identical after normalisation),
    # key-based matching is trivially 1.0 for all pairs → arbitrary Hungarian
    # assignment. Use value proximity instead so permuted points still score F1=1.0.
    data_P = [m for m in P if m.row != "__meta__"]
    data_T = [m for m in T if m.row != "__meta__"]
    scatter_degenerate = (
        data_P and data_T
        and all(isinstance(m.val, ScatterVal) for m in data_P + data_T)
        and len({m.row for m in data_P}) <= 1
        and len({m.row for m in data_T}) <= 1
    )

    # Key-similarity matrix (N × M)
    key_sim = np.zeros((N, M))
    for i, p in enumerate(P):
        for j, t in enumerate(T):
            if isinstance(p.val, ScatterVal) and isinstance(t.val, ScatterVal):
                if scatter_degenerate:
                    key_sim[i, j] = 1.0 - d_theta_scatter(p.val, t.val, theta, ranges)
                else:
                    key_sim[i, j] = 1.0 - nl_tau(p.row, t.row, tau)
            else:
                key_sim[i, j] = 1.0 - nl_tau(p.row + p.col, t.row + t.col, tau)

    # Hungarian assignment: minimise cost = 1 − similarity
    row_ind, col_ind = linear_sum_assignment(1.0 - key_sim)

    pair_sims = [entry_similarity(P[i], T[j], tau, theta, ranges)
                 for i, j in zip(row_ind, col_ind)]
    total_sim = sum(pair_sims)

    precision = total_sim / N
    recall    = total_sim / M
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    pairs = list(zip(row_ind.tolist(), col_ind.tolist(), pair_sims))

    return {"precision": precision, "recall": recall, "f1": f1,
            "matched_sim": total_sim, "pairs": pairs}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_rms(
    predicted: dict,
    target: dict,
    tau: float = 0.5,
    theta: float = 0.1,
) -> dict:
    """
    Compute RMS between two chart JSON dicts.

    Parameters
    ----------
    predicted : dict   Model-predicted chart JSON.
    target    : dict   Ground-truth chart JSON.
    tau       : float  NL distance threshold (default 0.5).
    theta     : float  Numeric distance threshold expressed as a fraction of
                       the axis range (default 0.1).

    Returns
    -------
    dict with keys:
        precision, recall, f1  — best scores across orientations
        orientation            — 'normal' or 'transposed' (scatter: always 'normal')
        chart_type             — detected chart type
        normal                 — scores for the normal orientation
        transposed             — scores for the transposed orientation
    """
    chart_type = _detect_chart_type(target)
    ranges     = _extract_ranges(target, chart_type)
    T_normal   = json_to_mappings(target, transpose=False, chart_type=chart_type)

    # Scatter has no meaningful transposition; all other types do
    orientations = [False] if chart_type == "scatter" else [False, True]

    results: dict = {}
    for transpose in orientations:
        P   = json_to_mappings(predicted, transpose=transpose, chart_type=chart_type)
        key = "transposed" if transpose else "normal"
        results[key] = _rms_single(P, T_normal, tau, theta, ranges)

    if "transposed" not in results:
        results["transposed"] = results["normal"]

    best = max(results, key=lambda k: results[k]["f1"])
    return {
        "precision":   results[best]["precision"],
        "recall":      results[best]["recall"],
        "f1":          results[best]["f1"],
        "orientation": best,
        "chart_type":  chart_type,
        "normal":      results["normal"],
        "transposed":  results["transposed"],
    }


def compute_rms_detailed(
    predicted: dict,
    target: dict,
    tau: float = 0.5,
    theta: float = 0.1,
) -> dict:
    """
    Like compute_rms but also returns per-mapping match details for the best orientation.

    Extra keys in the returned dict
    --------------------------------
    pairs         : list of {"gt": Mapping, "pred": Mapping, "similarity": float}
    unmatched_gt  : list[Mapping] — GT data mappings with no prediction counterpart
    unmatched_pred: list[Mapping] — predicted data mappings with no GT counterpart
    meta_pairs    : list of {"gt": Mapping, "pred": Mapping, "similarity": float}
    """
    chart_type = _detect_chart_type(target)
    ranges     = _extract_ranges(target, chart_type)
    T_normal   = json_to_mappings(target, transpose=False, chart_type=chart_type)

    orientations = [False] if chart_type == "scatter" else [False, True]

    results:  dict = {}
    P_by_key: dict = {}
    for transpose in orientations:
        P   = json_to_mappings(predicted, transpose=transpose, chart_type=chart_type)
        key = "transposed" if transpose else "normal"
        results[key]  = _rms_single(P, T_normal, tau, theta, ranges)
        P_by_key[key] = P

    if "transposed" not in results:
        results["transposed"]  = results["normal"]
        P_by_key["transposed"] = P_by_key["normal"]

    best         = max(results, key=lambda k: results[k]["f1"])
    P_best       = P_by_key[best]
    best_result  = results[best]

    def _is_meta(m: Mapping) -> bool:
        return m.row == "__meta__"

    matched_p: set[int] = set()
    matched_t: set[int] = set()
    data_pairs: list[dict] = []
    meta_pairs: list[dict] = []

    for pred_idx, gt_idx, sim in best_result["pairs"]:
        t_m = T_normal[gt_idx]
        p_m = P_best[pred_idx]
        entry = {"gt": t_m, "pred": p_m, "similarity": sim}
        if _is_meta(t_m) or _is_meta(p_m):
            meta_pairs.append(entry)
        else:
            data_pairs.append(entry)
        matched_p.add(pred_idx)
        matched_t.add(gt_idx)

    unmatched_gt   = [T_normal[j] for j in range(len(T_normal))
                      if j not in matched_t and not _is_meta(T_normal[j])]
    unmatched_pred = [P_best[i]   for i in range(len(P_best))
                      if i not in matched_p and not _is_meta(P_best[i])]

    return {
        "precision":      results[best]["precision"],
        "recall":         results[best]["recall"],
        "f1":             results[best]["f1"],
        "orientation":    best,
        "chart_type":     chart_type,
        "normal":         results["normal"],
        "transposed":     results["transposed"],
        "pairs":          data_pairs,
        "unmatched_gt":   unmatched_gt,
        "unmatched_pred": unmatched_pred,
        "meta_pairs":     meta_pairs,
    }


def compute_rms_from_files(
    predicted_path: str | Path,
    target_path:    str | Path,
    tau:   float = 0.5,
    theta: float = 0.1,
) -> dict:
    """Load two JSON files and compute RMS."""
    with open(predicted_path) as f:
        predicted = json.load(f)
    with open(target_path) as f:
        target = json.load(f)
    return compute_rms(predicted, target, tau=tau, theta=theta)
