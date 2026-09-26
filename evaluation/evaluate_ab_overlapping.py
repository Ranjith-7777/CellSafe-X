"""
Phase 7B — controlled A/B evaluation: frozen primary+backup core fusion
(Baseline A) vs. the new N-sensor reliability-aware core fusion (Variant B).

Run with:      python -m evaluation.evaluate_ab_overlapping

NEVER writes to results/metrics/evaluation_metrics.json (the canonical,
frozen Phase 1-6 artifact). Writes two clearly-labelled files instead:
  results/metrics/ab_baseline_a_legacy.json
  results/metrics/ab_variant_b_overlapping.json
plus a compact side-by-side comparison, so both results stay available and
the frozen baseline is never silently overwritten.

Both runs use IDENTICAL seeds/scenarios/steps (the same collect_records
defaults as the canonical evaluation), so any metric difference is
attributable only to the core temperature-fusion path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

from config import model_parameters as P
from evaluation.evaluate import compute_metrics, collect_records

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"

HEADLINE_KEYS = [
    ("hidden_state", "accuracy"),
    ("hidden_state", "macro_precision"),
    ("hidden_state", "macro_recall"),
    ("hidden_state", "macro_f1"),
    ("hidden_state", "multiclass_brier"),
    ("hidden_state", "negative_log_likelihood"),
]


def _get(d: dict, path) -> object:
    for k in path:
        d = d[k]
    return d


def main() -> int:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    print("Running Baseline A (legacy primary+backup fusion)...")
    df_a = collect_records(verbose=False, temperature_fusion="legacy")
    metrics_a = compute_metrics(df_a)

    print("Running Variant B (overlapping N-sensor fusion)...")
    df_b = collect_records(verbose=False, temperature_fusion="overlapping")
    metrics_b = compute_metrics(df_b)

    (METRICS_DIR / "ab_baseline_a_legacy.json").write_text(json.dumps(metrics_a, indent=2), encoding="utf-8")
    (METRICS_DIR / "ab_variant_b_overlapping.json").write_text(json.dumps(metrics_b, indent=2), encoding="utf-8")

    ac_a, ac_b = metrics_a["alarm_comparison"]["bayesian"], metrics_b["alarm_comparison"]["bayesian"]
    mb_a = ac_a["missed_dangerous_breakdown"]
    mb_b = ac_b["missed_dangerous_breakdown"]

    print("\n" + "=" * 78)
    print(f"{'Metric':38s} {'A: legacy':>16s} {'B: overlapping':>18s}")
    print("-" * 78)
    for group, key in HEADLINE_KEYS:
        va, vb = _get(metrics_a, [group, key]), _get(metrics_b, [group, key])
        print(f"{key:38s} {va:16.4f} {vb:18.4f}")
    print(f"{'ECE (raw)':38s} {metrics_a['calibration']['expected_calibration_error']:16.4f} "
          f"{metrics_b['calibration']['expected_calibration_error']:18.4f}")
    print(f"{'false_alarms':38s} {ac_a['false_alarms']:16d} {ac_b['false_alarms']:18d}")
    print(f"{'missed_dangerous (latent-state)':38s} {mb_a['missed_total']:16d} {mb_b['missed_total']:18d}")
    print(f"{'...of which genuinely hot':38s} "
          f"{mb_a['missed_temp_actually_crossed_pre_runaway_threshold']:16d} "
          f"{mb_b['missed_temp_actually_crossed_pre_runaway_threshold']:18d}")

    print("\nPer-scenario accuracy:")
    print(f"{'Scenario':30s} {'A':>10s} {'B':>10s}")
    for scenario in P.SCENARIOS:
        va = metrics_a["per_scenario"][scenario]["accuracy"]
        vb = metrics_b["per_scenario"][scenario]["accuracy"]
        print(f"{scenario:30s} {va:10.4f} {vb:10.4f}")
    print("=" * 78)
    print(f"\nWritten to {METRICS_DIR} (canonical evaluation_metrics.json untouched).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
