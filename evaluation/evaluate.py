"""
Offline evaluation of the CellSafe-X inference pipeline.

Run with:      python -m evaluation.evaluate

What it does
------------
1. Generates labelled synthetic sequences for every scenario across several
   seeds (fixed base seed => fully reproducible).
2. Runs the REAL pipeline on them - the same code the dashboard uses.
3. Scores the hidden-state posterior against the simulator's ground-truth
   labels: accuracy, macro precision / recall / F1, confusion matrix,
   multiclass Brier score and negative log-likelihood.
4. Scores the root-cause classifier against the scenario's true cause.
5. Compares false alarms and missed dangerous events against the fixed
   temperature-threshold baseline.
6. Writes metrics to results/metrics and figures to results/figures.

Nothing here is hardcoded: every number in the output files is computed from
the pipeline's actual behaviour on freshly generated data.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")  # headless: we only ever save files

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config import model_parameters as P
from data.battery_simulator import simulate_pack
from models import run_pipeline

# Windows-safe paths, resolved relative to this file rather than the CWD.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

BASE_SEED = 20240
DEFAULT_N_SEEDS = 5
DEFAULT_STEPS = 180

# A "dangerous declaration" for the Bayesian system: more than half the
# posterior mass sits on {Pre-Runaway, Thermal Runaway}.  For the baseline:
# the primary sensor has crossed the critical trip point.  Both are the
# system's own alarm condition, so the comparison is like for like.
BAYES_ALARM_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# data collection
# ---------------------------------------------------------------------------
def collect_records(
    n_seeds: int = DEFAULT_N_SEEDS,
    n_steps: int = DEFAULT_STEPS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the pipeline over every (scenario, seed, cell) and flatten the result."""
    records: List[Dict[str, object]] = []

    for scenario in P.SCENARIOS:
        for k in range(n_seeds):
            seed = BASE_SEED + k
            # Rotate the affected cell across seeds so the evaluation does not
            # only ever test one position in the pack.
            affected = k % P.N_CELLS
            sim = simulate_pack(
                scenario=scenario,
                affected_cell=affected,
                seed=seed,
                n_steps=n_steps,
            )
            outputs = run_pipeline(sim.frame)

            for cell, outs in outputs.items():
                truth = (
                    sim.frame[sim.frame["cell"] == cell]
                    .sort_values("step")
                    .reset_index(drop=True)
                )
                for i, o in enumerate(outs):
                    records.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "cell": cell,
                            "is_affected": bool(truth.loc[i, "is_affected"]),
                            "step": o.step,
                            "time_min": o.time_min,
                            "true_state": int(truth.loc[i, "true_state"]),
                            "pred_state": int(np.argmax(o.posterior)),
                            "p0": float(o.posterior[0]),
                            "p1": float(o.posterior[1]),
                            "p2": float(o.posterior[2]),
                            "p3": float(o.posterior[3]),
                            "p_dangerous": float(o.forecast.current_dangerous),
                            "true_dangerous": bool(
                                int(truth.loc[i, "true_state"]) in P.DANGEROUS_STATES
                            ),
                            "temp_primary": float(o.reading["temp_primary"]),
                            "temp_true": float(truth.loc[i, "temp_true"]),
                            "p_sensor_faulty": float(o.sensor.p_faulty),
                            "sensor_faulty_true": bool(truth.loc[i, "sensor_faulty_true"]),
                            "top_cause": o.cause.top_cause,
                            "true_cause": P.SCENARIO_TRUE_CAUSE[scenario],
                            "recommended_action": o.decision.recommended_action,
                            "threshold_triggered_warn": bool(
                                float(o.reading["temp_primary"]) >= P.THRESHOLD_WARN_C
                            ),
                            "threshold_triggered_critical": bool(
                                float(o.reading["temp_primary"]) >= P.THRESHOLD_CRITICAL_C
                            ),
                        }
                    )
            if verbose:
                print(f"  {scenario:<28} seed={seed} affected_cell={affected}  done")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def multiclass_brier(prob_matrix: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score: mean squared error between the predicted
    distribution and the one-hot truth, summed over classes.

    Range is [0, 2]; 0 is perfect.  Lower is better.
    """
    onehot = np.zeros_like(prob_matrix)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((prob_matrix - onehot) ** 2, axis=1)))


def negative_log_likelihood(prob_matrix: np.ndarray, y_true: np.ndarray) -> float:
    """Mean negative log probability assigned to the true class.

    Clipped at 1e-12 so a single over-confident mistake cannot return infinity
    (which would make the metric useless rather than merely bad).
    """
    p_true = prob_matrix[np.arange(len(y_true)), y_true]
    return float(-np.mean(np.log(np.clip(p_true, 1e-12, 1.0))))


def alarm_counts(pred_alarm: np.ndarray, true_dangerous: np.ndarray) -> Dict[str, float]:
    """False alarms and missed dangerous events for one alarm rule."""
    pred_alarm = np.asarray(pred_alarm, dtype=bool)
    true_dangerous = np.asarray(true_dangerous, dtype=bool)

    tp = int(np.sum(pred_alarm & true_dangerous))
    fp = int(np.sum(pred_alarm & ~true_dangerous))
    fn = int(np.sum(~pred_alarm & true_dangerous))
    tn = int(np.sum(~pred_alarm & ~true_dangerous))

    n_safe = fp + tn
    n_danger = tp + fn
    return {
        "true_positives": tp,
        "false_alarms": fp,
        "missed_dangerous": fn,
        "true_negatives": tn,
        "false_alarm_rate": float(fp / n_safe) if n_safe else 0.0,
        "miss_rate": float(fn / n_danger) if n_danger else 0.0,
        "precision": float(tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": float(tp / n_danger) if n_danger else 0.0,
    }


def reliability_curve(
    probs: np.ndarray, outcomes: np.ndarray, n_bins: int = 10
) -> Dict[str, List[float]]:
    """Bin predicted P(dangerous) and measure the empirical frequency in each bin."""
    probs = np.asarray(probs, dtype=float)
    outcomes = np.asarray(outcomes, dtype=bool)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, n_bins - 1)

    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        counts.append(n)
        if n == 0:
            mean_pred.append(float("nan"))
            frac_pos.append(float("nan"))
        else:
            mean_pred.append(float(probs[mask].mean()))
            frac_pos.append(float(outcomes[mask].mean()))

    # Expected Calibration Error over the non-empty bins.
    total = sum(counts)
    ece = 0.0
    for mp, fp_, n in zip(mean_pred, frac_pos, counts):
        if n and np.isfinite(mp) and np.isfinite(fp_):
            ece += (n / total) * abs(mp - fp_)

    return {
        "bin_mean_predicted": mean_pred,
        "bin_observed_frequency": frac_pos,
        "bin_counts": counts,
        "expected_calibration_error": float(ece),
    }


def _last_third_cause_accuracy(g: pd.DataFrame) -> float:
    """Root-cause accuracy once the fault has had time to develop.

    Most scenarios inject their fault several minutes in, so the opening steps
    are indistinguishable from normal operation by construction.  Reporting the
    late-run accuracy alongside the whole-run figure separates "the classifier
    is wrong" from "there was nothing to see yet".
    """
    if g.empty:
        return float("nan")
    cutoff = g["step"].max() * 2 // 3
    late = g[g["step"] >= cutoff]
    if late.empty:
        return float("nan")
    return float(np.mean(late["top_cause"] == late["true_cause"]))


def compute_metrics(df: pd.DataFrame) -> Dict[str, object]:
    """All headline metrics, computed from the collected records."""
    y_true = df["true_state"].to_numpy(dtype=int)
    y_pred = df["pred_state"].to_numpy(dtype=int)
    probs = df[["p0", "p1", "p2", "p3"]].to_numpy(dtype=float)
    labels = list(range(P.N_STATES))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-scenario accuracy tells a much more useful story than one global number.
    per_scenario = {}
    for scenario, g in df.groupby("scenario"):
        gt = g["true_state"].to_numpy(dtype=int)
        gp = g["pred_state"].to_numpy(dtype=int)
        gprob = g[["p0", "p1", "p2", "p3"]].to_numpy(dtype=float)
        # Macro-average only over the classes this scenario actually visits;
        # averaging over all four would silently divide by absent classes and
        # report a meaningless 0.25 for single-class scenarios.
        present = sorted(set(gt.tolist()) | set(gp.tolist()))
        aff = g[g["is_affected"]]
        per_scenario[str(scenario)] = {
            "n_samples": int(len(g)),
            "accuracy": float(np.mean(gt == gp)),
            "within_one_band_accuracy": float(np.mean(np.abs(gt - gp) <= 1)),
            "macro_f1_present_classes": float(
                f1_score(gt, gp, average="macro", labels=present, zero_division=0)
            ),
            "n_classes_present": len(set(gt.tolist())),
            "brier": multiclass_brier(gprob, gt, P.N_STATES),
            "nll": negative_log_likelihood(gprob, gt),
            # Root cause is a property of the FAULTY cell, so it is only scored
            # on that cell.  The other five cells are genuinely healthy and
            # correctly report benign heating; scoring them against the
            # scenario's fault label would be a labelling error, not a model
            # error.
            "root_cause_accuracy_affected_cell": float(
                np.mean(aff["top_cause"] == aff["true_cause"])
            ) if len(aff) else float("nan"),
            "root_cause_accuracy_affected_cell_last_third": _last_third_cause_accuracy(aff),
        }

    # Sensor-fault detection, scored only where a fault was actually injected
    # versus where it was not.
    sf_true = df["sensor_faulty_true"].to_numpy(dtype=bool)
    sf_pred = (df["p_sensor_faulty"].to_numpy(dtype=float) > 0.5)

    bayes_alarm = df["p_dangerous"].to_numpy(dtype=float) > BAYES_ALARM_THRESHOLD
    true_danger = df["true_dangerous"].to_numpy(dtype=bool)

    metrics: Dict[str, object] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_source": "SIMULATED - synthetic sequences from data/battery_simulator.py",
        "python_version": platform.python_version(),
        "config": {
            "base_seed": BASE_SEED,
            "n_scenarios": len(P.SCENARIOS),
            "n_cells": P.N_CELLS,
            "dt_seconds": P.DT_SECONDS,
            "bayes_alarm_threshold": BAYES_ALARM_THRESHOLD,
            "threshold_warn_c": P.THRESHOLD_WARN_C,
            "threshold_critical_c": P.THRESHOLD_CRITICAL_C,
        },
        "n_samples": int(len(df)),
        "hidden_state": {
            "accuracy": float(np.mean(y_true == y_pred)),
            "macro_precision": float(
                precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
            ),
            "multiclass_brier": multiclass_brier(probs, y_true, P.N_STATES),
            "negative_log_likelihood": negative_log_likelihood(probs, y_true),
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_labels": list(P.STATES),
            # Adjacent-band confusion (e.g. Abnormal vs Pre-Runaway) is far less
            # serious than confusing Healthy with Runaway, so we report it too.
            "within_one_band_accuracy": float(np.mean(np.abs(y_true - y_pred) <= 1)),
        },
        "root_cause": {
            # Scored on the affected cell only - see per_scenario above.
            "accuracy_affected_cell": float(
                np.mean(df[df["is_affected"]]["top_cause"] == df[df["is_affected"]]["true_cause"])
            ),
            "accuracy_affected_cell_last_third": _last_third_cause_accuracy(df[df["is_affected"]]),
            # Sanity check on the healthy cells, restricted to the scenarios
            # where the fault really is confined to one cell.  In the
            # cooling-failure and external-heat scenarios every cell in the pack
            # is genuinely affected, so reporting the pack-wide cause on a
            # "non-affected" cell is the correct answer there, not an error.
            "unaffected_cells_reporting_benign_heat_localised_faults": float(
                np.mean(
                    df[
                        (~df["is_affected"])
                        & (df["scenario"].isin(
                            ["Temperature Sensor Fault", "Internal Short Circuit"]
                        ))
                    ]["top_cause"] == "Normal/Fast-Charging Heat"
                )
            ),
        },
        "sensor_reliability": alarm_counts(sf_pred, sf_true),
        "alarm_comparison": {
            "bayesian": alarm_counts(bayes_alarm, true_danger),
            "threshold_warn": alarm_counts(
                df["threshold_triggered_warn"].to_numpy(dtype=bool), true_danger
            ),
            "threshold_critical": alarm_counts(
                df["threshold_triggered_critical"].to_numpy(dtype=bool), true_danger
            ),
        },
        "sensor_fault_scenario_only": _sensor_fault_scenario_comparison(df),
        "calibration": reliability_curve(
            df["p_dangerous"].to_numpy(dtype=float), true_danger
        ),
        "per_scenario": per_scenario,
    }
    return metrics


def _sensor_fault_scenario_comparison(df: pd.DataFrame) -> Dict[str, object]:
    """The headline demonstration: what each system does on the sensor-fault runs.

    In this scenario the pack is genuinely healthy the whole time, so every
    alarm raised by either system is a false alarm by construction.
    """
    g = df[df["scenario"] == "Temperature Sensor Fault"]
    if g.empty:
        return {}
    affected = g[g["is_affected"]]
    return {
        "n_samples_affected_cell": int(len(affected)),
        "true_dangerous_samples": int(affected["true_dangerous"].sum()),
        "bayesian_false_alarms": int(
            (affected["p_dangerous"] > BAYES_ALARM_THRESHOLD).sum()
        ),
        "threshold_warn_false_alarms": int(affected["threshold_triggered_warn"].sum()),
        "threshold_critical_false_alarms": int(
            affected["threshold_triggered_critical"].sum()
        ),
        "mean_p_sensor_faulty_after_onset": float(
            affected[affected["sensor_faulty_true"]]["p_sensor_faulty"].mean()
        )
        if affected["sensor_faulty_true"].any()
        else 0.0,
        "mean_p_sensor_faulty_before_onset": float(
            affected[~affected["sensor_faulty_true"]]["p_sensor_faulty"].mean()
        ),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
NAVY = "#12233f"
ACCENT = "#2b6cb0"


def _style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linestyle=":")


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    cm = np.asarray(cm, dtype=float)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(norm, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(P.N_STATES), P.STATES, rotation=30, ha="right")
    ax.set_yticks(range(P.N_STATES), P.STATES)
    ax.set_xlabel("Predicted hidden state (MAP of the filtered posterior)")
    ax.set_ylabel("True hidden state (simulator ground truth)")
    ax.set_title("CellSafe-X hidden-state confusion matrix\n(row-normalised, SIMULATED data)")

    for i in range(P.N_STATES):
        for j in range(P.N_STATES):
            ax.text(
                j, i,
                f"{norm[i, j]:.2f}\n({int(cm[i, j])})",
                ha="center", va="center",
                fontsize=9,
                color="white" if norm[i, j] > 0.55 else NAVY,
            )
    fig.colorbar(im, ax=ax, label="fraction of true-class samples")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_risk_over_time(df: pd.DataFrame, path: Path) -> None:
    """Mean P(dangerous) over time for the affected cell of each scenario."""
    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    for scenario in P.SCENARIOS:
        g = df[(df["scenario"] == scenario) & (df["is_affected"])]
        if g.empty:
            continue
        curve = g.groupby("time_min")["p_dangerous"].mean()
        ax.plot(curve.index, curve.to_numpy(), linewidth=2.0, label=scenario)

    ax.axhline(BAYES_ALARM_THRESHOLD, color="#c53030", linestyle="--", linewidth=1.2,
               label=f"alarm threshold ({BAYES_ALARM_THRESHOLD:g})")
    ax.set_xlabel("simulated time (minutes)")
    ax.set_ylabel("P(Pre-Runaway or Thermal Runaway)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Filtered dangerous-state probability, affected cell\n(mean over seeds, SIMULATED data)")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    _style_axes(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_calibration(cal: Dict[str, List[float]], path: Path) -> None:
    mp = np.array(cal["bin_mean_predicted"], dtype=float)
    of = np.array(cal["bin_observed_frequency"], dtype=float)
    counts = np.array(cal["bin_counts"], dtype=float)
    mask = np.isfinite(mp) & np.isfinite(of)

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(6.6, 6.8), height_ratios=[3, 1], sharex=True
    )
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.2,
            label="perfect calibration")
    ax.plot(mp[mask], of[mask], "o-", color=ACCENT, linewidth=2.0,
            markersize=7, label="CellSafe-X")
    ax.set_ylabel("observed frequency of a truly dangerous state")
    ax.set_title(
        "Reliability diagram for P(dangerous)\n"
        f"Expected Calibration Error = {cal['expected_calibration_error']:.4f} (SIMULATED data)"
    )
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=9)
    _style_axes(ax)

    centres = np.linspace(0.05, 0.95, len(counts))
    ax2.bar(centres, counts, width=0.085, color=NAVY, alpha=0.75)
    ax2.set_xlabel("predicted P(dangerous)")
    ax2.set_ylabel("samples")
    _style_axes(ax2)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate the CellSafe-X pipeline.")
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS,
                        help="number of random seeds per scenario (default: 5)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help="simulation steps per run (default: 180 = 30 minutes)")
    parser.add_argument("--quiet", action="store_true", help="suppress progress output")
    args = parser.parse_args(argv)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    verbose = not args.quiet
    if verbose:
        print("CellSafe-X evaluation  (all data is SIMULATED)")
        print(f"  scenarios={len(P.SCENARIOS)}  seeds={args.seeds}  "
              f"steps={args.steps}  cells={P.N_CELLS}")
        print("Generating data and running the Bayesian pipeline...")

    df = collect_records(n_seeds=args.seeds, n_steps=args.steps, verbose=verbose)

    if verbose:
        print(f"Collected {len(df):,} (cell, step) samples. Computing metrics...")

    metrics = compute_metrics(df)

    # --- save metrics ------------------------------------------------------
    json_path = METRICS_DIR / "evaluation_metrics.json"
    json_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    per_scenario_rows = [
        {"scenario": k, **v} for k, v in metrics["per_scenario"].items()
    ]
    pd.DataFrame(per_scenario_rows).to_csv(
        METRICS_DIR / "per_scenario_metrics.csv", index=False, encoding="utf-8"
    )

    cm = np.array(metrics["hidden_state"]["confusion_matrix"], dtype=int)
    pd.DataFrame(cm, index=P.STATES, columns=P.STATES).to_csv(
        METRICS_DIR / "confusion_matrix.csv", encoding="utf-8"
    )

    ac = metrics["alarm_comparison"]
    pd.DataFrame(
        [{"system": k, **v} for k, v in ac.items()]
    ).to_csv(METRICS_DIR / "alarm_comparison.csv", index=False, encoding="utf-8")

    # A compact raw-sample dump is useful for the dashboard and for debugging,
    # but the full frame would be large, so we keep the affected cells only.
    df[df["is_affected"]].to_csv(
        METRICS_DIR / "affected_cell_samples.csv", index=False, encoding="utf-8"
    )

    # --- save figures ------------------------------------------------------
    plot_confusion_matrix(cm, FIGURES_DIR / "confusion_matrix.png")
    plot_risk_over_time(df, FIGURES_DIR / "risk_over_time.png")
    plot_calibration(metrics["calibration"], FIGURES_DIR / "calibration_curve.png")

    # --- report ------------------------------------------------------------
    hs = metrics["hidden_state"]
    print("\n" + "=" * 66)
    print("HIDDEN-STATE INFERENCE")
    print(f"  samples                 : {metrics['n_samples']:,}")
    print(f"  accuracy                : {hs['accuracy']:.4f}")
    print(f"  within-one-band accuracy: {hs['within_one_band_accuracy']:.4f}")
    print(f"  macro precision         : {hs['macro_precision']:.4f}")
    print(f"  macro recall            : {hs['macro_recall']:.4f}")
    print(f"  macro F1                : {hs['macro_f1']:.4f}")
    print(f"  multiclass Brier        : {hs['multiclass_brier']:.4f}")
    print(f"  negative log-likelihood : {hs['negative_log_likelihood']:.4f}")
    rc = metrics["root_cause"]
    print(f"\nROOT CAUSE (affected cell): {rc['accuracy_affected_cell']:.4f}"
          f"  (last third of each run: {rc['accuracy_affected_cell_last_third']:.4f})")
    print(f"  healthy cells reporting benign heat (localised-fault scenarios): "
          f"{rc['unaffected_cells_reporting_benign_heat_localised_faults']:.4f}")
    print(f"CALIBRATION  ECE          : {metrics['calibration']['expected_calibration_error']:.4f}")

    print("\nALARM COMPARISON (false alarms / missed dangerous events)")
    for name, v in ac.items():
        print(f"  {name:<20} FA={v['false_alarms']:>6,} ({v['false_alarm_rate']:.4f})   "
              f"MISS={v['missed_dangerous']:>6,} ({v['miss_rate']:.4f})   "
              f"precision={v['precision']:.4f}")

    sf = metrics["sensor_fault_scenario_only"]
    if sf:
        print("\nSENSOR-FAULT SCENARIO, affected cell only (pack is truly healthy throughout)")
        print(f"  samples                              : {sf['n_samples_affected_cell']:,}")
        print(f"  Bayesian false alarms                : {sf['bayesian_false_alarms']:,}")
        print(f"  Threshold (warn) false alarms        : {sf['threshold_warn_false_alarms']:,}")
        print(f"  Threshold (critical) false alarms    : {sf['threshold_critical_false_alarms']:,}")
        print(f"  mean P(sensor faulty) after onset    : {sf['mean_p_sensor_faulty_after_onset']:.4f}")
        print(f"  mean P(sensor faulty) before onset   : {sf['mean_p_sensor_faulty_before_onset']:.4f}")

    print("\nPER SCENARIO")
    for k, v in metrics["per_scenario"].items():
        print(f"  {k:<28} acc={v['accuracy']:.3f}  +/-1band={v['within_one_band_accuracy']:.3f}  "
              f"Brier={v['brier']:.3f}  "
              f"cause(affected,late)={v['root_cause_accuracy_affected_cell_last_third']:.3f}")

    print("=" * 66)
    print(f"\nMetrics written to : {METRICS_DIR}")
    print(f"Figures written to : {FIGURES_DIR}")
    print("Reminder: all results are from SIMULATED data and are not validated "
          "against real battery hardware.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
