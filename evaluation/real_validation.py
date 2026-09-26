"""
Phase 6 — external validation: replay the Warwick thermal-runaway
experiments through the UNCHANGED CellSafe-X hidden-state filter.

Run with:      python -m evaluation.real_validation

This is a SEPARATE track from evaluation/evaluate.py (the synthetic,
quantitative benchmark). No number produced here is merged into, or
compared head-to-head against, the synthetic accuracy/Brier/NLL/ECE figures.
There is no four-state ground truth for real cells, so no accuracy-style
metric is computed - only danger-probability trajectories, entropy, and
event-aligned timing relative to the signal-derived proxies in
data/real_data_adapter.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import model_parameters as P
from data.real_data_adapter import N_EXPERIMENTS, load_experiment, raw_data_available
from models.active_sensing import posterior_entropy
from models.bayesian_filter import BayesianFilter
from models.risk_forecast import dangerous_probability

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "real_validation"


def replay_experiment(exp_id: int) -> Dict[str, object]:
    """Deterministic sequential replay: BayesianFilter has no RNG and no
    dependence on anything but the observation sequence, so re-running this
    on the same cached data always reproduces the same trajectory."""
    exp = load_experiment(exp_id)
    f = BayesianFilter()

    rows: List[Dict[str, float]] = []
    for t, obs in zip(exp.frame["time_s"], exp.observations):
        step = f.update(obs)
        p = step.posterior
        rows.append(
            {
                "time_s": float(t),
                "p_healthy": float(p[P.S_HEALTHY]),
                "p_abnormal": float(p[P.S_ABNORMAL]),
                "p_pre_runaway": float(p[P.S_PRE]),
                "p_thermal_runaway": float(p[P.S_RUNAWAY]),
                "p_dangerous": float(dangerous_probability(p)),
                "entropy_nats": float(posterior_entropy(p)),
                "map_state": P.STATES[int(np.argmax(p))],
                "n_channels_observed": len(obs),
                "temp_c_input": obs.get("temp_c", float("nan")),
            }
        )
    traj = pd.DataFrame(rows)

    events = exp.events
    onset = events.temp_runaway_onset_s
    summary = {
        "experiment_id": exp_id,
        "n_steps": len(traj),
        "max_p_dangerous": float(traj["p_dangerous"].max()),
        "max_p_dangerous_time_s": float(traj.loc[traj["p_dangerous"].idxmax(), "time_s"]),
        "temp_runaway_onset_s": onset,
        "pressure_vent_onset_s": events.pressure_vent_onset_s,
        "event_marker_note": events.note,
    }

    alarm = traj["p_dangerous"].to_numpy() > 0.5
    first_alarm_idx = np.argmax(alarm) if alarm.any() else None
    if first_alarm_idx is not None and alarm[first_alarm_idx]:
        first_alarm_t = float(traj["time_s"].iloc[first_alarm_idx])
        summary["first_alarm_time_s"] = first_alarm_t
        summary["warning_lead_time_s"] = (onset - first_alarm_t) if onset is not None else None
    else:
        summary["first_alarm_time_s"] = None
        summary["warning_lead_time_s"] = None

    if onset is not None:
        pre_event = traj[traj["time_s"] <= onset]
        summary["fraction_pre_event_flagged_dangerous"] = (
            float((pre_event["p_dangerous"] > 0.5).mean()) if len(pre_event) else float("nan")
        )
    else:
        summary["fraction_pre_event_flagged_dangerous"] = None

    return {"trajectory": traj, "summary": summary, "experiment": exp}


def plot_experiment(exp_id: int, result: Dict[str, object]) -> Path:
    exp = result["experiment"]
    traj: pd.DataFrame = result["trajectory"]
    summary = result["summary"]
    frame = exp.frame

    fig, axes = plt.subplots(4, 1, figsize=(9, 11), sharex=True)

    ax = axes[0]
    for ch, label in [("MidIntTemp", "internal (mid)"), ("MidSurfTemp", "surface (mid)"),
                       ("NegSurfTemp", "surface (-term)"), ("PosSurfTemp", "surface (+term)")]:
        ax.plot(frame["time_s"], frame[ch], label=label, linewidth=1.2)
    if summary["temp_runaway_onset_s"] is not None:
        ax.axvline(summary["temp_runaway_onset_s"], color="crimson", linestyle="--", label="runaway-onset proxy")
    ax.set_ylabel("Temperature (°C)")
    ax.set_title(f"Experiment {exp_id} — real measurements (Warwick/Faraday dataset)")
    ax.legend(fontsize=7, ncol=2)

    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(frame["time_s"], frame["CellVoltage"], color="tab:green", linewidth=1.2, label="Cell voltage")
    ax2.plot(frame["time_s"], frame["IntPre"], color="tab:purple", linewidth=1.0, label="Internal gas pressure")
    if summary["pressure_vent_onset_s"] is not None:
        ax.axvline(summary["pressure_vent_onset_s"], color="darkorange", linestyle="--", label="vent-onset proxy")
    ax.set_ylabel("Voltage (V)", color="tab:green")
    ax2.set_ylabel("Pressure (bar, gauge)", color="tab:purple")

    ax = axes[2]
    ax.plot(traj["time_s"], traj["p_healthy"], label="Healthy")
    ax.plot(traj["time_s"], traj["p_abnormal"], label="Abnormal Heating")
    ax.plot(traj["time_s"], traj["p_pre_runaway"], label="Pre-Runaway")
    ax.plot(traj["time_s"], traj["p_thermal_runaway"], label="Thermal Runaway")
    ax.set_ylabel("Posterior")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, ncol=4)

    ax = axes[3]
    ax.plot(traj["time_s"], traj["p_dangerous"], color="crimson", label="P(dangerous)")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1.0, label="alarm threshold")
    if summary["temp_runaway_onset_s"] is not None:
        ax.axvline(summary["temp_runaway_onset_s"], color="crimson", linestyle="--")
    if summary.get("first_alarm_time_s") is not None:
        ax.axvline(summary["first_alarm_time_s"], color="steelblue", linestyle="--", label="first alarm")
    ax.set_ylabel("P(dangerous)")
    ax.set_xlabel("Experiment time (s)")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7)

    fig.suptitle(
        "SIMULATION-FREE EXTERNAL VALIDATION — real Sony VTC6A thermal-runaway data\n"
        "Model parameters are UNCHANGED from the synthetic-calibrated CellSafe-X filter.",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = OUT_DIR / f"experiment_{exp_id}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path


def main() -> int:
    if not raw_data_available():
        print(
            "Real dataset not found. Run: "
            "python -m data.external.warwick_thermal_runaway.download"
        )
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for exp_id in range(N_EXPERIMENTS):
        print(f"Replaying experiment {exp_id}...")
        result = replay_experiment(exp_id)
        result["trajectory"].to_csv(OUT_DIR / f"experiment_{exp_id}_trajectory.csv", index=False)
        # Small (~100-130 row) 10-second-resampled input table, committed so
        # the replay is reproducible without re-downloading the 215 MB
        # source file or regenerating the ~140 MB raw-resolution cache.
        result["experiment"].frame.to_csv(OUT_DIR / f"experiment_{exp_id}_resampled_input.csv", index=False)
        plot_path = plot_experiment(exp_id, result)
        print(f"  wrote {plot_path}")
        all_summaries.append(result["summary"])

    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "data_source": "REAL - Warwick/Faraday Institution thermal-runaway dataset "
                "(DOI 10.17632/rgfhdhcd9k.1), NOT the synthetic simulator",
                "track": "external_validation_only_not_merged_with_synthetic_metrics",
                "experiments": all_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSummary written to {summary_path}")
    for s in all_summaries:
        print(
            f"  exp{s['experiment_id']}: max P(dangerous)={s['max_p_dangerous']:.3f} "
            f"at t={s['max_p_dangerous_time_s']:.0f}s, "
            f"runaway-onset proxy={s['temp_runaway_onset_s']}, "
            f"first alarm={s['first_alarm_time_s']}, "
            f"lead time={s['warning_lead_time_s']}"
        )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
