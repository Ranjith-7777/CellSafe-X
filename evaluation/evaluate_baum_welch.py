"""
Phase 7C — Baum-Welch training + engineered-vs-learned held-out comparison.

Run with:      python -m evaluation.evaluate_baum_welch

Never touches results/metrics/evaluation_metrics.json (the canonical,
frozen Phase 1-6 artifact) or config/model_parameters.py (the frozen
engineered HMM). Writes to results/baum_welch/.

SEED RANGES (trajectory-level split, never mixed):
  train      seed_base=40000, n_seeds=3   (EM parameter fitting)
  validation seed_base=50000, n_seeds=1   (state-label alignment only -
                                            ground truth used HERE, never
                                            inside the E/M updates)
  test       seed_base=20240, n_seeds=5   (the canonical evaluation seed
                                            range - held out from training
                                            and validation, used only for
                                            the final engineered-vs-learned
                                            comparison)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import model_parameters as P
from data.battery_simulator import simulate_pack
from evaluation.evaluate import alarm_counts, compute_metrics
from models import CellPipeline
from models.baum_welch import (
    LearnedHMMFilter,
    align_states,
    allowed_transition_mask,
    e_step,
    engineered_params,
    perturbed_init,
    run_baum_welch,
)
from models.bayesian_filter import BayesianFilter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "results" / "baum_welch"

TRAIN_SEED_BASE, N_TRAIN_SEEDS = 40000, 3
VAL_SEED_BASE, N_VAL_SEEDS = 50000, 1
TEST_SEED_BASE, N_TEST_SEEDS = 20240, 5  # canonical evaluation range
N_STEPS = 180


def _obs_streams_for_run(scenario: str, seed: int, affected: int):
    """One simulate_pack call per (scenario, seed) - matching
    evaluation/evaluate.py's collect_records EXACTLY (a single shared pack
    simulation with one fixed affected cell; the other five cells in that
    SAME run are genuinely healthy). Returns {cell: (obs_stream, truth,
    temp_true)} for all six cells of that one run.

    Same fused observation stream the engineered filter consumes
    (temp_c/temp_rate/voltage_dev/log_gas/neighbour_c), built via
    CellPipeline's unmodified "legacy" fusion path so training/test features
    are identical to what the production model sees. temp_true (ground-truth
    physical temperature) is also returned so held-out misses can be split
    into label-artifact vs genuinely-hot, exactly like evaluate.py.
    """
    sim = simulate_pack(scenario=scenario, affected_cell=affected, seed=seed, n_steps=N_STEPS)
    out_by_cell = {}
    for cell in range(P.N_CELLS):
        frame = sim.frame[sim.frame["cell"] == cell].sort_values("step").reset_index(drop=True)
        pipe = CellPipeline(cell=cell, temperature_fusion="legacy")
        obs_stream, truth, temp_true = [], [], []
        for _, row in frame.iterrows():
            out = pipe.step(row)
            obs_stream.append({
                "temp_c": out.fused_temp_c,
                "temp_rate": out.temp_rate,
                "voltage_dev": abs(out.reading["voltage_dev"]),
                "log_gas": out.reading["log_gas"],
                "neighbour_c": out.reading["neighbour_temp"],
            })
            truth.append(int(row["true_state"]))
            temp_true.append(float(row["temp_true"]))
        out_by_cell[cell] = (obs_stream, truth, temp_true)
    return out_by_cell


def _collect(seed_base: int, n_seeds: int):
    sequences, truths = [], []
    for k in range(n_seeds):
        seed = seed_base + k
        affected = k % P.N_CELLS
        for scenario in P.SCENARIOS:
            by_cell = _obs_streams_for_run(scenario, seed, affected)
            for cell in range(P.N_CELLS):
                obs, truth, _ = by_cell[cell]
                sequences.append(obs)
                truths.append(truth)
    return sequences, truths


def _predict_frame(params_or_none, use_learned: bool, sequences, truths, temp_trues, meta) -> pd.DataFrame:
    rows = []
    for seq, truth, temp_true, (scenario, seed, cell) in zip(sequences, truths, temp_trues, meta):
        if use_learned:
            f = LearnedHMMFilter(params_or_none)
            posteriors = [f.update(o) for o in seq]
        else:
            bf = BayesianFilter()
            posteriors = [bf.update(o).posterior for o in seq]
        for t, (p, y, tt) in enumerate(zip(posteriors, truth, temp_true)):
            rows.append({
                "scenario": scenario, "seed": seed, "cell": cell, "step": t,
                "time_min": t * P.DT_MINUTES,
                "is_affected": True,  # all cells scored; matches ablation-style scoring
                "true_state": y,
                "pred_state": int(np.argmax(p)),
                "p0": float(p[0]), "p1": float(p[1]), "p2": float(p[2]), "p3": float(p[3]),
                "p_dangerous": float(p[P.S_PRE] + p[P.S_RUNAWAY]),
                "true_dangerous": bool(y in P.DANGEROUS_STATES),
                "temp_true": tt,
                # Root-cause diagnosis is out of scope for this HMM-only
                # comparison; these placeholders are valid CAUSES labels
                # (so sklearn's confusion_matrix doesn't choke) but always
                # mismatched, so any root-cause figure in the output reads
                # as 0% rather than a fabricated 100% - not reported below.
                "top_cause": "Internal Short Circuit", "true_cause": "Normal/Fast-Charging Heat",
                "cause_posterior_entropy": 0.0,
                "p_sensor_faulty": 0.0, "sensor_faulty_true": False,
                "temp_primary": np.nan,
                "threshold_triggered_warn": False, "threshold_triggered_critical": False,
                "recommended_action": "n/a",
            })
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Collecting TRAIN sequences (seed_base={TRAIN_SEED_BASE}, n_seeds={N_TRAIN_SEEDS})...")
    train_seqs, _ = _collect(TRAIN_SEED_BASE, N_TRAIN_SEEDS)
    print(f"  {len(train_seqs)} trajectories")

    print(f"Collecting VALIDATION sequences (seed_base={VAL_SEED_BASE}, n_seeds={N_VAL_SEEDS})...")
    val_seqs, val_truth = _collect(VAL_SEED_BASE, N_VAL_SEEDS)

    mask = allowed_transition_mask()
    results = {}
    for init_name, init_params in [("engineered_init", engineered_params()), ("perturbed_init", perturbed_init(seed=7))]:
        print(f"Running Baum-Welch from {init_name}...")
        params, history = run_baum_welch(train_seqs, init_params, mask=mask, max_iter=40, tol=1e-2)
        results[init_name] = {"params": params, "history": history}
        print(f"  iterations={history.iterations} converged={history.converged} "
              f"initial_ll={history.log_likelihood[0]:.1f} final_ll={history.log_likelihood[-1]:.1f}")

    # Select the better-converged initialisation by TRAINING log-likelihood only.
    best_name = max(results, key=lambda k: results[k]["history"].log_likelihood[-1])
    print(f"\nSelected initialisation: {best_name} (by training log-likelihood)")
    learned_raw = results[best_name]["params"]

    print("Aligning learned states to semantic labels using VALIDATION ground truth...")
    learned, permutation = align_states(learned_raw, val_seqs, val_truth)
    print(f"  permutation (learned_idx -> semantic_idx): {permutation.tolist()}")

    # --- convergence CSV -----------------------------------------------
    conv_rows = []
    for name, r in results.items():
        for it, (ll, d) in enumerate(zip(r["history"].log_likelihood, r["history"].delta), start=1):
            conv_rows.append({"initialization": name, "iteration": it, "log_likelihood": ll, "delta": d})
    pd.DataFrame(conv_rows).to_csv(OUT_DIR / "convergence.csv", index=False)

    # --- held-out test ---------------------------------------------------
    print(f"\nCollecting TEST sequences (seed_base={TEST_SEED_BASE}, n_seeds={N_TEST_SEEDS}, "
          f"disjoint from train/validation)...")
    meta = []
    test_sequences: List[List[dict]] = []
    test_truths: List[List[int]] = []
    test_temp_trues: List[List[float]] = []
    for k in range(N_TEST_SEEDS):
        seed = TEST_SEED_BASE + k
        affected = k % P.N_CELLS
        for scenario in P.SCENARIOS:
            by_cell = _obs_streams_for_run(scenario, seed, affected)
            for cell in range(P.N_CELLS):
                obs, truth, temp_true = by_cell[cell]
                test_sequences.append(obs)
                test_truths.append(truth)
                test_temp_trues.append(temp_true)
                meta.append((scenario, seed, cell))
    print(f"  {len(test_sequences)} test trajectories, {sum(len(s) for s in test_sequences):,} samples")

    df_engineered = _predict_frame(None, False, test_sequences, test_truths, test_temp_trues, meta)
    df_learned = _predict_frame(learned, True, test_sequences, test_truths, test_temp_trues, meta)

    metrics_engineered = compute_metrics(df_engineered)
    metrics_learned = compute_metrics(df_learned)

    from models.baum_welch import forward_backward

    def _engineered_seq_loglik(seq: List[dict]) -> float:
        bf = BayesianFilter()
        return float(sum(bf.update(o).log_evidence for o in seq))

    avg_seq_ll_engineered = float(np.mean([_engineered_seq_loglik(seq) for seq in test_sequences]))
    avg_seq_ll_learned = float(np.mean([forward_backward(seq, learned)[3] for seq in test_sequences]))

    (OUT_DIR / "metrics_engineered.json").write_text(json.dumps(metrics_engineered, indent=2), encoding="utf-8")
    (OUT_DIR / "metrics_learned.json").write_text(json.dumps(metrics_learned, indent=2), encoding="utf-8")

    def _params_to_json(p):
        return {
            "pi": p.pi.tolist(), "A": p.A.tolist(),
            "means": {c: v.tolist() for c, v in p.means.items()},
            "stds": {c: v.tolist() for c, v in p.stds.items()},
        }

    (OUT_DIR / "learned_parameters.json").write_text(
        json.dumps({
            "selected_initialization": best_name,
            "permutation_learned_to_semantic": permutation.tolist(),
            "params": _params_to_json(learned),
        }, indent=2), encoding="utf-8",
    )
    (OUT_DIR / "engineered_parameters.json").write_text(
        json.dumps(_params_to_json(engineered_params()), indent=2), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print(f"{'Metric':32s} {'Engineered':>15s} {'Baum-Welch':>15s}")
    print("-" * 70)
    hs_e, hs_l = metrics_engineered["hidden_state"], metrics_learned["hidden_state"]
    for key in ["accuracy", "macro_precision", "macro_recall", "macro_f1", "multiclass_brier", "negative_log_likelihood"]:
        print(f"{key:32s} {hs_e[key]:15.4f} {hs_l[key]:15.4f}")
    print(f"{'ECE (raw)':32s} {metrics_engineered['calibration']['expected_calibration_error']:15.4f} "
          f"{metrics_learned['calibration']['expected_calibration_error']:15.4f}")
    ac_e = metrics_engineered["alarm_comparison"]["bayesian"]
    ac_l = metrics_learned["alarm_comparison"]["bayesian"]
    print(f"{'false_alarms':32s} {ac_e['false_alarms']:15d} {ac_l['false_alarms']:15d}")
    mb_e, mb_l = ac_e["missed_dangerous_breakdown"], ac_l["missed_dangerous_breakdown"]
    print(f"{'missed (latent-state)':32s} {mb_e['missed_total']:15d} {mb_l['missed_total']:15d}")
    print(f"{'...genuinely hot':32s} "
          f"{mb_e['missed_temp_actually_crossed_pre_runaway_threshold']:15d} "
          f"{mb_l['missed_temp_actually_crossed_pre_runaway_threshold']:15d}")
    print(f"{'avg sequence log-likelihood':32s} {avg_seq_ll_engineered:15.2f} {avg_seq_ll_learned:15.2f}")

    print("\nPer-scenario accuracy:")
    for scenario in P.SCENARIOS:
        va = metrics_engineered["per_scenario"][scenario]["accuracy"]
        vb = metrics_learned["per_scenario"][scenario]["accuracy"]
        print(f"  {scenario:28s} engineered={va:.4f}  baum_welch={vb:.4f}")
    print("=" * 70)
    print(f"\nWritten to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
