# CellSafe-X — Evaluation

All numbers below are from `python -m evaluation.evaluate` on SIMULATED data
(`data/battery_simulator.py`) — 6 scenarios × 5 seeds (base seed 20240) × 6
cells × 180 steps = 32,400 (cell, step) samples, rotating the affected cell
across seeds. Ground truth comes from the simulator's *internal* physical
state, invisible to the pipeline, so the filter is scored on recovering a
latent variable from noisy evidence, not against its own output.

## Final metrics (current, post Phase 1–4)

| Metric | Value |
|---|---|
| Hidden-state accuracy | 0.8840 |
| Within-one-band accuracy | 0.9806 |
| Macro precision | 0.8091 |
| Macro recall | 0.8928 |
| Macro F1 | 0.8425 |
| Multiclass Brier (raw → calibrated) | 0.2152 → 0.1933 |
| Negative log-likelihood (raw → calibrated) | 1.1217 → 0.4867 |
| Expected calibration error (raw → calibrated) | 0.0413 → 0.0468 (disclosed regression — see below) |
| Root-cause accuracy, affected cell (first third → whole run → last third) | 0.607 → 0.869 → 1.000 |
| False alarms (Bayesian / threshold-warn / threshold-critical) | 1,081 / 1,112 / 710 |
| Latent-state safety misses (sample-level, ground truth dangerous, not alarmed) | 364 |
| ...of which thermal-threshold misses (genuinely ≥58 °C, the safety-critical subset) | **0** |

Per scenario:

| Scenario | Accuracy | ±1 band | Brier | Cause (late) |
|---|---|---|---|---|
| Normal Operation | 0.997 | 1.000 | 0.005 | 1.000 |
| Temperature Sensor Fault | 0.997 | 1.000 | 0.005 | 1.000 |
| Fast Charging | 0.967 | 1.000 | 0.060 | 1.000 |
| External Heat Exposure | 0.770 | 1.000 | 0.426 | 1.000 |
| Internal Short Circuit | 0.769 | 0.884 | 0.451 | 1.000 |
| Cooling-System Failure | 0.803 | 1.000 | 0.344 | 1.000 |

Detection timing (runs that reach a dangerous ground-truth state):

| Scenario | n | Missed | Mean delay | Median delay |
|---|---|---|---|---|
| Cooling-System Failure | 30 | 0 | **-0.36 min (early warning)** | -1.25 min |
| External Heat Exposure | 30 | 0 | 2.38 min | 2.25 min |
| Internal Short Circuit | 5 | 0 | -0.43 min | -0.33 min |
| Fast Charging | 25 | 25 | n/a (label-flicker only, never physically hot — see below) | n/a |

Metric definitions (kept separate, never conflated; two different
granularities, both computed by `evaluation/evaluate.py`):
- **Latent-state safety miss (sample-level)** — a (cell, step) sample where
  ground truth ∈ {Pre-Runaway, Thermal Runaway} and `P(dangerous) ≤ 0.5` at
  that instant (`alarm_comparison.bayesian.missed_dangerous_breakdown.missed_total`,
  364 samples). This is defined purely by the LATENT STATE label, regardless
  of whether the cell was ever physically hot.
- **Thermal-threshold miss** — the SUBSET of the above where `temp_true ≥
  LABEL_PRE_C` (58 °C) - i.e. the cell really was hot, not just
  rate-escalation-labelled
  (`missed_dangerous_breakdown.missed_temp_actually_crossed_pre_runaway_threshold`,
  **0** in every evaluated run). This is the safety-critical number.
  Temperature is never used to *overrule* the latent-state label, only to
  annotate whether a miss was ever physically hot.
- **Run-level miss** (`detection_timing.*.n_missed` in the table above) is a
  STRICTER, separate metric: a whole (scenario, seed, cell) trajectory where
  the alarm never fires even once. Fast Charging's 25 "missed" runs are all
  label-flicker segments that never reach 58 °C (see the Phase-2 audit
  below), not a run where a genuinely hot cell went undetected.

## Baseline comparison (CellSafe-X vs. fixed-threshold BMS)

| System | False alarms | Missed dangerous events | Precision |
|---|---|---|---|
| CellSafe-X Bayesian, P(dangerous) > 0.5 | 1,081 | 364 (0 genuinely hot) | 0.834 |
| Threshold, warn (55 °C) | 1,112 | 637 | 0.823 |
| Threshold, critical (70 °C) | 710 | 2,735 | 0.812 |

Sensor-fault scenario, affected cell only (900 samples, pack genuinely
healthy throughout, so every alarm is a false alarm by construction):
CellSafe-X Bayesian **0** false alarms vs. threshold-critical **710**.

CellSafe-X is not uniformly better: it raises more *total* false alarms than
the critical threshold because it alarms earlier and at a lower bar. The
honest comparison is what each system represents, not a single scoreboard —
**CellSafe-X trades some simplicity for probabilistic uncertainty awareness,
diagnosis, forecasting and decision support**; a fixed threshold has none of
sensor-fault robustness, root-cause diagnosis, multi-step forecasting,
intervention comparison, propagation risk or active-sensing guidance.

## Root-cause confusion matrix

`results/metrics/root_cause_confusion_matrix.csv` (6×6, affected cell only).
Mean posterior entropy 0.157 nats of a possible ln(6) = 1.792 — confidently
peaked, consistent with the late-stage separability finding below.

## Calibration methodology

Temperature scaling (`models.bayesian_filter.calibrated_posterior`),
reporting-only — never fed back into `BayesianFilter.update`, `forecast_risk`
or `recommend_action` (its only caller anywhere in the codebase is
`evaluation/evaluate.py`).

- Fit partition: 3 seeds × 6 scenarios × 6 cells × 180 steps = 19,440 samples,
  `seed_base=90000`.
- Evaluation partition: 5 seeds × 6 × 6 × 180 = 32,400 samples,
  `seed_base=20240` — disjoint by construction
  (`CALIBRATION_SEED_BASE >= BASE_SEED + 1000`, asserted in
  `tests/test_evaluation.py`).
- Temperature chosen by a deterministic grid+refine minimising NLL on the
  calibration partition only, then applied unchanged to the evaluation
  partition. Fitted temperature ≈ 3.75.
- Disclosed trade-off: NLL and Brier improve substantially; ECE gets
  *slightly worse* (0.0413 → 0.0468). Temperature scaling minimises NLL, not
  ECE, and is not guaranteed to improve every calibration metric at once —
  reported honestly rather than hidden.

## Phase-by-phase audit trail

### Phase 1 — model and evaluation hardening

Diagnosed via per-channel log-likelihood decomposition:
- **Fast-Charging false-danger spike.** `temp_rate` is an EMA-smoothed slope,
  not a fresh 10 s reading. During the benign current-ramp onset its tempered
  log-likelihood swing reached ~48 nats — 5–10× every other channel — driving
  `P(dangerous)` to 1.00 while `temp_c` was still 41 °C. Grid search over
  `OBS_LIKELIHOOD_WEIGHTS["temp_rate"]` found a phase transition between 0.40
  (worst case 0.88) and 0.35 (worst case 0.29, below the alarm threshold).
  **Weight lowered 0.60 → 0.35.**
- **Cooling-Failure lag** traced to the transition matrix's strong diagonal
  interacting with a genuinely slow (14-minute) ramp — not a fixable
  emission-model bug (within-one-band accuracy stayed 1.000 throughout). No
  transition-matrix change was made.
- **Overconfidence** fixed with post-hoc temperature scaling (see above).
- **Root-cause 1.000 (final third)** investigated, not "fixed" — the six
  scenarios were deliberately given distinct late-stage signatures.

Before → after: accuracy 0.8608 → 0.8799, Brier 0.2546 → 0.2232, false alarms
1,349 → 1,061, Fast-Charging worst-case spike 1.00 → 0.29. **Safety check:**
every one of the missed-event samples, before and after, had `temp_true <
58 °C` — 0 genuinely hot misses in either configuration.

### Phase 2 — evaluation integrity and simulator label-stability audit

Audited the 566 Phase-1 missed events by ground-truth segment context:

| Category | Count (of 566) |
|---|---|
| Boundary/transition flicker (segment ≤3 steps, never physically hot) | 245 |
| Sustained but never physically hot (segment >3 steps, temp <58 °C) | 151 |
| Legitimate early dangerous-state onset (segment >3 steps, later reaches 58 °C) | 170 |
| Genuinely hazardous (temp_true ≥ 58 °C or true Runaway) | **0** |

Root cause: `_label_state`'s rate-escalation rule fired on a single noisy 10 s
finite difference (~0.6 °C/min 1-sigma noise). 288 dangerous-state
ground-truth segments existed; median length **1 step**; 405 one-step A→B→A
reversals. **Fix (simulator semantics):** escalation now requires the
elevated rate to persist ≥2 consecutive steps (20 s). Physical thresholds
untouched.

Effect: 288 → 145 segments, median length 1 → 8 steps, reversals 405 → 70,
missed events 566 → 364 (still 0 genuinely hazardous). Accuracy 0.8799 →
0.8840. Cooling-Failure's mean detection delay flipped from +5.4 min lag to
**-0.36 min (early warning)** once its ground truth stopped flickering.

Realism audit found `cooling_eff`/`current` are near-deterministic per-scenario
scalars (measurement noise only, no cross-seed variation) — this, not the
classifier, is why final-third root-cause accuracy reaches 1.000. Not changed
this phase (would need randomising per-seed scenario severity, out of scope
for a "minimal justified correction").

### Phase 3 — model-based intervention forecasting

Replaced the flat `INTERVENTION_EFFECTIVENESS` table with the
controlled-transition forecast (see ARCHITECTURE.md). Representative
counterfactuals (15-minute horizon):

| Scenario (step) | No action, +15min | Best action | Under action | Risk reduction |
|---|---|---|---|---|
| Normal Operation (t=15min) | 0.046 | Emergency Shutdown | 0.000 | 0.046 |
| Fast Charging (t=5min) | 0.189 | Emergency Shutdown | 0.004 | 0.185 |
| Cooling-System Failure (t=25min) | 0.652 | Emergency Shutdown | 0.143 | 0.509 |
| Internal Short Circuit (t=15min) | 0.977 | Emergency Shutdown | 0.913 | 0.065 |
| External Heat Exposure (t=25min) | 0.977 | Emergency Shutdown | 0.913 | 0.065 |

7,500 randomised (belief, horizon) monotonicity checks: 0 violations.

### Phase 4 — cross-cell propagation and active sensing

Propagation examples (15-minute horizon):

| Case | Source → neighbour | Baseline | Coupled | Increment |
|---|---|---|---|---|
| One dangerous cell, healthy neighbours (Internal Short) | 2→1 | 0.548 | 0.664 | 0.117 |
| One dangerous cell, healthy neighbours (Internal Short) | 2→3 | 0.471 | 0.595 | 0.124 |
| Cooling-System Failure (whole pack stressed) | 2→1 | 0.652 | 0.759 | 0.107 |
| Non-adjacent stressed cells (0 and 3 dangerous) | 3→2 | 0.058 | 0.097 | 0.039 |
| Isolation applied to source (Internal Short, cell 2) | 2→1 | – | 0.559 (was 0.664) | – |

Active-sensing (EIG) examples, `temp_c` channel:

| Posterior | Entropy (nats) | EIG (nats) | Recommend measuring? |
|---|---|---|---|
| Ambiguous [.3,.4,.2,.1] | 1.280 | 0.720 | Yes |
| Uniform [.25×4] | 1.386 | 0.848 | Yes |
| Confident [~0,~0,.003,.995] | 0.036 | 0.025 | No (confident + emergency-level danger) |

## Outputs

`results/metrics/evaluation_metrics.json`, `per_scenario_metrics.csv`,
`confusion_matrix.csv`, `root_cause_confusion_matrix.csv`,
`alarm_comparison.csv`, `detection_timing.csv`, `affected_cell_samples.csv`,
and three figures in `results/figures/`.

## Honest reading

- **Accuracy is 0.88, not 0.99, and that is the point** — the emission
  distributions overlap heavily by design; narrowing them would inflate
  accuracy while measuring nothing but synthetic separability.
- **The filter is over-confident**; its argmax is trustworthy, its exact
  probabilities near 0/1 should be read as "very likely", not literally.
- **Root-cause 1.000 late-stage is a synthetic-data artifact**, disclosed
  above, not a claim of real-world diagnostic perfection.
- **Calibration improves NLL/Brier but not ECE** — reported, not hidden.
