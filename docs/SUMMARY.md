# CellSafe-X — Final Project Summary (Faculty / Viva Preparation)

## Problem

A classical battery-management system trips on one number: primary-sensor
temperature. It cannot tell a cell heading into thermal runaway from a cell
with a broken thermistor — both look "very hot" to a single sensor. It must
either shut down on a false reading or raise the threshold and risk missing
a real event. There is no representation of uncertainty, no diagnosis of
*why* a cell is hot, and no forecasting of what happens next.

## Probabilistic model

- **Latent state** `Z_t ∈ {Healthy, Abnormal Heating, Pre-Runaway, Thermal
  Runaway}` per cell, exact 4-state HMM/one-slice-DBN forward filtering in
  log space.
- **Evidence** — five state-conditional Gaussian channels (temperature,
  heating rate, voltage deviation, vent gas, neighbour temperature), with a
  separate binary sensor-reliability hypothesis (reliable/faulty) fused in
  before the state filter ever sees the reading.
- **Diagnosis** — six-cause Bayesian root-cause classifier over anomaly
  features (residual not explained by temperature alone).
- **Decision** — minimum-expected-loss action selection over six actions and
  an explicit cost matrix, not a threshold ladder.

## Novel contributions (defensible)

1. A from-scratch, exact, numerically-stable Bayesian filter with a
   documented, tested emission-tempering correction for a known naive-Bayes
   over-confidence failure mode.
2. A three-stage audit-and-fix process (Phase 1–2) that traced a severe
   false-alarm bug to a specific channel's log-likelihood dominance, fixed it
   with a measured, grid-searched correction, and then traced a *ground-truth
   labelling* instability to the same root mechanism — fixing the simulator's
   semantics rather than post-processing labels.
3. A model-based, controlled-transition intervention forecast that is
   explicit about NOT being Pearlian causal inference, with monotonicity
   invariants checked over 7,500 randomised cases.
4. A cross-cell propagation forecasting layer and an EIG/EVI active-sensing
   module built on the *same* emission likelihood the filter already uses —
   no invented sensor, no arbitrary weights.
5. An evaluation suite that separates latent-state safety misses from
   thermal-threshold misses, reports calibration honestly (including where it
   gets worse), and audits its own missed-event counts down to individual
   ground-truth segments rather than reporting one aggregate number.

## Evaluation (synthetic, 32,400 samples)

Accuracy 0.884, macro F1 0.843, Brier 0.215→0.193 (calibrated), NLL 1.122→
0.487 (calibrated), 0 genuinely-hot missed dangerous events. Full table:
`docs/EVALUATION.md`.

## One important success

The Fast-Charging false-danger spike: a benign 2-minute current ramp used to
drive `P(dangerous)` to 1.00 (worst case, pre-Phase-1) purely from a
transient heating-rate reading, while absolute temperature stayed at 41 °C.
Diagnosed to a single over-weighted emission channel, corrected with a
measured (not guessed) weight change, verified to *improve* — not just
preserve — Cooling-Failure and Internal-Short accuracy as a side effect.
Worst case now 0.29, safely below the alarm threshold.

## One honest weakness

Cells are filtered independently; Phase 4's propagation forecast is a
one-step, documented approximation on top of that (each neighbour's own
independent forecast drives pressure on its neighbours — not a solved joint
fixed point), and every constant governing it is engineered, not fitted to
real or even richly-simulated multi-cell abuse data. The whole project is
synthetic-only: no number here has been validated against a real cell.

## Future work

Learn parameters from labelled data instead of hand-specifying them; a
properly fitted calibration that also improves ECE; a factored joint
multi-cell DBN (or particle filter) to replace the independent-cell +
propagation-forecast approximation; a structural causal model so intervention
effects could be described as truly causal; real telemetry and instrumented
abuse-test validation. Full list: `docs/LIMITATIONS.md`.
