# CellSafe-X

### A Bayesian Battery Safety Digital Twin for Thermal-Runaway Diagnosis, Forecasting and Decision Support

> **SIMULATION-BASED RESEARCH PROTOTYPE.** Every reading, scenario and
> evaluation number comes from a synthetic six-cell simulator
> (`data/battery_simulator.py`). No real cell, sensor or vehicle data is
> used anywhere, and no parameter here is fitted to or validated against
> real battery measurements. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Overview

A fixed-threshold battery-management system trips on one number — primary
sensor temperature — and cannot distinguish a cell heading into thermal
runaway from a cell with a broken thermistor; both simply read "very hot".
CellSafe-X treats battery safety as what it actually is: **inference of a
hidden thermal state from noisy, sometimes contradictory evidence**. It
filters that state exactly with a 4-state HMM, separately reasons about
whether its own primary sensor can be trusted, diagnoses *why* a cell is hot,
forecasts what happens next (with and without an intervention, and with and
without neighbour propagation), and recommends an action by minimum expected
loss rather than a trip point — all with calibrated, inspectable
probabilities on screen.

## Core contribution

| Capability | Method | Module |
|---|---|---|
| Trust the primary sensor? | Binary Bayesian hypothesis test, 5 residual cues | `models/fault_diagnosis.py` |
| What thermal state is this cell in? | Exact forward filtering, 4-state HMM/one-slice DBN | `models/bayesian_filter.py` |
| Why is it in that state? | Log-space Bayesian classifier, 6 causes | `models/fault_diagnosis.py` |
| What happens next? | Exact k-step forecast, `b_t @ A^k` | `models/risk_forecast.py` |
| What should I do? | Minimum expected loss, 6 actions | `models/decision_engine.py` |
| What if I act? | Model-based controlled-transition forecast (not causal) | `models/intervention.py` |
| Could it spread to another cell? | Topology-aware propagation forecast | `models/propagation.py` |
| Is another measurement worth it? | Expected information gain / decision-loss value | `models/active_sensing.py` |

Full mathematical design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Architecture

```
sensor reading -> reliability inference -> fused temp/rate -> exact HMM filter
   -> root-cause diagnosis -> risk forecast -> expected-loss decision
   -> intervention forecast -> propagation forecast -> active-sensing (EIG/EVI)
   -> Streamlit dashboard
```

Details, folder structure and dashboard page map: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Setup

Python 3.10+, from the `CellSafe-X` folder:

```bash
python -m venv venv
```

Windows PowerShell: `.\venv\Scripts\Activate.ps1` (if blocked, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use
CMD: `venv\Scripts\activate`).

```bash
pip install -r requirements.txt
```

## Run

```bash
python -m evaluation.evaluate   # generate results/metrics first
streamlit run app.py            # http://localhost:8501
```

## Evaluation

```bash
python -m evaluation.evaluate                  # full 5-seed run, 32,400 samples
python -m evaluation.evaluate --seeds 2 --steps 120   # faster pass while iterating
```

Methodology, full metrics, phase-by-phase audit trail and baseline
comparison: [docs/EVALUATION.md](docs/EVALUATION.md).

## Tests

```bash
pytest -q
```

**227 automated tests, all passing** (215 synthetic-track + 12 Phase-6
real-data-adapter tests, some skipped automatically if the real dataset has
not been downloaded).

## Final results

| Metric | Value |
|---|---|
| Hidden-state accuracy | 0.8840 |
| Macro F1 | 0.8425 |
| Brier (raw → calibrated) | 0.2152 → 0.1933 |
| NLL (raw → calibrated) | 1.1217 → 0.4867 |
| ECE (raw → calibrated) | 0.0413 → 0.0468 (disclosed — see docs/EVALUATION.md) |
| Root cause (first third → last third) | 0.607 → 1.000 |
| Latent-state safety misses (sample-level) / thermal-threshold misses (genuinely hot) | 364 / **0** |
| False alarms (Bayesian vs. critical threshold) | 1,081 vs. 710 |
| Missed dangerous events (Bayesian vs. critical threshold) | 364 vs. 2,735 |

Full tables including per-scenario results, detection timing, calibration
bins and the root-cause confusion matrix: [docs/EVALUATION.md](docs/EVALUATION.md).

## Demo scenarios

Six reproducible scenarios (Normal Operation, Fast Charging, Temperature
Sensor Fault, Cooling-System Failure, Internal Short Circuit, External Heat
Exposure), selectable from the dashboard sidebar with a fixed seed. Full
walk-through with exact steps and expected observations:
[docs/DEMO.md](docs/DEMO.md).

## Limitations

All data is synthetic and unvalidated against real cells; parameters are
hand-specified, not learned; cells are filtered independently (propagation is
a forecasting layer on top, not a joint model); the filter is over-confident;
intervention/propagation effects are engineered assumptions, not causal or
fitted. Full, honest list: [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Real experimental external validation

A separate, non-headline track. `evaluation/real_validation.py` replays
three real, externally-heated Sony VTC6A thermal-runaway experiments
(Warwick/Faraday Institution, Mendeley DOI `10.17632/rgfhdhcd9k.1`, CC BY
4.0) through the **unchanged** CellSafe-X filter — **synthetic simulator**
results (above) are for controlled model development and the full
quantitative benchmark; this **real dataset** is used only for
event-aligned validation on three actual experiments, with no exact
four-state ground truth, no pack topology, and no claim of deployment-level
generalisation. CellSafe-X was not trained or refit on this data. Run:

```bash
python data/external/warwick_thermal_runaway/download.py   # one-time, 215 MB
python -m evaluation.real_validation
```

Full write-up, provenance, field mapping and honest results:
[docs/REAL_VALIDATION.md](docs/REAL_VALIDATION.md).

## Documentation set

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — pipeline, math model, folder structure, dashboard
- [docs/EVALUATION.md](docs/EVALUATION.md) — methodology, final metrics, phase-by-phase audit trail
- [docs/REAL_VALIDATION.md](docs/REAL_VALIDATION.md) — real-experiment external validation (separate track)
- [docs/DEMO.md](docs/DEMO.md) — deterministic demo walk-through
- [docs/LIMITATIONS.md](docs/LIMITATIONS.md) — limitations and future work
- [docs/VIVA.md](docs/VIVA.md) — viva questions and concise answers
- [docs/SUMMARY.md](docs/SUMMARY.md) — one-page faculty/viva summary
