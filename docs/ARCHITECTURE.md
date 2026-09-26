# CellSafe-X — Architecture

> All data is SIMULATED (`data/battery_simulator.py`). Nothing here is fitted
> to or validated against a real cell. See [LIMITATIONS.md](LIMITATIONS.md).

Mermaid pipeline and state-transition diagrams: `assets/architecture.md`.

## Pipeline

```
raw sensor reading (per cell, per step)
    -> sensor-reliability inference        models/fault_diagnosis.py
    -> fused temperature + smoothed rate
    -> exact Bayesian filter over Z_t       models/bayesian_filter.py
    -> root-cause posterior                 models/fault_diagnosis.py
    -> multi-step risk forecast             models/risk_forecast.py
    -> minimum-expected-loss action         models/decision_engine.py
    -> controlled-transition intervention forecast   models/intervention.py
    -> cross-cell propagation forecast      models/propagation.py
    -> active-sensing / Value of Information  models/active_sensing.py
```

The reliability block runs FIRST: the hidden-state filter must not be fed a
temperature it has no reason to trust. The root-cause block, by contrast, is
given the RAW primary reading, because a sensor fault is only diagnosable
while its symptom is still visible.

## Folder structure

```
CellSafe-X/
├── app.py                          entry point: config, theme, session, navigation
├── conftest.py                     puts the project root on sys.path for pytest
├── requirements.txt
├── README.md
├── docs/                           this file + EVALUATION/DEMO/LIMITATIONS/VIVA/SUMMARY
├── config/
│   └── model_parameters.py         EVERY prior, transition, Gaussian, loss, intervention,
│                                    topology, propagation and active-sensing constant
├── data/
│   └── battery_simulator.py        six-cell synthetic pack, six scenarios
├── models/
│   ├── __init__.py                 CellPipeline - wires filter/diagnosis/forecast/decision
│   ├── bayesian_filter.py          exact forward filtering + posterior calibration
│   ├── fault_diagnosis.py          sensor reliability + root-cause posteriors
│   ├── risk_forecast.py            multi-step dangerous-state probabilities
│   ├── decision_engine.py          expected loss, action choice, threshold baseline
│   ├── intervention.py             controlled-transition intervention forecast (Phase 3)
│   ├── propagation.py              cross-cell propagation forecast (Phase 4)
│   └── active_sensing.py           EIG / EVI active sensing (Phase 4)
├── state/session_manager.py        shared simulation state + PageContext snapshot
├── pages/                          five pages, numbered 1/2/4/5/6 to match filenames
│   ├── 1_home.py                   command centre, three-column
│   ├── 2_live_battery_twin.py      neon battery twin + diagnostic profile
│   ├── 4_bayesian_analysis.py      one step of the recursion, opened up
│   ├── 5_risk_and_decisions.py     forecasting, decisions, interventions, propagation, VoI
│   └── 6_evaluation.py             saved offline results, rendered
├── dashboard/                      styles, layout, charts, components, navigation, twin SVG
├── evaluation/evaluate.py          labelled sequences -> metrics + figures
├── results/                        figures/, metrics/ (evaluation_metrics.json + CSVs)
└── tests/                          215 tests
```

`conftest.py` puts the project root on `sys.path` so `tests/` can import
`config`, `data` and `models` the same way `app.py` does.

### How the UI is wired

`app.py` is a shell only: page config, theme, shared session state, the
persistent left navigation, and the shared simulation controls (scenario,
affected cell, seed, length, Regenerate, Reset) — all rendered once in the
sidebar so every page sees the same simulation.

**The simulation is never rebuilt by navigating.** `state/session_manager.py`
keys the built twin on a signature of *(scenario, affected cell, seed, number
of steps)*. Changing page leaves that signature untouched, so the timeline
position, the selected cell and every filtered posterior survive the move.

Two implementation notes:
- **Every shared-state mutation is an `on_change`/`on_click` callback**, never
  an inline mutation + `st.rerun()`. Callbacks run *before* the script body,
  so the page renders correctly on the same run; a bare `st.rerun()` under
  `st.navigation` would silently bounce the user back to the default page.
- **Raw HTML goes through `dashboard.layout.render_html`**, which flattens
  indentation first — `st.markdown(..., unsafe_allow_html=True)` still runs
  the string through a Markdown parser, which turns 4-space-indented lines
  into literal code blocks.

## The mathematical model

### Hidden states

`Z_t ∈ {Healthy, Abnormal Heating, Pre-Runaway, Thermal Runaway}`. One step
is 10 seconds of simulated time.

### Prior `P(Z₀)`

`[0.900, 0.070, 0.025, 0.005]` — usually healthy at start, but every state
keeps non-zero mass so the filter reacts quickly and never divides by zero.

### Transition model `P(Z_t | Z_{t-1})`

|            | → Healthy | → Abnormal | → Pre-Runaway | → Runaway |
|------------|-----------|------------|---------------|-----------|
| **Healthy**  | 0.9960 | 0.0040 | 0.0000 | 0.0000 |
| **Abnormal** | 0.0300 | 0.9600 | 0.0100 | 0.0000 |
| **Pre-Runaway** | 0.0000 | 0.0150 | 0.9650 | 0.0200 |
| **Thermal Runaway** | 0.0000 | 0.0000 | 0.0010 | 0.9990 |

Strong diagonal (persistence), only-adjacent forward edges (gradual
deterioration, no teleporting), limited recovery edges, nearly-absorbing
Runaway. Every row sums to exactly 1, asserted at import time.

### Likelihood `P(Y_t | Z_t)`

Five state-conditional Gaussian channels: `temp_c`, `temp_rate`,
`voltage_dev`, `log_gas`, `neighbour_c`. Standard deviations are deliberately
wide so adjacent states overlap substantially — narrow Gaussians would give
artificially perfect evaluation numbers.

`current`, `soc`, `cooling_eff` are excluded from the emission model: they
are *inputs* to the system, not *emissions* of the thermal state (a coolant
pump can fail while the cell is still cold). They feed the root-cause
classifier instead.

Each channel's log-likelihood is tempered by a weight < 1 (`temp_rate` was
corrected 0.60 → 0.35 in Phase 1 — see [EVALUATION.md](EVALUATION.md)) because
the channels are correlated in reality and a raw naive-Bayes product would be
over-confident.

### Posterior — the filtering recursion

`models/bayesian_filter.py :: BayesianFilter.update`:

```
P(Z_t | Y_1:t)  ∝  P(Y_t | Z_t) × Σ_{z_{t-1}} P(Z_t | Z_{t-1}) P(Z_{t-1} | Y_1:t-1)
```

Exact (4×4 matrix-vector product), computed in log space with log-sum-exp
normalisation, so the posterior sums to 1 to float precision and cannot
underflow to NaN. A separate, reporting-only temperature-scaling step
(`calibrated_posterior`) can soften/sharpen a posterior for DISPLAY without
ever feeding back into this recursion.

### Sensor reliability

Binary hypothesis `H ∈ {reliable, faulty}` on the primary thermistor, prior
`P(faulty) = 0.03`, over five residual cues (`backup_delta`,
`neighbour_delta`, `gas_shortfall`, `volt_shortfall`, `jump_residual`). The
gas/voltage cues are deliberately one-sided: only a *shortfall* counts
against the sensor, because *more* gas/voltage than the temperature explains
is evidence of an electrical fault, not a broken thermistor. Fused
temperature: `E[T] = (1 − P(faulty))·T_primary + P(faulty)·T_backup`.

### Root-cause diagnosis

Log-space naive-Bayes classifier over six causes (Normal/Fast-Charging Heat,
Temperature Sensor Fault, Cooling-System Failure, Overcharging, Internal
Short Circuit, External Heat Exposure), fed the RAW primary reading (not the
fused one — a sensor fault must stay visible to be diagnosed). Two of the
eight features are *anomalies* (`vdev_anomaly`, `gas_anomaly` = observed minus
expected-at-this-temperature), which is what makes causes separable: raw gas
and voltage are near-deterministic functions of temperature alone.

### Risk forecasting

`P(Z_{t+k} | Y_{1:t}) = b_t @ A^k`, exact, no future observations assumed.
5/15/30-minute horizons via `A³⁰`, `A⁹⁰`, `A¹⁸⁰`.

### Expected loss

`ExpectedLoss(a) = Σ_s P(Z_t=s|Y_1:t) · L(a,s)`, `a* = argmin_a ExpectedLoss(a)`.
Six actions, four-state loss matrix (0–100 scale) combining safety loss,
battery damage, unavailability, false-shutdown cost and measurement delay.

### Controlled-transition intervention forecast (Phase 3)

CellSafe-X's transition matrix is a modelling assumption, not fit to
interventional data, so intervention effects are a **model-based,
controlled-transition counterfactual**, not Pearlian `do(X)` causal
identification. Each action gets an explicit alternative transition matrix
`A(action)`, built from the baseline by two multipliers
(`escalation_multiplier` on forward/more-dangerous entries,
`recovery_multiplier` on backward/recovery entries), diagonal recomputed so
every row still sums to 1. Forecast: `P_future(danger|evidence,a) = b_t @ A(a)^k`.
`Request Backup Measurement` is the identity (information-gathering, not
physical). Multipliers are monotonic across action "strength" by
construction and tested for it. `compare_interventions()` ranks by risk
reduction only; `recommend_action` (unchanged) ranks by expected loss
including operational cost — the two are never conflated.

### Cross-cell propagation forecast (Phase 4)

Cells stay independently filtered (no joint 4⁶-state DBN). The pack's linear-
chain topology (`config.model_parameters.pack_neighbours`, shared with the
simulator) drives a pressure term on each target cell from its neighbours'
own independent forecasts: `pressure_i = Σ_j COUPLING_STRENGTH · P(dangerous)_j`,
`multiplier_i = min(1 + GAIN·pressure_i, MAX_ESCALATION_MULTIPLIER)`, then the
same scale-and-renormalise transition-matrix construction as Phase 3.
`pack_at_least_one_dangerous` is an explicitly-labelled independence
approximation; `expected_dangerous_cells` is exact (linearity of
expectation). Isolate/Cooling/Shutdown interventions reuse Phase 3's
forecasts rather than inventing a second effect model.

### Active sensing / Value of Information (Phase 4)

`EIG(channel) = H(P(Z)) − E_y[H(P(Z|y))]` (nats), Monte Carlo estimated by
ancestral sampling from the current posterior's predictive mixture and a
single-channel Bayes update using the SAME tempered likelihood the filter
uses — deterministic seed, fixed sample count. Only the five `OBS_CHANNELS`
the filter already has a likelihood for are candidates (`current`/
`cooling_eff` are excluded — no filter likelihood exists for them).
`EVI(channel) = min_a E[Loss(a)] now − E_y[min_a E[Loss(a)|y]]`, decision-loss
units, computed and reported separately from EIG. `Request Backup
Measurement`'s *information* value is quantified here; its intervention
forecast (above) stays exactly neutral — the two capabilities are deliberately
separate.

## Dashboard

Five pages, deep-navy/lime theme, persistent left navigation + shared
simulation controls in the sidebar (scenario, affected cell, seed & length,
Regenerate, Reset).

1. **Overview** (`1_home.py`) — pack health, hidden state, root cause,
   dangerous-state probability, recommended action, six-cell miniature,
   forecast, root-cause/reliability posteriors, threshold comparison.
2. **Digital Twin** (`2_live_battery_twin.py`) — hand-written inline-SVG
   six-cell twin; every visual property is bound to a pipeline quantity
   (fill = SOC, glow = pack danger, cell colour = argmax state, hotspot =
   P(dangerous), conflict styling = sensor-fault flag).
3. **Bayesian Network** (`4_bayesian_analysis.py`) — the filtering recursion
   opened up: prior → prediction → likelihood → posterior, with the
   transition matrix, observation weights and evidence-attribution table.
4. **Analysis** (`5_risk_and_decisions.py`) — forecast, recommendation,
   intervention comparison (model-based, labelled non-causal), pack
   propagation, active sensing (EIG/EVI + gated recommendation), threshold
   comparison.
5. **Results** (`6_evaluation.py`) — saved offline evaluation: accuracy,
   calibration (raw vs. calibrated), confusion matrix, scenario-wise
   results, safety-miss breakdown, root-cause first/last-third accuracy.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'config'` | Run from the `CellSafe-X` folder; use `python -m evaluation.evaluate`, not `python evaluation/evaluate.py`. |
| PowerShell "running scripts is disabled" | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or activate from CMD. |
| `streamlit` not recognised | Activate the venv, or call `.\venv\Scripts\streamlit.exe run app.py` directly. |
| Results page says no results found | Run `python -m evaluation.evaluate`. |
| Port 8501 in use | `streamlit run app.py --server.port 8600`. |
| Dashboard slow on scenario/seed change | Expected — it re-simulates six cells and re-runs the pipeline; reduce "Simulation length" while iterating. |
| `Please replace use_container_width with width` | Upgrade: `pip install -U "streamlit>=1.49"`. |
| Play mode does not advance | It is at the last step — press Reset first. |
| Tests fail with import errors | Ensure `conftest.py` is present at the project root. |
