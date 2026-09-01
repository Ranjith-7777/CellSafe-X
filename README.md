# CellSafe-X

### A Dynamic Bayesian Digital Twin for Thermal-Runaway Diagnosis and Probabilistic Intervention in Electric Vehicle Battery Systems

> **SIMULATION-BASED RESEARCH PROTOTYPE.**
> Every reading, every scenario and every evaluation number in this project comes
> from a synthetic simulator (`data/battery_simulator.py`). No real cell, sensor
> or vehicle data is used anywhere. The model parameters are prototype
> assumptions chosen to demonstrate probabilistic reasoning; **they are not
> certified physical battery parameters** and have not been fitted to real
> battery measurements.

---

## 1. Project overview

A lithium-ion pack heading into thermal runaway and a lithium-ion pack with a
broken thermistor look identical to a classical battery-management system: both
report a very hot cell. A fixed temperature trip point cannot tell them apart,
so it must either shut the vehicle down on a single sensor's word or raise the
threshold and miss real events.

CellSafe-X treats the problem as what it actually is — **inference of a hidden
state from noisy, partially contradictory evidence** — and answers four
questions with explicit probability distributions rather than rules:

| Question | Method | Module |
|---|---|---|
| Can I trust this temperature reading? | Binary Bayesian hypothesis test over five evidence cues | `models/fault_diagnosis.py` |
| What thermal state is this cell in? | Exact forward filtering in a 4-state HMM / one-slice DBN | `models/bayesian_filter.py` |
| Why is it in that state? | Log-space Bayesian classifier over six candidate causes | `models/fault_diagnosis.py` |
| What should I do about it? | Minimum expected loss over six actions | `models/decision_engine.py` |

The result is a live digital twin of a six-cell pack that can hold two beliefs at
once — *"this sensor says 92 °C"* and *"this sensor is almost certainly
broken"* — and act on the second rather than the first.

---

## 2. Mid-review scope

### 2.1 What is implemented

- **Four-state hidden Markov model** (Healthy → Abnormal Heating → Pre-Runaway →
  Thermal Runaway) with an explicit initial distribution, transition matrix and
  state-conditional Gaussian emission model.
- **Exact forward filtering** in log space with log-sum-exp normalisation. No
  sampling, no variational approximation, no external inference library.
- **Bayesian sensor-reliability inference** over five residual cues, producing a
  posterior `P(faulty)` and a posterior-weighted fused temperature.
- **Bayesian root-cause diagnosis** across six causes, with the reliability
  posterior entering as *soft* evidence and a per-feature log-evidence
  attribution table.
- **Exact multi-step risk forecasting** by repeated multiplication of the
  transition matrix (5, 15 and 30-minute horizons).
- **Minimum-expected-loss decision engine** over six actions with a transparent,
  documented cost matrix.
- **Six-cell lumped-thermal simulator** with six reproducible scenarios and
  ground-truth hidden-state labels.
- **Threshold baseline** for side-by-side comparison.
- **Four-tab Streamlit dashboard** (Live Digital Twin, Bayesian Reasoning,
  Evaluation, Model Explanation).
- **Offline evaluation harness** producing accuracy, macro precision/recall/F1,
  confusion matrix, multiclass Brier score, negative log-likelihood, calibration
  curve and an alarm comparison against the threshold baseline.
- **76 automated tests.**

### 2.2 What is intentionally postponed

| Postponed | Why, and what it would take |
|---|---|
| **Parameter learning from data** | All parameters are hand-specified. Baum-Welch / EM on labelled sequences would replace the hand-tuned emission means and transition rates. |
| **Coupled multi-cell DBN** | Cells are filtered independently and interact only through the observed neighbour-temperature channel. A joint model would have 4⁶ = 4096 states and needs factored or approximate inference. |
| **Continuous-state hybrid model** | A switching Kalman filter over actual temperature would give a physical estimate rather than a four-band discretisation. |
| **Real telemetry integration** | Requires CAN-bus / BMS ingest, unit calibration and a real backup-sensor topology. |
| **Formal causal inference** | The current intervention model is an *assumed effectiveness* table (see §8). Genuine counterfactuals would need a structural causal model and identification assumptions. |
| **Calibration correction** | The filter is measurably over-confident (see §10). Temperature scaling or a properly tempered likelihood is the obvious next step. |
| **Validation against real cells** | Out of scope for a software prototype; would require instrumented abuse testing. |

---

## 3. Folder structure

```
CellSafe-X/
├── app.py                          entry point: config, theme, session, navigation
├── conftest.py                     puts the project root on sys.path for pytest
├── requirements.txt
├── README.md                       this file
├── assets/
│   └── architecture.md             Mermaid pipeline + state-transition diagrams
├── config/
│   └── model_parameters.py         EVERY prior, transition, Gaussian and loss value
├── data/
│   ├── __init__.py
│   └── battery_simulator.py        six-cell synthetic pack, six scenarios
├── models/
│   ├── __init__.py                 CellPipeline - wires the blocks together
│   ├── bayesian_filter.py          exact forward filtering + k-step prediction
│   ├── fault_diagnosis.py          sensor reliability + root-cause posteriors
│   ├── risk_forecast.py            multi-step dangerous-state probabilities
│   └── decision_engine.py          expected loss, action choice, threshold baseline
├── state/
│   ├── __init__.py
│   └── session_manager.py          shared simulation state + PageContext snapshot
├── pages/
│   ├── 1_home.py                   command centre, three-column
│   ├── 2_live_battery_twin.py      neon battery twin + diagnostic profile
│   ├── 3_sensor_monitoring.py      raw telemetry, channel by channel
│   ├── 4_bayesian_analysis.py      one step of the recursion, opened up
│   ├── 5_risk_and_decisions.py     forecasting + expected-loss decisions
│   ├── 6_evaluation.py             saved offline results, rendered
│   └── 7_model_guide.py            explanation layer + viva questions
├── dashboard/
│   ├── __init__.py
│   ├── styles.py                   palette, global CSS, shared Plotly layout
│   ├── layout.py                   page header, panels, cards, HTML-escaping
│   ├── charts.py                   every Plotly figure, on the dark theme
│   ├── components.py               reusable DOM components
│   ├── navigation.py               persistent left nav + shared controls
│   └── battery_twin.py             the neon six-cell twin (inline SVG)
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py                 labelled sequences -> metrics + figures
├── results/
│   ├── figures/                    confusion_matrix / risk_over_time / calibration_curve .png
│   ├── metrics/                    evaluation_metrics.json + CSVs
│   └── screenshots/                (for your own demo screenshots)
└── tests/
    ├── test_bayesian_filter.py     26 tests
    ├── test_fault_diagnosis.py     27 tests
    └── test_decision_engine.py     23 tests
```

`conftest.py` is not in the original specification but is required: its presence
at the project root is what puts the root directory on `sys.path` so that
`tests/` can import `config`, `data` and `models` the same way `app.py` does.

### How the UI is wired

`app.py` is a shell only. It configures the page, injects the theme, initialises
shared state and builds a `st.navigation` route table with `position="hidden"`,
so the seven pages get real URLs (`/home`, `/live-battery-twin`, …) while the
visible navigation is the custom one in `dashboard/navigation.py`.

**The simulation is never rebuilt by navigating.** `state/session_manager.py`
keys the built twin on a signature of *(scenario, affected cell, seed, number of
steps)*. Changing page leaves that signature untouched, so the timeline
position, the selected cell and every filtered posterior survive the move. The
twin is rebuilt only on a deliberate scenario change, an applied seed,
**Regenerate**, or a length change; **Reset** rewinds the timeline without
rebuilding.

Two implementation notes that are easy to get wrong and are worth knowing:

- **Every shared-state mutation is an `on_change` / `on_click` callback**, never
  an inline mutation followed by `st.rerun()`. Callbacks run *before* the script
  body, so the page renders correctly on the same run. A bare `st.rerun()` under
  `st.navigation` re-resolves the route and silently bounces the user back to
  the default page.
- **Raw HTML goes through `dashboard.layout.render_html`**, which flattens
  indentation first. `st.markdown(..., unsafe_allow_html=True)` still runs the
  string through a Markdown parser, and Markdown turns any line indented by four
  or more spaces into a literal code block — which silently mangled the battery
  SVG until it was fixed.

---

## 4. Installation

Open a terminal in the `CellSafe-X` folder. Python 3.10 or newer is required.

```bash
python -m venv venv
```

**Windows PowerShell:**

```bash
.\venv\Scripts\Activate.ps1
```

> If PowerShell refuses with *"running scripts is disabled on this system"*, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use the
> CMD activation below instead.

**Windows CMD:**

```bash
venv\Scripts\activate
```

Then, in either shell:

```bash
pip install -r requirements.txt
```

---

## 5. Commands to run

Generate the evaluation metrics and figures (do this first so the Evaluation tab
has something to display):

```bash
python -m evaluation.evaluate
```

Launch the dashboard:

```bash
streamlit run app.py
```

Run the test suite:

```bash
pytest -q
```

A faster evaluation pass, useful while iterating:

```bash
python -m evaluation.evaluate --seeds 2 --steps 120
```

Streamlit prints the URL it is serving on; by default this is
**http://localhost:8501**.

---

## 6. The mathematical model

### 6.1 Hidden states

`Z_t ∈ {Healthy, Abnormal Heating, Pre-Runaway, Thermal Runaway}` — the thermal
condition of one cell at time `t`. One step is 10 seconds of simulated time.

### 6.2 Prior — the initial distribution `P(Z₀)`

```
P(Z₀) = [0.900, 0.070, 0.025, 0.005]
```

A pack is usually healthy when the vehicle starts, but every state keeps
non-zero mass so the filter can react quickly and never divides by zero.

### 6.3 Transition model `P(Z_t | Z_{t-1})`

Rows are the state at `t-1`, columns the state at `t`:

|            | → Healthy | → Abnormal | → Pre-Runaway | → Runaway |
|------------|-----------|------------|---------------|-----------|
| **Healthy**  | 0.9960 | 0.0040 | 0.0000 | 0.0000 |
| **Abnormal** | 0.0300 | 0.9600 | 0.0100 | 0.0000 |
| **Pre-Runaway** | 0.0000 | 0.0150 | 0.9650 | 0.0200 |
| **Thermal Runaway** | 0.0000 | 0.0000 | 0.0010 | 0.9990 |

What the matrix encodes:

- **Persistence** — a strong diagonal, so cells do not flicker between states.
- **Gradual deterioration** — only forward edges to the *adjacent* worse state.
- **Limited recovery** — `Abnormal → Healthy` = 0.030 (cooling catches up) and,
  more rarely, `Pre-Runaway → Abnormal` = 0.015.
- **No teleporting failures** — `Healthy → Pre-Runaway` and `Healthy → Runaway`
  are exactly zero; a cell must pass through Abnormal Heating.
- **Nearly absorbing runaway** — a 0.999 self-loop.

Every row sums to exactly 1.000, asserted at import time by
`config/model_parameters.py :: _validate()`.

### 6.4 Likelihood — the emission model `P(Y_t | Z_t)`

Five continuous channels, each a state-conditional Gaussian:

| Channel | Healthy | Abnormal | Pre-Runaway | Runaway |
|---|---|---|---|---|
| `temp_c` (°C) | 32 ± 6 | 48 ± 8 | 68 ± 9 | 95 ± 12 |
| `temp_rate` (°C/min) | 0.05 ± 0.35 | 0.50 ± 0.60 | 1.80 ± 1.00 | 4.50 ± 1.80 |
| `voltage_dev` (V) | 0.010 ± 0.030 | 0.060 ± 0.050 | 0.180 ± 0.090 | 0.400 ± 0.150 |
| `log_gas` (log ppm) | 1.10 ± 0.80 | 2.56 ± 1.00 | 4.39 ± 1.10 | 5.86 ± 1.00 |
| `neighbour_c` (°C) | 31 ± 5 | 38 ± 7 | 50 ± 9 | 68 ± 11 |

The standard deviations are deliberately wide so that adjacent states overlap
substantially. Narrow, well-separated Gaussians would produce artificially
perfect evaluation numbers, which would be dishonest on synthetic data.

**Why three measured channels are excluded from `P(Y|Z)`.** `current`, `soc` and
`cooling_eff` are measured and displayed, but they are *inputs* to the system
rather than *emissions* of a cell's thermal state. A coolant pump can fail while
the cell it serves is still cold; a pack can be charged hard while every cell
stays healthy. An early version of this project did include `cooling_eff` in the
emission model, and the filter consequently declared "Pre-Runaway" the moment the
pump degraded — before any cell had actually heated up. These three channels feed
the root-cause classifier instead, where they belong.

**Likelihood tempering.** The five channels are modelled as conditionally
independent given `Z_t`, but they are genuinely correlated (gas and voltage
deviation are both driven by temperature). A raw naive-Bayes product would be
over-confident, so each channel's log-likelihood is multiplied by a weight below
one (0.35–0.70). This is a documented approximation, not a physical law: it
amounts to saying the five correlated channels carry about 2.6 independent
channels' worth of evidence.

### 6.5 Posterior — the filtering recursion

The online filtering equation implemented in
`models/bayesian_filter.py :: BayesianFilter.update`:

```
P(Z_t | Y_1:t)  ∝  P(Y_t | Z_t) × Σ_{z_{t-1}} P(Z_t | Z_{t-1}) P(Z_{t-1} | Y_1:t-1)
                   └── likelihood ──┘ └────── prior after the transition step ──────┘
```

with normalising constant

```
P(Y_t | Y_1:t-1) = Σ_{z_t} P(Y_t | z_t) P(z_t | Y_1:t-1)
```

Because there are only four hidden states, the sum is a 4×4 matrix–vector
product and the recursion is **exact**. Everything is computed in log space and
normalised with a log-sum-exp, so the posterior sums to one to float precision
even after thousands of steps and cannot underflow to NaN. The Model Explanation
tab displays `posterior.sum()` to fifteen decimal places so this can be checked
live.

### 6.6 Sensor reliability

A second, independent Bayesian problem on the binary hypothesis
`H ∈ {reliable, faulty}` for the primary thermistor:

```
P(H | cues) ∝ P(H) × Π_c P(cue_c | H)
```

with `P(faulty) = 0.03` as the prior. The five cues are residuals that should be
near zero if the primary sensor is telling the truth:

| Cue | Residual | σ under *reliable* | σ under *faulty* |
|---|---|---|---|
| `backup_delta` | T_primary − T_backup | 1.5 | 20.0 |
| `neighbour_delta` | T_primary − T_neighbour | 12.0 | 30.0 |
| `gas_shortfall` | min(0, log-gas − expected at T_primary) | 1.2 | 3.2 |
| `volt_shortfall` | min(0, \|Δ V\| − expected at T_primary) | 0.05 | 0.22 |
| `jump_residual` | T_primary(t) − T_primary(t−1) | 1.2 | 9.0 |

The whole mechanism is that single asymmetry. Under *reliable* the residuals are
tight, so a large disagreement is astronomically unlikely; under *faulty* they
are broad, so a large disagreement is merely unremarkable — but a *perfect*
agreement is comparatively less likely than under *reliable*. The likelihood
ratio therefore swings toward "faulty" exactly when the evidence conflicts, and
toward "reliable" when it corroborates. **Nothing in the code says "if primary is
hot and backup is cool then declare a sensor fault."**

Two design details that matter:

- **`neighbour_delta` is a deliberately weak cue** (σ = 12 under *reliable*). A
  cell in genuine runaway really is 40–50 °C hotter than its neighbours, so
  disagreement here is only mild evidence of a sensor problem. An earlier version
  used σ = 4.5 and consequently declared a sensor fault during a real internal
  short circuit — exactly the wrong answer.
- **The gas and voltage cues are one-sided.** Only a *shortfall* counts against
  the sensor. *More* gas or *more* voltage sag than the claimed temperature
  explains is evidence of an electrical fault in the cell, not of a broken
  thermistor.

The fused temperature handed to the filter is

```
E[T] = (1 − P(faulty)) × T_primary + P(faulty) × T_backup
```

### 6.7 Root-cause diagnosis

```
log P(C | E) = log P(C) + Σ_f w_f · log N(e_f ; μ_{C,f}, σ_{C,f})
                        + log P(sensor-fault evidence | C) + const
```

over six causes: Normal/Fast-Charging Heat, Temperature Sensor Fault,
Cooling-System Failure, Overcharging, Internal Short Circuit, External Heat
Exposure. Priors are 0.34 / 0.12 / 0.15 / 0.13 / 0.11 / 0.15.

The reliability posterior enters as **soft evidence**, not a hard flag:

```
P(evidence | C) = P(faulty) · P(faulty | C) + (1 − P(faulty)) · (1 − P(faulty | C))
```

so an uncertain reliability estimate only nudges the cause posterior while a
confident one dominates it.

**Two of the eight features are anomalies, and this is the key design decision:**

```
vdev_anomaly = |ΔV|    − the deviation expected at T_primary
gas_anomaly  = log-gas − the log-gas expected at T_primary
```

Raw gas and raw voltage deviation are almost pure functions of temperature — hot
electrolyte decomposes and vents whatever the reason it got hot — so as raw
features they carry no causal information at all: at 85 °C every cause looks
alike. The *anomaly* is the part temperature does not explain, and it is highly
diagnostic. An internal short vents and sags more than its temperature warrants;
a lying thermistor produces neither the gas nor the voltage drift that its
claimed temperature demands. Together with `cooling_eff` (which isolates cooling
failure) and `self_vs_neighbour` (which separates a single runaway cell from a
pack-wide event), this makes all six causes separable.

Note that root-cause diagnosis is given the **raw** primary reading, while the
filter is given the **fused** one. This looks inconsistent and is deliberate: a
sensor fault can only be diagnosed while its symptom is still visible. If the
corrected temperature were used here, `Temperature Sensor Fault` could never win,
because the evidence for it would already have been erased.

### 6.8 Risk forecasting

Given the filtered belief `b_t` and transition matrix `A`, the k-step-ahead
marginal is exactly `b_t A^k`, and the dangerous-state probability is the mass it
places on {Pre-Runaway, Thermal Runaway}. With a 10-second step, the 5/15/30
minute horizons are `A³⁰`, `A⁹⁰` and `A¹⁸⁰`. **No future percentage is assigned by
hand.** These are *prior* forecasts: they assume no future observations and no
intervention.

### 6.9 Expected loss

```
ExpectedLoss(a) = Σ_s P(Z_t = s | Y_1:t) · L(a, s)          a* = argmin_a ExpectedLoss(a)
```

The loss matrix (0–100 scale, combining safety loss, battery damage, vehicle
unavailability, false-shutdown cost and measurement delay):

| Action | Healthy | Abnormal | Pre-Runaway | Thermal Runaway |
|---|---|---|---|---|
| Continue Monitoring | 0 | 18 | 70 | 100 |
| Request Backup Measurement | 3 | 9 | 45 | 88 |
| Reduce Charging Current | 9 | 7 | 30 | 72 |
| Increase Cooling | 11 | 6 | 26 | 66 |
| Isolate Affected Module | 38 | 28 | 14 | 34 |
| Emergency Shutdown | 62 | 50 | 22 | 8 |

Doing nothing is free when healthy and catastrophic in runaway; Emergency
Shutdown is the reverse. Because the recommendation is an *expectation* over the
posterior, it moves smoothly as evidence accumulates instead of snapping at a
threshold — and a low-probability but catastrophic state can still dominate.

---

## 7. Dashboard

A seven-page application on a deep-navy / lime smart-energy theme, with a
persistent left navigation. Every page carries the same header: page title,
one-line purpose, current scenario, current simulated minute, a **Bayesian
Engine Active** indicator and the permanent **SIMULATION-BASED RESEARCH
PROTOTYPE** badge.

**Persistent sidebar** — brand mark and "Bayesian Safety Twin" at the top; the
seven navigation entries (active entry in lime with a tinted background and a
lime left indicator); the shared simulation controls (scenario, affected cell,
seed and length, Regenerate, Reset); and a live status footer with
**Simulation Active**, **Bayesian Engine Active** and the prototype version.

### 1 · Home — command centre
Three columns. Top: pack health, hidden state, root cause, dangerous-state
probability, recommended action. Left: active scenario, compact telemetry, the
six-cell miniature overview. Centre: circular pack-health gauge with state and
confidence, the 5/15/30-minute forecast, risk history. Right: root-cause
posterior, sensor reliability, the recommendation, and the threshold-versus-
Bayesian comparison. Closes with the pipeline strip *Sensor Evidence →
Reliability Inference → Bayesian Filtering → Risk Forecast → Expected-Loss
Decision*.

### 2 · Live Battery Twin — the neon pack model
A two-column hero: the battery twin on the left (~58%), the selected cell's full
diagnostic profile on the right (~42%), then timeline controls, temperature
evolution and Bayesian risk evolution.

The twin is hand-written inline SVG — no raster image, no progress bar, no
3-D library — and every visual property is **bound to a quantity the pipeline
computed**:

| Visual | Bound to |
|---|---|
| Energy fill height | measured state of charge |
| Outer neon glow colour and intensity | pack-level `P(dangerous)` from `pack_risk()` |
| Cell body colour | argmax of that cell's filtered posterior |
| Hotspot wash intensity | that cell's `P(Pre-Runaway or Thermal Runaway)` |
| Neighbour glow | the *neighbouring cells'* own inferred risk |
| Selected outline | the shared focus cell |
| Health / state / confidence labels | risk band, posterior argmax, posterior max |

**State of charge is not safety.** The fill shows stored energy only; all
thermal risk is carried by colour, glow and the hotspot wash. A full healthy
cell and a full running-away cell have identical fill and completely different
colouring.

**Conflicting evidence.** When the reliability posterior says the primary
thermistor is probably lying, the cell is drawn in the neutral conflict grey
with a dashed body and a `! CONFLICT` marker — deliberately *not* red, because
the model does not believe that cell is hot. This is what makes the sensor-fault
scenario look different from a genuine runaway at a glance.

The neighbour glow is worth noting: it is driven by the neighbours' own inferred
risk, so in the internal-short scenario it genuinely lags the faulty cell rather
than being an animation.

### 3 · Sensor Monitoring
Ten metric cards with per-channel one-minute trends, the pack overview, the
decisive primary-versus-backup thermistor chart beside the reliability
posterior, and per-cell trends for temperature, voltage deviation, vent gas
(log scale), cooling effectiveness, neighbour temperature and the sensor-fault
posterior over time.

### 4 · Bayesian Analysis
The recursion in four panels — *Previous Belief → Transition Prediction →
Sensor Likelihood → Updated Posterior* — then the numeric table, the posterior
sum to twelve decimal places, the log evidence, the stacked posterior timeline,
the observation vector with weights and per-state means, root-cause and
reliability posteriors with the evidence-attribution table, and the transition
matrix beside the filtering equation.

### 5 · Risk & Decisions
Current / 5 / 15 / 30-minute risk, the continuous risk trajectory, ranked
expected losses with the equation, the recommendation with its per-state
contributions and reason, the full action table including the probabilistic
intervention assessment, the threshold comparison, and the loss matrix.

### 6 · Evaluation
Reads `results/metrics/evaluation_metrics.json` and is labelled **Synthetic
Evaluation** in both the header and the banner. Accuracy, macro
precision/recall/F1, Brier, NLL, ECE, root-cause performance, confusion matrix,
calibration chart, scenario-wise results, the threshold comparison, the
sensor-fault headline, an explicit limitations section, the saved figures and
the raw JSON. If the metrics file is missing it shows the command and offers a
button that runs the evaluation in a subprocess.

### 7 · Model Guide
Problem statement, architecture, observed versus hidden variables, prior /
likelihood / posterior side by side with live values, transition matrix and
exact filtering, sensor reliability, root-cause diagnosis, forecasting and
expected loss, what "digital twin" means here, assumptions, limitations, the
roadmap, and sixteen viva questions in expanders.

### Responsive behaviour
Desktop keeps the persistent navigation, the three-column Home and the
two-column twin. Below 900 px the navigation narrows and the page header
stacks. Below 640 px every Streamlit column block switches to
`flex-direction: column`, so the layout becomes a single column with the battery
twin first and the diagnostics beneath it, with no horizontal overflow.

---

## 8. Demonstration scenarios

| Scenario | What the simulator does | What the system should conclude |
|---|---|---|
| **Normal Operation** | Steady moderate discharge, all cells near 33 °C | Healthy throughout, Continue Monitoring |
| **Fast Charging** | 85 A charge warms the whole pack to ~50 °C | Abnormal Heating, cause = benign charging heat, mild mitigation |
| **Temperature Sensor Fault** | Physics identical to Normal Operation; the affected cell's *primary thermistor* reports +55 °C from t = 6 min | `P(faulty) → 1.00`, cell still Healthy, **no shutdown** — while the threshold system trips |
| **Cooling-System Failure** | Cooling effectiveness collapses 0.90 → 0.12 over 14 min; whole pack heats | Progressive Abnormal → Pre-Runaway, cause = cooling failure, Isolate/Increase Cooling |
| **Internal Short Circuit** | Affected cell shorts at t = 5 min: voltage sags first, temperature runs away, gas appears with a delay, neighbours heat later by conduction | Thermal Runaway, cause = internal short, Emergency Shutdown — and `P(faulty)` stays *low* |
| **External Heat Exposure** | External source raises the effective ambient across the pack with a shallow gradient | Pre-Runaway/Runaway, cause = external heat, aggressive intervention |

**The headline demonstration** is the pair *Temperature Sensor Fault* versus
*Internal Short Circuit*. In both, the primary sensor reads far above the
critical trip point. The threshold baseline shuts down in both. CellSafe-X shuts
down in exactly one — and the two tests
`test_pipeline_does_not_shut_down_a_healthy_pack_with_a_broken_sensor` and
`test_pipeline_does_shut_down_a_genuine_runaway` pin both halves of that
behaviour.

---

## 9. Evaluation methodology

`evaluation/evaluate.py` generates labelled synthetic sequences for all six
scenarios across five seeds (base seed 20240), rotating the affected cell across
seeds so no single pack position is over-tested. It runs the **real** pipeline —
the same code the dashboard uses — over all six cells of every run, giving
32,400 (cell, step) samples.

**Ground-truth labels** come from the simulator's *internal* physical state (true
temperature and true heating rate), which the pipeline never sees. The filter is
therefore scored on recovering a latent variable from noisy, partly contradictory
observations, not against its own output.

**Metrics computed:** accuracy, within-one-band accuracy, macro precision, macro
recall, macro F1, confusion matrix, multiclass Brier score, negative
log-likelihood, expected calibration error, root-cause accuracy (scored on the
affected cell only — see below), sensor-fault detection rates, and false
alarms / missed dangerous events for the Bayesian system and both threshold
levels.

**Scoring caveats we handle explicitly:**

- Root cause is a property of the *faulty* cell, so it is scored only on that
  cell. The other five are genuinely healthy and correctly report benign heating;
  scoring them against the scenario's fault label would be a labelling error, not
  a model error.
- Root-cause accuracy is reported both whole-run and over the final third. Most
  scenarios inject their fault several minutes in, so the opening steps are
  indistinguishable from normal operation *by construction*.
- Per-scenario macro F1 averages only over the classes that scenario actually
  visits; averaging over all four would report a meaningless 0.25 for
  single-class scenarios.

**Outputs:** `results/metrics/evaluation_metrics.json`,
`per_scenario_metrics.csv`, `confusion_matrix.csv`, `alarm_comparison.csv`,
`affected_cell_samples.csv`, and three figures in `results/figures/`.

---

## 10. Verification results

All commands below were executed in this project folder on **Windows 11**,
Python **3.14.5**, in the bundled `venv`.

**Environment**

| Package | Version |
|---|---|
| streamlit | 1.61.1 |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| plotly | 6.9.0 |
| scikit-learn | 1.9.0 |
| matplotlib | 3.11.1 |
| pytest | 9.1.1 |

**`pytest -q` → 76 passed in 2.98s** (26 filter/forecast, 27 diagnosis/simulator,
23 decision/pipeline). Zero failures, zero skips.

**`python -m evaluation.evaluate`** — 32,400 samples, 6 scenarios × 5 seeds ×
6 cells × 180 steps:

| Metric | Value |
|---|---|
| Hidden-state accuracy | **0.8608** |
| Within-one-band accuracy | 0.9799 |
| Macro precision | 0.7952 |
| Macro recall | 0.8839 |
| Macro F1 | 0.8307 |
| Multiclass Brier score | 0.2546 |
| Negative log-likelihood | 1.2467 |
| Expected calibration error | 0.0476 |
| Root-cause accuracy (affected cell, whole run) | 0.8685 |
| Root-cause accuracy (affected cell, final third) | 1.0000 |

Per scenario:

| Scenario | Accuracy | ±1 band | Brier | Cause (late) |
|---|---|---|---|---|
| Normal Operation | 0.994 | 1.000 | 0.010 | 1.000 |
| Temperature Sensor Fault | 0.994 | 1.000 | 0.010 | 1.000 |
| Fast Charging | 0.940 | 0.995 | 0.103 | 1.000 |
| External Heat Exposure | 0.801 | 1.000 | 0.366 | 1.000 |
| Internal Short Circuit | 0.762 | 0.885 | 0.465 | 1.000 |
| Cooling-System Failure | 0.674 | 1.000 | 0.572 | 1.000 |

Alarm comparison (false alarms / missed dangerous events):

| System | False alarms | FA rate | Missed | Miss rate | Precision |
|---|---|---|---|---|---|
| CellSafe-X Bayesian, P(dangerous) > 0.5 | 1,349 | 0.0511 | 345 | **0.0574** | 0.8078 |
| Threshold, warn level (55 °C) | 1,101 | 0.0417 | 848 | 0.1410 | 0.8243 |
| Threshold, critical level (70 °C) | 710 | 0.0269 | 2,957 | 0.4917 | 0.8115 |

**Sensor-fault scenario, affected cell only** (900 samples, pack genuinely
healthy throughout, so *every* alarm is a false alarm by construction):

| System | False alarms |
|---|---|
| Threshold, critical level | **710** |
| Threshold, warn level | 714 |
| CellSafe-X Bayesian | **0** |

Mean `P(sensor faulty)` after the fault was injected: **1.0000**. Before it:
**0.0000**.

**`streamlit run app.py`** — launched successfully with no exceptions.

**Multi-page UI verification** (headless, via Streamlit's `AppTest` harness):

| Check | Result |
|---|---|
| All seven pages render, reached through the real navigation buttons | pass, zero exceptions |
| Navigation preserves step, seed, focus cell and simulation signature | pass across all 7 pages |
| Selected cell survives Twin → Bayesian Analysis → Risk & Decisions | pass |
| Timeline set on one page is visible on the next | pass |
| All six scenarios driven to their final step on the twin page | pass, twin SVG rendered in each |
| Root cause correct on the twin page for all six scenarios | pass |
| Sensor fault renders conflict styling and no critical red | pass, `P(faulty) = 1.0000` |
| Internal short renders critical red and no conflict styling | pass, `P(faulty) = 0.0394` |
| Displayed values trace to the pipeline objects | pass (cause, action, fused and primary temperature) |

**Internal-short escalation**, affected cell, seed 42 — the twin's colour and the
recommendation move together, both derived from the posterior:

| Step | Time | Inferred state | Twin colour | Recommended action |
|---|---|---|---|---|
| 0 | 0.00 min | Healthy | `#9AF75B` lime | Continue Monitoring |
| 36 | 6.00 min | Abnormal Heating | `#F7C95C` amber | Increase Cooling |
| 42 | 7.00 min | Pre-Runaway | `#FF8A55` orange | Isolate Affected Module |
| 53 | 8.83 min | Thermal Runaway | `#FF5E72` red | Emergency Shutdown |

**Browser verification** (Chromium, live server):

| Check | Result |
|---|---|
| Theme applied | `.stApp` background `rgb(7,11,31)`, Inter font stack |
| Default Streamlit header | collapsed to `0px` |
| Active navigation item | lime `rgb(154,247,91)` text + lime left indicator, `kind="primary"` |
| Inactive navigation items | muted `rgb(168,183,212)`, transparent background |
| Deep links | `/live-battery-twin`, `/model-guide`, `/evaluation` all open directly and initialise |
| Twin SVG in the DOM | 6 cells, 6 clip paths, 48 rects, 6 temperature labels, 5-item legend |
| Desktop 1512 px | hero columns 670 px / 481 px = 58% / 42% as specified |
| Mobile 390 px | `scrollWidth == 390`, **0** overflowing elements, all column blocks `flex-direction: column`, battery twin first |
| Model Guide page | 17 panels, 19 expanders, 5 KaTeX blocks, no leaked markup |
| Evaluation page | labelled "Synthetic Evaluation", 8 metric cards read from the saved JSON |

**Regression check.** After the UI rewrite, `pytest -q` still reports **76
passed** and `python -m evaluation.evaluate` reproduces byte-identical metrics
(accuracy 0.8608, macro F1 0.8307, Brier 0.2546, NLL 1.2467, ECE 0.0476),
confirming that no probabilistic behaviour changed.

### Honest reading of these numbers

- **Accuracy is 0.86, not 0.99, and that is the point.** The emission
  distributions overlap heavily by design. Within-one-band accuracy is 0.98 and
  the confusion matrix shows *no* Healthy↔Thermal-Runaway confusion at all —
  errors are adjacent-band and concentrated at the moments the true state crosses
  a labelling boundary.
- **Cooling-System Failure is the weakest scenario (0.674).** The cause is filter
  lag: the transition matrix's strong diagonal and small `Healthy → Abnormal`
  rate mean the belief takes several steps to follow a slow, monotone temperature
  ramp across a band boundary. Its ±1-band accuracy is 1.000, so the filter is
  never badly wrong, only late. This is a real, explainable trade-off between
  forecast realism and responsiveness, not a bug.
- **The filter is over-confident.** The reliability diagram
  (`results/figures/calibration_curve.png`) shows the posterior is strongly
  bimodal — almost all mass lands in the 0.0 and 1.0 bins — and in the top bin
  only ~83% of samples are truly dangerous. The low ECE (0.048) reflects that the
  populated extreme bins are roughly right, not that the middle is well
  calibrated. This is the expected behaviour of an HMM fed sustained consistent
  evidence, and the likelihood tempering already in place reduces but does not
  eliminate it. Fixing it properly is listed under future work.
- **Fast Charging produces a transient false alarm** around t ≈ 4 min (visible in
  `risk_over_time.png`, peaking near 0.86). The initial current ramp produces a
  heating rate that momentarily resembles Pre-Runaway; the filter then recovers
  as the temperature plateaus. This is also a genuine demonstration that the
  recovery edges in the transition matrix work.
- **Root-cause accuracy of 1.000 over the final third looks too good**, and the
  honest explanation is that the six synthetic scenarios were given deliberately
  distinct late-stage signatures. Real faults overlap far more. The whole-run
  figure of 0.8685 — which includes the ambiguous early phase — is the more
  meaningful number.
- **The Bayesian system has *more* raw false alarms than the critical threshold**
  (1,349 vs 710) because it alarms earlier and at a lower bar. It also misses
  8.6× fewer dangerous events (345 vs 2,957). The interesting comparison is not
  the totals but the sensor-fault scenario, where the threshold's false alarms
  are all of the specific kind that would strand a working vehicle.

---

## 11. Important limitations

1. **All data is synthetic.** Nothing here has been validated against a real
   cell, a real thermistor or a real vehicle.
2. **Parameters are prototype assumptions**, hand-specified rather than learned,
   and are not certified physical battery parameters.
3. **Cells are filtered independently.** The only coupling is the observed
   neighbour-temperature channel. A real pack has stronger thermal and electrical
   coupling.
4. **The emission model is naive Bayes with tempering** — a documented
   approximation of correlated channels, not a correlated multivariate model.
5. **The filter is over-confident** (§10). Its argmax is trustworthy; its exact
   probability values near 0 and 1 should be read as "very likely" rather than
   taken literally.
6. **The intervention model is not causal.** `INTERVENTION_EFFECTIVENESS` is an
   assumed table used only for the "probabilistic intervention assessment"
   display. It never affects the expected-loss ranking and must not be described
   as counterfactual inference.
7. **The loss matrix is a modelling choice.** Different cost assumptions produce
   different recommendations; the matrix is exposed in the dashboard precisely so
   this is visible rather than hidden.
8. **The ground-truth labels are themselves a discretisation** of a continuous
   temperature, chosen by the simulator's thresholds. Samples near a boundary are
   ambiguous by construction.
9. **No claim** is made about real-world deployment, certified fire prevention,
   experimentally prevented fires, formal causal counterfactual inference, or
   real sensor integration.

---

## 12. Future work

- Learn the transition and emission parameters with Baum-Welch / EM instead of
  hand-specifying them.
- Fix the over-confidence with proper calibration (temperature scaling on the
  posterior, or a principled likelihood-tempering weight fitted on held-out data)
  and re-measure the reliability diagram.
- Replace the four-band discretisation with a switching Kalman filter over
  continuous temperature, giving a physical estimate and a genuine uncertainty
  band.
- Move to a factored multi-cell DBN with explicit thermal coupling, using
  approximate inference (Boyen-Koller or particle filtering) to keep it tractable.
- Add a proper structural causal model so intervention effects can be reasoned
  about rather than assumed.
- Value-of-information analysis for "Request Backup Measurement": compute the
  expected reduction in loss from the extra measurement rather than pricing the
  delay with a fixed constant.
- Real telemetry ingest from a CAN-bus/BMS log, with unit calibration and a
  hardware-accurate backup-sensor topology.

---

## 13. Viva questions and concise answers

**Q. Where exactly is Bayes' rule in this code?**
Three places. `models/bayesian_filter.py :: BayesianFilter.update` (hidden-state
posterior), `models/fault_diagnosis.py :: sensor_reliability` (binary hypothesis
test), and `models/fault_diagnosis.py :: diagnose_root_cause` (six-way
classifier). All three compute `log prior + log likelihood`, then normalise with
a log-sum-exp.

**Q. Is your inference exact or approximate?**
Exact. With four hidden states the forward recursion is a 4×4 matrix–vector
product, so there is nothing to approximate — no sampling, no variational bound.
The only approximation anywhere is the *modelling* assumption of conditional
independence between observation channels, which we compensate for with
documented tempering weights.

**Q. Why did you not use pgmpy or another library?**
The inference is thirty lines of NumPy. Writing it directly means every step —
prediction, update, normalisation, log-space stability — is visible and
explainable, which matters more for a review than a dependency.

**Q. How do you know the posterior really sums to one?**
It is normalised by construction via `log_normalise`, asserted at every step in
`test_posterior_sums_to_one_every_step`, and displayed live to twelve decimal
places in the Bayesian Reasoning tab.

**Q. What stops the filter from underflowing?**
All arithmetic is in log space, and the normalisation uses the log-sum-exp trick
(subtract the max before exponentiating).
`test_filter_is_stable_over_a_long_run_of_extreme_evidence` runs 5,000 steps of an
observation far outside every state's support and checks the posterior stays
finite and normalised. Raw probability-space filtering would produce NaNs there.

**Q. Walk me through the sensor-fault case.**
The primary reads 92 °C. Under *reliable*, `backup_delta = 59` must be drawn from
`N(0, 1.5)` — a 39-sigma event. Under *faulty* it is drawn from `N(0, 20)` — about
3 sigma, unremarkable. The log-likelihood ratio is enormous and overwhelms the
0.03 prior, so `P(faulty) → 1`. The fused temperature then falls back to the
backup channel, the filter sees a normal 33 °C cell and stays Healthy, and the
decision engine recommends monitoring. The threshold system, which has only one
number to look at, shuts the vehicle down.

**Q. Then why does a real internal short not get dismissed as a sensor fault?**
Because the evidence pattern is different, and the cue model is built to notice
the difference. In a real short the *backup* sensor agrees (the strongest cue
points to *reliable*), and the gas and voltage cues are one-sided — they only
count against the sensor when the physical signature is *missing*, never when
there is *more* of it than temperature explains. An earlier version used
two-sided cues and did misdiagnose real shorts as sensor faults; that failure is
now pinned by `test_electrical_fault_does_not_look_like_a_sensor_fault` and
`test_pipeline_does_shut_down_a_genuine_runaway`.

**Q. Why is the filter fed the fused temperature but the root-cause classifier
the raw one?**
Because they answer different questions. The filter must not be poisoned by a
reading it has no reason to trust. The classifier must still be able to *see* the
implausible reading, because that reading is the evidence for "sensor fault". If
you corrected it first, that cause could never win.

**Q. Why did you leave cooling effectiveness out of the emission model?**
Because it is an input, not a symptom. A pump can fail while the cell it serves
is still cold. When it was included, the filter jumped to Pre-Runaway the moment
the pump degraded, before any cell had heated. It now feeds the root-cause
classifier, where low cooling effectiveness is the decisive signature of cooling
failure.

**Q. Why are `vdev_anomaly` and `gas_anomaly` anomalies rather than raw readings?**
Raw gas and raw voltage deviation are nearly deterministic functions of
temperature, so at 85 °C every cause produces the same values and the features
are useless for diagnosis. Subtracting the value expected at that temperature
leaves exactly the residual the temperature cannot explain, which is what
distinguishes an electrical fault from external heating.

**Q. Where do the 5/15/30-minute numbers come from?**
`b_t A^k` with `k = horizon / 10 s`, i.e. `A³⁰`, `A⁹⁰`, `A¹⁸⁰`. Nothing is
assigned by hand. `test_propagation_matches_repeated_multiplication` verifies the
cached matrix power equals the naive step-by-step loop.

**Q. Why minimise expected loss instead of just thresholding the probability?**
Because a threshold ignores consequences. A 3% chance of runaway and a 3% chance
of a mild overheat justify very different actions. Expected loss weighs each
state by what it would actually cost, so a low-probability catastrophic state can
correctly dominate the decision — and the recommendation moves smoothly with the
evidence instead of snapping.

**Q. Your accuracy is only 86%. Is that not weak?**
It is deliberate. The emission Gaussians overlap heavily so that the task is not
trivially separable. Within-one-band accuracy is 98%, and the confusion matrix
shows zero Healthy↔Runaway confusion. If I had narrowed the standard deviations I
could report 99%, but it would measure nothing except that I had made the
synthetic classes separable.

**Q. Is your model well calibrated?**
Not fully, and the dashboard shows it. The reliability diagram reveals a strongly
bimodal, over-confident posterior: in the top bin only ~83% of samples are truly
dangerous. That is the classic behaviour of an HMM given sustained consistent
evidence. The tempering weights reduce it; proper calibration is future work.

**Q. Is the intervention model causal inference?**
No, and I am careful not to call it that. It is a **probabilistic intervention
assessment** using an assumed effectiveness table. It is display-only and never
changes the expected-loss ranking. Real counterfactual reasoning would need a
structural causal model and identification assumptions I have not made.

**Q. What would you need to deploy this?**
Real telemetry, parameters learned from real abuse-test data rather than assumed,
validation against instrumented cells, calibration correction, a coupled
multi-cell model, and functional-safety certification. This prototype
demonstrates the reasoning architecture, not a shippable safety system.

---

## 14. Troubleshooting

**`ModuleNotFoundError: No module named 'config'` (or `models`, `data`)**
You are running from the wrong directory. All commands must be run from inside
the `CellSafe-X` folder — the one containing `app.py`. Use `python -m
evaluation.evaluate`, not `python evaluation/evaluate.py`.

**PowerShell: "running scripts is disabled on this system"**
Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal
first, or activate the virtual environment from CMD instead with
`venv\Scripts\activate`.

**`streamlit` is not recognised as a command**
The virtual environment is not activated. Activate it, or call it directly:
`.\venv\Scripts\streamlit.exe run app.py`.

**The Evaluation tab says no results were found**
Run `python -m evaluation.evaluate`, or press the button on that tab. It writes
`results/metrics/evaluation_metrics.json`.

**Port 8501 is already in use**
`streamlit run app.py --server.port 8600`.

**The dashboard is slow when I change scenario or seed**
Each change re-simulates six cells and re-runs the whole pipeline. This takes a
few seconds and is cached afterwards, so scrubbing the time slider is instant.
Reduce "Simulation length" if you need faster iteration.

**`Please replace use_container_width with width`**
You are on a Streamlit older than 1.49. Upgrade with
`pip install -U "streamlit>=1.49"`.

**Play mode does not advance**
Play advances one step per rerun. If the run is already at the last step it stops
automatically — press **Reset** first.

**Tests fail with import errors**
Make sure `conftest.py` is present at the project root; it is what puts the root
on `sys.path` for pytest.
