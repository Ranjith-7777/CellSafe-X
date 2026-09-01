# CellSafe-X — System Architecture

> All data flowing through this architecture is **SIMULATED**. No real cell,
> sensor or vehicle is involved at any point.

---

## 1. End-to-end inference pipeline

```mermaid
flowchart TD
    A["Six-Cell Battery Simulator<br/><i>data/battery_simulator.py</i><br/>SIMULATED sensor stream"]

    A --> B["<b>Sensor Reliability Inference</b><br/><i>models/fault_diagnosis.py</i><br/>P(reliable / faulty | cues)<br/>5 evidence cues, Bayes' rule"]

    B --> C["Sensor Fusion<br/>E[T] = (1-P<sub>faulty</sub>)·primary + P<sub>faulty</sub>·backup<br/>+ smoothed heating rate"]

    C --> D["<b>Exact Bayesian Filter</b><br/><i>models/bayesian_filter.py</i><br/>P(Z<sub>t</sub> | Y<sub>1:t</sub>) over 4 hidden states<br/>predict → update → normalise"]

    A -.->|"raw primary reading<br/>(uncorrected, on purpose)"| E
    B -.->|"soft evidence<br/>P(sensor faulty)"| E

    E["<b>Root-Cause Diagnosis</b><br/><i>models/fault_diagnosis.py</i><br/>P(Cause | Evidence) over 6 causes<br/>log-space naive Bayes"]

    D --> F["<b>Risk Forecast</b><br/><i>models/risk_forecast.py</i><br/>P(Z<sub>t+k</sub>) = P(Z<sub>t</sub>)·A<sup>k</sup><br/>5 / 15 / 30 minute horizons"]

    D --> G["<b>Expected-Utility Decision</b><br/><i>models/decision_engine.py</i><br/>E[loss](a) = Σ<sub>s</sub> P(s)·L(a,s)<br/>recommend argmin<sub>a</sub>"]

    A -.->|"raw primary temperature"| H
    H["Threshold Baseline<br/><i>models/decision_engine.py</i><br/>fixed trip point, single sensor"]

    D --> I
    E --> I
    F --> I
    G --> I
    H --> I
    B --> I

    I["<b>Dashboard</b><br/><i>app.py + dashboard/</i><br/>Live Digital Twin · Bayesian Reasoning<br/>Evaluation · Model Explanation"]

    style A fill:#e8eef7,stroke:#24406f,stroke-width:2px
    style B fill:#dbe9f5,stroke:#2b6cb0,stroke-width:2px
    style D fill:#dbe9f5,stroke:#2b6cb0,stroke-width:3px
    style E fill:#dbe9f5,stroke:#2b6cb0,stroke-width:2px
    style F fill:#dbe9f5,stroke:#2b6cb0,stroke-width:2px
    style G fill:#dbe9f5,stroke:#2b6cb0,stroke-width:2px
    style H fill:#f1f2f4,stroke:#94a3b8,stroke-width:1px,stroke-dasharray: 4 3
    style I fill:#0f1e38,stroke:#0f1e38,color:#ffffff,stroke-width:2px
```

### Why the ordering matters

**Reliability inference runs before the filter.** If a broken thermistor's
reading were fed straight into `P(Y | Z)`, a single failed sensor would drive
the hidden-state posterior to *Thermal Runaway* on its own, and the system would
shut down a perfectly healthy vehicle. Fusing first means the filter only ever
sees a temperature the system has a reason to trust.

**Root-cause diagnosis is given the RAW primary reading.** This looks
inconsistent with the paragraph above, and it is deliberate: a sensor fault can
only be *diagnosed* while its symptom — an implausibly hot reading — is still
visible. If the corrected temperature were used here, `Temperature Sensor Fault`
could never win, because the evidence for it would already have been erased.

---

## 2. Hidden-state transition model

Four latent thermal states with a first-order Markov transition structure.
Edge labels are the per-step probabilities from
`config/model_parameters.py :: TRANSITION_MATRIX`, where one step = 10 seconds
of simulated time.

```mermaid
stateDiagram-v2
    direction LR

    Healthy: Healthy
    Abnormal: Abnormal Heating
    Pre: Pre-Runaway
    Runaway: Thermal Runaway

    [*] --> Healthy: P(Z₀) = 0.900

    Healthy --> Healthy: 0.9960
    Healthy --> Abnormal: 0.0040

    Abnormal --> Abnormal: 0.9600
    Abnormal --> Healthy: 0.0300
    Abnormal --> Pre: 0.0100

    Pre --> Pre: 0.9650
    Pre --> Abnormal: 0.0150
    Pre --> Runaway: 0.0200

    Runaway --> Runaway: 0.9990
    Runaway --> Pre: 0.0010

    note right of Healthy
        No edge to Pre-Runaway or
        Thermal Runaway: deterioration
        must pass through Abnormal
        Heating first.
    end note

    note right of Runaway
        Nearly absorbing (0.999).
        Escape is possible in principle
        but vanishingly unlikely.
    end note
```

**Properties encoded in the matrix**

| Property | How it appears |
|---|---|
| Persistence | Strong diagonal (0.96 – 0.999) — cells do not flicker between states |
| Gradual deterioration | Only forward edges to the *adjacent* worse state |
| Limited recovery | `Abnormal → Healthy` = 0.030; `Pre → Abnormal` = 0.015 |
| No teleporting failures | `Healthy → Pre` and `Healthy → Runaway` are exactly 0 |
| Nearly absorbing runaway | `Runaway → Runaway` = 0.999 |
| Valid stochastic matrix | Every row sums to exactly 1.000 (asserted at import time) |

---

## 3. Module map

| Layer | File | Responsibility |
|---|---|---|
| Parameters | `config/model_parameters.py` | Every prior, transition, Gaussian and loss value, in one place |
| Data | `data/battery_simulator.py` | Six-cell lumped thermal simulation, six scenarios, ground-truth labels |
| Inference | `models/bayesian_filter.py` | Exact forward filtering and k-step prediction |
| Inference | `models/fault_diagnosis.py` | Sensor reliability + root-cause posteriors |
| Inference | `models/risk_forecast.py` | Multi-step dangerous-state probabilities |
| Decision | `models/decision_engine.py` | Expected loss, action selection, threshold baseline |
| Orchestration | `models/__init__.py` | `CellPipeline` — wires the blocks together per cell, per step |
| Evaluation | `evaluation/evaluate.py` | Labelled sequences, metrics, figures |
| Presentation | `dashboard/`, `app.py` | Theme, components, four-tab Streamlit dashboard |
