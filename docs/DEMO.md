# CellSafe-X — Demo Script

Deterministic, reproducible walk-through. Start with:

```bash
python -m evaluation.evaluate   # once, so the Results page has data
streamlit run app.py
```

All scenario/seed/cell/step controls live in the persistent left sidebar
(**Scenario**, **Affected cell**, **Seed & length** expander, timeline
slider on each page). The same seed always reproduces exactly the same run —
nothing below is manually nudged to force a nicer-looking output.

## Scenario reference

| Scenario | What the simulator does | What the system should conclude |
|---|---|---|
| Normal Operation | Steady moderate discharge, all cells near 33 °C | Healthy throughout, Continue Monitoring |
| Fast Charging | 85 A charge warms the whole pack to ~50 °C | Abnormal Heating, benign-charging cause, no severe false alarm |
| Temperature Sensor Fault | Physics identical to Normal Operation; affected cell's primary thermistor reports +55 °C from t=6 min | `P(faulty)→1.00`, cell stays Healthy, no shutdown |
| Cooling-System Failure | Cooling effectiveness collapses 0.90→0.12 over 14 min | Abnormal→Pre-Runaway, cooling-failure cause, cooling/isolation mitigation |
| Internal Short Circuit | Cell shorts at t=5 min: voltage sags, temperature runs away, gas appears late | Thermal Runaway, internal-short cause, Emergency Shutdown, `P(faulty)` stays low |
| External Heat Exposure | External source raises ambient across the pack, shallow gradient | Pre-Runaway/Runaway, external-heat cause, aggressive intervention |

## Demo A — Normal Operation

1. Sidebar → Scenario = **Normal Operation**, Seed = 42 (default), Length = 180.
2. **Overview** page: pack health stays green, hidden state Healthy, dangerous
   probability ≈ 0.
3. **Analysis** page: recommended action = Continue Monitoring; open
   "Active sensing" — `should_request_measurement = False` (posterior already
   confident, nothing worth asking about); open "Model-based intervention
   forecast" — every action's projected risk is ≈ baseline (nothing to
   mitigate).

**Expected story:** low risk, stable posterior, no unnecessary intervention,
no unnecessary backup measurement.

## Demo B — Fast Charging

1. Sidebar → Scenario = **Fast Charging**, Seed = 20240, Affected cell = 3.
2. **Analysis** page, scrub the timeline to ≈ t = 3–5 min: `P(dangerous)`
   rises but stays well below the 0.5 alarm threshold (worst case across the
   full evaluation grid is 0.29 — see `docs/EVALUATION.md`), unlike the
   pre-Phase-1 behaviour where the same transient reached 1.00.
3. Root cause on the **Overview** page reads Normal/Fast-Charging Heat, not a
   fault.

**Expected story:** benign heating no longer produces the old severe
false-danger spike.

## Demo C — Cooling-System Failure

1. Sidebar → Scenario = **Cooling-System Failure**, Seed = 20240.
2. **Overview** page, step through t ≈ 0 → 25 min: watch the posterior shift
   Healthy → Abnormal → Pre-Runaway as uncertainty evolves; per
   `docs/EVALUATION.md`, mean detection delay for this scenario is now
   **-0.36 min** (the alarm tends to fire at or before the ground-truth
   crossing, not after).
3. **Analysis** page → "Model-based intervention forecast": compare Increase
   Cooling against Continue Monitoring — cooling shows a materially lower
   projected +15 min danger.
4. Same page → "Pack propagation": with the affected cell dangerous, its
   immediate neighbours show a higher coupled-vs-independent forecast.

**Expected story:** evolving uncertainty, early (not late) detection,
intervention comparison, cooling-related mitigation.

## Demo D — Internal Short Circuit

1. Sidebar → Scenario = **Internal Short Circuit**, Seed = 20240, Affected
   cell = 2.
2. **Overview** page at t ≈ 15 min: hidden state Thermal Runaway, root cause
   Internal Short Circuit, `P(faulty)` (sensor reliability) stays low —
   the evidence pattern does not look like a sensor fault.
3. **Analysis** page → intervention comparison: Emergency Shutdown gives the
   largest risk reduction; note it does not claim to fully cure an
   already-runaway cell (see `docs/ARCHITECTURE.md`).
4. Same page → "Pack propagation": source = the affected cell, most
   vulnerable neighbour identified, then compare the coupled forecast with
   and without isolating the source (isolation lowers the neighbour's
   projected risk — see the reduction reported inline).

**Expected story:** high danger, correct root-cause identification, strong
intervention need, propagation risk to neighbours, isolation reduces it.

## Demo E — Temperature Sensor Fault

1. Sidebar → Scenario = **Temperature Sensor Fault**, Seed = 20240.
2. **Overview** / **Digital Twin** page after t = 6 min: the affected cell's
   primary reading is far above the critical trip point, but the twin renders
   it in the neutral conflict style (not red) and the hidden state stays
   Healthy.
3. **Analysis** page → threshold comparison expander: the naive
   fixed-threshold baseline shows **Critical/triggered → Emergency Shutdown**;
   CellSafe-X shows **Continue Monitoring**, explained by `P(faulty) ≈ 1.00`.

**Expected story:** sensor-reliability inference prevents a naive
threshold-style false alarm that would otherwise strand a healthy vehicle.

## Non-interactive reproduction

Every number quoted above is reproducible without the UI:

```bash
python -m evaluation.evaluate          # full 5-seed evaluation
pytest -q tests/test_intervention.py tests/test_propagation.py tests/test_active_sensing.py
```
