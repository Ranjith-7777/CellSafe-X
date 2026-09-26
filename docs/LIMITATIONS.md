# CellSafe-X — Limitations and Future Work

## Limitations

1. **All data is synthetic.** Nothing here has been validated against a real
   cell, a real thermistor or a real vehicle.
2. **Parameters are prototype assumptions**, hand-specified rather than
   learned, and are not certified physical battery parameters.
3. **Cells are filtered independently.** The only coupling is the observed
   neighbour-temperature channel plus the Phase-4 propagation *forecasting*
   layer (built on top of, not inside, the independent filters). A real pack
   has stronger, jointly-inferred thermal and electrical coupling.
4. **The emission model is naive Bayes with tempering** — a documented
   approximation of correlated channels, not a correlated multivariate model.
5. **The filter is over-confident.** Its argmax is trustworthy; its exact
   probability values near 0 and 1 should be read as "very likely" rather
   than taken literally. Post-hoc calibration improves NLL/Brier but not ECE.
6. **The intervention model is not Pearlian causal inference.** Each action's
   effect is a documented, engineered alternative transition matrix — a
   model-based, controlled-transition forecast, not `do(X)` identification
   from interventional data. It never overrides the expected-loss decision.
7. **The loss matrix is a modelling choice.** Different cost assumptions
   produce different recommendations; it is exposed in the dashboard
   precisely so this is visible rather than hidden.
8. **The ground-truth labels are themselves a discretisation** of a
   continuous temperature (plus a sustained-rate escalation rule fixed in
   Phase 2). Samples near a boundary are ambiguous by construction.
9. **No claim** is made about real-world deployment, certified fire
   prevention, experimentally prevented fires, formal (Pearlian) causal
   identification, or real sensor integration.
10. **Propagation and active-sensing constants are engineered, not learned.**
    `PROPAGATION_COUPLING_STRENGTH`/`PROPAGATION_GAIN` translate the
    simulator's continuous thermal-conduction coefficient into a
    discrete-state escalation multiplier by modelling choice, not fitting.
    Propagation uses each neighbour's independent forecast rather than
    solving a joint fixed point, so multi-hop cascades beyond one coupling
    step are not captured.
11. **`Overcharging`** has a prior and emission model in the root-cause
    classifier but no simulator scenario ever generates it — it is never
    validated against a positive example.

## Future work

- Learn the transition and emission parameters with Baum-Welch/EM instead of
  hand-specifying them.
- Properly fix the over-confidence (a fully tempered/fitted likelihood, or a
  calibration objective that improves ECE as well as NLL/Brier).
- Replace the four-band discretisation with a switching Kalman filter over
  continuous temperature, giving a physical estimate and genuine uncertainty
  band.
- Move to a factored multi-cell DBN with explicit joint thermal coupling
  (approximate inference — Boyen-Koller or particle filtering — to stay
  tractable), replacing the current independent-cell + propagation-forecast
  approximation.
- Add a proper structural causal model with explicit identification
  assumptions, so intervention effects could be described as truly causal.
- Fit `PROPAGATION_COUPLING_STRENGTH`/`PROPAGATION_GAIN` against controlled
  multi-cell simulation sweeps (or real data) instead of engineering them.
- Real telemetry ingest from a CAN-bus/BMS log, with unit calibration and a
  hardware-accurate backup-sensor topology, and validation against
  instrumented abuse testing.
