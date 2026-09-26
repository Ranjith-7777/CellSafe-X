# CellSafe-X — Viva Questions and Concise Answers

**Q. Where exactly is Bayes' rule in this code?**
Four places. `models/bayesian_filter.py :: BayesianFilter.update` (hidden-state
posterior), `models/fault_diagnosis.py :: sensor_reliability` (binary
hypothesis test), `models/fault_diagnosis.py :: diagnose_root_cause` (six-way
classifier), and `models/active_sensing.py` (posterior update inside the
EIG/EVI Monte Carlo estimator). All compute `log prior + log likelihood`,
then normalise with a log-sum-exp.

**Q. Is your inference exact or approximate?**
The 4-state hidden-state filter is exact (a 4×4 matrix-vector product each
step — no sampling, no variational bound). The Phase-4 additions are
explicitly approximate and labelled as such: propagation uses each
neighbour's independent forecast rather than a joint fixed point, and EIG/EVI
use Monte Carlo estimation with a fixed seed.

**Q. Why did you not use pgmpy or another library?**
The core inference is thirty lines of NumPy. Writing it directly means every
step — prediction, update, normalisation, log-space stability — is visible
and explainable.

**Q. How do you know the posterior really sums to one?**
Normalised by construction via `log_normalise`, asserted at every step in
tests, and displayed live to twelve decimal places on the Bayesian Network page.

**Q. What stops the filter from underflowing?**
All arithmetic is in log space with the log-sum-exp trick.
`test_filter_is_stable_over_a_long_run_of_extreme_evidence` runs 5,000 steps
of an observation far outside every state's support and checks the posterior
stays finite and normalised.

**Q. Walk me through the sensor-fault case.**
The primary reads 92 °C. Under *reliable*, `backup_delta=59` must be drawn
from `N(0,1.5)` — a 39-sigma event. Under *faulty* it is drawn from
`N(0,20)` — unremarkable. The likelihood ratio overwhelms the 0.03 prior, so
`P(faulty)→1`. The fused temperature falls back to the backup channel, the
filter stays Healthy, and the decision engine recommends monitoring — while
a fixed threshold shuts the vehicle down.

**Q. Then why does a real internal short not get dismissed as a sensor fault?**
Because the evidence pattern differs: the backup sensor agrees in a real
short, and the gas/voltage cues are one-sided (they only count against the
sensor when the physical signature is *missing*, never when there is *more*
than temperature explains).

**Q. Why is the filter fed the fused temperature but the root-cause
classifier the raw one?**
They answer different questions. The filter must not be poisoned by a
reading it has no reason to trust; the classifier must still *see* the
implausible reading, because that reading is the evidence for "sensor fault".

**Q. Why did you leave cooling effectiveness out of the emission model?**
It is an input, not a symptom — a pump can fail while the cell is still
cold. An early version included it and the filter jumped to Pre-Runaway the
moment the pump degraded, before any cell had heated.

**Q. Why are `vdev_anomaly`/`gas_anomaly` anomalies rather than raw readings?**
Raw gas/voltage deviation are nearly deterministic functions of temperature,
so at 85 °C every cause looks alike. The anomaly (observed minus
expected-at-this-temperature) is the residual that actually distinguishes causes.

**Q. Why minimise expected loss instead of thresholding the probability?**
A threshold ignores consequences: a 3% chance of runaway and a 3% chance of
mild overheat justify very different actions. Expected loss weighs each
state by its actual cost, so a low-probability catastrophic state can
correctly dominate.

**Q. Your accuracy is only 88%. Is that weak?**
Deliberate — the emission Gaussians overlap heavily so the task is not
trivially separable. Within-one-band accuracy is 98%, and the confusion
matrix shows zero Healthy↔Runaway confusion. Narrowing the Gaussians would
raise accuracy while measuring nothing but synthetic separability.

**Q. Is your model well calibrated?**
Not fully. The reliability diagram shows a bimodal, over-confident posterior.
Post-hoc temperature scaling improves NLL/Brier substantially but slightly
*worsens* ECE — disclosed, not hidden, in `docs/EVALUATION.md`.

**Q. Is the intervention model causal inference?**
No. It is a **model-based, controlled-transition counterfactual forecast**:
each action gets an explicit alternative transition matrix, and the forecast
is `b_t @ A(action)^k` from the current posterior. It is not Pearlian `do(X)`
identification from interventional data, and it never overrides the
expected-loss decision.

**Q. What is propagation, and is it a real joint model?**
No — cells remain independently filtered. Propagation is a forecasting layer
that nudges a target cell's escalation probability using its neighbours' own
independent forecasts and the pack's linear-chain topology. Multi-hop
cascades beyond one coupling step are not captured; this is disclosed, not a
claim of an exact joint solution.

**Q. What is EIG/EVI, and why does "Request Backup Measurement" not
physically help?**
EIG is the expected drop in posterior entropy from one more reading of a
channel the filter already models; EVI is the expected drop in minimum
decision loss. Both are Monte Carlo estimates over the predictive
distribution, deterministic seed. Requesting a measurement is
information-gathering only — its intervention-forecast effect stays exactly
zero by construction, separate from its (often positive) information value.

**Q. What would you need to deploy this?**
Real telemetry, parameters learned from real abuse-test data, validation
against instrumented cells, a fully fixed calibration, a coupled multi-cell
model, and functional-safety certification. This prototype demonstrates the
reasoning architecture, not a shippable safety system.
