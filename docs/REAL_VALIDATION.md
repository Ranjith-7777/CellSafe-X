# CellSafe-X — Real Experimental External Validation (Phase 6)

**Two tracks, never merged.** `evaluation/evaluate.py` is the synthetic,
controlled, quantitative benchmark (accuracy/Brier/NLL/ECE/etc. — see
`docs/EVALUATION.md`). `evaluation/real_validation.py` is a separate,
qualitative **external validation** track that replays three real
externally-heated thermal-runaway experiments through the SAME, UNCHANGED
CellSafe-X filter to see whether its danger estimate behaves sensibly. No
number from this track is combined with, or compared head-to-head against,
the synthetic metrics.

## Dataset

"Dataset of internal temperature and gas pressure in cylindrical lithium-ion
cells during thermal runaway", B. Gulsoy et al., University of Warwick /
Faraday Institution, Mendeley Data, DOI `10.17632/rgfhdhcd9k.1`, CC BY 4.0.
Three Sony VTC6A cells, externally heated to thermal runaway. Full
provenance, exact source field names/shapes, and the field→evidence mapping:
`data/external/warwick_thermal_runaway/PROVENANCE.md`.

Download: `python data/external/warwick_thermal_runaway/download.py`
(215 MB, sha256-verified, not committed to the repository).

## What is fed to the model, and what is not

Only `temp_c` (from the internal midpoint thermocouple, `MidIntTemp`) and a
causal `temp_rate` (first difference across 10-second, trailing-window-mean
resampled bins — matching CellSafe-X's own native step, no future sample
ever used) are passed to the unmodified `BayesianFilter`. `log_gas` and
`neighbour_c` are genuinely absent and simply omitted — the existing
`observation_log_likelihood` already marginalises missing channels
correctly (unchanged this phase). Cell voltage was tested as `voltage_dev`
and **excluded after diagnosis**: real cells rest at a freshly-charged
~4.1–4.2 V, while `NOMINAL_VOLTAGE=3.70` represents the synthetic
simulator's narrow monitoring-baseline assumption, not a real cell's
full-charge voltage — feeding it through that formula produced a ~45-nat
log-likelihood swing that swamped every other channel from step one. This
is a data-adapter mapping decision, not a change to any frozen model
parameter. Sensor-reliability fusion, root-cause diagnosis, interventions,
propagation and active-sensing are **not run** on real data — none of their
required inputs (backup sensor, current, SOC, cooling effectiveness, pack
topology) exist in a single externally-heated cell.

## Event markers

No separate event-annotation file exists in this dataset entry, and the
companion paper's own stated pre-vent/soft-vent/runaway timestamps were not
accessible from this environment. Two **signal-derived proxies** are used
instead, computed directly from the measured channels and explicitly labelled
as proxies, never as the model's hidden-state ground truth:
`temp_runaway_onset` = first time `MidIntTemp ≥ 150 °C`;
`pressure_vent_onset` = first time internal gas pressure rises ≥1 bar above
its own first-60-second baseline.

## Results

| Experiment | Max P(dangerous) | First alarm (t, s) | Runaway-onset proxy (t, s) | "Lead time" (s) |
|---|---|---|---|---|
| 0 | 1.000 | 60 | 700 | 640 |
| 1 | 1.000 | 50 | 730 | 680 |
| 2 | 1.000 | 20 | 720 | 700 |

Full trajectories/plots: `results/real_validation/experiment_{0,1,2}.png`
and `..._trajectory.csv`; resampled inputs actually fed to the model:
`..._resampled_input.csv`; machine-readable summary: `summary.json`.

**Honest reading.** In all three experiments the alarm fires within the
first 1–2 resampled steps (10–60 s), hundreds of seconds before the
150 °C runaway-onset proxy and the dataset's visible flame/explosion event
(~800 s). The model never returns to a low-risk state once triggered. This
is driven almost entirely by `temp_rate`: the externally-applied heating
protocol itself produces a sustained heating rate (6–17+ °C/min from very
early on) that exceeds the synthetic Thermal-Runaway calibration
(4.50 ± 1.80 °C/min) while the absolute temperature is still only 20–30 °C.
This is a genuine, informative finding, not a bug: CellSafe-X's `temp_rate`
emission model was calibrated against the SYNTHETIC simulator's scenarios,
none of which include a sustained, externally-forced heating ramp at
low absolute temperature — so an unmodified, synthetic-calibrated filter is
extremely (arguably over-) sensitive to this specific real stimulus. It never
missed an event (no false negative), but it also cannot distinguish "being
externally heated" from "genuine electrochemical runaway" at this
sensitivity — exactly the kind of finding external validation is for.

## Relationship to Phase 7A's overlapping sensors

The Warwick dataset genuinely has multiple temperature channels (internal
midpoint, surface midpoint, and near-terminal surface probes), but they are
**not** treated as overlapping measurements of one latent quantity the way
Phase 7A's synthetic primary/surface/backup sensors are. They are
**spatially distinct** thermal points on the same cell (core vs. case vs.
near each terminal) that can legitimately read differently even when every
sensor is working correctly - fusing them with
`models.sensor_fusion.fuse_overlapping_sensors` would implicitly (and
wrongly) assume they share one consensus temperature. This phase therefore
still uses only `MidIntTemp` as the single `temp_c` reading for real-data
replay (see above); Phase 7A's reliability-aware fusion remains a
synthetic-track capability, not applied to this dataset.

## Limitations (read before drawing any conclusion)

1. **Only three cells**, one cell format (Sony VTC6A cylindrical), one lab.
2. **Externally heated** runaway, not an internal electrical/thermal fault —
   the dominant real CellSafe-X demonstration scenario (Internal Short
   Circuit) has no real-data counterpart here.
3. **Different sensor configuration**: internal + multi-point surface
   thermocouples and a gas-pressure transducer, not the synthetic model's
   primary/backup thermistor pair, vent-gas concentration sensor, or current/
   SOC/cooling instrumentation.
4. **No exact CellSafe-X four-state ground truth** exists for real cells —
   only signal-derived proxies, so no accuracy/Brier/NLL/ECE-style metric is
   computed here, by design (see `docs/EVALUATION.md` for where those
   metrics DO apply — the synthetic track only).
5. **No multi-cell pack topology** — propagation, intervention and
   active-sensing remain synthetic-only; this dataset cannot validate them.

**This is external, event-aligned validation on three real cells — it is
not proof of deployment-level generalisation**, and CellSafe-X was not
trained or refit on this data (no parameter in `config/model_parameters.py`
was changed as a result of this phase).
