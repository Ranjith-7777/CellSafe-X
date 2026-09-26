"""Bayesian network and live belief update."""
import numpy as np
import pandas as pd
import streamlit as st
from config import model_parameters as P
from dashboard.layout import page_header,panel,metric_grid,pct,num
from dashboard.navigation import cell_selector,render_timeline
from dashboard.network_graph import render_bayesian_network
from dashboard.styles import ACCENT_CYAN,ACCENT_GREEN
from models.active_sensing import recommend_measurement
from models.sensor_fusion import fuse_overlapping_sensors
from state import session_manager as SM

ctx=SM.get_context();focus=ctx.focus_out;fs=focus.filter_step
page_header("Bayesian Network","How CellSafe-X updates its belief from uncertain battery evidence.",ctx.scenario,ctx.time_min)
c1,c2=st.columns([1,3],gap="medium")
with c1: cell_selector(ctx,"bn_cell",label="Cell")
with c2: render_timeline(ctx,"bn_time")

with panel("Live inference graph"):
    render_bayesian_network()

prior_state=P.STATES[int(np.argmax(fs.prior))]; like_state=P.STATES[int(np.argmax(fs.likelihood_norm))]; post_state=P.STATES[int(np.argmax(fs.posterior))]
metric_grid([("Prior",prior_state,pct(np.max(fs.prior)),ACCENT_CYAN,True),("Likelihood",like_state,pct(np.max(fs.likelihood_norm)),ACCENT_CYAN,True),("Posterior",post_state,pct(np.max(fs.posterior)),ACCENT_GREEN,True)])

EXPLAIN={"Current Hidden State":"The unobserved thermal condition inferred from the previous belief and today’s evidence.","Sensor Reliability":"A separate posterior controls how strongly the primary and backup temperature channels are trusted.","Updated Posterior":"The normalized product of the predicted prior and the current evidence likelihood.","Risk Forecast":"The posterior is propagated through the transition matrix without assuming future observations.","Decision":"The action with minimum expected loss under the updated state distribution."}
node=st.selectbox("Explain node",list(EXPLAIN))
st.caption(EXPLAIN[node])
with st.expander("Bayesian equation"):
    st.latex(r"P(Z_t\mid Y_{1:t}) \propto P(Y_t\mid Z_t)\sum_{z_{t-1}}P(Z_t\mid z_{t-1})P(z_{t-1}\mid Y_{1:t-1})")

with st.expander("Sensor set (Phase 7B)"):
    st.caption(
        "The canonical set of sensors CellSafe-X understands "
        "(config.model_parameters.SENSOR_REGISTRY), which ones overlap "
        "(observe the same latent quantity) and are fused rather than "
        "double-counted, and which is currently contributing to inference."
    )
    row = ctx.sim.frame[(ctx.sim.frame["cell"] == ctx.focus_cell) & (ctx.sim.frame["step"] == ctx.step)].iloc[0]
    readings = {s: float(row[f"temp_{s}"]) for s in P.OVERLAPPING_SENSOR_NAMES}
    fusion = fuse_overlapping_sensors(readings)

    reg_rows = []
    for sensor_key, meta in P.SENSOR_REGISTRY.items():
        overlap_group = meta["family"] if meta["overlaps"] else "-"
        reliability = pct(fusion.reliability[sensor_key.removeprefix("temp_")]) if sensor_key.removeprefix("temp_") in fusion.reliability else "-"
        reg_rows.append({
            "Sensor": sensor_key,
            "Family": meta["family"],
            "Overlap group": overlap_group,
            "HMM evidence": "yes" if meta["hmm_eligible"] else "no",
            "Active-sensing eligible": "yes" if meta["active_sensing_eligible"] else "no",
            "Current reliability": reliability,
        })
    st.dataframe(pd.DataFrame(reg_rows), width="stretch", hide_index=True)

    metric_grid([
        ("Fused temperature", num(fusion.fused_temp_c, 2) + " °C", "Overlapping-sensor Bayesian consensus", ACCENT_GREEN, False),
        ("Sensor disagreement", num(fusion.disagreement_c, 2) + " °C", "std. dev. of the 3 raw readings", ACCENT_CYAN, False),
    ])

    rec = recommend_measurement(focus.posterior)
    st.caption(
        f"Next measurement (EIG/EVI): **{rec.best_channel}** "
        f"({rec.information_gain[0].sensor_family} family) - "
        f"{'worth requesting' if rec.should_request_measurement else 'not worth requesting'} "
        f"(EIG={rec.information_gain[0].expected_information_gain:.3f} nats, "
        f"EVI={rec.value_of_information.expected_value_of_information:.2f} loss units)."
    )
