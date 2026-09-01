"""Bayesian network and live belief update."""
import numpy as np
import streamlit as st
from config import model_parameters as P
from dashboard.layout import page_header,panel,metric_grid,pct
from dashboard.navigation import cell_selector,render_timeline
from dashboard.network_graph import render_bayesian_network
from dashboard.styles import ACCENT_CYAN,ACCENT_GREEN
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
