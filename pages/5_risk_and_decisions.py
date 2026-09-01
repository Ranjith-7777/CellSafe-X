"""Focused state, diagnosis, risk, and decision analysis."""
import pandas as pd
import streamlit as st
from config import model_parameters as P
from dashboard import charts,components as C
from dashboard.layout import page_header,panel,metric_card,metric_grid,pct,num,kv_rows,section
from dashboard.navigation import cell_selector,render_timeline
from dashboard.styles import ACCENT_CYAN,CONFLICT,risk_color
from models import compare_with_threshold
from state import session_manager as SM

ctx=SM.get_context();out=ctx.focus_out
page_header("Analysis","Inspect the current state, diagnosis and safety decision.",ctx.scenario,ctx.time_min)
c1,c2=st.columns([1,3],gap="medium")
with c1: cell_selector(ctx,"analysis_cell",label="Cell")
with c2: render_timeline(ctx,"analysis_time")

section("State Probabilities")
with panel(None): st.plotly_chart(charts.posterior_chart(out.posterior,height=220,title=""),width="stretch",key="analysis_state")

section("Diagnosis")
left,right=st.columns(2,gap="medium")
with left:
    with panel("Root-cause posterior",tall=True):
        st.plotly_chart(charts.root_cause_chart(out.cause.posterior,height=245),width="stretch",key="analysis_cause")
with right:
    with panel("Sensor reliability",tall=True):
        st.plotly_chart(charts.sensor_reliability_chart(out.sensor.p_reliable,out.sensor.p_faulty,height=210),width="stretch",key="analysis_reliability")
        kv_rows([("Assessment",SM.evidence_consistency(out.sensor.p_faulty)),("Reliable",pct(out.sensor.p_reliable)),("Faulty",pct(out.sensor.p_faulty))])

section("Risk and Decision")
fc=out.forecast; risks=[("Now",fc.current_dangerous),("+5 minutes",fc.dangerous_by_horizon[5.0]),("+15 minutes",fc.dangerous_by_horizon[15.0]),("+30 minutes",fc.dangerous_by_horizon[30.0])]
metric_grid([(label,pct(value),"Dangerous-state probability",risk_color(value),False) for label,value in risks])
rec,loss=st.columns([2,1],gap="medium")
with rec:
    with panel("Recommendation"): C.recommendation_card(out.decision)
with loss:
    with panel("Expected loss"): metric_card("Minimum",num(out.decision.expected_losses[out.decision.recommended_action],2),out.decision.recommended_action,ACCENT_CYAN)

with st.expander("Threshold comparison"):
    C.comparison_panel(compare_with_threshold(out.reading["temp_primary"],out.decision,out.sensor.p_faulty))
with st.expander("Full expected-loss table"):
    st.dataframe(pd.DataFrame({"Action":list(out.decision.expected_losses),"Expected loss":list(out.decision.expected_losses.values())}),width="stretch",hide_index=True)
with st.expander("Detailed evidence"):
    st.dataframe(C.cue_table(out.sensor.cues),width="stretch",hide_index=True)
with st.expander("Transition matrix"):
    st.dataframe(pd.DataFrame(P.TRANSITION_MATRIX,index=P.STATES,columns=P.STATES),width="stretch")
with st.expander("Observation likelihoods"):
    st.dataframe(pd.DataFrame({"State":P.STATES,"Likelihood":out.filter_step.likelihood_norm}),width="stretch",hide_index=True)
