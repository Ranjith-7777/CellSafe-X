"""Compact pack-level overview."""
import streamlit as st
from dashboard import charts
from dashboard.battery_twin import render_battery_twin
from dashboard.layout import page_header, metric_grid, panel, kv_rows, pct
from dashboard.styles import CONFLICT, STATE_COLORS, risk_color, risk_label
from state import session_manager as SM

ctx=SM.get_context(); worst=ctx.worst_out; danger=ctx.pack_danger
state=ctx.state_of(ctx.worst_cell); colour=CONFLICT if SM.is_conflicted(worst.sensor.p_faulty) else STATE_COLORS.get(state,CONFLICT)
page_header("Battery Safety Overview","Live pack condition and the next safest action.",ctx.scenario,ctx.time_min)
items=[("Pack Status",risk_label(danger),f"Worst cell {ctx.worst_cell+1}",risk_color(danger),False),("Hidden State",state,f"{pct(ctx.confidence_of(ctx.worst_cell))} confidence",colour,True),("Runaway Risk",pct(danger),"Dangerous-state probability",risk_color(danger),False),("Recommended Action",worst.decision.recommended_action,"Minimum expected loss",risk_color(danger),True)]
metric_grid(items)

with panel("Battery Pack Status"):
    left,right=st.columns([1.75,1],gap="large")
    with left: render_battery_twin(ctx, compact=True)
    with right:
        kv_rows([("Assessment",risk_label(danger)),("Focus cell",f"Cell {ctx.worst_cell+1}"),("State",state),("Confidence",pct(ctx.confidence_of(ctx.worst_cell))),("Pack risk",pct(danger))])

left,right=st.columns([3,2],gap="medium")
with left:
    with panel("Risk trend"):
        times,values=ctx.danger_history(ctx.worst_cell)
        st.plotly_chart(charts.risk_history_chart(times,values,ctx.time_min,ctx.total_minutes,height=220),width="stretch",key="overview_risk")
with right:
    with panel("Assessment"):
        fc=worst.forecast
        kv_rows([("Root cause",worst.cause.top_cause),("Sensor reliability",pct(worst.sensor.p_reliable)),("Current risk",pct(fc.current_dangerous)),("+15 minutes",pct(fc.dangerous_by_horizon[15.0])),("+30 minutes",pct(fc.dangerous_by_horizon[30.0])),("Recommendation",worst.decision.recommended_action)])
