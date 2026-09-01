"""Interactive battery digital twin."""
import numpy as np
import streamlit as st
from config import model_parameters as P
from dashboard import charts
from dashboard.battery_twin import render_battery_twin
from dashboard.layout import page_header,panel,kv_rows,pct,num
from dashboard.navigation import render_timeline
from state import session_manager as SM

ctx=SM.get_context()
page_header("Digital Twin","Explore the simulated pack and inspect one cell at a time.",ctx.scenario,ctx.time_min)

def scenario_changed(): SM.set_scenario(st.session_state["dt_scenario"])
def affected_changed(): SM.set_affected_cell(st.session_state["dt_affected"]-1)
def focus_changed(): SM.set_focus_cell(st.session_state["dt_focus"]-1)
def seed_changed(): SM.set_seed(st.session_state["dt_seed"])
def steps_changed(): SM.set_n_steps(st.session_state["dt_steps"])

for key,value in (("dt_scenario",ctx.scenario),("dt_affected",ctx.affected_cell+1),("dt_focus",ctx.focus_cell+1),("dt_seed",ctx.seed),("dt_steps",ctx.n_steps)):
    if key not in st.session_state: st.session_state[key]=value
with panel("Simulation controls"):
    controls=st.columns([1.35,1,2.5,.65],gap="small")
    controls[0].selectbox("Scenario",P.SCENARIOS,key="dt_scenario",on_change=scenario_changed)
    controls[1].selectbox("Affected cell",range(1,ctx.n_cells+1),key="dt_affected",on_change=affected_changed)
    with controls[2]: render_timeline(ctx,key="dt")
    controls[3].button("Reset",icon=":material/restart_alt:",width="stretch",on_click=SM.reset_timeline)
    with st.expander("Advanced",icon=":material/tune:"):
        a,b,c=st.columns([1,1,.7])
        a.number_input("Seed",0,1_000_000,key="dt_seed",on_change=seed_changed)
        b.slider("Simulation length",60,360,step=30,key="dt_steps",on_change=steps_changed)
        c.button("Regenerate",icon=":material/refresh:",width="stretch",on_click=SM.regenerate)

left,right=st.columns([3,2],gap="medium")
with left:
    with panel("Battery twin"):
        render_battery_twin(ctx)
        st.selectbox("Selected cell",range(1,ctx.n_cells+1),key="dt_focus",on_change=focus_changed)
with right:
    out=ctx.focus_out;r=out.reading
    with panel(f"Selected-cell details · Cell {ctx.focus_cell+1}"):
        kv_rows([("Primary temperature",f"{num(r['temp_primary'],1)} °C"),("Backup temperature",f"{num(r['temp_backup'],1)} °C"),("Voltage deviation",f"{num(r['voltage_dev'],3)} V"),("Gas",f"{num(np.expm1(r['log_gas']),1)} ppm"),("Cooling",pct(r['cooling_eff'])),("SOC",pct(r['soc'])),("Hidden state",ctx.state_of(ctx.focus_cell)),("Dangerous-state probability",pct(ctx.cell_danger(ctx.focus_cell))),("Sensor reliability",pct(out.sensor.p_reliable)),("Recommended action",out.decision.recommended_action)])

with panel("Cell trend"):
    mode=st.segmented_control("Measure",["Temperature","Risk","Voltage"],default="Temperature",label_visibility="collapsed")
    if mode=="Temperature": fig=charts.primary_vs_backup_chart(ctx.sim.frame,ctx.focus_cell,ctx.step,height=260)
    elif mode=="Risk":
        times,values=ctx.danger_history(ctx.focus_cell);fig=charts.risk_history_chart(times,values,ctx.time_min,ctx.total_minutes,height=260)
    else: fig=charts.sensor_series_chart(ctx.sim.frame,"voltage_dev","Voltage deviation","Voltage deviation (V)",ctx.step,ctx.focus_cell,height=260)
    st.plotly_chart(fig,width="stretch",key=f"dt_{mode}")
