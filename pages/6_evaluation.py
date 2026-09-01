"""Concise synthetic evaluation results."""
from pathlib import Path
import numpy as np
import streamlit as st
from dashboard import charts
from dashboard.evaluation_data import EvaluationDataError, load_evaluation_metrics
from dashboard.layout import page_header,metric_grid,panel,pct,num,render_html
from dashboard.styles import ACCENT_CYAN,ACCENT_GREEN,WARNING,CONFLICT
from state import session_manager as SM

ctx=SM.get_context(); path=Path("results/metrics/evaluation_metrics.json")
try:
    metrics=load_evaluation_metrics(path)
except EvaluationDataError as exc:
    st.error(str(exc),icon=":material/error:")
    st.stop()
hs=metrics["hidden_state"];cal=metrics["calibration"]
page_header("Results","Measured performance across the synthetic evaluation suite.",ctx.scenario,ctx.time_min)
render_html('<span class="csx-chip cyan">Synthetic Evaluation</span>')
items=[("Accuracy",pct(hs["accuracy"]),"Hidden-state classification",ACCENT_GREEN),("Macro F1",pct(hs["macro_f1"]),"Across four states",ACCENT_CYAN),("Brier Score",num(hs["multiclass_brier"],3),"Lower is better",WARNING),("Calibration Error",pct(cal["expected_calibration_error"]),"Expected calibration error",CONFLICT)]
metric_grid([(label,value,note,colour,False) for label,value,note,colour in items])

left,right=st.columns(2,gap="medium")
with left:
    with panel("Confusion matrix"):
        st.plotly_chart(charts.confusion_heatmap(np.array(hs["confusion_matrix"]),hs["confusion_matrix_labels"],height=300),width="stretch",key="results_cm")
with right:
    with panel("Scenario-wise accuracy"):
        st.plotly_chart(charts.scenario_bar_chart(metrics["per_scenario"],height=300),width="stretch",key="results_scenario")

left,right=st.columns(2,gap="medium")
with left:
    with panel("Sensor-fault false-alarm comparison"):
        sf=metrics["sensor_fault_scenario_only"]
        render_html(f'''<div style="display:flex;gap:18px;align-items:flex-end;padding:10px 0 18px"><div><div style="font-size:11px;color:#6F7F78">CellSafe-X</div><div style="font-size:26px;font-weight:750;color:#246B4B">{sf['bayesian_false_alarms']:,}</div></div><div><div style="font-size:11px;color:#6F7F78">Critical threshold</div><div style="font-size:26px;font-weight:750;color:#D7A442">{sf['threshold_critical_false_alarms']:,}</div></div></div>''')
        st.caption("False alarms on the affected cell during the temperature-sensor-fault scenario.")
with right:
    with panel("Current limitations"):
        render_html('<div class="csx-limitations"><ul><li>Evaluation uses synthetic scenarios and hand-specified parameters.</li><li>Posterior probabilities remain over-confident in the highest-risk bin.</li><li>Cells are filtered independently; thermal coupling is observed, not jointly inferred.</li><li>Results do not establish real-world safety or deployment readiness.</li></ul></div>')
