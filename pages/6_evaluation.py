"""Concise synthetic evaluation results."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
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

cc=metrics.get("calibration_correction");mb=metrics["alarm_comparison"]["bayesian"].get("missed_dangerous_breakdown");rc=metrics["root_cause"]
if cc and mb:
    items2=[
        ("ECE (raw → calibrated)",f"{cc['uncalibrated']['ece_p_dangerous']*100:.1f}% → {cc['calibrated']['ece_p_dangerous']*100:.1f}%","Temperature-scaled posterior, fit on a disjoint seed range",ACCENT_CYAN),
        ("Latent-state safety misses",f"{mb['missed_total']:,}","Ground truth says Pre-Runaway/Runaway but no alarm fired, see docs/EVALUATION.md",WARNING),
        ("...of which thermal-threshold misses",f"{mb['missed_temp_actually_crossed_pre_runaway_threshold']:,}","Genuinely ≥58°C and missed - the safety-critical subset",ACCENT_GREEN),
        ("Root cause (first third → last third)",f"{pct(rc['accuracy_affected_cell_first_third'])} → {pct(rc['accuracy_affected_cell_last_third'])}","Improves as each scenario's signature develops",ACCENT_CYAN),
    ]
    metric_grid([(label,value,note,colour,False) for label,value,note,colour in items2])

left,right=st.columns(2,gap="medium")
with left:
    with panel("Sensor-fault false-alarm comparison"):
        sf=metrics["sensor_fault_scenario_only"]
        render_html(f'''<div style="display:flex;gap:18px;align-items:flex-end;padding:10px 0 18px"><div><div style="font-size:11px;color:#6F7F78">CellSafe-X</div><div style="font-size:26px;font-weight:750;color:#246B4B">{sf['bayesian_false_alarms']:,}</div></div><div><div style="font-size:11px;color:#6F7F78">Critical threshold</div><div style="font-size:26px;font-weight:750;color:#D7A442">{sf['threshold_critical_false_alarms']:,}</div></div></div>''')
        st.caption("False alarms on the affected cell during the temperature-sensor-fault scenario.")
with right:
    with panel("Current limitations"):
        render_html('<div class="csx-limitations"><ul><li>Evaluation uses synthetic scenarios and hand-specified parameters.</li><li>Posterior probabilities remain over-confident in the highest-risk bin.</li><li>Cells are filtered independently; thermal coupling is observed, not jointly inferred.</li><li>Results do not establish real-world safety or deployment readiness.</li></ul></div>')

bw_dir = Path("results/baum_welch")
if (bw_dir / "metrics_engineered.json").exists() and (bw_dir / "metrics_learned.json").exists():
    with st.expander("Baum-Welch learned HMM (Phase 7C, experimental - not the production model)"):
        st.caption(
            "The engineered HMM above remains the production model everywhere else in "
            "this dashboard. This section only compares it against an EM-learned "
            "variant on the same held-out trajectories - see docs and "
            "`python -m evaluation.evaluate_baum_welch`."
        )
        me = json.loads((bw_dir / "metrics_engineered.json").read_text(encoding="utf-8"))
        ml = json.loads((bw_dir / "metrics_learned.json").read_text(encoding="utf-8"))
        hse, hsl = me["hidden_state"], ml["hidden_state"]
        st.dataframe(pd.DataFrame({
            "Metric": ["Accuracy", "Macro F1", "Brier", "NLL", "False alarms", "Genuinely-hot misses"],
            "Engineered": [
                num(hse["accuracy"], 3), num(hse["macro_f1"], 3), num(hse["multiclass_brier"], 3),
                num(hse["negative_log_likelihood"], 3),
                str(me["alarm_comparison"]["bayesian"]["false_alarms"]),
                str(me["alarm_comparison"]["bayesian"]["missed_dangerous_breakdown"]["missed_temp_actually_crossed_pre_runaway_threshold"]),
            ],
            "Baum-Welch": [
                num(hsl["accuracy"], 3), num(hsl["macro_f1"], 3), num(hsl["multiclass_brier"], 3),
                num(hsl["negative_log_likelihood"], 3),
                str(ml["alarm_comparison"]["bayesian"]["false_alarms"]),
                str(ml["alarm_comparison"]["bayesian"]["missed_dangerous_breakdown"]["missed_temp_actually_crossed_pre_runaway_threshold"]),
            ],
        }), width="stretch", hide_index=True)

        if (bw_dir / "convergence.csv").exists():
            conv = pd.read_csv(bw_dir / "convergence.csv")
            st.caption("EM convergence (training log-likelihood per iteration)")
            st.line_chart(conv.pivot(index="iteration", columns="initialization", values="log_likelihood"))

        if (bw_dir / "learned_parameters.json").exists() and (bw_dir / "engineered_parameters.json").exists():
            lp = json.loads((bw_dir / "learned_parameters.json").read_text(encoding="utf-8"))
            ep = json.loads((bw_dir / "engineered_parameters.json").read_text(encoding="utf-8"))
            st.caption(f"Selected initialisation: **{lp['selected_initialization']}**")
            c1, c2 = st.columns(2)
            with c1:
                st.caption("Engineered transition matrix")
                st.dataframe(pd.DataFrame(np.array(ep["A"]).round(4)), width="stretch")
            with c2:
                st.caption("Baum-Welch learned transition matrix")
                st.dataframe(pd.DataFrame(np.array(lp["params"]["A"]).round(4)), width="stretch")
