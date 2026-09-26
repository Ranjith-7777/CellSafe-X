"""Focused state, diagnosis, risk, and decision analysis."""
import pandas as pd
import streamlit as st
from config import model_parameters as P
from dashboard import charts,components as C
from dashboard.layout import page_header,panel,metric_card,metric_grid,pct,num,kv_rows,section
from dashboard.navigation import cell_selector,render_timeline
from dashboard.styles import ACCENT_CYAN,CONFLICT,risk_color
from models import compare_interventions, compare_with_threshold, forecast_pack_propagation, recommend_measurement
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

with st.expander("Model-based intervention forecast (not causal inference - see note)"):
    st.caption(
        "Each action gets an explicit, documented alternative transition matrix "
        "(config.model_parameters.INTERVENTION_TRANSITION_EFFECTS). This is a "
        "controlled-transition forecast from the current posterior, not Pearlian "
        "do(X) causal identification, and it never overrides the expected-loss "
        "recommendation above."
    )
    ranking = compare_interventions(out.posterior, horizon_min=15.0)
    st.dataframe(
        pd.DataFrame(
            {
                "Action": [r.action for r in ranking],
                "P(dangerous) +15min if no action": [r.baseline_dangerous for r in ranking],
                "P(dangerous) +15min under action": [r.intervention_dangerous for r in ranking],
                "Risk reduction": [r.risk_reduction for r in ranking],
            }
        ),
        width="stretch", hide_index=True,
    )
with st.expander("Pack propagation (model-based, not experimentally validated)"):
    st.caption(
        "Cells are filtered independently; this is a forecasting layer on top "
        "of those independent posteriors using the pack's linear-chain "
        "topology, not an exact joint multi-cell model. See "
        "config.model_parameters section 10b for the full disclosure."
    )
    cell_posteriors = {o.cell: o.posterior for o in ctx.cell_states}
    prop = forecast_pack_propagation(cell_posteriors, horizon_min=15.0)
    src = prop.most_likely_propagation_source
    st.dataframe(
        pd.DataFrame(
            {
                "Cell": list(prop.per_cell),
                "Now": [f.current_dangerous for f in prop.per_cell.values()],
                "+15min, no coupling": [f.future_dangerous_independent for f in prop.per_cell.values()],
                "+15min, with coupling": [f.future_dangerous_coupled for f in prop.per_cell.values()],
                "Propagation increment": [f.propagation_increment for f in prop.per_cell.values()],
            }
        ),
        width="stretch", hide_index=True,
    )
    kv_rows([
        ("Most likely propagation source", f"Cell {src}" if src is not None else "n/a"),
        ("Most vulnerable neighbour", f"Cell {prop.most_vulnerable_neighbour}" if prop.most_vulnerable_neighbour is not None else "n/a"),
        ("P(at least one cell dangerous)", pct(prop.pack_at_least_one_dangerous)),
        ("Expected number of dangerous cells", num(prop.expected_dangerous_cells, 2)),
    ])
    if src is not None:
        iso = forecast_pack_propagation(cell_posteriors, horizon_min=15.0, isolated_cells=frozenset({src}))
        st.caption(
            f"If Cell {src} were isolated: neighbour risk would move to "
            + ", ".join(
                f"cell {n}: {iso.per_cell[n].future_dangerous_coupled:.3f} (was {prop.per_cell[n].future_dangerous_coupled:.3f})"
                for n in P.pack_neighbours(src)
            )
        )

with st.expander("Active sensing (Value of Information)"):
    st.caption(
        "Expected information gain uses the same emission likelihood the "
        "filter uses - not an arbitrary sensor weight. Requesting a "
        "measurement never physically changes the pack (see the Phase 3 "
        "intervention forecast above, where this action stays neutral)."
    )
    rec = recommend_measurement(out.posterior)
    kv_rows([
        ("Posterior confidence", pct(rec.current_confidence)),
        ("P(dangerous) now", pct(rec.current_dangerous_probability)),
        ("Best next measurement", rec.best_channel),
        ("Expected information gain", f"{rec.information_gain[0].expected_information_gain:.4f} nats"),
        ("Expected value of information", num(rec.value_of_information.expected_value_of_information, 3)),
        ("Worth requesting?", "Yes" if rec.should_request_measurement else "No"),
    ])
    for reason in rec.reasons:
        st.caption(f"- {reason}")

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
