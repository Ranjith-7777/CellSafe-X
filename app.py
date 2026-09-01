"""
CellSafe-X — Bayesian Battery Safety Digital Twin
=================================================

Application entry point.  Run with:

    streamlit run app.py

This module owns only the shell: page configuration, global styling, shared
session-state initialisation and the persistent left navigation.  Each of the
seven pages lives in `pages/` and reads its data from one immutable
`PageContext` snapshot produced by `state.session_manager`.

Every probability shown anywhere in this application is produced by the models
package — exact Bayesian filtering, Bayesian sensor-reliability inference, a
Bayesian root-cause classifier, exact k-step forecasting and minimum-expected-
loss decision making.  Nothing on screen is a hardcoded percentage, and all data
is SIMULATED.
"""

from __future__ import annotations

import streamlit as st

from dashboard.battery_twin import TWIN_CSS
from dashboard.components import COMPONENTS_CSS
from dashboard.network_graph import NETWORK_CSS
from dashboard.navigation import render_brand, render_nav, render_status_footer
from dashboard.styles import CSS
from state import session_manager as SM

st.set_page_config(
    page_title="CellSafe-X | Bayesian Battery Safety Twin",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Global styling is injected once per run, before any page renders.
st.markdown(CSS, unsafe_allow_html=True)
st.markdown(COMPONENTS_CSS, unsafe_allow_html=True)
st.markdown(TWIN_CSS, unsafe_allow_html=True)
st.markdown(NETWORK_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# pages
# ---------------------------------------------------------------------------
PAGES = [
    st.Page("pages/1_home.py", title="Overview", icon=":material/dashboard:", url_path="overview", default=True),
    st.Page("pages/2_live_battery_twin.py", title="Digital Twin", icon=":material/battery_charging_full:", url_path="digital-twin"),
    st.Page("pages/4_bayesian_analysis.py", title="Bayesian Network", icon=":material/account_tree:", url_path="bayesian-network"),
    st.Page("pages/5_risk_and_decisions.py", title="Analysis", icon=":material/analytics:", url_path="analysis"),
    st.Page("pages/6_evaluation.py", title="Results", icon=":material/assessment:", url_path="results"),
]


def main() -> None:
    # `position="hidden"` suppresses Streamlit's built-in page list so that the
    # custom navigation below is the only one on screen, while keeping real
    # multi-page routing (each page has its own URL and can be opened directly).
    nav = st.navigation(PAGES, position="hidden")

    # Initialise shared state BEFORE the page body runs, so a deep link to any
    # page initialises exactly as if the user had arrived through Home.
    SM.init_session()
    ctx = SM.get_context()

    render_brand()
    render_nav(PAGES, nav.url_path)
    render_status_footer()

    nav.run()


if __name__ == "__main__":
    main()
