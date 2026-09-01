"""Presentation layer for the CellSafe-X dashboard.

    styles.py        palette, global CSS, shared Plotly layout
    layout.py        page header, panels, cards, HTML-escaping helpers
    charts.py        every Plotly figure, on the dark theme
    components.py    reusable DOM components (cell strip, banners, tables)
    navigation.py    persistent left navigation and shared simulation controls
    battery_twin.py  the neon six-cell battery digital twin (inline SVG)

Nothing in this package computes a probability; it renders values that the
`models` package already produced.
"""

from dashboard.styles import CSS

__all__ = ["CSS"]
