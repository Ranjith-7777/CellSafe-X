"""Responsive, bounded SVG for the CellSafe-X Bayesian dependency graph."""

from dashboard.layout import render_html


def render_bayesian_network() -> None:
    """Render the complete network in a compact left-to-right layout."""
    render_html(
        """
        <div class="bn-scroll">
        <svg class="bn-graph" viewBox="0 0 1400 540" preserveAspectRatio="xMidYMid meet"
             role="img" aria-label="CellSafe-X Bayesian network">
          <defs>
            <marker id="bn-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">
              <path d="M0,0 L8,4 L0,8 Z" fill="#9BAEA5"/>
            </marker>
          </defs>
          <g class="bn-edges">
            <path d="M180 92 H230 V183 H280"/><path d="M180 219 H280"/>
            <path d="M180 448 H215 V205 H280"/>
            <path d="M470 205 H510 V60 H565"/><path d="M470 205 H520 V145 H565"/>
            <path d="M470 205 H530 V230 H565"/><path d="M470 205 H530 V315 H565"/>
            <path d="M470 205 H520 V400 H565"/><path d="M470 205 H510 V485 H565"/>
            <path d="M180 448 H500 V60 H565"/><path d="M500 448 V145 H565"/>
            <path d="M765 60 H815 V236 H880"/><path d="M765 145 H825 V246 H880"/>
            <path d="M765 230 H880"/><path d="M765 315 H825 V266 H880"/>
            <path d="M765 400 H815 V276 H880"/><path d="M765 485 H805 V286 H880"/>
            <path d="M1080 260 H1135 V180 H1180"/><path d="M1080 260 H1135 V340 H1180"/>
          </g>
          <g class="bn-node teal"><rect x="20" y="65" width="160" height="54"/><text x="100" y="92">Previous State</text></g>
          <g class="bn-node amber"><rect x="20" y="192" width="160" height="54"/><text x="100" y="219">Root Cause</text></g>
          <g class="bn-node purple"><rect x="20" y="421" width="160" height="54"/><text x="100" y="448">Sensor Reliability</text></g>
          <g class="bn-node teal strong"><rect x="280" y="178" width="190" height="54"/><text x="375" y="205">Current Hidden State</text></g>
          <g class="bn-node evidence"><rect x="565" y="33" width="200" height="54"/><text x="665" y="60">Primary Temperature</text></g>
          <g class="bn-node evidence"><rect x="565" y="118" width="200" height="54"/><text x="665" y="145">Backup Temperature</text></g>
          <g class="bn-node evidence"><rect x="565" y="203" width="200" height="54"/><text x="665" y="230">Voltage Evidence</text></g>
          <g class="bn-node evidence"><rect x="565" y="288" width="200" height="54"/><text x="665" y="315">Gas Evidence</text></g>
          <g class="bn-node evidence"><rect x="565" y="373" width="200" height="54"/><text x="665" y="400">Cooling Evidence</text></g>
          <g class="bn-node evidence"><rect x="565" y="458" width="200" height="54"/><text x="665" y="485">Neighbour Evidence</text></g>
          <g class="bn-node posterior"><rect x="880" y="230" width="200" height="60"/><text x="980" y="260">Updated Posterior</text></g>
          <g class="bn-node outcome"><rect x="1180" y="153" width="180" height="54"/><text x="1270" y="180">Risk Forecast</text></g>
          <g class="bn-node outcome"><rect x="1180" y="313" width="180" height="54"/><text x="1270" y="340">Decision</text></g>
        </svg></div>
        """
    )


NETWORK_CSS = """
<style>
.bn-scroll{width:100%;overflow-x:auto;overflow-y:hidden}
.bn-graph{display:block;width:100%;height:auto;max-height:560px;min-height:430px;background:#fff}
.bn-edges path{fill:none;stroke:#9BAEA5;stroke-width:1.35;opacity:.72;marker-end:url(#bn-arrow)}
.bn-node rect{rx:10;ry:10;stroke:#C9D6D0;stroke-width:1.2}
.bn-node text{font-family:Inter,Aptos,"Segoe UI",sans-serif;font-size:15px;font-weight:500;fill:#173C2A;text-anchor:middle;dominant-baseline:middle}
.bn-node.teal rect{fill:#DDF2EA}.bn-node.amber rect{fill:#F7EACB}.bn-node.purple rect{fill:#E8E1F8}
.bn-node.evidence rect{fill:#E4F0F6}.bn-node.outcome rect{fill:#DDEFD9}
.bn-node.strong rect{fill:#BFE3D5;stroke:#5FA77D}.bn-node.posterior rect{fill:#285F42;stroke:#285F42}
.bn-node.posterior text{fill:#fff;font-weight:650}
@media(max-width:800px){.bn-graph{width:1000px;min-height:385px}.bn-scroll{padding-bottom:8px}}
</style>
"""
