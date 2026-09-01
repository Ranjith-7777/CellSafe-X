"""Restrained six-cell battery digital-twin rendering."""
from dashboard.layout import esc, num, pct, render_html, unit_interval
from dashboard.styles import CONFLICT, STATE_COLORS
from state import session_manager as SM

def render_battery_twin(ctx, compact=False):
    cells=[]
    for i,out in enumerate(ctx.cell_states):
        reading=out.reading; soc=unit_interval(reading["soc"])
        state=ctx.state_of(i); conflict=SM.is_conflicted(out.sensor.p_faulty)
        colour=CONFLICT if conflict else STATE_COLORS.get(state,CONFLICT)
        selected=" selected" if i==ctx.focus_cell else ""
        cells.append(f'''<div class="bt-cell{selected}"><div class="bt-cap"></div><div class="bt-body"><div class="bt-fill" style="height:{soc*100:.1f}%;background:{colour}"></div><div class="bt-copy"><b>Cell {i+1}</b><span>{num(reading['temp_primary'],1)} °C</span><small>{pct(soc,0)} SOC</small></div></div></div>''')
    render_html(f'''<div class="bt-wrap{' compact' if compact else ''}"><div class="bt-pack">{''.join(cells)}</div><div class="bt-legend"><span><i style="background:#43A777"></i>Healthy</span><span><i style="background:#D7A442"></i>Heating</span><span><i style="background:#E47D4D"></i>Pre-runaway</span><span><i style="background:#D95362"></i>Runaway</span><span><i style="background:#8067C7"></i>Sensor conflict</span></div></div>''')

TWIN_CSS='''<style>
.bt-wrap{background:#FAFCFB;border:1px solid #DFE7E2;border-radius:12px;padding:20px 18px 15px}.bt-pack{display:grid;grid-template-columns:repeat(6,minmax(76px,1fr));gap:12px;align-items:end}.bt-cell{position:relative;padding-top:8px}.bt-cap{width:34%;height:8px;margin:auto;background:#D9E3DE;border:1px solid #CAD7D1;border-bottom:0;border-radius:4px 4px 0 0}.bt-body{height:176px;position:relative;overflow:hidden;background:#fff;border:1px solid #CBD8D2;border-radius:8px}.bt-fill{position:absolute;bottom:0;left:0;right:0;opacity:.23}.bt-copy{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;color:#173C2A}.bt-copy b{font-size:14px;font-weight:650}.bt-copy span{font-size:18px;font-weight:700;color:#173C2A;margin-top:6px}.bt-copy small{font-size:13px;color:#60736A;margin-top:4px}.bt-cell.selected .bt-body{border:2px solid #5FA77D;box-shadow:0 0 0 3px rgba(95,167,125,.13)}.bt-legend{display:flex;justify-content:center;flex-wrap:wrap;gap:16px;margin-top:15px}.bt-legend span{font-size:13px;color:#60736A}.bt-legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}.bt-wrap.compact .bt-body{height:142px}
@media(max-width:700px){.bt-wrap{overflow-x:auto}.bt-pack{min-width:610px}.bt-body{height:150px}.bt-legend{min-width:610px;justify-content:flex-start}}
</style>'''
