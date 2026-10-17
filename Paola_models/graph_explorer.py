"""Build a standalone offline graph and uncertainty explorer from saved diagnostics."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.offline import get_plotlyjs


PAGE = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>Paola · graph model explorer</title>
<style>
body{font:16px system-ui;background:#f5f7fa;color:#152238;margin:24px auto;max-width:1150px;padding:0 20px}
h1{font-size:28px}p{line-height:1.5}select{font:inherit;padding:8px;margin:5px;border:1px solid #aebdcb;border-radius:6px}
.card{background:white;border:1px solid #dce3eb;border-radius:12px;padding:18px;margin:16px 0}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}svg{width:100%;height:350px}
.muted{color:#51627a;font-size:14px}.stats{font-weight:600} @media(max-width:750px){.grid{grid-template-columns:1fr}}
</style>
<h1>How this graph model uses neighboring zones</h1>
<p>Four anonymous price zones · June–August 2026 development confirmation · clustered Bayesian graph.
Arrows run from <b>source → receiving zone</b>. The official data does not identify countries or transmission links.
These are learned dependencies and model sensitivities, not evidence of causal trade effects.</p>
<div class="card"><label>View <select id="view">
<option value="attention">Learned attention</option><option value="ablation">Effect of removing an edge on MAE</option>
<option value="sensitivity">Sensitivity to a +10 €/MWh source curve shift</option></select></label>
<label>Date <select id="date"></select></label><label>Hour <select id="hour"></select></label>
<p id="explain"></p><div class="grid"><svg id="network" viewBox="0 0 440 350" role="img" aria-label="Directed model graph"></svg>
<div id="matrix"></div></div><p class="muted">Node colors show training-derived zone similarity clusters for the selected fold.
Self-attention appears in the matrix only. Width reflects absolute edge magnitude; compare values within the same view.
Removing an edge masks it in both graph blocks and renormalizes attention; alternate two-hop paths can remain.</p></div>
<div class="card"><h2>Quarter-hour predictions and uncertainty</h2><label>Zone <select id="zone"></select></label>
<div id="stats" class="stats"></div><div id="prices" style="height:410px"></div>
<p class="muted">Shaded band: empirically calibrated marginal 90% interval. Coverage is measured against the saved actual quarter prices.
It is not a simultaneous whole-day guarantee. Point predictions and every raw sampled scenario preserve the supplied hourly averages;
the interval endpoints do not have that constraint. No inference-time fitting.</p></div>
<p class="muted">Point MAE comparison: see advanced_results.md. Bayesian inference covers output weights and adaptive graph offsets,
with a deterministic encoder. Attention is averaged over the two blocks. All displayed results were already inspected development data.</p>
<script>__PLOTLY__</script><script>
const D=__DATA__, zones=['ZONE_0','ZONE_1','ZONE_2','ZONE_3'];
const el=id=>document.getElementById(id), opt=(v,t)=>{let o=document.createElement('option');o.value=v;o.textContent=t;return o};
const dates=[...new Set(D.prob.map(r=>r.delivery_date))].sort();
el('date').append(opt('all','All confirmation dates')); dates.forEach(d=>el('date').append(opt(d,d)));
el('hour').append(opt('all','All hours'));for(let h=1;h<=24;h++)el('hour').append(opt(String(h),h));
zones.forEach(z=>el('zone').append(opt(z,z)));
const avg=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:0;
function update(){
 const day=el('date').value,h=el('hour').value,view=el('view').value;
 el('hour').disabled=view!=='attention';
 const source=view==='attention'?D.graph:(view==='ablation'?D.ablation:D.sensitivity);
 const rows=source.filter(r=>(day==='all'||r.delivery_date===day)&&(view!=='attention'||h==='all'||r.hour_index===+h));
 const key=view==='attention'?'attention':(view==='ablation'?'MAE_increase_when_edge_removed':'mean_absolute_target_shape_change');
 const matrix=zones.map(t=>zones.map(s=>{let a=rows.filter(r=>r.target===t&&r.source===s);return a.length?avg(a.map(r=>r[key])):null}));
 const texts={attention:'Attention is a normalized message weight, not a measured effect. Each receiving row sums to one.',
 ablation:'Positive: removing this edge increased the receiving zone’s MAE (the edge helped). Negative: removal improved MAE. Units: €/MWh. This is a frozen-model ablation, not retraining.',
 sensitivity:'Mean absolute change in the receiving zone’s predicted quarter-hour shape after increasing all supplied source-zone hourly prices by 10 €/MWh. Includes changes through shared regime probabilities; this is an input stress test, not a causal estimate.'};
 el('explain').textContent=texts[view];
 Plotly.react('matrix',[{type:'heatmap',x:zones,y:zones,z:matrix,colorscale:view==='ablation'?'RdBu':'Blues',
 zmid:view==='ablation'?0:undefined,reversescale:view==='ablation',hovertemplate:'Source %{x}<br>Target %{y}<br>Value %{z:.5f}<extra></extra>'}],
 {margin:{l:72,r:30,t:28,b:65},xaxis:{title:'Source'},yaxis:{title:'Receiver',autorange:'reversed'},height:350}, {responsive:true,displaylogo:false});
 const pos=[[105,75],[335,75],[105,275],[335,275]], colors=['#276caa','#d47b2d','#607d35','#8953a6'];
 const finite=matrix.flat().filter(v=>v!==null);const max=Math.max(...finite.map(Math.abs),1e-8);
 let svg='<defs><marker id="arrow" markerWidth="7" markerHeight="7" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="context-stroke"/></marker></defs>';
 for(let t=0;t<4;t++)for(let s=0;s<4;s++)if(s!==t){
  let value=matrix[t][s];if(value===null)continue;
  const [x1,y1]=pos[s],[x2,y2]=pos[t],dx=x2-x1,dy=y2-y1,len=Math.hypot(dx,dy),nx=-dy/len,ny=dx/len;
  const a=[x1+dx/len*38+nx*7,y1+dy/len*38+ny*7],b=[x2-dx/len*38+nx*7,y2-dy/len*38+ny*7];
  svg+=`<path d="M${a[0]},${a[1]} L${b[0]},${b[1]}" stroke="${value<0?'#bb5268':'#3575a5'}" opacity=".75" stroke-width="${.5+5*Math.abs(value)/max}" marker-end="url(#arrow)"><title>${zones[s]} → ${zones[t]}: ${value.toFixed(5)}</title></path>`;
 }
 let graphDay=day==='all'?dates[dates.length-1]:day;
 for(let z=0;z<4;z++){
  let row=D.graph.find(r=>r.delivery_date===graphDay&&r.source===zones[z]);let cluster=row?row.source_cluster:0;
  svg+=`<circle cx="${pos[z][0]}" cy="${pos[z][1]}" r="33" fill="${colors[cluster%colors.length]}"/><text x="${pos[z][0]}" y="${pos[z][1]+5}" text-anchor="middle" fill="white" font-size="13">${zones[z]}</text>`;
 }
 el('network').innerHTML=svg;
 const zone=el('zone').value,plotDay=day==='all'?dates[0]:day;
 const p=D.prob.filter(r=>r.delivery_date===plotDay&&r.market_id===zone).sort((a,b)=>a.hour_index-b.hour_index||a.quarter_index-b.quarter_index);
 const x=p.map(r=>(r.hour_index-1)+(r.quarter_index-1)/4),cov=avg(p.map(r=>+(r.price_eur_mwh>=r.lower90&&r.price_eur_mwh<=r.upper90)));
 el('stats').textContent=`${plotDay} · ${zone} · MAE ${avg(p.map(r=>Math.abs(r.point-r.price_eur_mwh))).toFixed(3)} €/MWh · marginal coverage ${(100*cov).toFixed(1)}% · mean interval width ${avg(p.map(r=>r.upper90-r.lower90)).toFixed(2)} €/MWh`;
 Plotly.react('prices',[
 {x,y:p.map(r=>r.lower90),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip'},
 {x,y:p.map(r=>r.upper90),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:'rgba(39,108,170,.17)',name:'Calibrated 90% band'},
 {x,y:p.map(r=>r.point),mode:'lines',line:{color:'#276caa'},name:'Bayesian graph point'},
 {x,y:p.map(r=>r.price_eur_mwh),mode:'lines',line:{color:'#242d3a'},name:'Actual'}],
 {margin:{l:65,r:15,t:30,b:50},xaxis:{title:'Hour of day'},yaxis:{title:'€/MWh'},legend:{orientation:'h'},hovermode:'x unified'}, {responsive:true,displaylogo:false});
}
['view','date','hour','zone'].forEach(id=>el(id).onchange=update);update();
</script></html>'''


def build(folder):
    files={'graph':'hourly_graph','ablation':'edge_ablation','sensitivity':'neighbor_sensitivity',
           'prob':'probabilistic_predictions'}
    frames={key:pd.read_csv(folder/(value+'.csv')) for key,value in files.items()}
    payload={key:json.loads(frame.to_json(orient='records',double_precision=8)) for key,frame in frames.items()}
    # All inputs originate in this local experiment; escaping prevents embedded closing tags.
    serialized=json.dumps(payload,separators=(',',':')).replace('</','<\\/')
    path=folder/'graph_explorer.html'
    path.write_text(PAGE.replace('__PLOTLY__',get_plotlyjs()).replace('__DATA__',serialized))
    static_plot(folder,frames)
    return path


def static_plot(folder,frames):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    keys=[('graph','attention','Mean learned attention'),
          ('ablation','MAE_increase_when_edge_removed','MAE change when edge removed'),
          ('sensitivity','mean_absolute_target_shape_change','Source +10 €/MWh sensitivity')]
    for ax,(name,key,title) in zip(axes,keys):
        table=frames[name].pivot_table(index='target',columns='source',values=key,aggfunc='mean').reindex(index=[f'ZONE_{i}' for i in range(4)],columns=[f'ZONE_{i}' for i in range(4)])
        bound=max(np.nanmax(np.abs(table)),1e-8)
        im=ax.imshow(table,cmap='RdBu_r' if name=='ablation' else 'Blues',
                     vmin=-bound if name=='ablation' else 0,vmax=bound)
        for i in range(4):
            for j in range(4):
                if np.isfinite(table.iloc[i,j]):ax.text(j,i,f'{table.iloc[i,j]:.3f}',ha='center',va='center',fontsize=9)
        ax.set(xticks=range(4),yticks=range(4),xticklabels=table.columns,yticklabels=table.index,
               xlabel='Source →',ylabel='Receiving zone',title=title)
        fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('Bayesian graph · June–August development · model reliance, not causal trade flows',fontsize=12)
    fig.savefig(folder/'graph_diagnostics.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=Path('Paola_models/results/adaptive_probabilistic_study'))
    print(build(p.parse_args().results))
