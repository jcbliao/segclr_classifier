#!/usr/bin/env python3
"""Build standalone Plotly pages for the two analysis notebooks.

The pages contain their data and controls; GitHub Pages only serves static files.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def records(frame: pd.DataFrame, columns=None) -> str:
    if columns is not None:
        frame = frame[[c for c in columns if c in frame.columns]]
    value = frame.replace({np.nan: None}).to_dict("records")
    return json.dumps(value, separators=(",", ":")).replace("</", "<\\/")


STYLE = """
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f6f8fb}
body{margin:0}.top{padding:18px 24px;background:white;border-bottom:1px solid #dce2ea}
h1{font-size:20px;margin:0 0 5px}.sub{font-size:13px;color:#596579}
.controls{display:flex;flex-wrap:wrap;gap:12px;padding:14px 24px;background:#fff;border-bottom:1px solid #dce2ea}
label{font-size:12px;color:#596579;display:flex;flex-direction:column;gap:4px}
select{font:inherit;color:#172033;background:white;border:1px solid #b8c2d1;border-radius:6px;padding:6px 28px 6px 8px}
#plot{min-height:620px;background:white;margin:18px;border-radius:8px;box-shadow:0 1px 5px #23304818}
.home{color:#315ec9;text-decoration:none}.note{padding:0 24px 20px;color:#596579;font-size:12px}
"""


COMMON_JS = """
const gd=document.getElementById('plot');
const palette=['#4C78A8','#F58518','#54A24B','#E45756','#72B7B2','#B279A2','#FF9DA6','#9D755D','#BAB0AC'];
function uniq(xs){return [...new Set(xs.filter(x=>x!==null&&x!==undefined))]}
function control(id,label,values,shown){const wrap=document.createElement('label');wrap.textContent=label;
 const s=document.createElement('select');s.id=id; values.forEach((v,i)=>{const o=document.createElement('option');
 o.value=i;o.textContent=shown?shown(v):String(v);o._value=v;s.appendChild(o)});wrap.appendChild(s);
 document.querySelector('.controls').appendChild(wrap);s.addEventListener('change',draw);return s}
function val(s){return s.options[s.selectedIndex]._value}
const config={responsive:true,displaylogo:false,toImageButtonOptions:{format:'png',scale:2}};
"""


def page(slug: str, title: str, subtitle: str, data: str, script: str):
    html = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><script src="https://cdn.plot.ly/plotly-3.1.0.min.js"></script><style>{STYLE}</style></head>
<body><div class="top"><h1>{title}</h1><div class="sub">{subtitle} · <a class="home" href="index.html">all figures</a></div></div>
<div class="controls"></div><div id="plot"></div><div class="note">Use the toolbar to pan, zoom, reset, or download a PNG. Controls and hover work without a notebook kernel.</div>
<script>const rows={data};{COMMON_JS}{script}</script></body></html>"""
    (OUT / f"{slug}.html").write_text(html)


def architecture_pages():
    from analysis import architecture_comparison as ac
    payloads = ac.load_confusions(ROOT / "results")
    frame = ac.load_best_epochs(ROOT / "results", payloads)
    avg = ac.mean_over_folds(frame)
    cols = ["architecture", "model", "n_embeddings", "folds"] + [c for c in avg if c.startswith(("window_", "cell_"))]
    data = records(avg, cols)
    page("architecture-models-by-count", "Every model at each node count",
         "architecture_comparison.ipynb · fold-averaged", data, r"""
function draw(){let traces=[];for(const n of [10,20,40]){const p=rows.filter(r=>r.n_embeddings===n).sort((a,b)=>b.window_macro_f1-a.window_macro_f1);
 traces.push({type:'bar',orientation:'h',x:p.map(r=>r.window_macro_f1),y:p.map(r=>r.model),name:`n=${n}`,
 customdata:p.map(r=>[r.architecture,r.folds]),hovertemplate:'%{y}<br>macro F1 %{x:.3f}<br>%{customdata[0]} · %{customdata[1]} fold(s)<extra></extra>'})}
 Plotly.react(gd,traces,{height:Math.max(650,rows.length*12),barmode:'group',xaxis:{title:'window macro F1'},yaxis:{automargin:true},margin:{l:210,t:35},legend:{orientation:'h'}},config)}draw();""")

    classes = [None] + ac.class_names(payloads)
    page("architecture-across-counts", "Architecture comparison across node counts",
         "architecture_comparison.ipynb · class, level, and metric controls", data, r"""
const cs=control('class','Class',""" + json.dumps(classes) + r""",x=>x===null?'All classes (macro)':x);
const level=control('level','Level',['window','cell']);const metric=control('metric','Metric',['f1','recall','precision','accuracy']);
function draw(){const c=val(cs),scope=val(level),m=val(metric);let col;if(c===null){col=scope+'_'+({f1:'macro_f1',recall:'balanced_accuracy',precision:'macro_precision',accuracy:'accuracy'}[m])}else col=scope+'_'+(m==='accuracy'?'recall':m)+'_'+c;
 const traces=[];uniq(rows.map(r=>r.model)).forEach((model,i)=>{const p=rows.filter(r=>r.model===model&&r[col]!=null).sort((a,b)=>a.n_embeddings-b.n_embeddings);if(p.length)traces.push({x:p.map(r=>r.n_embeddings),y:p.map(r=>r[col]),mode:'lines+markers',name:model,legendgroup:model,hovertemplate:`${model}<br>n=%{x}<br>${m}=%{y:.3f}<extra></extra>`})});
 Plotly.react(gd,traces,{height:720,xaxis:{title:'Number of embeddings',tickvals:[10,20,40]},yaxis:{title:(c||'macro')+' '+m},margin:{l:70,r:25,t:35,b:65},legend:{orientation:'h',y:-.18}},config)}draw();""")

    packed=[]
    for run,p in payloads.items():
        desc=ac.describe_run(run)
        if desc is None: continue
        packed.append({"architecture":desc[0],"model":ac.model_label(run),"n":desc[2],"classes":p["classes"],
                       "window":p["window_test_metrics"],"cell":p["test_metrics"]})
    page("architecture-confusion-matrices", "Confusion matrices across node count",
         "architecture_comparison.ipynb · architecture, ablation, level, and normalization controls",
         json.dumps(packed,separators=(",", ":")), r"""
const arch=control('arch','Architecture',uniq(rows.map(r=>r.architecture)));const model=control('model','Ablation',[]);
const level=control('level','Level',['window','cell']);const norm=control('norm','Scale',['normalized','raw']);
function sync(){const old=model.options[model.selectedIndex]?val(model):null;model.innerHTML='';uniq(rows.filter(r=>r.architecture===val(arch)).map(r=>r.model)).forEach((v,i)=>{const o=new Option(v.startsWith(val(arch))?(v.slice(val(arch).length).replace(/^ \+ /,'')||'no position, no LPE'):v,i);o._value=v;model.add(o)});if(old){[...model.options].forEach((o,i)=>{if(o._value===old)model.selectedIndex=i})}}
arch.addEventListener('change',()=>{sync();draw()});model.addEventListener('change',draw);function draw(){const p=rows.filter(r=>r.architecture===val(arch)&&r.model===val(model)).sort((a,b)=>a.n-b.n);let traces=[];
 p.forEach((r,i)=>{const m=r[val(level)],z=m.confusion_matrix.map(x=>x.slice());if(val(norm)==='normalized')z.forEach(row=>{const s=row.reduce((a,b)=>a+b,0);row.forEach((x,j)=>row[j]=s?x/s:0)});traces.push({type:'heatmap',z,x:r.classes,y:r.classes,xaxis:'x'+(i?i+1:''),yaxis:'y'+(i?i+1:''),coloraxis:'coloraxis',text:z,hovertemplate:'true %{y}<br>predicted %{x}<br>%{z:.3f}<extra></extra>'})});
 const layout={height:650,grid:{rows:1,columns:Math.max(1,p.length),pattern:'independent'},coloraxis:{colorscale:'Blues',cmin:0,cmax:val(norm)==='normalized'?1:undefined},margin:{l:100,r:25,t:60,b:120},annotations:p.map((r,i)=>({text:`n=${r.n}`,xref:`x${i?i+1:''} domain`,yref:'paper',x:.5,y:1.08,showarrow:false}))};Plotly.react(gd,traces,layout,config)}sync();draw();""")


def feature_pages():
    from analysis import feature_prediction_correlation as fp
    runs=fp.cached_runs(fp.usable_runs())
    frames=fp.load_frames(runs)
    bins,quant,overall,cells=frames["bins"],frames["quantiles"],frames["overall"],frames["cells"]
    class_values=[None]+list(dict.fromkeys(bins.class_name.dropna().astype(str)))
    def feature_page(slug,title,features,family=False):
        b=bins[bins.feature.isin(features)]
        q=quant[quant.feature.isin(features)] if len(quant) else quant
        packed={"bins":json.loads(records(b,["architecture","model","n_embeddings","feature","class_name","bin","feature_median","f1"])),
                "quantiles":json.loads(records(q,["architecture","model","n_embeddings","feature","class_name","p25","p50","p75","p90"]))}
        controls=(r"""const arch=control('arch','Architecture',uniq(rows.bins.map(r=>r.architecture)));const model=control('model','Ablation',[]);""" if family else "")+r"""
const cls=control('class','Class',"""+json.dumps(class_values)+r""",x=>x===null?'All classes (macro)':x);
"""+(r"""function sync(){model.innerHTML='';uniq(rows.bins.filter(r=>r.architecture===val(arch)).map(r=>r.model)).forEach((v,i)=>{const o=new Option(v,i);o._value=v;model.add(o)})}arch.addEventListener('change',()=>{sync();draw()});model.addEventListener('change',draw);""" if family else "")+r"""
function draw(){const c=val(cls);let p=rows.bins.filter(r=>(c===null?r.class_name===null:r.class_name===c));"""+(r"p=p.filter(r=>r.model===val(model));" if family else "")+r"""
 const traces=[];const groups={};p.forEach(r=>{const key="""+("r.n_embeddings+'|'+r.feature" if family else "r.model+'|'+r.n_embeddings")+r""";(groups[key]??=[]).push(r)});Object.entries(groups).forEach(([key,g],i)=>{g.sort((a,b)=>a.bin-b.bin);traces.push({x:g.map(r=>r.feature_median),y:g.map(r=>r.f1),mode:'lines+markers',name:key.replace('|',' · n='),hovertemplate:key+'<br>x=%{x:.3g}<br>F1=%{y:.3f}<extra></extra>'})});
 Plotly.react(gd,traces,{height:700,xaxis:{title:'feature value'},yaxis:{title:(c||'macro')+' F1'},margin:{l:70,r:20,t:30,b:70},legend:{orientation:'h',y:-.18}},config)}"""+("sync();" if family else "")+"draw();"
        page(slug,title,"feature_prediction_correlation.ipynb · live HTML controls",json.dumps(packed,separators=(",",":")),controls)
    feature_page("feature-path-length","F1 against path length",["path_distance_um"])
    feature_page("feature-node-density","F1 against node density",["node_density_per_um"])
    for family,features in fp.FEATURE_FAMILIES.items():
        feature_page("feature-"+family.lower().replace(" ","-").replace("(","").replace(")","").replace("^","").replace("/","-"),
                     "F1 against "+family,list(features),True)

    data=records(overall,["architecture","model","n_embeddings","class_name","coarse_class","granular","f1"])
    page("feature-granular-class-bars","Window F1 per granular cell type","feature_prediction_correlation.ipynb · architecture and ablation controls",data,r"""
const arch=control('arch','Architecture',uniq(rows.map(r=>r.architecture)));const model=control('model','Ablation',[]);function sync(){model.innerHTML='';uniq(rows.filter(r=>r.architecture===val(arch)).map(r=>r.model)).forEach((v,i)=>{const o=new Option(v,i);o._value=v;model.add(o)})}arch.addEventListener('change',()=>{sync();draw()});model.addEventListener('change',draw);
function draw(){const p=rows.filter(r=>r.model===val(model)&&r.granular===true);const names=uniq(p.map(r=>r.class_name));const traces=[10,20,40].map((n,i)=>({type:'bar',name:`n=${n}`,x:names,y:names.map(x=>{const q=p.find(r=>r.n_embeddings===n&&r.class_name===x);return q?q.f1:null}),hovertemplate:'%{x}<br>F1 %{y:.3f}<extra></extra>'}));Plotly.react(gd,traces,{height:700,barmode:'group',yaxis:{range:[0,1],title:'window F1'},xaxis:{tickangle:-55},margin:{l:65,r:20,t:30,b:170}},config)}sync();draw();""")

    data=records(cells,["architecture","model","n_embeddings","root_id","cell_type","f1"])
    page("feature-cell-f1-cdf","Cumulative distribution of per-cell F1","feature_prediction_correlation.ipynb · architecture and ablation controls",data,r"""
const arch=control('arch','Architecture',uniq(rows.map(r=>r.architecture)));const model=control('model','Ablation',[]);function sync(){model.innerHTML='';uniq(rows.filter(r=>r.architecture===val(arch)).map(r=>r.model)).forEach((v,i)=>{const o=new Option(v,i);o._value=v;model.add(o)})}arch.addEventListener('change',()=>{sync();draw()});model.addEventListener('change',draw);
function draw(){const p=rows.filter(r=>r.model===val(model));const traces=[10,20,40].map(n=>{const x=p.filter(r=>r.n_embeddings===n).map(r=>r.f1).sort((a,b)=>a-b);return{x,y:x.map((_,i)=>(i+1)/x.length),mode:'lines',line:{shape:'hv'},name:`n=${n}`,hovertemplate:'F1 %{x:.3f}<br>fraction ≤ %{y:.3f}<extra></extra>'}});Plotly.react(gd,traces,{height:680,xaxis:{range:[0,1],title:'per-cell F1'},yaxis:{range:[0,1],title:'fraction of held-out cells'},shapes:[{type:'line',x0:.5,x1:.5,y0:0,y1:1,line:{dash:'dash',color:'#555'}}],margin:{l:70,r:20,t:30,b:65}},config)}sync();draw();""")


def index():
    pages=[]
    for path in sorted(OUT.glob("*.html")):
        if path.name=="index.html": continue
        text=path.read_text(); title=text.split("<title>",1)[1].split("</title>",1)[0]
        pages.append(f'<li><a href="{path.name}">{title}</a></li>')
    (OUT/"index.html").write_text(f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'><title>Interactive analysis figures</title><style>{STYLE}main{{max-width:900px;margin:40px auto;background:white;padding:30px;border-radius:8px}}li{{margin:12px 0}}a{{color:#315ec9}}</style><main><h1>Interactive analysis figures</h1><p>Standalone Plotly exports from the analysis notebooks.</p><ul>{''.join(pages)}</ul></main>")
    (OUT/".nojekyll").touch()


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,default=OUT);args=parser.parse_args()
    OUT=args.output.resolve(); OUT.mkdir(parents=True,exist_ok=True)
    for old in OUT.glob("*.html"): old.unlink()
    architecture_pages();feature_pages();index()
    print(f"Wrote {len(list(OUT.glob('*.html')))-1} figure pages and {OUT/'index.html'}")


if __name__=="__main__": main()
