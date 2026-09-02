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
 const i=[10,20,40].indexOf(n),axis=i?String(i+1):'';traces.push({type:'bar',orientation:'h',x:p.map(r=>r.window_macro_f1),y:p.map(r=>r.model),xaxis:'x'+axis,yaxis:'y'+axis,name:`n=${n}`,showlegend:false,
 customdata:p.map(r=>[r.architecture,r.folds]),hovertemplate:'%{y}<br>macro F1 %{x:.3f}<br>%{customdata[0]} · %{customdata[1]} fold(s)<extra></extra>'})}
 const layout={height:760,grid:{rows:1,columns:3,pattern:'independent'},margin:{l:210,r:25,t:65,b:65},annotations:[10,20,40].map((n,i)=>({text:`${n} embeddings`,xref:`x${i?i+1:''} domain`,yref:'paper',x:.5,y:1.06,showarrow:false,font:{size:16}}))};for(let i=1;i<=3;i++){layout['xaxis'+(i===1?'':i)]={title:'window macro F1',gridcolor:'#d9d9d9'};layout['yaxis'+(i===1?'':i)]={automargin:true,autorange:'reversed'}}Plotly.react(gd,traces,layout,config)}draw();""")

    classes = [None] + ac.class_names(payloads)
    page("architecture-across-counts", "Architecture comparison across node counts",
         "architecture_comparison.ipynb · class, level, and metric controls", data, r"""
const cs=control('class','Class',""" + json.dumps(classes) + r""",x=>x===null?'All classes (macro)':x);
const level=control('level','Level',['window','cell']);const metric=control('metric','Metric',['f1','recall','precision','accuracy']);
function draw(){const c=val(cs),scope=val(level),m=val(metric);let col;if(c===null){col=scope+'_'+({f1:'macro_f1',recall:'balanced_accuracy',precision:'macro_precision',accuracy:'accuracy'}[m])}else col=scope+'_'+(m==='accuracy'?'recall':m)+'_'+c;
 const architectures=uniq(rows.map(r=>r.architecture)),traces=[];architectures.forEach((a,ai)=>{uniq(rows.filter(r=>r.architecture===a).map(r=>r.model)).forEach((model,mi)=>{const p=rows.filter(r=>r.architecture===a&&r.model===model&&r[col]!=null).sort((x,y)=>x.n_embeddings-y.n_embeddings),axis=ai?String(ai+1):'';if(p.length)traces.push({x:p.map(r=>r.n_embeddings),y:p.map(r=>r[col]),xaxis:'x'+axis,yaxis:'y'+axis,mode:'lines+markers',name:model,legendgroup:model,showlegend:true,hovertemplate:`${model}<br>n=%{x}<br>${m}=%{y:.3f}<extra></extra>`})})});
 const ncols=3,nrows=Math.ceil(architectures.length/ncols),layout={height:500*nrows,grid:{rows:nrows,columns:ncols,pattern:'independent'},margin:{l:75,r:25,t:75,b:110},legend:{orientation:'h',y:-.08},annotations:architectures.map((a,i)=>({text:a,xref:`x${i?i+1:''} domain`,yref:`y${i?i+1:''} domain`,x:.5,y:1.13,showarrow:false,font:{size:15}}))};architectures.forEach((_,i)=>{const k=i?'axis'+(i+1):'axis';layout['x'+k]={title:'Number of embeddings',tickvals:[10,20,40],gridcolor:'#d9d9d9'};layout['y'+k]={title:(c||'macro')+' '+m,gridcolor:'#d9d9d9'}});Plotly.react(gd,traces,layout,config)}draw();""")

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
 const panels="""+("[10,20,40]" if family else "uniq(rows.bins.map(r=>r.model))")+r""",traces=[];panels.forEach((panel,pi)=>{const variants="""+("uniq(p.map(r=>r.feature))" if family else "[10,20,40]")+r""";variants.forEach((variant,vi)=>{let g=p.filter(r=>"""+("r.n_embeddings===panel&&r.feature===variant" if family else "r.model===panel&&r.n_embeddings===variant")+r""").sort((a,b)=>a.bin-b.bin);if(!g.length)return;const axis=pi?String(pi+1):'';traces.push({x:g.map(r=>r.feature_median),y:g.map(r=>r.f1),xaxis:'x'+axis,yaxis:'y'+axis,mode:'lines+markers',marker:{size:6},line:{color:palette[vi]},name:"""+("variant.replaceAll('_',' ')" if family else "'n='+variant")+r""",legendgroup:String(variant),showlegend:pi===0,hovertemplate:"""+("variant" if family else "'n='+variant")+r"""+'<br>x=%{x:.3g}<br>F1=%{y:.3f}<extra></extra>'});
 const q=rows.quantiles.find(r=>r.model==="""+("val(model)&&r.n_embeddings===panel&&r.feature===variant" if family else "panel&&r.n_embeddings===variant&&r.feature===g[0].feature")+r"""&&(c===null?r.class_name===null:r.class_name===c));if(q){['p25','p50','p75','p90'].forEach(mark=>{if(q[mark]!=null)traces.push({x:[q[mark],q[mark]],y:[0,Math.max(...g.map(r=>r.f1))],xaxis:'x'+axis,yaxis:'y'+axis,mode:'lines',line:{color:palette[vi],dash:'dot',width:1},showlegend:false,hoverinfo:'skip'})})}})});
 const ncols=3,nrows=Math.ceil(panels.length/ncols),layout={height:"""+("520" if family else "400*nrows")+r""",grid:{rows:nrows,columns:ncols,pattern:'independent'},margin:{l:75,r:20,t:70,b:90},legend:{orientation:'h',y:-.08},annotations:panels.map((x,i)=>({text:"""+("'n='+x" if family else "x")+r""",xref:`x${i?i+1:''} domain`,yref:`y${i?i+1:''} domain`,x:.5,y:1.13,showarrow:false,font:{size:14}}))};panels.forEach((_,i)=>{const k=i?'axis'+(i+1):'axis';layout['x'+k]={title:p.length?p[0].feature.replaceAll('_',' '):'feature',gridcolor:'#d9d9d9'};layout['y'+k]={title:(c||'macro')+' F1',gridcolor:'#d9d9d9',matches:'y'}});Plotly.react(gd,traces,layout,config)}"""+("sync();" if family else "")+"draw();"
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
function draw(){const p=rows.filter(r=>r.model===val(model)),types=uniq(p.map(r=>r.cell_type)).sort(),traces=[],layout={height:550+280*Math.ceil(types.length/5),margin:{l:70,r:20,t:55,b:65},shapes:[],annotations:[]};
 function curve(part,n,axis,show){const x=part.filter(r=>r.n_embeddings===n).map(r=>r.f1).sort((a,b)=>a-b);if(x.length)traces.push({x,y:x.map((_,i)=>(i+1)/x.length),xaxis:'x'+axis,yaxis:'y'+axis,mode:'lines',line:{shape:'hv',color:palette[[10,20,40].indexOf(n)]},name:`n=${n}`,legendgroup:`n=${n}`,showlegend:show,hovertemplate:'F1 %{x:.3f}<br>fraction ≤ %{y:.3f}<extra></extra>'})}
 [10,20,40].forEach(n=>curve(p,n,'',true));layout.xaxis={domain:[0,1],anchor:'y',range:[0,1],title:'per-cell F1'};layout.yaxis={domain:[.58,1],anchor:'x',range:[0,1],title:'fraction of held-out cells'};layout.annotations.push({text:'All held-out cells',xref:'paper',yref:'paper',x:.5,y:1.05,showarrow:false,font:{size:16}});layout.shapes.push({type:'line',xref:'x',yref:'y',x0:.5,x1:.5,y0:0,y1:1,line:{dash:'dash',color:'#555'}});
 const cols=5,nrows=Math.ceil(types.length/cols),gap=.025,w=(1-gap*(cols-1))/cols,rowH=.48/nrows;types.forEach((type,i)=>{const id=i+2,a=String(id),col=i%cols,row=Math.floor(i/cols),x0=col*(w+gap),y1=.50-row*rowH,y0=y1-rowH*.78;layout['xaxis'+a]={domain:[x0,x0+w],anchor:'y'+a,range:[0,1],title:'per-cell F1'};layout['yaxis'+a]={domain:[y0,y1],anchor:'x'+a,range:[0,1]};[10,20,40].forEach(n=>curve(p.filter(r=>r.cell_type===type),n,a,false));layout.annotations.push({text:type,xref:'paper',yref:'paper',x:x0+w/2,y:y1+.02,showarrow:false,font:{size:11}});layout.shapes.push({type:'line',xref:'x'+a,yref:'y'+a,x0:.5,x1:.5,y0:0,y1:1,line:{dash:'dash',color:'#555'}})});Plotly.react(gd,traces,layout,config)}sync();draw();""")


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
