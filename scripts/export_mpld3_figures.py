#!/usr/bin/env python3
"""Export the notebooks' actual Matplotlib Figures as static mpld3 pages."""
from __future__ import annotations

import json
import multiprocessing
import re
import shutil
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpld3
from mpld3._display import NumpyEncoder

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "figures"
ASSETS = OUT / "assets"
sys.path.insert(0, str(ROOT))

STYLE = """
:root{font-family:Inter,ui-sans-serif,system-ui,sans-serif;color:#172033;background:#f6f8fb}
body{margin:0}.top{padding:18px 24px;background:white;border-bottom:1px solid #dce2ea}
h1{font-size:20px;margin:0 0 5px}.sub{font-size:13px;color:#596579}.home{color:#315ec9}
.controls{display:flex;flex-wrap:wrap;gap:12px;padding:14px 24px;background:#fff;border-bottom:1px solid #dce2ea}
label{font-size:12px;color:#596579;display:flex;flex-direction:column;gap:4px}select{font:inherit;padding:6px}
.plotwrap{margin:18px;overflow:auto;background:white;border-radius:8px;box-shadow:0 1px 5px #23304818}
#plot{display:inline-block}.note{padding:0 24px 20px;color:#596579;font-size:12px}
"""


def clean(value):
    return None if value is None else str(value)


class Export:
    def __init__(self, slug, title, subtitle, controls):
        self.slug, self.title, self.subtitle = slug, title, subtitle
        self.controls = controls
        self.states = []
        self.directory = ASSETS / slug
        self.directory.mkdir(parents=True, exist_ok=True)

    def add(self, figure, **params):
        if figure is None:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            payload = mpld3.fig_to_dict(figure)
        self.add_payload(payload, **params)
        plt.close(figure)

    def add_payload(self, payload, **params):
        if payload is None:
            return
        index = len(self.states)
        path = self.directory / f"{index:04d}.json"
        path.write_text(json.dumps(payload, cls=NumpyEncoder, separators=(",", ":")))
        self.states.append({"params": {k: clean(v) for k, v in params.items()},
                            "src": f"assets/{self.slug}/{index:04d}.json"})

    def write(self):
        spec = json.dumps({"controls": self.controls, "states": self.states}, separators=(",", ":"))
        html = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{self.title}</title><script src="https://d3js.org/d3.v5.min.js"></script><script src="https://mpld3.github.io/js/mpld3.v0.5.12.js"></script><style>{STYLE}</style></head>
<body><div class="top"><h1>{self.title}</h1><div class="sub">{self.subtitle} · <a class="home" href="index.html">all figures</a></div></div>
<div class="controls"></div><div class="plotwrap"><div id="plot"></div></div><div class="note">This is the notebook's actual Matplotlib figure serialized with mpld3. Narrow embeds scroll instead of resizing the figure.</div>
<script>const spec={spec},box=document.querySelector('.controls'),selects={{}};
function values(name,partial){{return [...new Set(spec.states.filter(s=>Object.entries(partial).every(([k,v])=>s.params[k]===v)).map(s=>s.params[name]))]}}
function rebuild(){{let partial={{}};for(const c of spec.controls){{const previous=selects[c.name]?.value;let options=values(c.name,partial);let s=selects[c.name];if(!s){{const l=document.createElement('label');l.textContent=c.label;s=document.createElement('select');s.onchange=()=>{{rebuild();draw()}};l.appendChild(s);box.appendChild(l);selects[c.name]=s}}s.innerHTML='';options.forEach(v=>{{const o=new Option(v===null?'All classes (macro)':v,v===null?'__null__':v);s.add(o)}});if([...s.options].some(o=>o.value===previous))s.value=previous;partial[c.name]=s.value==='__null__'?null:s.value}}}}
async function draw(){{const params={{}};for(const c of spec.controls)params[c.name]=selects[c.name].value==='__null__'?null:selects[c.name].value;const state=spec.states.find(s=>Object.entries(params).every(([k,v])=>s.params[k]===v));if(!state)return;document.getElementById('plot').innerHTML='';const fig=await fetch(state.src).then(r=>r.json());mpld3.draw_figure('plot',fig)}}
rebuild();draw();</script></body></html>'''
        (OUT / f"{self.slug}.html").write_text(html)


_FEATURE_CONTEXT = None


def render_feature_state(task):
    """Worker-side rendering; fork shares the large summary frames read-only."""
    fp, bins, quant, overall, cells, classes = _FEATURE_CONTEXT
    kind, args = task
    if kind == "feature":
        feature, cls, marks = args
        fig = fp.plot_feature_by_model(bins, feature, cls, quantiles=quant if marks else None)
    elif kind == "family":
        family, model, cls = args
        fig = fp.plot_family_by_count(bins, family, model, cls, quantiles=quant)
    elif kind == "bars":
        model, = args
        fig = fp.plot_class_bars(overall, model, classes)
    else:
        model, = args
        fig = fp.plot_score_cdf(cells, model, overall)
    if fig is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        payload = mpld3.fig_to_dict(fig)
    plt.close(fig)
    return payload


def architecture_pages():
    from analysis import architecture_comparison as ac
    payloads = ac.load_confusions(ROOT / "results")
    frame = ac.load_best_epochs(ROOT / "results", payloads)

    out = Export("architecture-models-by-count", "Every model at each node count",
                 "architecture_comparison.ipynb", [])
    out.add(ac.plot_within_count(frame)); out.write()

    out = Export("architecture-across-counts", "Architecture comparison across node counts",
                 "architecture_comparison.ipynb",
                 [{"name":"class","label":"Class"},{"name":"level","label":"Level"},{"name":"metric","label":"Metric"}])
    for cls in [None] + ac.class_names(payloads):
        for level in ("window", "cell"):
            for metric in ("f1", "recall", "precision", "accuracy"):
                column = ac.metric_column(metric, level, cls)
                if column in frame:
                    out.add(ac.plot_across_counts(frame, column), **{"class":cls,"level":level,"metric":metric})
    out.write()

    pairs = sorted({(ac.describe_run(r)[0], ac.model_label(r)) for r in payloads if ac.describe_run(r)}, key=lambda x:(ac.architecture_rank(x[0]),x[1]))
    out = Export("architecture-confusion-matrices", "Confusion matrices across node count",
                 "architecture_comparison.ipynb",
                 [{"name":"architecture","label":"Architecture"},{"name":"model","label":"Ablation"},{"name":"level","label":"Level"},{"name":"scale","label":"Scale"}])
    for architecture, model in pairs:
        for level in ("window", "cell"):
            for scale in ("normalized", "raw"):
                out.add(ac.plot_confusion_grid(payloads, model, level, scale=="normalized"),
                        architecture=architecture, model=model, level=level, scale=scale)
    out.write()


def feature_pages():
    global _FEATURE_CONTEXT
    from analysis import feature_prediction_correlation as fp
    frames = fp.load_frames(fp.cached_runs(fp.usable_runs()))
    bins, quant, overall, cells = (frames[k] for k in ("bins","quantiles","overall","cells"))
    classes = [None] + list(dict.fromkeys(bins.class_name.dropna().astype(str)))
    models = fp.model_order(bins)
    pairs = [(str(bins[bins.model==m].architecture.iloc[0]), m) for m in models]
    _FEATURE_CONTEXT = (fp, bins, quant, overall, cells, frames["classes"])
    pool = ProcessPoolExecutor(max_workers=4, mp_context=multiprocessing.get_context("fork"))

    for feature, slug, title, marks in (
        ("path_distance_um","feature-path-length","F1 against path length",None),
        ("node_density_per_um","feature-node-density","F1 against node density",quant)):
        out=Export(slug,title,"feature_prediction_correlation.ipynb",[{"name":"class","label":"Class"}])
        tasks=[("feature",(feature,cls,marks is not None)) for cls in classes]
        for cls,payload in zip(classes,pool.map(render_feature_state,tasks)):
            out.add_payload(payload,**{"class":cls})
        out.write()

    for family, features in fp.FEATURE_FAMILIES.items():
        slug={"clearance radius (nm)":"feature-clearance-radius-nm",
              "embedding-box volume (um^3)":"feature-embedding-box-volume-um3",
              "distance from soma (um)":"feature-distance-from-soma-um"}[family]
        out=Export(slug,"F1 against "+family,"feature_prediction_correlation.ipynb",
                   [{"name":"architecture","label":"Architecture"},{"name":"model","label":"Ablation"},{"name":"class","label":"Class"}])
        combinations=[(architecture,model,cls) for architecture,model in pairs for cls in classes]
        tasks=[("family",(family,model,cls)) for architecture,model,cls in combinations]
        for (architecture,model,cls),payload in zip(combinations,pool.map(render_feature_state,tasks)):
            out.add_payload(payload,architecture=architecture,model=model,**{"class":cls})
        out.write()

    out=Export("feature-granular-class-bars","Window F1 per granular cell type","feature_prediction_correlation.ipynb",
               [{"name":"architecture","label":"Architecture"},{"name":"model","label":"Ablation"}])
    for (architecture,model),payload in zip(pairs,pool.map(render_feature_state,[("bars",(m,)) for _,m in pairs])):
        out.add_payload(payload,architecture=architecture,model=model)
    out.write()

    cell_models=fp.model_order(cells)
    out=Export("feature-cell-f1-cdf","Cumulative distribution of per-cell F1","feature_prediction_correlation.ipynb",
               [{"name":"architecture","label":"Architecture"},{"name":"model","label":"Ablation"}])
    for model,payload in zip(cell_models,pool.map(render_feature_state,[("cdf",(m,)) for m in cell_models])):
        architecture=str(cells[cells.model==model].architecture.iloc[0])
        out.add_payload(payload,architecture=architecture,model=model)
    out.write()
    pool.shutdown()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for path in OUT.glob("*.html"): path.unlink()
    if ASSETS.exists(): shutil.rmtree(ASSETS)
    ASSETS.mkdir()
    architecture_pages(); feature_pages()
    links=[]
    for path in sorted(OUT.glob("*.html")):
        if path.name=="index.html": continue
        title=path.read_text().split("<title>",1)[1].split("</title>",1)[0]
        links.append(f'<li><a href="{path.name}">{title}</a></li>')
    (OUT/"index.html").write_text(f"<!doctype html><meta charset=utf-8><title>Interactive analysis figures</title><style>{STYLE}main{{max-width:900px;margin:40px auto;background:white;padding:30px}}</style><main><h1>Interactive analysis figures</h1><ul>{''.join(links)}</ul></main>")
    (OUT/".nojekyll").touch()
    print(f"Wrote {len(links)} mpld3 pages with {sum(1 for _ in ASSETS.rglob('*.json'))} Matplotlib states")

if __name__=="__main__": main()
