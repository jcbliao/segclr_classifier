"""Cache the component distributions and append executed comparisons to both notebooks."""
import os
from pathlib import Path
import sys
import nbformat
from nbclient import NotebookClient

REPO=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(REPO))
from analysis.presynaptic.component_site_distribution import component_site_counts as counts

def main():
    frame=counts.load_components()
    summary=counts.summary_table(frame)
    print(summary.to_string(),flush=True)
    expected={'Non-native':(16657,149925558,5439,136241017),
              'Native resampled':(16765,308397752,5830,269133542)}
    for label,g in frame.groupby('dataset'):
        nc,ne,kc,ke=expected[label]
        assert len(g)==nc and int(g.n_edges.sum())==ne
        kept=g.n_presynaptic_sites>5
        assert int(kept.sum())==kc and int(g.loc[kept,'n_edges'].sum())==ke
    out=counts.HERE
    summary.to_csv(out/'summary.csv')
    counts.cutoff_table(frame).to_csv(out/'cutoff_retention.csv',index=False)
    for name,fig in [('presynaptic_sites_per_component',counts.plot_distribution(frame)),
                     ('component_cutoff_retention',counts.plot_cutoff_retention(frame))]:
        fig.savefig(out/f'{name}.png',dpi=180,bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
    md=nbformat.v4.new_markdown_cell("""## Presynaptic sites per connected component — before the >5 filter

Here **component means an undirected connected component after the 5 µm soma
cut**, including isolated nodes. This cutoff-diagnostic section deliberately
includes **all components, including zero-site components**, before either the
>5-site filter or upstream root/tree exclusions. It does not change the filter
used by the preceding analyses.

Count distinct presynaptic IDs mapped within 2 µm, using each dataset's existing
mapping: non-native to a mapped SegCLR-bearing node, native directly to a
resampled node. Components are counted once each and pooled across the same
2,209 cells. The histograms cover the complete range, including zero-site components,
using equal-width 100-site bins and linear axes. Heights are component counts
per bin. Change HISTOGRAM_BIN_WIDTH in the cell to adjust the bin width; the
range always extends through the maximum observed count. The retention curves apply **strictly >k sites**.""")
    setup=nbformat.v4.new_code_cell("""from pathlib import Path
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'gnn').is_dir())
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from analysis.presynaptic.component_site_distribution import component_site_counts
component_counts = component_site_counts.load_components()
display(component_site_counts.summary_table(component_counts))
HISTOGRAM_BIN_WIDTH = 100
component_distribution_figure = component_site_counts.plot_distribution(
    component_counts, bin_width=HISTOGRAM_BIN_WIDTH
)
plt.close(component_distribution_figure)
component_distribution_figure""")
    retention=nbformat.v4.new_code_cell("""display(component_site_counts.cutoff_table(component_counts).round(4))
component_retention_figure = component_site_counts.plot_cutoff_retention(component_counts)
plt.close(component_retention_figure)
component_retention_figure""")
    # Execute only the new appendix; preserve all existing outputs and analyses.
    section=nbformat.v4.new_notebook(cells=[md,setup,retention],
        metadata={'kernelspec':{'display_name':'Python 3 (ossify)','language':'python','name':'python3'}})
    NotebookClient(section,timeout=1800,kernel_name='python3',
        resources={'metadata':{'path':str(counts.HERE)}}).execute()
    for folder in ('new_skeletons','new_skeletons_native'):
        path=counts.HERE.parent/folder/'skeleton_stats.ipynb'
        nb=nbformat.read(path,as_version=4)
        start=next((i for i,c in enumerate(nb.cells)
                    if c.cell_type=='markdown' and c.source.startswith('## Presynaptic sites per connected component')),len(nb.cells))
        nb.cells=nb.cells[:start]+section.cells
        tmp=path.with_suffix('.tmp.ipynb')
        nbformat.write(nb,tmp); os.replace(tmp,path)
        print('Updated',path,flush=True)
    print('Component distribution validation passed',flush=True)

if __name__=='__main__':
    main()
