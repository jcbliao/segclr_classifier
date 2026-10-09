"""Plot hard window vote fractions for the audited Martinotti test cells."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
frame=pd.read_csv(ROOT/'all_martinotti_cells.csv',dtype={'root_id':'string'})
runs=frame[['experiment','condition','architecture','run']].drop_duplicates()
fig,axes=plt.subplots(4,4,figsize=(16,14),sharex=True,sharey=True)
colors={'864691135340141542':'#d62728','864691135474179378':'#ff7f0e','864691135114295961':'#9467bd','864691135875919502':'#2ca02c'}
for ax,(_,r) in zip(axes.flat,runs.iterrows()):
 selected=frame[(frame.experiment==r.experiment)&(frame.run==r.run)]
 ax.scatter(selected.pyramidal_fraction*100,selected.martinotti_fraction*100,c='#bbbbbb',s=25)
 for rid,color in colors.items():
  s=selected[selected.root_id==rid]
  ax.scatter(s.pyramidal_fraction*100,s.martinotti_fraction*100,c=color,s=45,label=rid)
 ax.plot([0,100],[0,100],color='black',lw=.5,ls='--')
 title={'native_skeletons_pre_post':'Original','native_pre_post_axon_filtered':'Filtered','native_pre_post_axon_filtered_equal_sampling':'Filtered equal'}[r.experiment]
 ax.set_title(f'{title} · {r.condition}\n{r.architecture}',fontsize=10)
 ax.grid(alpha=.2); ax.set_xlim(0,100); ax.set_ylim(0,100)
for ax in list(axes.flat)[len(runs):]: ax.set_visible(False)
fig.supxlabel('Windows predicted pyramidal (%)'); fig.supylabel('Windows predicted Martinotti (%)')
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=2,fontsize=9)
fig.suptitle('True Martinotti test cells: hard window vote fractions',y=.995)
fig.tight_layout(rect=[0,0,.99,.95]);fig.savefig(ROOT/'window_vote_fractions.png',dpi=180);fig.savefig(ROOT/'window_vote_fractions.pdf')
