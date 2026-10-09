"""Plot compact cached statistics; no database reads or inference."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerBase
COLORS=plt.get_cmap('tab10').colors[:5]
BINS=np.linspace(0,1,51)

class GradientHandler(HandlerBase):
    def create_artists(self,legend,orig,xdescent,ydescent,width,height,fontsize,trans):
        color=orig.get_color(); artists=[]
        for fraction,alpha in [(1,.10),(.72,.16),(.42,.23)]:
            artists.append(Rectangle((-xdescent,-ydescent+height*(1-fraction)/2),width,height*fraction,
                                     facecolor=color,alpha=alpha,edgecolor='none',transform=trans))
        artists.append(Line2D([-xdescent,width-xdescent],[height/2-ydescent]*2,color=color,transform=trans))
        return artists

def histograms(values,title,counts=False,*,class_ids=None,class_names=None,class_title="Cell types"):
    if class_ids is not None:
        if counts or class_names is None:
            raise ValueError("Cell-type panels require raw probabilities and class names")
        if len(class_ids)!=len(values) or any(len(a)!=len(b) for a,b in zip(class_ids,values)):
            raise ValueError("Cell-type labels must align with probability rows")
    hist=np.asarray(values) if counts else np.asarray([np.histogram(x,BINS)[0] for x in values])
    panels=3 if class_ids is not None else 2
    fig,axes=plt.subplots(1,panels,figsize=(6.5*panels,4),sharex=True)
    pooled=hist.sum(0); width=np.diff(BINS)
    axes[0].stairs(pooled/(pooled.sum()*width),BINS,fill=True,alpha=.5,color='slateblue')
    axes[0].set_title(f'Composite · {pooled.sum():,} fold predictions')
    for fold,h in enumerate(hist):
        axes[1].stairs(h/(h.sum()*width),BINS,color=COLORS[fold],label=f'Fold {fold}',linewidth=1.7)
    axes[1].set_title('Five folds');axes[1].legend()
    if class_ids is not None:
        probabilities=np.concatenate(values); labels=np.concatenate(class_ids)
        for index,name in enumerate(class_names):
            h=np.histogram(probabilities[labels==index],BINS)[0]
            if not h.sum():continue
            axes[2].stairs(h/(h.sum()*width),BINS,color=plt.get_cmap('tab10')(index),
                           label=f'{name} (n={h.sum():,})',linewidth=1.7)
        axes[2].set_title(class_title);axes[2].legend(fontsize=8)
    for ax in axes:ax.set(xlabel='Probability',ylabel='Density',xlim=(0,1));ax.grid(alpha=.15)
    fig.suptitle(title);fig.tight_layout();plt.show()

def probability_lines(ax,caches,column):
    handles=[]
    for fold,z in enumerate(caches):
        if column==0:
            n=z['n'].astype(float);total=z['sums'][:,0];ss=z['squares'][:,0]
        else:
            n=z['selected_class_n'][:,column-1].astype(float)
            total=z['selected_class_sums'][:,column-1];ss=z['selected_class_squares'][:,column-1]
        mean=np.divide(total,n,out=np.full(10,np.nan),where=n>0)
        variance=np.divide(ss-total*mean,n-1,out=np.full(10,np.nan),where=n>1)
        se=np.sqrt(np.maximum(variance,0)/np.maximum(n,1));x=np.arange(1,11)
        # Nested 95%, 80%, and 50% intervals make a stepped opacity gradient.
        for q,alpha in [(1.959964,.10),(1.281552,.16),(.674490,.23)]:
            ax.fill_between(x,np.clip(mean-q*se,0,1),np.clip(mean+q*se,0,1),color=COLORS[fold],alpha=alpha)
        line,=ax.plot(x,mean,'o-',color=COLORS[fold],label=f'Fold {fold}',markersize=3)
        handles.append(line)
    ax.set(xlabel='Presynaptic nodes / embeddings used',ylabel='Mean probability',xticks=range(1,11),ylim=(0,1))
    ax.grid(alpha=.15)
    ax.legend(handles=handles,handler_map={Line2D:GradientHandler()},fontsize=8)

def accuracy_plot(probabilities,correct,title,*,class_ids=None,class_names=None,class_title="Cell types",xlim=(0,1)):
    if class_ids is not None:
        if class_names is None or len(class_ids)!=len(probabilities) or any(len(a)!=len(b) for a,b in zip(class_ids,probabilities)):
            raise ValueError("Cell-type labels must align with probability rows")
    edges=np.linspace(0,1,21);centers=(edges[:-1]+edges[1:])/2
    panels=3 if class_ids is not None else 2
    fig,axes=plt.subplots(1,panels,figsize=(6.5*panels,4))
    groups=[('Composite',np.concatenate(probabilities),np.concatenate(correct),'black')]
    groups += [(f'Fold {i}',p,c,COLORS[i]) for i,(p,c) in enumerate(zip(probabilities,correct))]
    for name,p,c,color in groups:
        n=np.histogram(p,edges)[0];k=np.histogram(p[c],edges)[0]
        y=np.divide(k,n,out=np.full(20,np.nan),where=n>0)
        ax=axes[0] if name=='Composite' else axes[1]
        ax.plot(centers,y,'o-',color=color,label=name,markersize=3)
        if name=='Composite':
            # Wilson binomial 95% interval.
            d=1+1.96**2/np.maximum(n,1)
            mid=(y+1.96**2/(2*np.maximum(n,1)))/d
            half=1.96*np.sqrt(y*(1-y)/np.maximum(n,1)+1.96**2/(4*np.maximum(n,1)**2))/d
            ax.fill_between(centers,mid-half,mid+half,color=color,alpha=.15)
    axes[0].set_title('Composite');axes[1].set_title('Five folds')
    if class_ids is not None:
        p=np.concatenate(probabilities); c=np.concatenate(correct); labels=np.concatenate(class_ids)
        for index,name in enumerate(class_names):
            mask=labels==index
            n=np.histogram(p[mask],edges)[0]; k=np.histogram(p[mask & c],edges)[0]
            y=np.divide(k,n,out=np.full(len(centers),np.nan),where=n>0)
            axes[2].plot(centers,y,'o-',color=plt.get_cmap('tab10')(index),
                         label=name,markersize=3)
        axes[2].set_title(class_title)
    for ax in axes:
        ax.set(xlabel='Probability bin',ylabel='Accuracy within bin',xlim=xlim,ylim=(0,1));ax.legend();ax.grid(alpha=.2)
    fig.suptitle(title);fig.tight_layout();plt.show()

def probability_fold_mean(ax,caches,column):
    """Equal-weight fold means; Student-t intervals across available folds."""
    from scipy.stats import t
    means=[]
    for z in caches:
        n=z['selected_class_n'][:,column-1]
        means.append(np.divide(z['selected_class_sums'][:,column-1],n,
                              out=np.full(10,np.nan),where=n>0))
    means=np.asarray(means);k=np.isfinite(means).sum(0)
    mean=np.divide(np.nansum(means,axis=0),k,out=np.full(10,np.nan),where=k>0)
    ss=np.nansum((means-mean)**2,axis=0)
    variance=np.divide(ss,k-1,out=np.full(10,np.nan),where=k>1)
    se=np.sqrt(variance/np.maximum(k,1));x=np.arange(1,11)
    color='tab:blue'
    for coverage,alpha in [(.95,.10),(.80,.16),(.50,.23)]:
        q=t.ppf((1+coverage)/2,np.maximum(k-1,1))
        ax.fill_between(x,np.clip(mean-q*se,0,1),np.clip(mean+q*se,0,1),color=color,alpha=alpha)
    line,=ax.plot(x,mean,'o-',color=color,label='Mean across folds (50/80/95% CI)',markersize=4)
    ax.set(xlabel='Presynaptic nodes / embeddings used',ylabel='Mean selected-class probability',
           xticks=range(1,11),ylim=(0,1))
    ax.grid(alpha=.15)
    ax.legend(handles=[line],handler_map={Line2D:GradientHandler()},fontsize=8)
