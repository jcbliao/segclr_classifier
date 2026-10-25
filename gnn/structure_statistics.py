"""Activation statistics; magnitudes are descriptive, interventions test reliance."""
import torch


class StructureStatistics:
    def __init__(self):
        self.values={};self.histograms={};self.pending={};self.radial_order=None

    def add(self,key,value,mask=None):
        value=value.detach().float()
        if mask is not None:value=value[mask.expand_as(value)]
        value=value.flatten()
        total=torch.stack((value.sum(),torch.tensor(value.numel(),device=value.device),value.square().sum()))
        if key not in self.values:self.values[key]=total.double()
        else:self.values[key]+=total
        bounds=None
        if key.endswith('_mass') or key.endswith('_fraction'):bounds=(0.,1.)
        elif key.endswith('/entropy'):bounds=(0.,3.)
        elif key.endswith('/effective_keys'):bounds=(1.,11.)
        elif key.endswith('/self_neighbor_cosine'):bounds=(-1.,1.)
        if bounds is not None:
            counts=torch.histc(value,bins=50,min=bounds[0],max=bounds[1])
            if key not in self.histograms:self.histograms[key]=(bounds,counts.double())
            else:self.histograms[key][1].add_(counts)

    def attention(self,layer,content,topology,weights,permitted,gamma,adj,valid):
        b,h,n,_=weights.shape
        query=valid[:,None,:].expand(b,h,n)
        for head in range(h):
            w=weights[:,head]
            kinds={'node':valid.clone(),'cls':torch.zeros_like(valid)}
            kinds['node'][:,0]=False;kinds['cls'][:,0]=True
            for kind,queries in kinds.items():
                prefix=f'GT/layer{layer}/head{head}/{kind}'
                entropy=-(w*w.clamp_min(1e-30).log()).sum(-1)
                self.add(prefix+'/entropy',entropy,queries)
                self.add(prefix+'/effective_keys',entropy.exp(),queries)
                self.add(prefix+'/largest_key_mass',w.max(-1).values,queries)
                self.add(prefix+'/cls_key_mass',w[:,:,0],queries)
                self_mask=torch.eye(n,device=w.device,dtype=torch.bool)[None]
                neighbor=(adj>0)&~self_mask
                nonneighbor=(adj==0)&valid[:,None,:]
                nonneighbor[:,:,0]=False
                for name,mask in [('self',self_mask),('neighbor',neighbor),('other_node',nonneighbor)]:
                    self.add(prefix+'/'+name+'_mass',(w*mask).sum(-1),queries)
                p=permitted.expand_as(weights)[:,head]
                count=p.sum(-1).clamp_min(1)
                for term,name in [(content[:,head],'qk'),(topology.expand_as(content)[:,head],'adjacency')]:
                    mean=(term*p).sum(-1)/count
                    rms=(((term-mean[:,:,None]).square()*p).sum(-1)/count).sqrt()
                    self.add(prefix+'/'+name+'_centered_logit_rms',rms,queries)
                if gamma is not None:
                    for index,name in [(0,'qk_gamma'),(1,'adjacency_gamma')]:
                        self.add(prefix+'/'+name,gamma[:,0,:,index],queries)
            if self.radial_order is not None:
                radial=w[:,0,1:].gather(1,self.radial_order)
                radial_valid=valid[:,1:].gather(1,self.radial_order)
                for rank in range(radial.shape[1]):
                    self.add(f'GT/layer{layer}/head{head}/cls/radial_rank{rank}_mass',radial[:,rank],radial_valid[:,rank])

    def message(self,layer,branch,value):
        prefix=f'MPNN/layer{layer}'
        self.add(prefix+'/'+branch+'_norm',value.norm(dim=-1))
        if branch=='neighbor':self.pending[layer]=value.detach()
        else:
            neighbor=self.pending.pop(layer)
            a=neighbor.norm(dim=-1);b=value.norm(dim=-1)
            self.add(prefix+'/neighbor_norm_fraction',a/(a+b).clamp_min(1e-30))
            self.add(prefix+'/self_neighbor_cosine',torch.nn.functional.cosine_similarity(neighbor,value,dim=-1))

    def export(self):
        result={}
        for key,total in self.values.items():
            s,n,sq=total.cpu().tolist()
            mean=s/max(n,1)
            result[key]={'mean':mean,'std':max(0,sq/max(n,1)-mean*mean)**.5,'count':int(n)}
            if key in self.histograms:
                bounds,counts=self.histograms[key]
                result[key]['histogram']={'edges':torch.linspace(*bounds,51).tolist(),'counts':counts.cpu().tolist()}
        return result


def parameter_statistics(model):
    result={}
    for name,p in model.named_parameters():
        v=p.detach().float()
        result[name]={'shape':list(v.shape),'parameters':v.numel(),'rms':float(v.square().mean().sqrt()),
                      'frobenius_norm':float(v.norm())}
    return result
