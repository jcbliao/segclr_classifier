"""Read-only, evaluation-time interventions on cross-node computation."""
from contextlib import contextmanager
import types
import torch


def attention_parts(module,x,adj,valid,attn_mask=None):
    b,n,_=x.shape
    q,k,v=module.qkv_projection(x).view(b,n,3,module.num_heads,module.dim).permute(0,3,1,2,4).unbind(3)
    qk=(q@k.transpose(-2,-1))*module.scale
    gamma=module.predict_gamma(x)[:,None] if module.use_adj_bias else None
    if gamma is not None and module.use_exp:gamma=gamma.exp()
    content=qk if gamma is None else gamma[:,:,:,0:1]*qk
    topology=torch.zeros_like(content) if gamma is None else gamma[:,:,:,1:2]*adj[:,None]
    permitted=valid[:,None,None,:]
    if attn_mask is not None:permitted=permitted & attn_mask[:,None]
    return content,topology,v,permitted,gamma


@contextmanager
def intervention(model,condition,observer=None):
    """Restore every method/hook after each intervention, including exceptions."""
    methods=[];hooks=[]
    try:
        if model.graph_transformer is not None:
            for layer,block in enumerate(model.graph_transformer.blocks):
                module=block.attn
                original=module.forward
                def forward(self,x,adj,key_padding_mask,attn_mask=None,layer=layer):
                    content,topology,value,permitted,gamma=attention_parts(self,x,adj,key_padding_mask,attn_mask)
                    logits=content+topology
                    if condition=='no_qk':logits=topology.expand_as(content)
                    if condition=='no_adjacency_bias':logits=content
                    if condition=='uniform_attention':logits=torch.zeros_like(content)
                    weights=logits.masked_fill(~permitted,float('-inf')).softmax(-1)
                    if condition=='nodewise_mean_readout':
                        weights=torch.zeros_like(weights)
                        keys=key_padding_mask.clone();keys[:,0]=False
                        weights[:,:,0,:]=keys[:,None,:]/keys.sum(-1)[:,None,None].clamp_min(1)
                    if observer is not None:
                        observer.attention(layer,content,topology,weights,permitted,gamma,adj,key_padding_mask)
                    output=self.proj((weights@value).transpose(1,2).reshape(x.shape[0],x.shape[1],-1))
                    if condition=='nodewise_mean_readout':
                        output=output.clone();output[:,1:]=0
                    return torch.zeros_like(output) if condition=='no_attention_update' else output
                module.forward=types.MethodType(forward,module)
                methods.append((module,original))
        if model.encoder is not None:
            for layer,conv in enumerate(model.encoder.convs):
                for branch,linear in [('neighbor',conv.lin_l),('self',conv.lin_r)]:
                    def hook(module,inputs,output,layer=layer,branch=branch):
                        if observer is not None:observer.message(layer,branch,output)
                        if condition==('no_neighbors' if branch=='neighbor' else 'no_self'):
                            return torch.zeros_like(output)
                    hooks.append(linear.register_forward_hook(hook))
        yield
    finally:
        for module,original in methods:module.forward=original
        for hook in hooks:hook.remove()


def randomized_edges(batch,generator=None):
    """Uniform simple undirected graphs within windows, preserving edge counts."""
    sizes=batch.ptr[1:]-batch.ptr[:-1];b=len(sizes);k=int(sizes.max())
    rows=torch.arange(k,device=sizes.device)
    valid=rows[None,:]<sizes[:,None]
    candidates=(valid[:,:,None]&valid[:,None,:])&torch.triu(torch.ones(k,k,device=sizes.device,dtype=torch.bool),1)
    edge_graph=batch.batch[batch.edge_index[0]]
    counts=torch.bincount(edge_graph,minlength=b)//2
    scores=torch.rand((b,k,k),device=sizes.device,generator=generator).masked_fill(~candidates,float('inf'))
    order=scores.flatten(1).argsort(1)
    ranks=torch.empty_like(order).scatter_(1,order,torch.arange(k*k,device=sizes.device).expand(b,-1))
    selected=(ranks<counts[:,None]).view(b,k,k)&candidates
    graph,src,dst=selected.nonzero(as_tuple=True)
    src=src+batch.ptr[graph];dst=dst+batch.ptr[graph]
    return torch.cat((torch.stack((src,dst)),torch.stack((dst,src))),1)
