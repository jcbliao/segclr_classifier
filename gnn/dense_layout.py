"""Dense packing with CPU-known dimensions and no device-to-host checks."""
import torch

def dense_layout(batch,batch_size,max_nodes):
    counts=torch.zeros(batch_size,dtype=torch.long,device=batch.device)
    counts.scatter_add_(0,batch,torch.ones_like(batch))
    ptr=torch.cat((counts.new_zeros(1),counts.cumsum(0)))
    flat=batch*max_nodes+torch.arange(len(batch),device=batch.device)-ptr[batch]
    mask=torch.zeros(batch_size*max_nodes,dtype=torch.bool,device=batch.device)
    mask.index_fill_(0,flat,True)
    return flat,mask.view(batch_size,max_nodes),ptr,batch_size,max_nodes

def pack_dense(values,layout):
    flat,mask,ptr,batch_size,max_nodes=layout
    out=values.new_zeros((batch_size*max_nodes,)+tuple(values.shape[1:]))
    out.index_copy_(0,flat,values)
    return out.view((batch_size,max_nodes)+tuple(values.shape[1:])),mask

def pack_adjacency(edges,batch,layout):
    _,_,ptr,batch_size,max_nodes=layout
    source,target=edges
    graph=batch[source]
    row=source-ptr[graph];col=target-ptr[batch[target]]
    flat=(graph*max_nodes+row)*max_nodes+col
    adj=torch.zeros(batch_size*max_nodes*max_nodes,device=edges.device)
    adj.scatter_add_(0,flat,torch.ones(len(source),device=edges.device))
    return adj.view(batch_size,max_nodes,max_nodes)
