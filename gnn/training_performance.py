"""Exact hierarchy routing prepared on CPU to avoid per-head CUDA synchronizations."""
import numpy as np
import torch
import torch.nn.functional as F


def prepare_routes(head, targets):
    if not hasattr(head,'nodes'):return None
    if targets.device.type!='cpu':raise ValueError('Prepare routing before copying targets to CUDA')
    values=targets.numpy();routes=[]
    for i,node in enumerate(head.nodes):
        rows=(np.arange(len(values),dtype=np.int64) if node['parent_level']==-1 else
              np.flatnonzero(values[:,node['parent_level']]==node['parent_global_idx']))
        if not len(rows):continue
        mapping=np.full(len(head.hierarchy.level_classes[node['child_level']]),-1,np.int64)
        mapping[node['child_global_indices']]=np.arange(node['n_children'])
        local=mapping[values[rows,node['child_level']]]
        routes.append((i,torch.from_numpy(rows),torch.from_numpy(local)))
    return routes


def move_routes(routes,device):
    if routes is None:return None
    return [(i,rows.to(device,non_blocking=True),targets.to(device,non_blocking=True)) for i,rows,targets in routes]


def routed_loss(head,hidden,targets,routes):
    if routes is None:return head.compute_loss(hidden,targets)
    features=hidden if head.trunk is None else head.trunk(hidden)
    loss=features.new_zeros(());count=0
    for i,rows,local in routes:
        loss=loss+F.cross_entropy(head.heads[i](features.index_select(0,rows)),local,
                                  weight=head._class_weight(i),reduction='sum')
        count+=rows.numel()
    return loss/(count.clamp_min(1) if isinstance(count,torch.Tensor) else max(count,1))


@torch.no_grad()
def static_top_down(head,hidden):
    if not hasattr(head,'nodes'):return head.predict_top_down(hidden)
    features=hidden if head.trunk is None else head.trunk(hidden)
    predictions=torch.zeros((len(features),head.n_levels),dtype=torch.long,device=features.device)
    current=torch.full((len(features),),-1,dtype=torch.long,device=features.device)
    keys=sorted(set(head.node_lookup)|set(head.passthrough))
    for level in range(head.n_levels):
        following=torch.zeros_like(current)
        for parent_level,parent in keys:
            if parent_level!=level-1:continue
            mask=current==parent;key=(parent_level,parent)
            if key in head.node_lookup:
                i=head.node_lookup[key]
                local=head.heads[i](features).argmax(-1)
                children=torch.as_tensor(head.nodes[i]['child_global_indices'],device=features.device)
                following=torch.where(mask,children[local],following)
            else:following=torch.where(mask,head.passthrough[key],following)
        predictions[:,level]=following;current=following
    return predictions

def model_training_loss(model,x,edge_index,batch,pos_enc,rel_pos,thickness,has_segclr,postsynaptic_embedding,batch_shape,targets,routes):
    hidden=model(x,edge_index,batch,pos_enc=pos_enc,rel_pos=rel_pos,thickness=thickness,
                 has_segclr=has_segclr,postsynaptic_embedding=postsynaptic_embedding,batch_shape=batch_shape)
    return routed_loss(model.cls_head,hidden,targets,routes)

def prepare_dense_routes(head,targets):
    sparse=prepare_routes(head,targets)
    if sparse is None:return None
    result=[];count=0
    for i,rows,labels in sparse:
        full=torch.full((len(targets),),-100,dtype=torch.long)
        full[rows]=labels;result.append((i,full));count+=len(rows)
    return result,count


def dense_routed_loss(head,hidden,routes,count):
    features=hidden if head.trunk is None else head.trunk(hidden)
    loss=features.new_zeros(())
    for i,targets in routes:
        loss=loss+F.cross_entropy(head.heads[i](features),targets,weight=head._class_weight(i),ignore_index=-100,reduction='sum')
    return loss/(count.clamp_min(1) if isinstance(count,torch.Tensor) else max(count,1))
