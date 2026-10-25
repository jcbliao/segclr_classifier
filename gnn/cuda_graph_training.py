"""Replay complete optimizer steps; preserve active heads and ignore other rows."""
import torch
import time
from gnn.training_performance import dense_routed_loss

class CUDAGraphTrainer:
    def __init__(self,model,optimizer,scaler,amp=True,max_graphs=2048):
        if model.config.architecture not in ("mean", "linear", "pointwise_mlp", "graph_transformer", "graph_transformer_teasar") or model.config.cls_resnet_bn or model.config.cls_resnet_dropout or (model.config.architecture.startswith("graph_transformer") and model.config.gt_dropout):
            raise ValueError("Full-step capture requires the tested deterministic model configuration")
        self.model=model;self.optimizer=optimizer;self.scaler=scaler;self.amp=amp
        self.max_graphs=max_graphs;self.cache={};self.pool=torch.cuda.graph_pool_handle()
        self.stream=torch.cuda.Stream();self.warmed=False
        self.loss_sum=torch.zeros((),device='cuda',dtype=torch.float64)
        self.hits=0;self.misses=0;self.fallbacks=0;self.capture_seconds=0.0
        # Initialize persistent Adam buffers outside capture, including previously unused heads.
        for group in optimizer.param_groups:
            group['capturable']=True
            for parameter in group['params']:
                state=optimizer.state[parameter]
                if not state:
                    state['step']=torch.zeros((),device=parameter.device)
                    state['exp_avg']=torch.zeros_like(parameter)
                    state['exp_avg_sq']=torch.zeros_like(parameter)
                    if group.get('amsgrad'):state['max_exp_avg_sq']=torch.zeros_like(parameter)
        scaler.scale(torch.zeros((),device='cuda'))

    def _loss(self,values,active,count,shape):
        x,edges,batch,pos,rel,thickness,has,post=values[:8]
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.amp,cache_enabled=False):
            hidden=self.model(x,edges,batch,pos_enc=pos,rel_pos=rel,thickness=thickness,has_segclr=has,postsynaptic_embedding=post,batch_shape=shape)
            return dense_routed_loss(self.model.cls_head,hidden,list(zip(active,values[8:])),count)

    def step(self,data,shape,dense):
        routes,count=dense;active=tuple(i for i,_ in routes)
        geometry=self.model.config.architecture not in ('mean','pointwise_mlp','linear')
        values=[data.x,data.edge_index if geometry else None,data.batch,
                getattr(data,'pos_enc',None) if geometry else None,getattr(data,'rel_pos',None) if geometry else None,
                getattr(data,'thickness',None),getattr(data,'has_segclr',None) if geometry else None,getattr(data,'postsynaptic_embedding',None)]
        values.extend(t for _,t in routes)
        signature=(shape,active,count,tuple(None if t is None else (tuple(t.shape),t.dtype) for t in values))
        cached=self.cache.get(signature)
        if cached is None:
            if len(self.cache)>=self.max_graphs or torch.cuda.mem_get_info()[0]<4*2**30:
                self.fallbacks+=1
                self.optimizer.zero_grad(set_to_none=True)
                loss=self._loss(values,active,count,shape)
                self.scaler.scale(loss).backward();self.scaler.step(self.optimizer);self.scaler.update()
                self.loss_sum.add_(loss.detach().double()*shape[0]);return
            capture_started=time.perf_counter()
            static=[None if t is None else torch.empty_like(t) for t in values]
            torch._foreach_copy_([t for t in static if t is not None],[t for t in values if t is not None])
            self.stream.wait_stream(torch.cuda.current_stream())
            if not self.warmed:
                with torch.cuda.stream(self.stream):
                    for _ in range(3):
                        self.optimizer.zero_grad(set_to_none=True)
                        self.scaler.scale(self._loss(static,active,count,shape)).backward()
                torch.cuda.current_stream().wait_stream(self.stream)
                self.warmed=True
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool,stream=self.stream,capture_error_mode='thread_local'):
                self.optimizer.zero_grad(set_to_none=True)
                loss=self._loss(static,active,count,shape)
                self.scaler.scale(loss).backward();self.scaler.step(self.optimizer);self.scaler.update()
                self.loss_sum.add_(loss.detach().double()*shape[0])
            cached=(graph,static);self.cache[signature]=cached;self.misses+=1
            self.capture_seconds+=time.perf_counter()-capture_started
        else:
            self.hits+=1
            graph,static=cached
            torch._foreach_copy_([t for t in static if t is not None],[t for t in values if t is not None])
        cached[0].replay()
