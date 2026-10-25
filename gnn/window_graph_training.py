"""Capture fixed-capacity steps with masked padding; only real windows contribute."""
import time
import numpy as np
import torch
from gnn.cuda_graph_training import CUDAGraphTrainer
from gnn.training_performance import prepare_dense_routes,dense_routed_loss

class WindowGraphTrainer(CUDAGraphTrainer):
    def __init__(self,model,optimizer,scaler,dataset,amp=True,bucket_size=128):
        super().__init__(model,optimizer,scaler,amp)
        self.dataset=dataset;self.bucket_size=bucket_size;p=dataset._packed
        self.fake=len(dataset);self.k=int(dataset.window_sizes.max());self.fields=p['device_fields']
        self.transfer_stream=torch.cuda.Stream();self.batch_counter=0
        required=(len(dataset)+1)*self.k*8
        if model.config.architecture.startswith('graph_transformer'):required+=(len(dataset)+1)*self.k*self.k*4
        if required+8*2**30>torch.cuda.mem_get_info()[0]:
            raise MemoryError('Insufficient GPU memory for masked-window graph buffers')
        nodes=int(p['ptr'][-1]);graphs=np.repeat(np.arange(len(dataset)),dataset.window_sizes)
        members=np.full((len(dataset)+1,self.k),-1,np.int64)
        members[graphs,np.arange(nodes)-p['ptr'][graphs]]=np.arange(nodes)
        self.members=torch.from_numpy(members).cuda();del graphs,members
        self.geometry=model.config.architecture.startswith('graph_transformer');self.adjacency=None
        if self.geometry:
            adjacency=np.zeros((len(dataset)+1,self.k,self.k),np.float32)
            edge_graphs=np.repeat(np.arange(len(dataset)),np.diff(p['edge_offsets']))
            np.add.at(adjacency,(edge_graphs,p['edges'][0],p['edges'][1]),1)
            self.adjacency=torch.from_numpy(adjacency).cuda();del adjacency,edge_graphs
        routes,count=prepare_dense_routes(model.cls_head,torch.from_numpy(p['targets']))
        self.cpu_targets={i:labels.numpy() for i,labels in routes}
        self.active_bits=np.zeros(len(dataset),np.uint64) if max(self.cpu_targets,default=0)<64 else None
        if self.active_bits is not None:
            for i,labels in self.cpu_targets.items():self.active_bits|=(labels>=0).astype(np.uint64)<<np.uint64(i)
        self.targets={i:torch.cat((labels,torch.tensor([-100]))).cuda() for i,labels in routes}
        self.post=None if p['post'] is None else torch.from_numpy(np.vstack((p['post'],np.zeros((1,p['post'].shape[1]),np.float32)))).cuda()

    def build(self,window_ids,bucket):
        member_ids=self.members.index_select(0,window_ids).flatten();valid=member_ids>=0;safe=member_ids.clamp_min(0)
        values={name:torch.where(valid.view((-1,)+(1,)*(field.ndim-1)),field.index_select(0,safe),field.new_zeros(())) for name,field in self.fields.items()}
        if 'device_augmentation_bank' in self.dataset._packed:
            # Captured CUDA RNG generates fresh, independent views on every replay.
            values['x']=torch.where(valid[:,None],self.dataset.sample_packed_x(safe,device=True),values['x'].new_zeros(()))
        batch=torch.arange(bucket,device=window_ids.device).repeat_interleave(self.k)
        adjacency=None if self.adjacency is None else self.adjacency.index_select(0,window_ids)
        post=None if self.post is None else self.post.index_select(0,window_ids)
        return values,batch,valid,adjacency,post

    def _window_loss(self,window_ids,bucket,active):
        values,batch,valid,adjacency,post=self.build(window_ids,bucket)
        with torch.autocast('cuda',dtype=torch.float16,enabled=self.amp,cache_enabled=False):
            hidden=self.model(values['x'],None,batch,pos_enc=values.get('pos_enc'),rel_pos=values.get('rel_pos'),thickness=values.get('thickness'),has_segclr=values['has_segclr'],postsynaptic_embedding=post,batch_shape=(bucket,self.k),node_valid=valid,dense_adjacency=adjacency)
            routes=[(i,self.targets[i].index_select(0,window_ids)) for i in active]
            count=sum((labels!=-100).sum() for _,labels in routes)
            return dense_routed_loss(self.model.cls_head,hidden,routes,count)

    def step_indices(self,indices):
        indices=np.asarray(indices,np.int64);real=len(indices);bucket=((real+self.bucket_size-1)//self.bucket_size)*self.bucket_size
        bits=None if self.active_bits is None else int(np.bitwise_or.reduce(self.active_bits[indices]))
        active=tuple(i for i,labels in self.cpu_targets.items() if (np.any(labels[indices]>=0) if bits is None else bits & (1<<i)));key=(bucket,active,self.batch_counter%2)
        self.batch_counter+=1
        pinned=torch.empty(bucket,dtype=torch.long,pin_memory=True);pinned.fill_(self.fake);pinned[:real].copy_(torch.from_numpy(indices))
        cached=self.cache.get(key)
        if cached is None:
            started=time.perf_counter();ids=torch.empty(bucket,dtype=torch.long,device='cuda')
            with torch.cuda.stream(self.transfer_stream):
                ids.copy_(pinned,non_blocking=True)
                ready=torch.cuda.Event();ready.record()
            torch.cuda.current_stream().wait_event(ready)
            self.stream.wait_stream(torch.cuda.current_stream())
            if not self.warmed:
                with torch.cuda.stream(self.stream):
                    for _ in range(3):
                        self.optimizer.zero_grad(set_to_none=True);self.scaler.scale(self._window_loss(ids,bucket,active)).backward()
                torch.cuda.current_stream().wait_stream(self.stream);self.warmed=True
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,pool=self.pool,stream=self.stream,capture_error_mode='thread_local'):
                self.optimizer.zero_grad(set_to_none=True);loss=self._window_loss(ids,bucket,active)
                self.scaler.scale(loss).backward();self.scaler.step(self.optimizer);self.scaler.update()
                self.loss_sum.add_(loss.detach().double()*(ids!=self.fake).sum())
            cached=[graph,ids,torch.cuda.Event(),ready];self.cache[key]=cached;self.misses+=1;self.capture_seconds+=time.perf_counter()-started
        else:
            self.hits+=1
            with torch.cuda.stream(self.transfer_stream):
                self.transfer_stream.wait_event(cached[2])
                cached[1].copy_(pinned,non_blocking=True)
                ready=cached[3];ready.record()
            torch.cuda.current_stream().wait_event(ready)
        cached[0].replay()
        cached[2].record()
        return real
