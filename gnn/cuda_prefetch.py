"""Bounded CPU preparation and H2D transfer pipeline on a dedicated CUDA stream."""
import copy
import queue
import threading
import torch
from gnn.training_performance import prepare_routes, prepare_dense_routes

class CUDABatchPrefetcher:
    def __init__(self,loader,device,head=None,dense=False):
        self.loader=loader;self.device=device;self.head=head;self.dense=dense
    def __len__(self):return len(self.loader)
    def __iter__(self):
        source=iter(self.loader)  # Start multiprocessing workers on the caller thread.
        pending=queue.Queue(maxsize=2);stop=threading.Event();stream=torch.cuda.Stream(device=self.device)
        def put(value):
            while not stop.is_set():
                try:pending.put(value,timeout=.1);return
                except queue.Full:pass
        def produce():
            try:
                with torch.cuda.device(self.device),torch.cuda.stream(stream):
                    for cpu in source:
                        if stop.is_set():break
                        shape=(cpu.num_graphs,int((cpu.ptr[1:]-cpu.ptr[:-1]).max()))
                        routes=(prepare_dense_routes(self.head,cpu.y_levels) if self.dense else prepare_routes(self.head,cpu.y_levels)) if self.head is not None else None
                        cpu=cpu.apply(lambda value:value.pin_memory() if value.device.type=='cpu' else value)
                        gpu=copy.copy(cpu).to(self.device,non_blocking=True)
                        if self.dense and routes is not None:
                            full,count=routes
                            routes=([(i,t.pin_memory().to(self.device,non_blocking=True)) for i,t in full],count)
                        elif routes is not None:
                            routes=[(i,r.pin_memory().to(self.device,non_blocking=True),t.pin_memory().to(self.device,non_blocking=True)) for i,r,t in routes]
                        ready=torch.cuda.Event();ready.record(stream)
                        put((gpu,shape,routes,ready,cpu))
                put(None)
            except BaseException as error:put(error)
        worker=threading.Thread(target=produce,name='cuda-batch-prefetch',daemon=True);worker.start()
        try:
            while True:
                value=pending.get()
                if value is None:break
                if isinstance(value,BaseException):raise value
                gpu,shape,routes,ready,cpu=value
                current=torch.cuda.current_stream(self.device);current.wait_event(ready)
                gpu.record_stream(current)
                if self.dense and routes is not None:
                    for _,targets in routes[0]:targets.record_stream(current)
                elif routes is not None:
                    for _,rows,targets in routes:rows.record_stream(current);targets.record_stream(current)
                yield gpu,shape,routes
        finally:
            stop.set();worker.join()


class CUDATuplePrefetcher(CUDABatchPrefetcher):
    """The same bounded transfer pipeline for the paired trainer's tensor tuples."""
    def __iter__(self):
        source = iter(self.loader)
        pending = queue.Queue(maxsize=2)
        stop = threading.Event()
        stream = torch.cuda.Stream(device=self.device)

        def put(value):
            while not stop.is_set():
                try:
                    pending.put(value, timeout=.1)
                    return
                except queue.Full:
                    pass

        def produce():
            try:
                with torch.cuda.device(self.device), torch.cuda.stream(stream):
                    for cpu in source:
                        if stop.is_set():
                            break
                        pinned = tuple(t.pin_memory() if t.device.type == 'cpu' and not t.is_pinned() else t for t in cpu)
                        gpu = tuple(t.to(self.device, non_blocking=True) for t in pinned)
                        event = torch.cuda.Event()
                        event.record(stream)
                        put((gpu, event, pinned))
                put(None)
            except BaseException as error:
                put(error)

        worker = threading.Thread(target=produce, name='cuda-tuple-prefetch', daemon=True)
        worker.start()
        try:
            while True:
                item = pending.get()
                if item is None:
                    break
                if isinstance(item, BaseException):
                    raise item
                gpu, event, pinned = item
                current = torch.cuda.current_stream(self.device)
                current.wait_event(event)
                for tensor in gpu:
                    tensor.record_stream(current)
                yield gpu
        finally:
            stop.set()
            worker.join()
