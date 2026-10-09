"""Window-local center preservation without globally densifying synaptic regions."""
import heapq
import numpy as np
from scipy.spatial import cKDTree
from data.geodesic_window import _window_laplacian_pos_enc, DEFAULT_POS_DIM

CENTER_POLICY = 'nearest-full-resolution-window-local-v2'
DEFAULT_SYNAPSE_MATCH_CUTOFF_NM = 5_000.0


class CenteredWindows:
    """Shift a nearby degree-2 sample to the true center, preserving cable paths.

    Branch points/endpoints are never shifted. If both endpoints of the containing
    coarse edge are anchors, insert only this window's center on that edge instead.
    All returned membership IDs address the full post-soma geometry.
    """
    def __init__(self, pos, edges, lengths, original_ids, coarse, xyz):
        self.pos=pos
        self.distances,self.centers=cKDTree(pos).query(xyz)
        self.full_ids=np.asarray(original_ids)
        self.coarse_full=np.searchsorted(original_ids,coarse['original_node_ids'])
        np.testing.assert_array_equal(self.full_ids[self.coarse_full],coarse['original_node_ids'])
        self.lookup=np.full(len(pos),-1,np.int64)
        self.lookup[self.coarse_full]=np.arange(len(self.coarse_full))
        self.adj=[[] for _ in self.coarse_full]
        for (a,b),length in zip(coarse['edges'],coarse['edge_length_nm']):
            self.adj[a].append((int(b),float(length)))
            self.adj[b].append((int(a),float(length)))
        # Locate removed centers on the original cable path of a coarse edge.
        self.left=np.full(len(pos),-1,np.int64)
        self.right=np.full(len(pos),-1,np.int64)
        self.offset=np.zeros(len(pos),np.float64)
        self.total=np.zeros(len(pos),np.float64)
        full_adj=[[] for _ in pos]
        for (a,b),length in zip(edges,lengths):
            full_adj[a].append((int(b),float(length)))
            full_adj[b].append((int(a),float(length)))
        for a,start in enumerate(self.coarse_full):
            for nxt,length in full_adj[start]:
                if self.lookup[nxt]>=0 or self.left[nxt]>=0:
                    continue
                prev,current=int(start),nxt
                chain=[]; arc=length
                while self.lookup[current]<0:
                    chain.append((current,arc))
                    assert len(full_adj[current])==2, 'Coarse geometry lost a branch or endpoint'
                    neighbor,weight=next(item for item in full_adj[current] if item[0]!=prev)
                    prev,current=current,neighbor
                    arc+=weight
                b=int(self.lookup[current])
                for node,offset in chain:
                    self.left[node]=a; self.right[node]=b
                    self.offset[node]=offset; self.total[node]=arc
        assert np.all((self.lookup>=0)|(self.left>=0))

    def select(self, full_center, k):
        center=int(self.lookup[full_center]); override={}
        if center<0:
            a,b=int(self.left[full_center]),int(self.right[full_center])
            t,L=float(self.offset[full_center]),float(self.total[full_center])
            candidates=[(distance,node) for distance,node in ((t,a),(L-t,b)) if len(self.adj[node])==2]
            if candidates:
                shift,center=min(candidates)
                towards=b if center==a else a
                other,old=next(item for item in self.adj[center] if item[0]!=towards)
                override[center]=[(towards,L-shift),(other,old+shift)]
                override[towards]=[(v,L-shift if v==center else w) for v,w in self.adj[towards]]
                override[other]=[(v,old+shift if v==center else w) for v,w in self.adj[other]]
            else:
                center=len(self.adj)
                override[center]=[(a,t),(b,L-t)]
                override[a]=[(v,w) for v,w in self.adj[a] if v!=b]+[(center,t)]
                override[b]=[(v,w) for v,w in self.adj[b] if v!=a]+[(center,L-t)]
        def neighbors(node):
            return override[node] if node in override else self.adj[node]
        queue=[(0.,center)]; best={center:0.}; selected=[]; seen=set(); radius=0.
        while queue and len(selected)<k:
            distance,node=heapq.heappop(queue)
            if node in seen: continue
            seen.add(node); selected.append(node); radius=distance
            for nxt,weight in neighbors(node):
                candidate=distance+weight
                if candidate<best.get(nxt,np.inf):
                    best[nxt]=candidate; heapq.heappush(queue,(candidate,nxt))
        if len(selected)<k: return None
        local={node:i for i,node in enumerate(selected)}
        pairs=[]; cable=0.
        for node in selected:
            for nxt,weight in neighbors(node):
                if nxt in local:
                    pairs.append((local[node],local[nxt]))
                    if local[node]<local[nxt]: cable+=weight
        members=np.array([full_center if node==center else self.coarse_full[node] for node in selected],np.int32)
        pairs=np.asarray(pairs,np.int32).reshape(-1,2)
        assert len(np.unique(members))==k and len(pairs)==2*(k-1)
        return members,radius,cable,pairs

    def arrays(self, synapse_ids, k, compute_lpe=True,
               match_cutoff_nm=DEFAULT_SYNAPSE_MATCH_CUTOFF_NM):
        import torch
        n=len(self.centers)
        valid=np.zeros(n,bool); radii=np.full(n,np.nan,np.float32)
        cable=np.full(n,np.nan,np.float64)
        offsets=[0]; edge_offsets=[0]; members=[]; edges=[]; lpes=[]; cache={}
        for i,center in enumerate(self.centers):
            if self.distances[i] <= match_cutoff_nm:
                center=int(center)
                if center not in cache:
                    result=self.select(center,k)
                    if result is not None:
                        nodes,radius,length,pairs=result
                        pe=(_window_laplacian_pos_enc(torch.from_numpy(pairs.T.copy()).long(),k,DEFAULT_POS_DIM).numpy()
                            if compute_lpe else None)
                        result=nodes,radius,length,pairs,pe
                    cache[center]=result
                result=cache[center]
                if result is not None:
                    nodes,radius,length,pairs,pe=result
                    valid[i]=True; radii[i]=radius; cable[i]=length
                    members.append(nodes); edges.append(pairs)
                    if pe is not None: lpes.append(pe)
            offsets.append(offsets[-1]+(k if valid[i] else 0))
            edge_offsets.append(edge_offsets[-1]+(2*(k-1) if valid[i] else 0))
        return dict(new_synapse_id=np.asarray(synapse_ids),
            new_nearest_observed_node=self.centers.astype(np.int32),
            new_nearest_observed_distance_nm=self.distances,
            new_valid_k_window=valid,new_radius_nm=radii,new_cable_length_nm=cable,
            new_window_offsets=np.asarray(offsets,np.int64),
            new_window_members=np.concatenate(members) if members else np.empty(0,np.int32),
            new_window_edge_offsets=np.asarray(edge_offsets,np.int64),
            new_window_edges=np.concatenate(edges) if edges else np.empty((0,2),np.int32),
            new_window_lpe=np.concatenate(lpes) if lpes else np.empty((0,DEFAULT_POS_DIM),np.float32))
