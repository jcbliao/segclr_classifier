"""Prepare, train and validate the conf0 counterparts of the four deployed models."""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
BASE = Path(os.environ.get('CONF0_BASE','/orcd/scratch/orcd/013/jcbliao/presynaptic_axons'))
STATE = Path(os.environ.get('CONF0_STATE',str(ROOT/'analysis/db_deployment/presynaptic/conf0')))
ORIGINAL = Path(os.environ.get('CONF0_FRAGMENT_SOURCE','/orcd/compute/sdorkenw/001/jcbliao/segclr_workflows/incoming_presynaptic_fragments_k10_collina_v2'))
OUTPUT = Path(os.environ.get('CONF0_OUTPUT',str(ORIGINAL.with_name('incoming_presynaptic_fragments_k10_collina_conf0_v1'))))
POST = Path(os.environ.get('CONF0_POST_SOURCE','/orcd/scratch/orcd/013/jcbliao/segclr/postsynaptic_site_embeddings_v1718'))
RESULTS_ROOT = os.environ.get('CONF0_RESULTS_ROOT')
REUSE_POST_CACHE = os.environ.get('CONF0_REUSE_POST_CACHE','1')=='1'
CLASS_SETS = tuple(os.environ.get('CONF0_CLASS_SETS','three,six').split(','))
TRAINING_ONLY = os.environ.get('CONF0_TRAIN_ONLY')=='1'
GROUPS = {'BasketFam':'BasketFam','BipFam':'nonBasketInhibitory','MartFam':'nonBasketInhibitory','NglFam':'nonBasketInhibitory','pyramidal':'Excitatory','thalamocortical':'Excitatory'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    temp.replace(path)


def link(dest,target):
    if dest.is_symlink():
        if dest.resolve() != target.resolve():
            raise ValueError(f'wrong existing link: {dest}')
    elif dest.exists():
        raise FileExistsError(dest)
    else:
        dest.symlink_to(target.resolve(),target_is_directory=target.is_dir())


def cache(family,database,roots,frame,grouped,checkpoint_hash):
    import numpy as np
    output = (BASE/'cave_skeletons_pre_post/k10/conf0/postsynaptic' if family=='cave_n10' else BASE/'native_skeletons_pre_post/scale16/k17/conf0')
    old = Path(str(output).replace('/conf0/','/conf0.7/'))
    if output.name == 'conf0':
        old = output.with_name('conf0.7')
    (output/'cells').mkdir(parents=True,exist_ok=True)
    expected = dict(format='postsynaptic-native-site-cache-v1',variant='cave' if family=='cave_n10' else 'new',
        database=str(database.resolve()),embeddings=str(POST.resolve()),embedding_dim=64,checkpoint_sha256=checkpoint_hash,
        label_confidence_min=0,exclusion_policy='unresolved_root_or_empty_mask; unchanged synapse and window eligibility')
    marker = output/'complete.json'
    if marker.exists():
        previous = json.loads(marker.read_text())
        if any(previous.get(k)!=v for k,v in expected.items()) or previous['n_cells']!=len(roots):
            raise ValueError('conf0 postsynaptic cache provenance mismatch')
        return output
    if REUSE_POST_CACHE:
        old_marker = json.loads((old/'complete.json').read_text())
        if old_marker['checkpoint_sha256'] != checkpoint_hash:
            raise ValueError('old postsynaptic cache uses different weights')
    counts = dict(n_cells=0,n_site_rows=0,n_eligible_sites=0,reused_cells=0)
    variant = 'cave' if family=='cave_n10' else 'new'
    for rid in sorted(roots):
        with np.load(database/'cells'/f'{rid}.npz') as z:
            ids = z[f'{variant}_synapse_id'].copy()
            valid = z[f'{variant}_valid_k_window'].copy()
        destination = output/'cells'/f'{rid}.npz'
        reusable = old/'cells'/f'{rid}.npz'
        if REUSE_POST_CACHE and reusable.exists():
            with np.load(reusable) as z:
                np.testing.assert_array_equal(z['synapse_ids'],ids)
                if not np.isfinite(z['post_x']).all():
                    raise ValueError('nonfinite cached embedding')
                eligible = z['eligible'].copy()
            link(destination,reusable)
            counts['reused_cells'] += 1
        else:
            rows = frame.iloc[grouped.get(rid,[])].set_index('synapse_id')
            positions = rows.index.get_indexer(ids)
            if np.any((valid if variant=='cave' else np.ones(len(ids),bool)) & (positions<0)):
                raise ValueError(f'{family}/{rid}: required postsynaptic sites absent from embedding source')
            present = np.flatnonzero(positions>=0)
            aligned = rows.iloc[positions[present]]
            good = aligned.postsynaptic_root_id.ne(0).to_numpy() & aligned.status.eq('ok').to_numpy()
            eligible = np.zeros(len(ids),bool)
            eligible[present[good]] = True
            vectors = np.zeros((len(ids),64),np.float32)
            if good.any():
                vectors[present[good]] = np.stack(aligned.embedding.to_numpy()[good]).astype(np.float32)
            if not np.isfinite(vectors).all():
                raise ValueError('nonfinite post vectors')
            temp = destination.with_suffix('.tmp.npz')
            np.savez_compressed(temp,synapse_ids=ids,eligible=eligible,post_x=vectors)
            temp.replace(destination)
        counts['n_cells'] += 1
        counts['n_site_rows'] += len(ids)
        counts['n_eligible_sites'] += int((valid & eligible).sum())
        if counts['n_cells']%100==0:
            print(family,counts,flush=True)
    save(marker,dict(expected,**counts))
    return output


def prepare():
    import duckdb
    import numpy as np
    import pandas as pd
    from data.dataset_lcpn import load_hierarchy
    from data.dataset_presynaptic import PresynapticWindowDataset
    from scripts.predict_incoming_fragment_cell_types import checkpoint
    state_models = []
    databases = {'cave_n10':BASE/'casey_confidence_cave/k10/conf0','single_pre_post':BASE/'casey_coarse_confidence_with_tc_folds/scale16/k17/conf0'}
    cohorts = {}
    for family,database in databases.items():
        cohorts[family] = [json.loads((database/f'fold{f}/manifest.json').read_text()) for f in range(5)]
        roots = set(cohorts[family][0]['cells'])
        for f,manifest in enumerate(cohorts[family]):
            if manifest['casey_coarse_confidence_min']!=0 or set(manifest['cells'])!=roots or manifest['fold_index']!=f:
                raise ValueError('conf0 cohort/fold mismatch')
            original = json.loads((Path(str(database).replace('/conf0','/conf0.7'))/f'fold{f}/manifest.json').read_text())
            if not set(original['cells']) < roots:
                raise ValueError('conf0 must strictly include the conf0.7 cohort')
            for rid,info in original['cells'].items():
                if info != manifest['cells'][rid]:
                    raise ValueError(f'label/split changed for existing cell {rid}')
        for other in cohorts.get('cave_n10',[]):
            if set(other['cells'])!=roots:
                raise ValueError('family cohorts differ')
    for f in range(5):
        if cohorts['cave_n10'][f]['cells'] != cohorts['single_pre_post'][f]['cells']:
            raise ValueError('family labels or splits differ')
    roots = set(cohorts['cave_n10'][0]['cells'])
    manifest = json.loads((POST/'manifest.json').read_text())
    summary = json.loads((POST/'summary.json').read_text())
    if manifest['materialization_version']!=1718 or summary['n_sites']!=manifest['n_sites']:
        raise ValueError('postsynaptic embedding verification incomplete')
    wanted = pd.DataFrame({'presynaptic_root_id':np.asarray(sorted(map(int,roots)),np.int64)})
    with duckdb.connect() as con:
        con.execute('SET threads=4')
        con.register('wanted',wanted)
        frame = con.execute('SELECT e.synapse_id,e.presynaptic_root_id,e.postsynaptic_root_id,e.status,e.embedding FROM read_parquet(?) e JOIN wanted USING(presynaptic_root_id)',[str(POST/'embeddings/*.parquet')]).df()
    if frame.duplicated(['presynaptic_root_id','synapse_id']).any() or ((frame.postsynaptic_root_id!=0)&~frame.status.isin(['ok','empty_mask'])).any():
        raise ValueError('duplicate or unfinished postsynaptic source rows')
    grouped = frame.groupby('presynaptic_root_id').indices
    for family,database in databases.items():
        post_cache = cache(family,database/'fold0',set(map(int,roots)),frame,grouped,manifest['checkpoint_sha256'])
        for classes in CLASS_SETS:
            for fold,source in enumerate(cohorts[family]):
                data = deepcopy(source)
                if classes=='three':
                    hierarchy = load_hierarchy(source)
                    buckets = {name:[] for name in set(GROUPS.values())}
                    for label,path in hierarchy.label_paths.items():
                        buckets[GROUPS[path[-1]]].append(label)
                    data['hierarchy_tree'] = {name:{'_labels_':sorted(labels)} for name,labels in buckets.items()}
                    data['hierarchy_levels_dropped'] = 0
                    data['three_class_provenance'] = dict(source_manifest=str(database/f'fold{fold}/manifest.json'),source_class_mapping=GROUPS)
                target = STATE/'manifests'/family/classes/f'fold{fold}.json'
                save(target,data)
                template_path = checkpoint(family,classes,fold).parent.with_suffix('.json')
                template = json.loads(template_path.read_text())['args']
                args = {k:v.replace('/conf0.7/','/conf0/') if isinstance(v,str) else v for k,v in template.items()}
                args.update(manifest=str(target),presynaptic_database=str(database/f'fold{fold}'),postsynaptic_cache=str(post_cache),resume=True)
                if RESULTS_ROOT:
                    args['results_dir']=str(checkpoint(family,classes,fold,confidence='0',results_root=RESULTS_ROOT).parent.parent)
                    # Legacy memory maps contain old SegCLR features.
                    args['presynaptic_memmap_root']=None
                if args.get('presynaptic_memmap_root') and not all((Path(args['presynaptic_memmap_root'])/rid/'complete.json').exists() for rid in roots):
                    args['presynaptic_memmap_root'] = None
                # Resolved old reports include optional nulls; the JSON parser
                # validates choices and rejects null for choice-valued flags.
                # Omission restores their existing None defaults.
                args = {key:value for key,value in args.items() if value is not None}
                if not args['use_postsynaptic'] or args['architecture']!='pointwise_mlp' or args['num_embeddings']!=(10 if family=='cave_n10' else 1) or args['epochs']!=30:
                    raise ValueError('training template scientific settings changed')
                path = checkpoint(family,classes,fold,confidence='0',results_root=RESULTS_ROOT)
                config = STATE/'configs'/f'{len(state_models):02d}.json'
                save(config,args)
                from scripts.train_gnn import build_training_parser
                from scripts.training_config import parse_training_args
                parsed = parse_training_args(build_training_parser(),['--config',str(config)])
                if parsed.manifest != str(target):
                    raise ValueError('training parser changed the manifest path')
                # Exercise real loader alignment and target encoding without inference.
                for split in ('train','test'):
                    rid = next(r for r,info in data['cells'].items() if info['split']==split)
                    probe = dict(data,cells={rid:data['cells'][rid]})
                    dataset = PresynapticWindowDataset(probe,split,'cave' if family=='cave_n10' else 'new',database=database/f'fold{fold}',postsynaptic_cache=post_cache,use_postsynaptic=True,single_presynaptic_embedding=family=='single_pre_post')
                    example = dataset[0]
                    if example.postsynaptic_embedding.shape != (1,64):
                        raise ValueError('post feature dimensions changed')
                state_models.append(dict(index=len(state_models),family=family,classes=classes,fold=fold,
                    config=str(config),config_sha256=sha(config),manifest_sha256=sha(target),template=str(template_path),template_sha256=sha(template_path),checkpoint=str(path)))
    if not TRAINING_ONLY:
        OUTPUT.mkdir(exist_ok=True,mode=0o2775)
        for name in ('plan.json','parts','cohort','postsynaptic_points','summary.json','fragments.duckdb','status'):
            link(OUTPUT/name,ORIGINAL/name)
        (OUTPUT/'cell_types').mkdir(exist_ok=True,mode=0o2775)
        for name in ('post_shards','post_shards_complete.json'):
            if (ORIGINAL/'cell_types'/name).exists():
                link(OUTPUT/'cell_types'/name,ORIGINAL/'cell_types'/name)
    record=dict(label_confidence_min=0,cells=len(roots),models=state_models,output=str(OUTPUT),fold_policy='unchanged original assignments')
    if TRAINING_ONLY:
        record['training_feature_plan_sha256']=sha(os.environ['CONF0_FEATURE_PLAN'])
    else:
        record.update(deployment_axon_filter='unchanged hard argmax axon per fold',source_fragment_plan_sha256=sha(ORIGINAL/'plan.json'))
    save(STATE/'plan.json',record)
    print('PREPARATION PASSED',len(state_models),'models;',len(roots),'cells',flush=True)


def train(index):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('training requires the allocated GPU')
    plan = json.loads((STATE/'plan.json').read_text())
    m = plan['models'][index]
    config = Path(m['config'])
    args = json.loads(config.read_text())
    if sha(config)!=m['config_sha256'] or sha(args['manifest'])!=m['manifest_sha256'] or sha(m['template'])!=m['template_sha256']:
        raise ValueError('frozen training inputs changed')
    report = Path(m['checkpoint']).parent.with_suffix('.json')
    if report.exists():
        saved = json.loads(report.read_text())
        if any(saved['args'].get(k)!=v for k,v in args.items()):
            raise ValueError('existing training result differs from config')
        if not Path(m['checkpoint']).exists():
            raise ValueError('completed report lacks checkpoint')
        print('Matching completed result exists',report,flush=True)
        return
    os.execv(sys.executable,[sys.executable,'-u',str(ROOT/'scripts/train_gnn.py'),'--config',str(config)])


def pilot():
    import torch
    from scripts.predict_incoming_fragment_cell_types import load_checkpoint,vectorized_predictor
    if not torch.cuda.is_available():
        raise RuntimeError('parity check requires the allocated GPU')
    torch.set_num_threads(1)
    torch.manual_seed(73)
    plan = json.loads((OUTPUT/'cell_types/plan.json').read_text())
    checked = []
    for family in ('cave_n10','single_pre_post'):
        for classes in ('three','six'):
            infos = [m for m in plan['models'] if m['family']==family and m['classes']==classes]
            models = [load_checkpoint(Path(m['path']))[0] for m in infos]
            k = 10 if family=='cave_n10' else 1
            x = torch.randn(8,k,64,device='cuda')
            counts = torch.arange(8,device='cuda')%k+1
            mask = (torch.arange(k,device='cuda')[None,:]<counts[:,None]).float()
            post = torch.randn(8,64,device='cuda')
            predictor = vectorized_predictor(models,'cuda')
            with torch.inference_mode():
                raw = predictor(x,mask,post)
                flat = x[mask.bool()]
                batch = torch.repeat_interleave(torch.arange(8,device='cuda'),counts)
                for f,model in enumerate(models):
                    hidden = model(flat,None,batch,postsynaptic_embedding=post)
                    expected = model.cls_head.predict_distribution(hidden)
                    for actual,wanted in zip(raw,expected['head_logits']):
                        torch.testing.assert_close(actual[f],wanted,rtol=2e-5,atol=2e-5)
                    checked.append(dict(family=family,classes=classes,fold=infos[f]['fold'],sha256=sha(infos[f]['path'])))
    save(OUTPUT/'cell_types/inference_parity.json',dict(passed=True,models=checked,device='cuda',comparison='vectorized inference versus training model forward and heads; varied bag sizes'))
    print('ALL 20 MODEL/FOLD PARITY CHECKS PASSED',flush=True)


def finish():
    import duckdb
    import pyarrow.parquet as pq
    from scripts.predict_incoming_fragment_cell_types import finalize
    finalize(OUTPUT,4)
    plan = json.loads((OUTPUT/'cell_types/plan.json').read_text())
    summary = json.loads((OUTPUT/'cell_types/summary.json').read_text())
    parity = json.loads((OUTPUT/'cell_types/inference_parity.json').read_text())
    if not parity['passed'] or len(parity['models'])!=20:
        raise ValueError('inference parity has not passed')
    files = sorted((OUTPUT/'cell_types/parts').glob('*/*.parquet'))
    digest = sha(OUTPUT/'cell_types/plan.json')
    for path in files:
        metadata = pq.read_schema(path).metadata
        if metadata.get(b'cell_type_plan_sha256')!=digest.encode() or metadata.get(b'fragment_plan_sha256')!=plan['fragment_plan_sha256'].encode():
            raise ValueError('prediction part provenance mismatch')
    for m in plan['models']:
        if sha(m['path'])!=m['sha256'] or sha(m['manifest'])!=m['manifest_sha256']:
            raise ValueError('model artifact changed')
    with duckdb.connect() as c:
        c.execute('SET threads=4')
        c.execute("SET memory_limit='20GB'")
        c.read_parquet(str(OUTPUT/'cell_types/parts/*/*.parquet')).create_view('predictions')
        original = c.read_parquet(str(ORIGINAL/'parts/*/*.parquet'))
        columns = [name for name in original.columns if name not in ('fragment_embeddings','mean_embedding')]
        original.create_view('original')
        cols = ','.join('"'+name+'"' for name in columns)
        for left,right in [('predictions','original'),('original','predictions')]:
            if c.sql(f'SELECT count(*) FROM ((SELECT {cols} FROM {left}) EXCEPT ALL (SELECT {cols} FROM {right}))').fetchone()[0]:
                raise ValueError('original fragment fields changed')
    save(OUTPUT/'cell_types/completion_audit.json',dict(passed=True,synapses=summary['rows'],parts=len(files),
        prediction_records=sum(summary['counts'].values()),model_folds=20,label_confidence_min=0,
        inference_parity_verified=True,source_fragment_fields_preserved=True,model_hashes_verified=True,
        part_metadata_verified=True,score_columns_validated=True,shared_database_predictions_written=False))
    print('CONF0 DEPLOYMENT AUDIT PASSED',flush=True)


def repair_status():
    """Validate existing saved shards before restoring their completion markers."""
    import pyarrow.parquet as pq
    digest=sha(OUTPUT/'cell_types/plan.json')
    fragment_digest=sha(OUTPUT/'plan.json')
    for rank in range(32):
        expected=json.loads((ORIGINAL/'status'/f'{rank:03d}_complete.json').read_text())
        saved=json.loads((OUTPUT/'cell_types/status'/f'{rank:03d}.json').read_text())
        files=sorted((OUTPUT/'cell_types/parts'/f'{rank:03d}').glob('*.parquet'))
        rows=0
        for path in files:
            part=pq.ParquetFile(path)
            md=part.schema_arrow.metadata or {}
            if md.get(b'cell_type_plan_sha256')!=digest.encode() or md.get(b'fragment_plan_sha256')!=fragment_digest.encode():
                raise ValueError('stale saved shard')
            rows+=part.metadata.num_rows
        if len(files)!=expected['parts'] or rows!=expected['synapses'] or saved['parts']!=len(files) or saved['rows']!=rows or saved['plan_sha256']!=digest:
            raise ValueError(f'incomplete saved inference shard {rank}')
        save(OUTPUT/'cell_types/status'/f'{rank:03d}_complete.json',saved)
    print('ALL 32 SAVED INFERENCE SHARDS VERIFIED',flush=True)


if __name__=='__main__':
    os.chdir(ROOT)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('prepare','train','prepare-inference','pilot','finish','repair-status'))
    p.add_argument('--index',type=int,default=0)
    a = p.parse_args()
    if a.stage=='prepare':
        prepare()
    elif a.stage=='train':
        train(a.index)
    elif a.stage=='prepare-inference':
        from scripts.predict_incoming_fragment_cell_types import prepare as prepare_inference
        prepare_inference(OUTPUT,4,confidence='0',results_root=RESULTS_ROOT)
        from scripts.predict_incoming_fragment_cell_types import prepare_posts
        prepare_posts(OUTPUT,4)
    elif a.stage=='pilot':
        pilot()
    elif a.stage=='repair-status':
        repair_status()
    else:
        finish()
