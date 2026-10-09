"""Compare v1718/v1928 axons and find Casey families outside the v1718 store."""
import json
from datetime import datetime, timezone
from pathlib import Path

import lance
import pandas as pd
from caveclient import CAVEclient

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / 'analysis/casey_additional_axons_v1928'
CASEY = Path('/home/jcbliao/rotation/segclr/misc/casey_class_hierarchy/microns_public_v1822_ct_csm_v2_sep15.parquet')
STORE = Path('/orcd/compute/sdorkenw/001/segclr-db/microns')
STRATEGIES = ['axon_partially_extended', 'axon_fully_extended']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    token = json.loads((Path.home()/'.cloudvolume/secrets/global.daf-apis.com-cave-secret.json').read_text())['token']
    client = CAVEclient('minnie65_phase3_v1', auth_token=token, version=1928)
    frames = {}
    metadata = {}
    for version in [1718, 1928]:
        metadata[version] = client.materialize.get_version_metadata(version)
        for table in ['proofreading_status_and_strategy', 'nucleus_detection_v0']:
            path = OUT/f'{table}_v{version}.parquet'
            if path.exists():
                frame = pd.read_parquet(path)
            else:
                print('Query', version, table, flush=True)
                frame = client.materialize.query_table(table, materialization_version=version, split_positions=True)
                frame.to_parquet(path, index=False)
            frames[version, table] = frame
            print(version, table, len(frame), flush=True)
    store = lance.dataset(str(STORE/'dims/cells.lance')).to_table(columns=['root_id', 'mat_version']).to_pandas()
    assert set(store.mat_version) == {1718}
    old_nuclei = frames[1718, 'nucleus_detection_v0'][['id','pt_root_id']].rename(columns={'id':'nucleus_id'})
    existing = old_nuclei[old_nuclei.pt_root_id.isin(store.root_id)]
    existing.to_csv(OUT/'existing_v1718_store_nuclei.csv', index=False)
    excluded = set(existing.nucleus_id)
    inventories = {}
    counts = {}
    for version in [1718,1928]:
        proof = frames[version,'proofreading_status_and_strategy']
        extended = proof[proof.strategy_axon.isin(STRATEGIES)].copy()
        nuclei = frames[version,'nucleus_detection_v0'][['id','pt_root_id']].rename(columns={'id':'nucleus_id'})
        joined = extended.merge(nuclei, on='pt_root_id', how='left')
        inventories[version] = joined
        joined.to_parquet(OUT/f'extended_axons_v{version}.parquet',index=False)
        counts[version] = {'unique_roots':int(extended.pt_root_id.nunique()),
            'strategy_unique_roots':{k:int(g.pt_root_id.nunique()) for k,g in extended.groupby('strategy_axon')},
            'unique_nuclei':int(joined.nucleus_id.nunique()),
            'roots_without_nucleus':int(joined.loc[joined.nucleus_id.isna(),'pt_root_id'].nunique()),
            'status_axon_false_roots':int(extended.loc[~extended.status_axon.eq(True),'pt_root_id'].nunique())}
    old_ids = set(inventories[1718].nucleus_id.dropna().astype('int64'))
    new_ids = set(inventories[1928].nucleus_id.dropna().astype('int64'))
    added = inventories[1928][inventories[1928].nucleus_id.isin(new_ids-old_ids)]
    removed = inventories[1718][inventories[1718].nucleus_id.isin(old_ids-new_ids)]
    added.to_csv(OUT/'newly_extended_nuclei.csv',index=False)
    removed.to_csv(OUT/'no_longer_extended_nuclei.csv',index=False)
    a = inventories[1718][['nucleus_id','strategy_axon']].dropna().drop_duplicates()
    b = inventories[1928][['nucleus_id','strategy_axon']].dropna().drop_duplicates()
    transitions = a.merge(b,on='nucleus_id',suffixes=('_1718','_1928')).groupby(['strategy_axon_1718','strategy_axon_1928']).nucleus_id.nunique()
    casey = pd.read_parquet(CASEY)
    candidates = casey[casey.ct_coarse.isin(['BipFam','MartFam','NglFam']) & ~casey.cell_id.isin(excluded)].copy()
    candidates = candidates.rename(columns={'cell_id':'nucleus_id','root_id':'casey_v1822_root_id'})
    current = frames[1928,'nucleus_detection_v0'][['id','pt_root_id']].rename(columns={'id':'nucleus_id','pt_root_id':'root_id_v1928'})
    current['root_id_v1928'] = current.root_id_v1928.astype('Int64')
    resolved = candidates.merge(current,on='nucleus_id',how='left',validate='many_to_one')
    proof = frames[1928,'proofreading_status_and_strategy'][['pt_root_id','status_axon','strategy_axon','status_dendrite','strategy_dendrite']].rename(columns={'pt_root_id':'root_id_v1928'})
    resolved = resolved.merge(proof,on='root_id_v1928',how='left')
    resolved.to_parquet(OUT/'casey_candidates_audit.parquet',index=False)
    eligible = resolved[resolved.strategy_axon.isin(STRATEGIES)].copy()
    assert not set(eligible.nucleus_id) & excluded
    for col in ['nucleus_id','casey_v1822_root_id','root_id_v1928']:
        eligible[col] = eligible[col].astype('int64')
    eligible = eligible.sort_values(['ct_coarse','ct_coarse_confidence','nucleus_id'],ascending=[True,False,True])
    eligible.to_csv(OUT/'additional_casey_extended_axons.csv',index=False)
    eligible.to_parquet(OUT/'additional_casey_extended_axons.parquet',index=False)
    summary = {'checked_at':datetime.now(timezone.utc).isoformat(),'datastack':'minnie65_phase3_v1',
        'metadata':metadata,'existing_store':str(STORE),'casey_source':str(CASEY),
        'existing_store_roots':len(store),'existing_store_roots_without_nucleus':int((~store.root_id.isin(existing.pt_root_id)).sum()),
        'excluded_existing_nuclei':len(excluded),'confidence_cutoff':None,
        'extended_axon_inventories':counts,'newly_extended_nuclei':len(new_ids-old_ids),
        'no_longer_extended_nuclei':len(old_ids-new_ids),
        'transitions':{' -> '.join(k):int(v) for k,v in transitions.items()},
        'casey_candidates_after_exclusion':candidates.groupby('ct_coarse').nucleus_id.nunique().to_dict(),
        'casey_candidates_unresolved_v1928':int(resolved.loc[resolved.root_id_v1928.isna(),'nucleus_id'].nunique()),
        'additional_casey_extended_axons':eligible.groupby('ct_coarse').nucleus_id.nunique().to_dict(),
        'additional_casey_by_strategy':{' / '.join(k):int(g.nucleus_id.nunique()) for k,g in eligible.groupby(['ct_coarse','strategy_axon'])}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    print(json.dumps(summary,indent=2,default=str),flush=True)


if __name__ == '__main__':
    main()
