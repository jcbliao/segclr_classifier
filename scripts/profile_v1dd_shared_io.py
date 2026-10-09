import json,time
from pathlib import Path
import numpy as np
base=Path('/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196')
rows=[]
for group in [base/'chunks_v1196_v1'/'image_mip3',next((base/'chunks_v1196_v1'/'mask_mip2').iterdir())]:
    files=[]
    for bucket in group.iterdir():
        if bucket.is_dir(): files.extend(list(bucket.glob('*.npy'))[:16])
        if len(files)>=128: break
    for repeat in range(2):
        started=time.perf_counter(); total=0
        for path in files: total+=np.load(path,allow_pickle=False).nbytes
        rows.append(dict(path=str(group),repeat=repeat,files=len(files),bytes=total,seconds=time.perf_counter()-started))
# Probe production destination writing; remove only this temporary probe.
p=base/f'profile_io_probe_{__import__("os").getpid()}.npy'
try:
    data=np.zeros((128,129,129,129),dtype=np.uint8); started=time.perf_counter(); np.save(p,data)
    rows.append(dict(write_bytes=data.nbytes,seconds=time.perf_counter()-started))
except OSError as e: rows.append(dict(write_error=str(e)))
finally: p.unlink(missing_ok=True)
Path('analysis/v1dd_cell_types/input_profile/shared_io.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(rows),flush=True)
