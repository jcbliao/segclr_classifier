import sqlite3,tempfile,time,json
from pathlib import Path
base=Path('/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196')
with tempfile.TemporaryDirectory(prefix='sqlite_profile_',dir=base) as folder:
    payload=bytes(262144);out=[]
    for group in [1,32]:
        c=sqlite3.connect(Path(folder)/f'{group}.sqlite');c.execute('PRAGMA synchronous=NORMAL');c.execute('CREATE TABLE chunks (key TEXT PRIMARY KEY, data BLOB)');c.commit();start=time.monotonic()
        for i in range(128):
            c.execute('INSERT INTO chunks VALUES (?,?)',(str(i),payload))
            if (i+1)%group==0:c.commit()
        c.commit();elapsed=time.monotonic()-start;c.close();out.append(dict(chunks=128,chunks_per_commit=group,seconds=elapsed))
    print(json.dumps(out),flush=True)
