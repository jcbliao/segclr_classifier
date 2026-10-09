import fcntl,time,json,sys,os
from pathlib import Path
p=Path('/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196')/'lock_benchmark'
p.mkdir(exist_ok=True)
file=p/(sys.argv[1] if len(sys.argv)>1 else 'shared.lock')
if len(sys.argv)>2:time.sleep(max(0,float(sys.argv[2])-time.time()))
start=time.monotonic();wait=0
for i in range(50):
 with file.open('a') as f:
  t=time.monotonic();fcntl.flock(f,fcntl.LOCK_EX);wait+=time.monotonic()-t
  time.sleep(.002)
print(json.dumps(dict(node=os.uname().nodename,iterations=50,elapsed=time.monotonic()-start,wait=wait)),flush=True)
