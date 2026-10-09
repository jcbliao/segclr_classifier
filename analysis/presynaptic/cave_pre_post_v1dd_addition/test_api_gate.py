"""Unrestricted request concurrency and transient-only retry behavior."""
import sys,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import v1dd_api_gate as gate

# All sixteen operations must enter together: an eight-slot gate would deadlock.
barrier=threading.Barrier(16)
def concurrent_operation():
 barrier.wait(timeout=10)
 return 'ok'
with ThreadPoolExecutor(max_workers=16) as pool:
 assert list(pool.map(lambda _:gate.retry_api(concurrent_operation,'concurrent'),range(16)))==['ok']*16
attempt=[0];flushed=[]
def operation():
 attempt[0]+=1
 if attempt[0]<3:
  response=requests.Response();response.status_code=502;raise requests.HTTPError(response=response)
 return 'ok'
with patch.object(gate.time,'sleep'):
 assert gate.retry_api(operation,'test',on_wait=lambda:flushed.append(True))=='ok'
assert attempt[0]==3 and len(flushed)==2
response=requests.Response();response.status_code=401
def forbidden():raise requests.HTTPError(response=response)
try:gate.retry_api(forbidden,'test_auth')
except requests.HTTPError:pass
else:raise AssertionError('Authentication failures must not retry as transient outages')
print('PASS unrestricted concurrency, transient retry with buffer publication, non-transient error propagation')
