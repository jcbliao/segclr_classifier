"""Transient-error retries for V1DD, without a global concurrency gate."""
from contextlib import contextmanager
import random,time
import requests

@contextmanager
def api_slot(on_wait=None):
 # Compatibility for diagnostic callers; requests no longer acquire NFS slots.
 yield

def retry_api(operation,label,on_wait=None):
 attempt=0
 while True:
  try:
   return operation()
  except (requests.ConnectionError,requests.Timeout) as exc:
   failure=type(exc).__name__
  except requests.HTTPError as exc:
   if exc.response is None or exc.response.status_code not in [408,429,500,502,503,504]:raise
   failure=f'HTTP_{exc.response.status_code}'
  attempt+=1;delay=min(60,2**min(attempt,6))+random.uniform(0,2)
  if on_wait is not None:on_wait()
  print(f'V1DD API RETRY operation={label} attempt={attempt} error={failure} delay_s={delay:.1f}',flush=True)
  time.sleep(delay)
