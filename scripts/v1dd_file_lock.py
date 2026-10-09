"""Acquire an NFS advisory lock without its slow blocking handover path."""
import fcntl
import random
import time


def acquire_lock(handle):
    """Keep flock mutual exclusion and crash recovery; retry denied claims."""
    delay=0.01
    while True:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            return
        except BlockingIOError:
            time.sleep(random.uniform(delay/2,delay))
            delay=min(delay*1.7,0.2)
