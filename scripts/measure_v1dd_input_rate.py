"""Measure production of complete GPU-ready input batches, not storage chunks."""
import argparse
import json
import subprocess
import time
from pathlib import Path

INPUTS = Path('/orcd/scratch/orcd/013/jcbliao/segclr/v1dd_candidates_v1196/inputs_v1')


def snapshot(job_id):
    batches = {}
    for path in INPUTS.glob('*/batch_[0-9]*/metadata.json'):
        metadata = json.loads(path.read_text())
        batches[str(path.parent)] = int(metadata['n_nodes'])
    queue = None
    if job_id:
        result = subprocess.run(['squeue', '-h', '-r', '-j', job_id, '-o', '%i %T %C'],
                                capture_output=True, text=True, timeout=15)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        queue = result.stdout.splitlines()
    return dict(time=time.time(), batches=batches, queue=queue)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--job-id')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error('--seconds must be nonnegative')
    before = snapshot(args.job_id)
    time.sleep(args.seconds)
    after = snapshot(args.job_id)
    new = {key: value for key, value in after['batches'].items() if key not in before['batches']}
    elapsed = after['time']-before['time']
    result = dict(elapsed_seconds=elapsed, new_batches=len(new), new_ready_inputs=sum(new.values()),
                  ready_inputs_per_minute=sum(new.values())*60/elapsed,
                  total_ready_inputs=sum(after['batches'].values()),
                  queue_before=before['queue'], queue_after=after['queue'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
