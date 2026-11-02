"""Launch an explicitly supplied local job detached, recording real Linux PID."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--log', type=Path, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
a = p.parse_args()
a.log.parent.mkdir(parents=True, exist_ok=True)
pidfile = a.log.with_suffix('.process.json')
if pidfile.exists():
    prior = json.loads(pidfile.read_text())
    proc = Path('/proc')/str(prior['pid'])
    if proc.exists() and (proc/'stat').read_text().split()[21] == prior.get('start_ticks'):
        raise RuntimeError('Recorded job is still alive')
with a.log.open('ab', buffering=0) as out, open(os.devnull, 'rb') as inp:
    job = subprocess.Popen(a.command, stdin=inp, stdout=out, stderr=subprocess.STDOUT,
                           start_new_session=True, close_fds=True)
    ticks = (Path('/proc')/str(job.pid)/'stat').read_text().split()[21]
record = dict(pid=job.pid, start_ticks=ticks, command=a.command, cwd=os.getcwd(), started=time.time())
pidfile.write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)

