"""Local sequential experiment queue. Stop on failure; no automatic retries."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def write(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)
def alive(record):
    p=Path('/proc')/str(record['pid'])
    try:
        bits=(p/'stat').read_text().split()
        return bits[21]==record['start_ticks'] and bits[2]!='Z'
    except FileNotFoundError:return False
def complete(output):
    p=Path(output)
    if not (p/'complete.json').exists():return False
    c=json.loads((p/'complete.json').read_text());e=json.loads((p/'evaluation.json').read_text())
    if c['updates']!=15650 or c['presentations']!=500000 or c['pipeline_test']:raise RuntimeError('Incorrect completed budget')
    if e['clean']['n']!=10000 or e['corrupted_examples']!=750000 or len(e['corruptions'])!=15:raise RuntimeError('Incomplete evaluation')
    return True

def run(manifest_path):
    manifest_path=Path(manifest_path).resolve();folder=manifest_path.parent;manifest=json.loads(manifest_path.read_text())
    lock=(folder/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    status={'pid':os.getpid(),'started':time.time(),'state':'waiting_for_parent','runs':[]}
    status_path=folder/'status.json';write(status_path,status)
    try:
        parent=manifest['parent']
        while alive(parent['process']):time.sleep(10)
        if not complete(parent['output']):raise RuntimeError('Parent stopped without successful training and evaluation')
        for spec in manifest['runs']:
            for name,wanted in manifest['source_hashes'].items():
                if hashlib.sha256((folder/'source'/name).read_bytes()).hexdigest()!=wanted:raise RuntimeError('Queue source snapshot changed')
            row={'name':spec['name'],'output':spec['output']};status['runs'].append(row)
            if complete(spec['output']):row['state']='already_complete';write(status_path,status);continue
            status['state']='running';row['state']='running';row['started']=time.time()
            with open(spec['log'],'ab',buffering=0) as log:
                child=subprocess.Popen(spec['command'],cwd=manifest['cwd'],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                row['pid']=child.pid
                row['start_ticks']=(Path('/proc')/str(child.pid)/'stat').read_text().split()[21]
                write(status_path,status)
                print(json.dumps(row),flush=True)
                code=child.wait()
            row['exit_code']=code;row['finished']=time.time()
            if code or not complete(spec['output']):raise RuntimeError('Run failed: '+spec['name'])
            row['state']='complete';write(status_path,status)
        # Verify identical sampled inputs/operations across all three variants.
        reference=None
        for spec in manifest['runs']:
            hashes=[json.loads(line)['sampling_sha256'] for line in (Path(spec['output'])/'trace.jsonl').open()]
            if len(hashes)!=15650:raise RuntimeError('Wrong trace length')
            if reference is not None and hashes!=reference:raise RuntimeError('Cross-run sampling mismatch')
            reference=hashes
        comparison={}
        for name,output in [('learned_js',parent['output'])]+[(r['name'],r['output']) for r in manifest['runs']]:
            e=json.loads((Path(output)/'evaluation.json').read_text())
            comparison[name]={'clean':e['clean'],'corrupted_mean':e['corrupted_mean'],'output':output}
        write(folder/'comparison.json',{'results':comparison,'sampling_verified_new_runs':True,'caveat':manifest['caveat']})
        status['state']='complete';status['finished']=time.time();write(status_path,status)
    except Exception as exc:
        status['state']='failed';status['error']=str(exc);status['finished']=time.time();write(status_path,status);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args();run(a.manifest)
