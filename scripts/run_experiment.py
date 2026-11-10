"""Run an explicit configuration; defaults to printing its command."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('config',type=Path)
p.add_argument('--run',action='store_true',help='Execute; otherwise print only')
p.add_argument('--data-root',type=Path)
p.add_argument('--output',type=Path)
p.add_argument('--device',choices=['cuda','cpu','mps'])
p.add_argument('--backbone', choices=['smallcnn','resnet18'])
p.add_argument('--severity-upperbound', type=float, help='Override the per-operation AugMix level upper bound for a mixer run')
a=p.parse_args()
cfg=json.loads(a.config.read_text())
allowed={'virtual_mixer.py','chain_aware_mixing.py','matched_infonce_js.py','matched_mixing.py','metamix_jointbn.py','augce_cifar10.py','feature_mse_cifar10.py','infonce_cifar10.py',
         'infonce_classneg_cifar10.py','mse_infonce_cifar10.py'}
if cfg['runner'] not in allowed:raise ValueError('Unknown runner')
args=cfg['args'].copy()
if a.severity_upperbound is not None:
    if cfg['runner'] not in {'chain_aware_mixing.py', 'virtual_mixer.py'}:
        p.error('--severity-upperbound requires a chain-aware or virtual mixer runner')
    if not 0.1 <= a.severity_upperbound <= 10:
        p.error('--severity-upperbound must be between 0.1 and 10')
    args['severity_upperbound'] = a.severity_upperbound
for key,value in [('root',a.data_root),('output',a.output),('device',a.device),('backbone',a.backbone)]:
    if value is not None:args[key]=str(value)
command=[sys.executable,str(ROOT/'reproduce'/cfg['runner'])]
for key,value in args.items():
    if isinstance(value,bool):
        if value:command.append('--'+key.replace('_','-'))
    else:command.extend(['--'+key.replace('_','-'),str(value)])
print(shlex.join(command),flush=True)
if a.run:subprocess.run(command,cwd=ROOT,check=True)
