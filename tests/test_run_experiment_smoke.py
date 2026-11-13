"""Smoke test for scripts/run_experiment.py; needs no CIFAR data.

Every config in configs/ is launched through run_experiment.py with the runners'
--pipeline-test and --stop-after-epochs 2 flags. A run therefore trains for 2 epochs
in total, one on each of the first two tasks, using the runner's synthetic
17-images-per-class dataset. Stopping after epoch 2 skips final evaluation.

Runners whose --device option does not accept this machine's device (for example,
CUDA-only runners on a machine without CUDA) are skipped.
"""
import functools
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'scripts'/'run_experiment.py'
CONFIGS=sorted((ROOT/'configs').glob('*.json'))
EPOCHS=2
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'


def write_smoke_config(source,directory):
    config=json.loads(source.read_text())
    config['args'].update(pipeline_test=True,stop_after_epochs=EPOCHS)
    path=directory/source.name
    path.write_text(json.dumps(config))
    return path,config


def launch(path,directory,device,*flags):
    return subprocess.run(
        [sys.executable,str(SCRIPT),str(path),'--device',device,
         '--data-root',str(directory/'no_data'),'--output',str(directory/'output'),*flags],
        cwd=ROOT,capture_output=True,text=True,timeout=300)


@functools.cache
def runner_help(runner):
    """Read --help, so wrappers that reuse another runner's parser are covered too."""
    return subprocess.run([sys.executable,str(runner),'--help'],
                          cwd=ROOT,capture_output=True,text=True,timeout=120)


def numbers(value):
    if isinstance(value,dict):
        value=list(value.values())
    if isinstance(value,list):
        for item in value:
            yield from numbers(item)
    elif isinstance(value,(int,float)) and not isinstance(value,bool):
        yield value


class RunExperimentSmokeTests(unittest.TestCase):
    def assert_succeeded(self,result):
        self.assertEqual(result.returncode,0,f'{result.stdout[-2000:]}\n{result.stderr[-2000:]}')

    def check_dispatch(self,source):
        """Without --run, the command is printed and nothing is trained."""
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            path,config=write_smoke_config(source,directory)
            result=launch(path,directory,'cpu')
            self.assert_succeeded(result)
            command=shlex.split(result.stdout.splitlines()[-1])
            self.assertEqual(command[1],str(ROOT/'reproduce'/config['runner']))
            self.assertIn('--pipeline-test',command)
            self.assertEqual(command[command.index('--stop-after-epochs')+1],str(EPOCHS))
            self.assertEqual(command[command.index('--device')+1],'cpu')
            self.assertEqual(command[command.index('--root')+1],str(directory/'no_data'))
            self.assertEqual(command[command.index('--output')+1],str(directory/'output'))
            self.assertFalse((directory/'output').exists())

    def check_two_epochs(self,source):
        runner=ROOT/'reproduce'/json.loads(source.read_text())['runner']
        help_result=runner_help(runner)
        self.assert_succeeded(help_result)
        choices=re.search(r'--device \{([^}]*)\}',help_result.stdout)
        self.assertIsNotNone(choices,f'{runner.name} has no --device choices')
        devices=choices.group(1).split(',')
        if DEVICE not in devices:
            self.skipTest(f'{runner.name} only accepts --device {"/".join(devices)}')
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            path,_=write_smoke_config(source,directory)
            result=launch(path,directory,DEVICE,'--run')
            self.assert_succeeded(result)
            rows=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
            epochs=[row for row in rows if row.get('event')=='epoch']
            self.assertEqual([(row['task'],row['epoch']) for row in epochs],[(1,1),(2,1)])
            self.assertTrue(0<epochs[0]['updates']<epochs[1]['updates'])
            for row in epochs:
                self.assertTrue(all(math.isfinite(n) for n in numbers(row)),row)
            saved=json.loads((directory/'output'/'config.json').read_text())
            self.assertTrue(saved['pipeline_test'])
            self.assertEqual(saved['images_per_class'],17)
            self.assertEqual(saved['device'],DEVICE)


for _source in CONFIGS:
    setattr(RunExperimentSmokeTests,f'test_dispatch_{_source.stem}',
            lambda self,source=_source:self.check_dispatch(source))
    setattr(RunExperimentSmokeTests,f'test_two_epochs_{_source.stem}',
            lambda self,source=_source:self.check_two_epochs(source))


if __name__=='__main__':
    unittest.main()
