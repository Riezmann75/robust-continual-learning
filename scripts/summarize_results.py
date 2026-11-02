"""Print completed held-out results; no training or evaluation is launched."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'results'
print('| Run | Clean class-IL | C10-C class-IL | Clean task-IL | C10-C task-IL |')
print('|---|---:|---:|---:|---:|')
for entry in json.loads((root/'manifest.json').read_text()):
    e=json.loads((root/entry['evaluation']).read_text())
    values=[e['clean']['class_il'],e['corrupted_mean']['class_il'],
            e['clean']['task_il'],e['corrupted_mean']['task_il']]
    print('| '+entry['run']+' | '+' | '.join(f'{100*x:.2f}%' for x in values)+' |')
