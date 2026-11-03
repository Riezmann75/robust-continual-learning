"""Batch-weighted epoch summaries and the complete learned-m trajectory."""
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def report(output, committed_updates):
    output = Path(output)
    groups = {}
    batch_m, batch_step = [], []
    with (output/'trace.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['update'] > committed_updates:
                continue
            key = (row['task'], row['epoch'], row['global_epoch'])
            group = groups.setdefault(key, {'m': [], 'seen': [], 'outer': [], 'last': None, 'per_example': []})
            group['m'].append(row['m_after'])
            group['per_example'].extend(row['m_after_per_example'])
            group['seen'].append(row['seen_js'])
            group['outer'].append(row['outer_js_after'])
            group['last'] = row
            batch_m.append(row['m_after'])
            batch_step.append(row['update'])
    epochs = []
    for (task, epoch, ge), group in groups.items():
        m = np.array(group['m'])
        per_example=np.array(group['per_example'])
        epochs.append(dict(task=task, epoch=epoch, global_epoch=ge, batches=len(m),
            m_mean=float(m.mean()),
            per_example_m_mean=float(per_example.mean()),per_example_m_std=float(per_example.std()),
            per_example_m_min=float(per_example.min()),per_example_m_max=float(per_example.max()),
            per_example_fraction_above_095=float((per_example>.95).mean()), m_min=float(m.min()), m_max=float(m.max()),
            m_end=float(m[-1]), fraction_above_095=float((m > .95).mean()),
            seen_js_mean=float(np.mean(group['seen'])),
            outer_js_after_mean=float(np.mean(group['outer'])),
            updates=group['last']['update'], presentations=group['last']['presentations']))
    result = dict(committed_updates=committed_updates, summary_weighting='m_* are batch mean statistics; per_example_* count current+replay entries including repeated presentations',
                  epochs=epochs, latest=epochs[-1] if epochs else None,
                  caveat='Revised per-example first-order interpretation; author gradient/pairing details unverified; low JS alone is not robustness.')
    temp = output/'progress.json.tmp'
    temp.write_text(json.dumps(result, indent=2))
    os.replace(temp, output/'progress.json')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(batch_step, batch_m, color='steelblue', alpha=.35, linewidth=.4, label='Every batch mean (after mixer update)')
    ax.plot([r['updates'] for r in epochs], [r['m_mean'] for r in epochs],
            color='navy', linewidth=1.4, label='Epoch mean over batches')
    ax.axhline(.5, color='gray', linestyle=':', label='Initialization')
    for prev, cur in zip(epochs, epochs[1:]):
        if prev['task'] != cur['task']:
            ax.axvline(prev['updates'], color='gray', linestyle='--', linewidth=.8)
    ax.set(xlabel='Classifier updates', ylabel='Original-image weight m', ylim=(0,1),
           title='MetaMix revised small CNN: learned original-image weight')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    temp = output/'m_progress.tmp.png'
    fig.savefig(temp, dpi=150)
    plt.close(fig)
    os.replace(temp, output/'m_progress.png')
    return result

