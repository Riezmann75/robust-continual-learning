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
            group = groups.setdefault(key, {'m': [], 'seen': [], 'outer': [], 'last': None, 'per_example': [], 'entropy': [], 'max_weight': []})
            group['m'].append(row['m_after'])
            group['per_example'].extend(row['m_after_per_example'])
            group['seen'].append(row['seen_consistency'])
            outer_key = 'outer_consistency_after'
            if 'solver_variant' in row:
                outer_key = 'outer_query_js'
            elif row['outer_consistency'] == 'virtual_query_ce':
                outer_key = 'mixer_loss'
            group['outer'].append(row[outer_key])
            group['entropy'].append(row['weight_entropy_after'])
            group['max_weight'].append(row['weight_max_after'])
            group['last'] = row
            batch_m.append(row['m_after'])
            batch_step.append(row['update'])
    epochs = []
    for (task, epoch, ge), group in groups.items():
        m = np.array(group['m'])
        per_example=np.array(group['per_example'])
        epochs.append(dict(task=task, epoch=epoch, global_epoch=ge, batches=len(m),
            m_mean=float(m.mean()),weight_entropy_mean=float(np.mean(group['entropy'])),weight_max_mean=float(np.mean(group['max_weight'])),
            per_example_m_mean=float(per_example.mean()),per_example_m_std=float(per_example.std()),
            per_example_m_min=float(per_example.min()),per_example_m_max=float(per_example.max()),
            per_example_fraction_above_095=float((per_example>.95).mean()), m_min=float(m.min()), m_max=float(m.max()),
            m_end=float(m[-1]), fraction_above_095=float((m > .95).mean()),
            seen_consistency_mean=float(np.mean(group['seen'])),
            outer_consistency_after_mean=float(np.mean(group['outer'])),
            updates=group['last']['update'], presentations=group['last']['presentations']))
    consistency = next(iter(groups.values()))['last']['consistency'] if groups else 'unknown'
    outer_consistency = next(iter(groups.values()))['last']['outer_consistency'] if groups else 'unknown'
    result = dict(committed_updates=committed_updates, summary_weighting='m_* are batch mean statistics; per_example_* count weight draws: two candidate-conditioned weight draws per entry; fixed m=.5',
                  epochs=epochs, latest=epochs[-1] if epochs else None,
                  classifier_consistency=consistency,
                  outer_consistency=outer_consistency,
                  caveat=f"New chain-aware method; seen {consistency}, unseen {outer_consistency}, fixed m=.5; direct consistency minimization is not a retention hypergradient or a guarantee of robustness.")
    if outer_consistency == 'virtual_query_ce':
        result['caveat'] = 'Query CE measured at virtual classifier parameters before the mixer step.'
        for epoch in epochs:
            epoch['query_loss_mean'] = epoch.pop('outer_consistency_after_mean')
    if outer_consistency == 'feature_match':
        result['caveat'] = 'Seen mixtures match detached unmixed unseen target features; fixed m=.5; no hypergradient or robustness guarantee.'
        for epoch in epochs:
            epoch['feature_match_after_mean'] = epoch['outer_consistency_after_mean']
    temp = output/'progress.json.tmp'
    variant = group['last'].get('solver_variant') if epochs else None
    if variant:
        result.update(solver_variant=variant, summary_weighting='Epoch means over batches',
                      caveat='Outer JS is measured at the final virtual step before its mixer update.')
        for epoch in epochs:
            epoch['outer_query_js_mean'] = epoch.pop('outer_consistency_after_mean')
    temp.write_text(json.dumps(result, indent=2))
    os.replace(temp, output/'progress.json')
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(batch_step, batch_m, color='steelblue', alpha=.35, linewidth=.4, label='Every batch mean (after mixer update)')
    ax.plot([r['updates'] for r in epochs], [r['m_mean'] for r in epochs],
            color='navy', linewidth=1.4, label='Epoch mean over batches')
    if variant != 'metamix':
        ax.axhline(.5, color='gray', linestyle=':', label='Fixed m = 0.5')
    for prev, cur in zip(epochs, epochs[1:]):
        if prev['task'] != cur['task']:
            ax.axvline(prev['updates'], color='gray', linestyle='--', linewidth=.8)
    ax.set(xlabel='Classifier updates', ylabel='Original-image weight m', ylim=(0,1),
           title=variant or 'Chain-aware mixer')
    ax.legend(loc='best', fontsize=8)
    fig.tight_layout()
    temp = output/'m_progress.tmp.png'
    fig.savefig(temp, dpi=150)
    plt.close(fig)
    os.replace(temp, output/'m_progress.png')
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    axes[0].plot([r['global_epoch'] for r in epochs],[r['weight_entropy_mean'] for r in epochs])
    axes[0].axhline(float(np.log(3)),linestyle=':',color='gray')
    axes[0].set(xlabel='Global epoch',ylabel='Mean chain weight entropy')
    axes[1].plot([r['global_epoch'] for r in epochs],[r['weight_max_mean'] for r in epochs])
    axes[1].set(xlabel='Global epoch',ylabel='Mean largest chain weight',ylim=(1/3,1))
    fig.tight_layout();fig.savefig(output/'weights_progress.tmp.png',dpi=150);plt.close(fig)
    os.replace(output/'weights_progress.tmp.png',output/'weights_progress.png')
    return result

