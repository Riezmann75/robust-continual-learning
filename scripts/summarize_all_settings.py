"""Generate the complete experiment inventory from archived evaluations."""
import json
from pathlib import Path
from summarize_corruption_categories import aggregate,table,GROUPS,LABELS
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results'
NAMES={
'metamix_seed17':'Original first-order MetaMix pilot',
'infonce_seed17':'Standard InfoNCE (1)',
'classaware_infonce_seed17':'Class-aware InfoNCE (1)',
'mse05_nce05_seed17':'Class-aware InfoNCE (0.5) + feature MSE (0.5)',
'mse02_nce1_seed17':'Class-aware InfoNCE (1) + feature MSE (0.2)',
'augce_seed17':'Augmented CE (1)',
'barlow_seed17':'Augmented CE (1) + BT augmented pair (1)',
'bt_threepair_seed17':'BT three pairs (1), no augmented CE',
'multihead_seed17':'Four heads + augmented CE (1) + diversity (0.1)',
'infonce_recon_seed17':'Class-aware InfoNCE (1) + own-view pixel MSE (1)',
'learned_js':'Learned mixer: classifier JS / outer JS',
'random_js':'Random mixer: classifier JS / no outer loss',
'random_infonce':'Random mixer: classifier class-aware InfoNCE / no outer loss',
'learned_infonce':'Learned mixer: classifier class-aware InfoNCE / outer InfoNCE',
'learned_infonce_js':'Learned mixer: classifier class-aware InfoNCE / outer JS',
'chain_aware_fixed_m':'Chain-aware, m=0.5: classifier class-aware InfoNCE / outer JS',
}

def main():
    entries=json.loads((RES/'all_settings_manifest.json').read_text());data={};groups={}
    for entry in entries:
        run=entry['run'];c=json.loads((RES/entry['config']).read_text());d=json.loads((RES/entry['config']).with_name('complete.json').read_text())
        assert c['epochs_per_task']==10 and c['images_per_class']==5000 and c['batch']==32
        assert c['tasks']==[[0,1],[2,3],[4,5],[6,7],[8,9]]
        assert d['updates']==15650 and d['presentations']==500000 and not d['pipeline_test']
        name=NAMES.get(run,'Feature MSE ('+str(c.get('lambda_feat','?'))+')')
        results=aggregate(json.loads((RES/entry['evaluation']).read_text()))
        data[run]={'label':name,'family':entry['family'],'seed':c['seed'],'parameters':c['parameters'],'config':entry['config'],'evaluation':entry['evaluation'],'metrics':results}
        groups.setdefault(entry['family'],[]).append(run)
    intro="""# Results for all completed training settings

This index covers **22 completed runs**, including repeated seeds, across all
currently completed training settings. It separates pipeline families: rows in
different families are not controlled single-factor comparisons. All results are
our implementations, not verified reproductions of the MetaMix authors' code.

## Shared budget and evaluation

Every included run uses all 50,000 clean CIFAR-10 training images, native 32x32,
five sequential class pairs (0,1), (2,3), (4,5), (6,7), (8,9), 10 epochs/task,
current batch 32 including the last short batch, and 500 reservoir slots.
Each completed 15,650 updates and 500,000 current-image presentations. Base
training includes clean CE and DER++ (replay CE 0.5, stored-logit mean MSE 0.2),
with constant classifier SGD 0.03, no momentum or weight decay.

Evaluate the final checkpoint on 10,000 clean images plus 750,000 CIFAR-10-C
images (15 corruptions, five severities). Class-IL predicts among all ten classes;
Task-IL receives the true class pair. No test-time mixing or test-based selection.
Accuracy columns are percentages. Parenthesized objective weights are coefficients,
not accuracies. Each row links its original configuration, preserving exact settings.

## What differs between families

- Historical MetaMix pilot: batch-pooled controller and earlier replay/BatchNorm
  conventions. It is not the corrected MetaMix reference.
- Random AugMix objectives: all nine operations available without the corrected
  pseudo-seen/unseen split; independent per-image/view random weights and joint
  original/two-view BatchNorm. MSE=0 retains those forwards, so it is not a pure
  no-augmentation DER++ baseline. Normalization/RNG conventions differ from the
  corrected split-pool pipeline.
- Corrected split-pool pipeline: current and replay augmentation, joint forwards,
  and the documented context/state corrections. Learned weights are shared across
  views; random weights are independent across views. Mixer SGD 0.05, J=3 when learned.
- Chain-aware extension: adds candidate descriptors, fixes m=0.5, uses per-view
  learned weights and unseen JS. Multiple components change together.

Most classifiers have 288,746 parameters. The four-head experiment has 305,288
and stores four sets of replay logits per image (more bytes despite 500 slots).
InfoNCE+reconstruction adds a 429,667-parameter decoder during training only.
Feature MSE is dimension-scaled on normalized features; pixel reconstruction uses
ordinary mean MSE. Class-aware InfoNCE excludes other same-class entries from
negatives; standard InfoNCE does not. Both use temperature 0.1. Barlow Twins uses
coordinate scaling /128 and off-diagonal coefficient 0.0051; it is not InfoNCE.

Only feature-MSE and its zero-weight control have three seeds (17,18,19).
Other settings have one seed. Do not infer statistical superiority from these rows.
[Three-seed paired results](../results/three_seed_comparison.json) are retained.
"""
    doc=intro
    for family,runs in groups.items():
        doc+='\n## '+family+'\n\n'
        rows=[]
        for run in runs:
            a=data[run];m=a['metrics'];label='['+a['label']+'](../results/'+a['config']+')'
            rows.append([label,str(a['seed'])]+[f'{100*v:.2f}' for v in [m['class_il']['clean'],m['class_il']['overall'],m['task_il']['clean'],m['task_il']['overall']]])
        doc+=table(['Training setting','Seed','Clean class-IL','CIFAR-10-C class-IL','Clean task-IL','CIFAR-10-C task-IL'],rows)+'\n'
    doc+="""
## Settings without a completed full evaluation

The original 100-epochs/task MetaMix run and the intermediate separate-forward
corrected run were stopped; neither has a final accuracy to report. The prepared
augmented-CE plus reconstruction-decoder recipe has no completed full-budget run
in this inventory. It must not be confused with the completed InfoNCE plus
reconstruction setting. Tiny pipeline/smoke tests are excluded.

## Corruption categories and regeneration

[All-settings category and per-corruption tables](all_settings_corruptions.md)
cover both Class-IL and Task-IL for every row above. Original evaluation files
retain every corruption/severity combination; no retraining was required.

```sh
python scripts/summarize_all_settings.py
```

[Machine-readable summary](../results/all_settings_summary.json) and
[artifact manifest](../results/all_settings_manifest.json).
"""
    (ROOT/'docs/all_training_results.md').write_text(doc)
    detail="""# All training settings: corruption category results

[Training objectives, seeds, budgets and comparison limitations](all_training_results.md).
Each row is a completed run, identified by the same run key as the manifest.

Class-IL predicts among ten classes; Task-IL uses the correct pair. Each corruption
averages five severities; each category averages its corruptions. Overall weights
the four groups by (3,4,4,4), not equally. Groups follow MetaMix Tables 6-7:
Noise = Gaussian/shot/impulse; Blur = defocus/glass/motion/zoom;
Weather = snow/frost/fog/brightness; Digital = contrast/elastic/pixelate/JPEG.
The five two-class task structure matches the paper; exact author class order is
unverified. Our budget, architecture, seed counts and implementation differ.

All values are percentages. These are aggregates of saved full evaluations, not
new training, new corruptions, or a claim to reproduce the paper's accuracy.
"""
    for metric,title in [('class_il','Class-IL'),('task_il','Task-IL')]:
        detail+='\n## '+title+' category means\n\n'
        rows=[]
        for run,a in data.items():
            m=a['metrics'][metric];rows.append([run,str(a['seed'])]+[f'{100*v:.2f}' for v in [*[m['categories'][k] for k in GROUPS],m['overall']]])
        detail+=table(['Run','Seed',*GROUPS,'Overall'],rows)+'\n'
        for group,names in GROUPS.items():
            detail+='\n### '+title+' — '+group+' per corruption\n\n'
            rows=[[run,str(a['seed'])]+[f'{100*a["metrics"][metric]["per_corruption"][c]:.2f}' for c in names] for run,a in data.items()]
            detail+=table(['Run','Seed',*[LABELS[c] for c in names]],rows)+'\n'
    (ROOT/'docs/all_settings_corruptions.md').write_text(detail)
    (RES/'all_settings_summary.json').write_text(json.dumps({'units':'fraction','groups':GROUPS,'runs':data},indent=2)+'\n')
    print('Generated all-setting summary and both category/per-corruption tables for',len(data),'completed runs.')

if __name__=='__main__':main()
