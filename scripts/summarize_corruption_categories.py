"""Aggregate already completed CIFAR-10-C evaluations; no training or inference."""
import json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/corrected_mixing_seed17'
GROUPS={
    'Noise':['gaussian_noise','shot_noise','impulse_noise'],
    'Blur':['defocus_blur','glass_blur','motion_blur','zoom_blur'],
    'Weather':['snow','frost','fog','brightness'],
    'Digital':['contrast','elastic_transform','pixelate','jpeg_compression'],
}
RUNS={
    'learned_js':'MetaMix-guided mixer: JS / JS',
    'random_js':'Random mixer: JS / none',
    'random_infonce':'Random mixer: InfoNCE / none',
    'learned_infonce':'MetaMix-guided mixer: InfoNCE / InfoNCE',
    'learned_infonce_js':'MetaMix-guided mixer: InfoNCE / JS',
    'chain_aware_fixed_m':'Chain-aware mixer (m=0.5): InfoNCE / JS',
}
LABELS={'gaussian_noise':'Gauss','shot_noise':'Shot','impulse_noise':'Impulse','defocus_blur':'Defocus','glass_blur':'Glass','motion_blur':'Motion','zoom_blur':'Zoom','snow':'Snow','frost':'Frost','fog':'Fog','brightness':'Bright','contrast':'Contrast','elastic_transform':'Elastic','pixelate':'Pixel','jpeg_compression':'JPEG'}

def aggregate(e):
    expected=[c for names in GROUPS.values() for c in names]
    assert len(set(expected))==15 and set(e['corruptions'])==set(expected)
    assert e['clean']['n']==10000 and e['corrupted_examples']==750000
    result={}
    for metric in ('class_il','task_il'):
        per={}
        for name in expected:
            sev=e['corruptions'][name];assert set(sev)==set(map(str,range(1,6)))
            assert all(v['n']==10000 and 0<=v[metric]<=1 for v in sev.values())
            per[name]=sum(sev[str(s)][metric] for s in range(1,6))/5
        categories={group:sum(per[c] for c in names)/len(names) for group,names in GROUPS.items()}
        overall=sum(per.values())/15
        assert math.isclose(overall,e['corrupted_mean'][metric],abs_tol=1e-12)
        assert math.isclose(overall,sum(categories[g]*len(GROUPS[g]) for g in GROUPS)/15,abs_tol=1e-12)
        result[metric]={'clean':e['clean'][metric],'per_corruption':per,'categories':categories,'overall':overall}
    return result

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']+['---:']*(len(headers)-1))+' |']+['| '+' | '.join(row)+' |' for row in rows])

def main():
    results={}
    for run in RUNS:
        folder=BASE/run;cfg=json.loads((folder/'config.json').read_text());done=json.loads((folder/'complete.json').read_text())
        assert cfg['tasks']==[[0,1],[2,3],[4,5],[6,7],[8,9]]
        assert cfg['seed']==17 and cfg['epochs_per_task']==10 and cfg['images_per_class']==5000
        assert done['updates']==15650 and done['presentations']==500000 and not done['pipeline_test']
        results[run]=aggregate(json.loads((folder/'evaluation.json').read_text()))
    output={'groups':GROUPS,'units':'fraction; Markdown tables show percentages','aggregation':'average five severities per corruption, then average corruptions within category; overall gives each of the 15 corruptions equal weight','task_pairs':[[0,1],[2,3],[4,5],[6,7],[8,9]],'results':results}
    (BASE/'corruption_categories.json').write_text(json.dumps(output,indent=2)+'\n')
    text="""# CIFAR-10-C results by category: Class-IL and Task-IL

These tables aggregate the saved full evaluations of six completed seed-17 runs;
no model was retrained and no new inference or substitute corruptions were needed.
Method labels show **classifier consistency / mixer objective**; all include CE + DER++.
MetaMix-guided denotes our paper-guided implementation, not verified author code.
The chain-aware model also fixes m=0.5 and changes controller inputs and per-view
weighting, so it is not an isolated outer-objective ablation.

## Splits and interpretation

- All configurations use five sequential tasks: (0,1), (2,3), (4,5), (6,7), (8,9).
  These are airplane/automobile, bird/cat, deer/dog, frog/horse, ship/truck in CIFAR-10.
- Each run uses all 50,000 clean training images, the custom CNN, and 10 epochs/task.
- **Class-IL:** predict among all ten output classes without a task identity.
- **Task-IL:** mask outputs to the true class pair. Correct task information is supplied at evaluation; it is not inferred by the model.
- Evaluate the final checkpoint on 10,000 clean and 750,000 corrupted test images;
  no test-time mixing and no test-based checkpoint selection.

The grouping matches [MetaMix supplement, Tables 6 and 7](https://openaccess.thecvf.com/content/CVPR2023/supplemental/Wang_MetaMix_Towards_Corruption-Robust_Continual_Learning_With_Temporally_Self-Adaptive_Data_Transformation_CVPR_2023_supplemental.pdf).
The [main paper's evaluation section](https://openaccess.thecvf.com/content/CVPR2023/papers/Wang_MetaMix_Towards_Corruption-Robust_Continual_Learning_With_Temporally_Self-Adaptive_Data_Transformation_CVPR_2023_paper.pdf)
describes five two-class tasks and averaging over five severities. The text checked
does not establish the exact class order used in their runs. Our fixed pairing is
explicit above; identical ordering with the authors is not claimed. Our CNN,
10-epoch budget, single seed and implementation conventions differ from the paper,
so these are our results, not reproductions of its numerical tables.

## Aggregation

For each corruption, average accuracy over severities 1-5 (50,000 predictions).
Each category is the mean of its corruption accuracies. The overall corrupted
average is the mean of all 15 corruptions, **not** an unweighted average of four
categories: Noise has 3 corruptions, while Blur, Weather and Digital each have 4.
Thus overall = (3*Noise + 4*Blur + 4*Weather + 4*Digital)/15.

CIFAR-10 has 1,000 test examples per class, so each two-class task has 2,000.
Our full-test accuracy therefore equals the equally weighted average of the five
task accuracies. The artifacts do not contain a separate per-task breakdown or
backward-transfer metric. Category grouping is an analysis of test corruptions,
not the pseudo-seen/unseen augmentation split used during training.

All values below are **percent accuracy**; calculations use unrounded values.
"""
    for metric,title in [('task_il','Task-IL'),('class_il','Class-IL')]:
        text+='\n## '+title+' category summary\n\n'
        rows=[]
        for run,label in RUNS.items():
            a=results[run][metric]
            rows.append([label]+[f'{100*v:.2f}' for v in [a['clean'],*[a['categories'][g] for g in GROUPS],a['overall']]])
        text+=table(['Method','Clean',*GROUPS,'Overall corrupted'],rows)+'\n'
        text+='\n### '+title+' per-corruption results\n'
        for group,names in GROUPS.items():
            text+='\n#### '+group+'\n\n'
            rows=[[label]+[f'{100*results[run][metric]["per_corruption"][c]:.2f}' for c in names] for run,label in RUNS.items()]
            text+=table(['Method',*[LABELS[c] for c in names]],rows)+'\n'
    text+='\n## Reproduce the report\n\n```sh\npython scripts/summarize_corruption_categories.py\n```\n\n[Machine-readable category results](../results/corrected_mixing_seed17/corruption_categories.json) retain fractional accuracies. Original evaluation.json files retain all 75 corruption/severity combinations for both evaluation modes.\n'
    (ROOT/'docs/corruption_category_results.md').write_text(text)
    print(json.dumps({run:{metric:a['categories'] for metric,a in values.items()} for run,values in results.items()},indent=2))

if __name__=='__main__':main()
