"""Regenerate paper tables/figure from the eight archived full evaluations."""
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('categories', ROOT/'scripts/summarize_corruption_categories.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
IDS = dict(zip(report.RUNS, 'ABCDEFGH'))
NAMES = ['Guided', 'Random', 'Random', 'Guided', 'Guided', 'Chain-aware', 'Chain-aware', 'Chain-aware']
OBJECTIVES = ['JS / JS', 'JS / none', 'NCE / none', 'NCE / NCE', 'NCE / JS', 'NCE / JS', 'JS / JS', 'NCE / NCE']


def row(cells):
    highlight = r'\rowcolor{ourrow} ' if cells[0] == 'F' else ''
    return highlight + ' & '.join(cells) + r' \\' + '\n'


def cells(values, columns):
    formatted = []
    for value, column in zip(values, columns):
        ranks = sorted(set(column), reverse=True)
        score = f'{100*value:.2f}'
        if value == ranks[0]:
            score = r'\textbf{' + score + '}'
        elif len(ranks) > 1 and value == ranks[1]:
            score = r'\underline{' + score + '}'
        formatted.append(score)
    return formatted


def generate():
    (HERE/'tables').mkdir(exist_ok=True)
    (HERE/'figures').mkdir(exist_ok=True)
    results, summaries, manifest = {}, {}, {}
    for name in report.RUNS:
        folder = report.BASE/name
        cfg = json.loads((folder/'config.json').read_text())
        done = json.loads((folder/'complete.json').read_text())
        assert cfg['seed'] == 17 and cfg['epochs_per_task'] == 10
        assert done['updates'] == 15650 and done['presentations'] == 500000 and not done['pipeline_test']
        for filename, expected in cfg['source_hashes'].items():
            assert hashlib.sha256((folder/'source'/filename).read_bytes()).hexdigest() == expected
        results[name] = report.aggregate(json.loads((folder/'evaluation.json').read_text()))
        summaries[name] = json.loads((folder/'progress.json').read_text())
        manifest[name] = {filename: hashlib.sha256((folder/filename).read_bytes()).hexdigest()
                          for filename in ['config.json','complete.json','evaluation.json','progress.json']}
    (HERE/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    values = [[a['class_il']['clean'],a['class_il']['overall'],a['task_il']['clean'],a['task_il']['overall']]
              for a in results.values()]
    text = r'''\begin{table*}[t]
\centering\small
\setlength{\tabcolsep}{7pt}
\begin{tabular}{lllrrrr}
\toprule
ID & Mixer & Classifier / mixer & Clean class-IL & Corrupt class-IL & Clean task-IL & Corrupt task-IL \\
\midrule
'''
    for i, (name, v) in enumerate(zip(results, values)):
        text += row([IDS[name],NAMES[i],OBJECTIVES[i],*cells(v,zip(*values))])
    text += r'''\bottomrule
\end{tabular}
\caption{\textbf{All completed core experiments.} Accuracy (\%), seed 17.
Guided denotes our MetaMix-guided implementation, not verified author code.
NCE denotes class-aware InfoNCE. All chain-aware runs fix $m=0.5$; all rows use
CE + DER++. Corrupt scores average all 15 corruptions and five severities.
Blue highlights our focal chain-aware InfoNCE/JS configuration.
Bold marks the best score and underlining the second-best, ranked using unrounded values per column.}
\label{tab:main}
\end{table*}
'''
    (HERE/'tables/main_results.tex').write_text(text)

    text = '\\begin{table*}[t]\n\\centering\\small\n'
    text += '\\begin{tabular}{lllrrrrr}\n\\toprule\n'
    for metric, title in [('class_il','Class-IL'),('task_il','Task-IL')]:
        text += row([r'\multicolumn{8}{l}{\textbf{'+title+'}}'])
        text += row(['ID','Mixer','Objectives',*report.GROUPS,'Overall']) + '\\midrule\n'
        values = [[*a[metric]['categories'].values(),a[metric]['overall']] for a in results.values()]
        for i,(name,v) in enumerate(zip(results,values)):
            text += row([IDS[name],NAMES[i],OBJECTIVES[i],*cells(v,zip(*values))])
        text += '\\midrule\n' if metric == 'class_il' else '\\bottomrule\n'
    text += r'''\end{tabular}
\caption{\textbf{Corruption-category accuracy (\%).} Each category averages
five-severity corruption means. Overall weights Noise, Blur, Weather, and
Digital by 3, 4, 4, and 4 corruptions, respectively. NCE is class-aware InfoNCE;
all chain-aware runs fix $m=0.5$. Blue highlights chain-aware InfoNCE/JS;
bold marks the best score and underlining the second-best, ranked using unrounded values per column.}
\label{tab:categories}
\end{table*}
'''
    (HERE/'tables/categories.tex').write_text(text)

    text = ''
    for group,names in report.GROUPS.items():
        n = len(names)
        if group == 'Weather': text += '\\clearpage\n'
        text += '\\begin{center}\n\\begin{minipage}{\\textwidth}\n\\centering\\small\n\\setlength{\\tabcolsep}{6pt}\n'
        text += '\\begin{tabular}{lll'+'r'*(2*n)+'}\n\\toprule\n'
        text += row(['','','',r'\multicolumn{'+str(n)+r'}{c}{Class-IL}',r'\multicolumn{'+str(n)+r'}{c}{Task-IL}'])
        text += row(['ID','Mixer','Objectives',*[report.LABELS[c] for c in names]*2])+'\\midrule\n'
        values = [[a[metric]['per_corruption'][c] for metric in ['class_il','task_il'] for c in names]
                  for a in results.values()]
        for i,(name,v) in enumerate(zip(results,values)):
            text += row([IDS[name],NAMES[i],OBJECTIVES[i],*cells(v,zip(*values))])
        text += '\\bottomrule\n\\end{tabular}\n\\captionof{table}{'+group+' corruptions: accuracy (\\%) averaged over five severities. Bold: best; underlined: second-best (unrounded ranks).}\n\\end{minipage}\n\\end{center}\n'
    (HERE/'tables/per_corruption.tex').write_text(text)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(7,2.5),layout='constrained')
    offsets = [(5,-9),(-12,-10),(-11,5),(-11,5),(-12,-10),(5,0),(4,-13),(-21,6)]
    for i,(name,a) in enumerate(results.items()):
        color,marker = ('#2563a0','o') if NAMES[i]=='Guided' else ('#cf7820','s') if NAMES[i]=='Random' else ('#19806d','^')
        x,y=100*a['class_il']['clean'],100*a['class_il']['overall']
        axes[0].scatter(x,y,s=40,color=color,marker=marker)
        axes[0].annotate(IDS[name],(x,y),xytext=offsets[i],textcoords='offset points',fontsize=8,
                         arrowprops={'arrowstyle':'-','color':color,'lw':.5} if i in (1,6,7) else None)
    axes[0].set(xlabel='Clean class-IL accuracy (%)',ylabel='CIFAR-10-C class-IL accuracy (%)',xlim=(45,58),ylim=(36,50))
    axes[0].grid(alpha=.15)
    for name,color in [('learned_js','#2563a0'),('learned_infonce','#b64e45'),('learned_infonce_js','#8563a0')]:
        epochs=summaries[name]['epochs']
        axes[1].plot([e['global_epoch'] for e in epochs],[e['m_mean'] for e in epochs],label=IDS[name]+': '+OBJECTIVES[list(results).index(name)],color=color,lw=1.5)
    axes[1].axhline(.5,color='#19806d',ls='--',lw=1,label='F-H: fixed 0.5')
    axes[1].set(xlabel='Completed epoch',ylabel='Mean original-image weight m',ylim=(.45,1.02))
    axes[1].legend(fontsize=7,loc='lower right',bbox_to_anchor=(1,.14),frameon=True,facecolor='white',edgecolor='none',framealpha=1)
    axes[1].grid(alpha=.15)
    fig.savefig(HERE/'figures/overview.pdf')
    plt.close(fig)
    print('Generated tables, figure, and verified data manifest for eight runs.')


if __name__ == '__main__':
    generate()
