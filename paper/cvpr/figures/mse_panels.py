"""Build the illustrated manuscript's panels from archived batch-32 runs."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
here = Path(__file__).resolve().parent
results = here.parents[2] / 'results'
archive = results / 'corrected_mixing_seed17'
plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42, 'axes.spines.top': False, 'axes.spines.right': False})
teal, blue, gray = '#176b59', '#265885', '#485563'
fig, ax = plt.subplots(figsize=(7, 1.85))
ax.set(xlim=(0, 10), ylim=(0, 2.5)); ax.axis('off')
def box(x, y, width, text, color):
    ax.add_patch(FancyBboxPatch((x,y),width,.63,boxstyle='round,pad=.04',ec=color,fc='white',lw=1))
    ax.text(x+width/2,y+.315,text,ha='center',va='center',color=color,fontsize=8)
def arrow(a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color=gray,lw=1))
box(.08,.93,1.12,'Image x',gray)
for y, label, op, color in [(1.72,'Seen chains','Learned mixture',teal),(.17,'Unseen chains','Unmixed targets',blue)]:
    box(1.86,y,1.48,label,color); box(3.91,y,1.9,op,color)
    box(6.38,y,1.52,'Classifier\nfeatures',color)
    arrow((1.25,1.23),(1.81,y+.31)); arrow((3.4,y+.31),(3.85,y+.31))
    arrow((5.87,y+.31),(6.32,y+.31)); arrow((7.96,y+.31),(8.54,1.24))
box(8.6,.93,1.24,'Feature\nMSE',gray)
fig.savefig(here/'mse_branches.pdf',bbox_inches='tight',pad_inches=.02); plt.close(fig)
fig,(a,b)=plt.subplots(1,2,figsize=(7,2.45),layout='constrained')
for key,label,color in [('learned_js','Guided JS / JS',blue),('learned_infonce_js','Guided InfoNCE / JS','#a25a20')]:
    epochs=json.loads((archive/key/'progress.json').read_text())['epochs']
    a.plot([e['global_epoch'] for e in epochs],[e['m_mean'] for e in epochs],label=label,color=color)
a.axhline(.5,color=teal,ls='--',label='Feature MSE (fixed)')
a.set(xlabel='Completed epoch',ylabel='Original-image weight m',ylim=(.45,1.035),title='(a) Mixture strength')
a.legend(frameon=False,fontsize=7,loc='center right')
for i,(folder,label) in enumerate([(archive/'learned_infonce_js','Guided JS'),(archive/'learned_infonce','Guided NCE'),(results/'feature_matching_evaluation','Feature MSE'),(archive/'random_infonce','Random')]):
    r=json.loads((folder/'evaluation.json').read_text()); cs=[100*r[k]['class_il'] for k in ('corrupted_mean','clean')]
    b.plot(cs,[i,i],color='#a9b0b6',lw=1.5)
    for j,(v,c) in enumerate(zip(cs,[teal,blue])):
        b.scatter(v,i,c=c,s=23,label=['Corrupted','Clean'][j] if i==0 else None)
b.set(yticks=range(4),yticklabels=['Guided JS','Guided NCE','Feature MSE','Random'],xlabel='Class-IL accuracy (%)',title='(b) Final evaluation',ylim=(-.5,3.8))
b.legend(loc='upper left',fontsize=7,ncol=2,frameon=False)
fig.savefig(here/'mse_evidence.pdf',bbox_inches='tight',pad_inches=.03); plt.close(fig)
