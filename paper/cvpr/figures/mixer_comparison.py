"""Render a compact vector comparison of the two augmentation mixers."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
OUT=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(12,5.6))
fig.subplots_adjust(.015,.02,.985,.98)
ax.set(xlim=(0,12),ylim=(0,5.6));ax.axis('off')
ink='#26384A';blue='#356B9B';teal='#078579';gray='#7C8995'
def text(x,y,s,size=10,weight='normal'):
 ax.text(x,y,s,ha='center',va='center',color=ink,fontsize=size,fontweight=weight)
def arrow(a,b,color=gray):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=11,lw=1.2,color=color))
def box(x,y,w,h,label,color=blue):
 ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.02,rounding_size=.06',fc='#F5F8FA',ec=color,lw=1.2))
 text(x+w/2,y+h/2,label)
def cells(x,y,n,w=.22,h=.25,color=blue,values=None):
 for i in range(n):
  ax.add_patch(Rectangle((x+i*w,y),w,h,fc=color,ec='white',lw=1,alpha=(values[i] if values else .55)))
 ax.add_patch(Rectangle((x,y),n*w,h,fc='none',ec=color,lw=.7))
ax.plot([6,6],[.35,5.25],color='#DDE3E8',lw=1)
text(3,5.32,'MetaMix baseline',14,'bold');text(9,5.32,'Our chain-aware mixer',14,'bold')
for center in (3,9):
 text(center,4.94,r'Context $I_i$ · 258-D',10)
 cells(center-1.25,4.48,2,.25,color=gray)
 cells(center-.70,4.48,4,.25,color=blue)
 cells(center+.35,4.48,4,.25,color=blue)
 text(center-1.0,4.32,r'$g_t^\top g_r,\ L_r$',8)
 text(center-.2,4.32,r'$e_r$',9);text(center+.85,4.32,r'$e_t$',9)
 arrow((center,4.15),(center,3.96))
 box(center-.65,3.48,1.3,.45,'LSTM')
 text(center-1.65,3.7,r'$h_{t-1},c_{t-1}$',10)
 arrow((center-1.12,3.7),(center-.69,3.7))
 arrow((center,3.44),(center,3.18))
 cells(center-.55,2.89,5,.22,color=blue)
 text(center-1.2,3.02,r'$h_t$ · 5-D',9)
# MetaMix: context-only head, shared weights
arrow((3,2.85),(3,2.52));box(2.15,2.06,1.7,.43,'Linear + softmax')
arrow((3,2.02),(3,1.7));cells(2.55,1.37,3,.3,.3,blue,[.45,.55,.75])
text(3.62,1.52,r'$w_i$',11)
arrow((2.7,1.32),(1.65,.82));arrow((3.3,1.32),(4.35,.82))
box(.75,.31,1.8,.47,'Seen mixtures');box(3.45,.31,1.8,.47,'Unseen mixtures')
text(3,.04,'Shared across views and pools',9)
# Our mixer: descriptors enter a scorer after the LSTM
# Route hidden state down-left into scorer.
ax.plot([8.95,8.95,7.8],[2.85,2.65,2.65],color=blue,lw=1.2)
arrow((7.8,2.65),(7.8,2.37),blue)
text(10.64,3.9,r'Chain descriptors $d_k$',10)
for k in range(3):
 cells(10.0,3.39-k*.3,6,.22,.23,teal)
 text(9.77,3.5-k*.3,str(k+1),8)
text(10.66,2.61,r'$[\Delta h_k,\Delta p_k]$ · 138-D each',8)
ax.plot([10.66,10.66,9.1],[2.45,2.4,2.4],color=teal,lw=1.2)
arrow((9.1,2.4),(9.1,2.37),teal)
box(7.0,1.9,2.7,.44,r'Shared scorer $g(h_t,d_k)$',teal)
text(8.35,1.69,'143 → 32 → 1, for each chain',8)
arrow((8.35,1.54),(8.35,1.29),teal)
cells(7.9,.99,3,.3,.25,teal,[.3,.9,.55]);text(8.35,.83,'3 scores → softmax',8)
arrow((8.35,.69),(8.35,.58),teal)
box(7.0,.08,2.7,.47,r'Seen mixtures: $w_{iv}$',teal)
text(10.9,1.54,'Dirichlet(1,1,1)',9)
cells(10.45,1.0,3,.3,.25,gray,[.7,.35,.55])
arrow((10.9,.95),(10.9,.58))
box(10.0,.08,1.8,.47,'Unseen mixtures',gray)
for ext in ('pdf','svg','png'):
 fig.savefig(OUT/f'mixer_comparison.{ext}',dpi=200,facecolor='white')
plt.close(fig)
