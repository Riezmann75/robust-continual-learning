"""Compact mixer comparison; module colors and symbols are defined once."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42,'svg.fonttype':'none'})
fig, ax = plt.subplots(figsize=(12, 5.9))
fig.subplots_adjust(.01,.01,.99,.99)
ax.set(xlim=(0,12),ylim=(0,5.9)); ax.axis('off')
ink='#26384A'; gray='#788490'
colors={'L':'#4477AA','H':'#AA3377','C':'#BB7733','G':'#228877','S':'#7755AA'}
symbols={'L':r'$\mathcal{L}$','H':r'$H$','C':r'$\oplus$','G':r'$g_\phi$','S':r'$\sigma$'}
def text(x,y,s,size=11,bold=False):
 ax.text(x,y,s,ha='center',va='center',color=ink,fontsize=size,fontweight='bold' if bold else 'normal')
def arrow(a,b):
 ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,lw=1.15,color=gray,shrinkA=0,shrinkB=0))
def line(xs,ys): ax.plot(xs,ys,color=gray,lw=1.15)
def module(x,y,key,w=.82,h=.34):
 color=colors[key]
 ax.add_patch(FancyBboxPatch((x-w/2,y),w,h,boxstyle='round,pad=0,rounding_size=.055',fc=color,ec=color,lw=1))
 ax.text(x,y+h/2,symbols[key],ha='center',va='center',color='white',fontsize=12)
def cells(x,y,label,n=5):
 w=.14; h=.19
 for i in range(n): ax.add_patch(Rectangle((x-n*w/2+i*w,y),w,h,fc='#DCE3E9',ec='white',lw=.8))
 ax.add_patch(Rectangle((x-n*w/2,y),n*w,h,fc='none',ec=gray,lw=.6))
 text(x+.5,y+h/2,label,11)
text(2.1,5.64,'MetaMix',14,True);text(8.1,5.64,'Our chain-aware mixer',14,True)
line([4.1,4.1],[.78,5.65])
for c in (2.1,8.1):
 cells(c,5.12,r'$I_i$')
 arrow((c,5.12),(c,4.88));module(c,4.54,'L')
 text(c-1.25,4.71,r'$h_{t-1},c_{t-1}$',10)
 arrow((c-.72,4.71),(c-.41,4.71))
 arrow((c,4.54),(c,4.3));cells(c,4.11,r'$h_t$')
# MetaMix predicts all logits with one head.
arrow((2.1,4.11),(2.1,3.25));module(2.1,2.91,'H')
arrow((2.1,2.91),(2.1,2.55));cells(2.1,2.36,r'$\mathbf{s}_i$',3)
arrow((2.1,2.36),(2.1,1.99));module(2.1,1.65,'S')
arrow((2.1,1.65),(2.1,1.39));cells(2.1,1.2,r'$w_i$',3)
text(2.1,.94,'Seen + unseen views',10)
# The same current hidden state feeds three independent scoring paths.
xs=[5.35,7.55,9.75]
line([8.1,8.1],[4.11,3.89]);line([xs[0]-.23,xs[-1]-.23],[3.89,3.89])
for k,x in enumerate(xs,1):
 line([x-.23,x-.23],[3.89,3.25])
 arrow((x-.23,3.25),(x-.23,3.11))
 cells(x+.55,3.48,rf'$d_{k}$',4)
 line([x+.55,x+.55,x+.23],[3.48,3.29,3.29])
 arrow((x+.23,3.29),(x+.23,3.11))
 module(x,2.77,'C')
 arrow((x,2.77),(x,2.53));module(x,2.19,'G')
 arrow((x,2.19),(x,1.98));text(x,1.83,rf'$s_{k}$')
 arrow((x,1.68),(x,1.48))
module(7.55,1.14,'S',w=5.25)
arrow((7.55,1.14),(7.55,.94));cells(7.55,.75,r'$w_{iv}$',3)
text(6.2,.84,'Seen views',10)
text(11.1,1.8,r'$w\sim\mathrm{Dir}(\mathbf{1})$',10)
arrow((11.1,1.58),(11.1,1.15));text(11.1,.94,'Unseen views',10)
# One legend replaces repeated module names throughout the architecture.
line([.2,11.8],[.61,.61])
entries=[('L','LSTM',.65),('H','Linear head',2.55),('C','Concatenate',4.95),('G','Shared scorer',7.55),('S','Softmax',10.3)]
for key,label,x in entries:
 module(x,.12,key,w=.55,h=.3)
 ax.text(x+.4,.27,label,ha='left',va='center',color=ink,fontsize=10)
for ext in ('pdf','svg','png'):
 fig.savefig(OUT/f'mixer_comparison.{ext}',dpi=180,facecolor='white')
plt.close(fig)
