"""Epoch diagnostics for Feature MSE + DER++ (random AugMix views)."""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def report(output,epochs):
    payload=dict(method='Feature MSE + DER++ (random AugMix views)',epochs=epochs,
        latest=epochs[-1] if epochs else None,
        note='Training diagnostics, not test scores. Mixture weights are random, not learned.')
    tmp=output/'progress.json.tmp'
    tmp.write_text(json.dumps(payload,indent=2))
    os.replace(tmp,output/'progress.json')
    if not epochs:
        return
    x=[e['global_epoch'] for e in epochs]
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    def line(ax,key,label):
        ax.plot(x,[e['metrics'][key] for e in epochs],label=label)
    line(axes[0,0],'current_ce','Current CE')
    line(axes[0,0],'feature_loss','Feature loss')
    axes[0,0].set_title('Loss components')
    line(axes[0,1],'clean_accuracy','Clean originals (current + replay)')
    line(axes[0,1],'augmented_accuracy','Random augmented views')
    axes[0,1].set_title('Training accuracy')
    axes[0,1].set_ylim(0,1)
    line(axes[0,2],'cosine_view1','Original / view 1')
    line(axes[0,2],'cosine_view2','Original / view 2')
    axes[0,2].set_title('Paired feature cosine')
    axes[1,0].plot(x,[e['sampling_m']['mean'] for e in epochs],label='Sampled mean m')
    axes[1,0].axhline(.5,color='gray',linestyle='--',label='Expected mean')
    axes[1,0].set_ylim(0,1)
    axes[1,0].set_title('Random weights (not learned)')
    for i,label in enumerate(('Original','View 1','View 2')):
        axes[1,1].plot(x,[e['metrics']['raw_feature_norms'][i] for e in epochs],label=label)
        axes[1,2].plot(x,[e['metrics']['normalized_feature_variance'][i] for e in epochs],label=label)
    axes[1,1].set_title('Raw feature norm')
    axes[1,2].set_title('Normalized feature variance across examples')
    for ax in axes.flat:
        ax.set_xlabel('Completed epoch')
        ax.legend(fontsize=7)
        for before,after in zip(epochs,epochs[1:]):
            if before['task']!=after['task']:
                ax.axvline(before['global_epoch']+.5,color='gray',linestyle=':',linewidth=.8)
    fig.suptitle('Feature MSE + DER++ (random AugMix views)')
    fig.tight_layout()
    tmp=output/'diagnostics.tmp.png'
    fig.savefig(tmp,dpi=140)
    plt.close(fig)
    os.replace(tmp,output/'diagnostics.png')

