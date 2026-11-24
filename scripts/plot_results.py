"""Rebuild four scientific figures from published CSV, into a separate folder."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from common import ROOT, COUNTS, BEAMS, rows, sha256
from statistics_utils import pareto
from verify_results import verify


def render(root=ROOT, output=None):
    root, output = Path(root), Path(output or Path(root) / 'outputs/figures')
    verify(root)
    output.mkdir(parents=True, exist_ok=True)
    data = rows(root)
    colors = {2:'#8455a5',4:'#1765a1',8:'#db7624',16:'#16806a'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    for filename, key, ylabel in [('recall-vs-beam.png','recall@100','Full-test Recall@100'),
            ('unique-vs-beam.png','mean_unique_beam_items','Mean unique valid items across all beams')]:
        fig, ax = plt.subplots(figsize=(9,5), constrained_layout=True)
        for n in COUNTS:
            selected = sorted([r for r in data if r['n'] == n], key=lambda r:r['beam'])
            line, = ax.plot([r['beam'] for r in selected],[r[key] for r in selected],marker='o',color=colors[n],label='N='+str(n))
            np.testing.assert_array_equal(line.get_ydata(), [r[key] for r in selected])
        if key == 'mean_unique_beam_items':
            ax.axhline(100,ls='--',color='#555555',label='100 unique items')
        ax.set_xticks(BEAMS)
        ax.set(xlabel='Fixed actual beam size',ylabel=ylabel,title='Latte: frozen epoch30 / seed2024')
        ax.grid(alpha=.25)
        ax.legend()
        fig.savefig(output / filename,dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,key,title,fmt in zip(axes.flat,('recall@100','ndcg@10','latency_p50_ms','mean_returned'),
            ('Recall@100','NDCG@10','Shared-GPU batch1 P50 (ms)','Returned unique items (max100)'),('.4f','.5f','.1f','.1f')):
        matrix = np.array([[next(r[key] for r in data if r['n']==n and r['beam']==b) for b in BEAMS] for n in COUNTS])
        image = ax.imshow(matrix,aspect='auto',cmap='viridis')
        ax.set_xticks(range(4),BEAMS)
        ax.set_yticks(range(4),COUNTS)
        ax.set(xlabel='Fixed beam size',ylabel='Latent token types N',title=title)
        for i,j in np.ndindex(matrix.shape):
            ax.text(j,i,format(matrix[i,j],fmt),ha='center',va='center',color='white' if image.norm(matrix[i,j])<.55 else 'black')
        fig.colorbar(image,ax=ax,shrink=.85)
    fig.savefig(output / 'heatmap.png',dpi=180)
    plt.close(fig)
    front = pareto(data)
    fig, ax = plt.subplots(figsize=(10,6),constrained_layout=True)
    markers = {250:'o',500:'s',1000:'^',2000:'D'}
    for r in data:
        ax.scatter(r['latency_p50_ms'],r['recall@100'],c=colors[r['n']],marker=markers[r['beam']],s=55,
            edgecolors='black' if r['point'] in front else 'none',linewidths=1.2)
    selected = sorted([r for r in data if r['point'] in front],key=lambda r:r['latency_p50_ms'])
    ax.plot([r['latency_p50_ms'] for r in selected],[r['recall@100'] for r in selected],'k--',alpha=.5,lw=1)
    for i,r in enumerate(selected):
        ax.annotate('N%d/B%d'%(r['n'],r['beam']),(r['latency_p50_ms'],r['recall@100']),xytext=(8,12 if i%2==0 else -20),
            textcoords='offset points',fontsize=8,arrowprops=dict(arrowstyle='-',lw=.5))
    handles = [Line2D([],[],marker='o',ls='',color=colors[n],label='N='+str(n)) for n in COUNTS]
    handles += [Line2D([],[],marker=markers[b],ls='',color='gray',label='B='+str(b)) for b in BEAMS]
    ax.legend(handles=handles,loc='lower right',ncol=2,fontsize=8)
    ax.set(xlabel='Shared-GPU batch1 P50 latency (ms)',ylabel='Full-test Recall@100',title='Observed Pareto frontier under recorded shared GPU load')
    ax.margins(x=.15,y=.18)
    ax.grid(alpha=.25)
    fig.savefig(output / 'pareto.png',dpi=180)
    plt.close(fig)
    audit = dict(status='passed',csv_sha256=sha256(root/'data/joint16/results.csv'),
        images={p.name:sha256(p) for p in sorted(output.glob('*.png'))},
        description='Figures reconstructed from verified CSV; chart arrays checked against source data.')
    (output/'figure-audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(audit))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    render(args.root,args.output)
