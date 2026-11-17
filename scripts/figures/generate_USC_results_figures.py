#!/usr/bin/env python3
"""Recreate USC Results figures 4-9 from the uploaded analysis files.

Usage:
    python generate_USC_results_figures.py --data-dir /mnt/data --out-dir /mnt/data/USC_figures

Requires pandas, numpy, matplotlib, openpyxl. Optional: WMSC per-turn CSV
with columns file, turn_number, WMSC (pass --wmsc-csv path).

IMPORTANT: Uses REPRODUCED 39-file results, NOT the unreconciled values printed
in the manuscript. This script is for reproducible replotting, not a pixel-level
reproduction of the paper's original illustrations.
"""
import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

METRICS=['BERTScore_F1','Raw_SBERT','TaskAware_SBERT','Consecutive_BLEU','Consecutive_ROUGE_L','WMSC']
LABELS={'BERTScore_F1':'BERTScore F1','Raw_SBERT':'Raw SBERT','TaskAware_SBERT':'Task-Aware SBERT','Consecutive_BLEU':'BLEU','Consecutive_ROUGE_L':'ROUGE-L','WMSC':'WMSC'}
COLORS=dict(zip(METRICS,plt.get_cmap('tab10').colors[:6]))

def save(fig, out, name):
    fig.savefig(out/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(out/(name+'.png'),dpi=300,bbox_inches='tight')
    plt.close(fig)

def annotations(path):
    a=pd.read_excel(path).dropna(subset=['File Name']).copy()
    vote_cols=[c for c in a if str(c).upper().endswith('_STOP')]
    for col in vote_cols:
        a[col]=a[col].astype(str).str.strip()
    a['votes']=a[vote_cols].apply(lambda x:[v for v in x if re.fullmatch(r'\d+\s*[-–]\s*[SFC]',v,re.I)],axis=1)
    a['stops']=a['votes'].apply(lambda vs:[int(re.match(r'(\d+)',v).group(1)) for v in vs])
    a['outcomes']=a['votes'].apply(lambda vs:[re.search(r'([SFC])$',v,re.I).group(1).upper() for v in vs])
    a['max_row_votes']=a['stops'].apply(lambda xs:max([xs.count(x) for x in set(xs)],default=0))
    a['strict']=a.apply(lambda r:len(r.stops)==7 and len(set(r.outcomes))==1 and r.outcomes[0]!='C' and max(r.stops)-min(r.stops)<=2,axis=1)
    a['majority']=a['max_row_votes']>=5
    return a

def fig4(a,out):
    # Sort by agreement and then by task; plot each vote and outcome counts.
    a=a.copy().sort_values(['max_row_votes','File Name'],ascending=[False,True]).reset_index(drop=True)
    fig,(ax,bx)=plt.subplots(1,2,figsize=(13,max(8,len(a)*.28)),gridspec_kw={'width_ratios':[3,1.15]})
    oc={'S':'tab:green','F':'tab:red','C':'0.55'}
    for i,r in a.iterrows():
        for x,o in zip(r.stops,r.outcomes):ax.scatter(x,i,color=oc[o],marker={'S':'o','F':'s','C':'x'}[o],s=27,alpha=.75)
        counts=[r.outcomes.count(z) for z in ['S','F','C']]
        left=0
        for n,z in zip(counts,['S','F','C']):
            bx.barh(i,n,left=left,color=oc[z],height=.65);left+=n
    ax.set_yticks(range(len(a)),[str(x)[:43] for x in a['File Name']],fontsize=6)
    ax.invert_yaxis();bx.invert_yaxis();ax.set_xscale('symlog',linthresh=1)
    ax.set_xlabel('Human stopping row (symlog scale)');bx.set_xlabel('Number of annotators')
    bx.set_xlim(0,7);bx.set_yticks([]);ax.set_title('(a) Individual stopping rows');bx.set_title('(b) Outcome votes')
    ax.legend(handles=[Line2D([0],[0],marker={'S':'o','F':'s','C':'x'}[z],color=oc[z],linestyle='None',label={'S':'Success','F':'Failure','C':'Continue'}[z]) for z in ['S','F','C']],loc='lower right',fontsize=8)
    fig.suptitle('Human stopping judgments (all annotation entries)');save(fig,out,'Fig4_human_stopping_judgments')

def fig5(out):
    # Paper's Fig. 5 is an explanatory case diagram, not a numerical plot.
    cases=[('Fibonacci game','Same stopping row; different outcome','Compilable guessing logic; disagreement on game completeness'),('Palindrome','Same failure outcome; different row','Early incorrect implementation vs. later unproductive loop'),('Anagram game','Stop / failure / continue disagreement','Interactive game with unresolved anagram validation flaw')]
    fig,ax=plt.subplots(figsize=(10,4.5));ax.axis('off')
    for i,(name,issue,detail) in enumerate(cases):
        y=.83-i*.31
        ax.text(.03,y,name,weight='bold',fontsize=12,transform=ax.transAxes)
        ax.text(.31,y,issue,fontsize=11,transform=ax.transAxes)
        ax.text(.31,y-.10,detail,fontsize=9,transform=ax.transAxes)
        if i<2:ax.plot([.02,.98],[y-.17,y-.17],color='.75',transform=ax.transAxes)
    ax.set_title('Illustrative reasons for annotation disagreement',fontsize=13)
    save(fig,out,'Fig5_annotation_disagreement_examples')

def fig6(au,diffs,out):
    au=au.melt(id_vars=['setting','conversation'],value_vars=[m for m in METRICS if m in au],var_name='metric',value_name='AUROC')
    means=au.groupby(['setting','metric'])['AUROC'].mean()
    ds=diffs.groupby(['setting','metric'])['stop_minus_poststop'].mean()
    fig,(ax,bx)=plt.subplots(1,2,figsize=(12,5.2),sharey=True)
    names=[m for m in METRICS if m in au.metric.unique()]
    y=np.arange(len(names))
    for off,setting in [(-.16,'strict'),(.16,'majority')]:
        ax.scatter([means.get((setting,m),np.nan) for m in names],y+off,label=setting,s=52)
        bx.scatter([ds.get((setting,m),np.nan) for m in names],y+off,label=setting,s=52)
    ax.axvline(.5,ls='--',color='.5');bx.axvline(0,ls='--',color='.5')
    ax.set_yticks(y,[LABELS[m] for m in names]);ax.invert_yaxis();ax.set_xlim(0,1)
    ax.set_xlabel('Macro-AUROC');bx.set_xlabel('Mean stop minus post-stop score')
    ax.set_title('(a) Stopping-turn ranking');bx.set_title('(b) Stop-to-post-stop difference');ax.legend()
    fig.suptitle('Reproduced RQ2 metric comparisons (39 uploaded workbooks)')
    save(fig,out,'Fig6_AUROC_and_stop_poststop')

def fig7(prox,out):
    p=prox[prox.metric.isin(METRICS[:3])].copy()
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharey=True)
    for ax,setting in zip(axes,['strict','majority']):
        data=[p[(p.setting==setting)&(p.metric==m)].distance_rows.dropna().values for m in METRICS[:3]]
        ax.boxplot(data,tick_labels=[LABELS[m] for m in METRICS[:3]],showfliers=True)
        ax.set_title(setting.capitalize()+' agreement');ax.set_ylabel('Peak distance from stopping region (rows)')
        ax.tick_params(axis='x',rotation=20)
    fig.suptitle('Stopping-point proximity: earliest metric maximum')
    save(fig,out,'Fig7_stopping_point_proximity')

def load_traces(audit,datadir,wmsc_csv=None):
    extra=None
    if wmsc_csv:
        extra=pd.read_csv(wmsc_csv)
        if not {'file','turn_number','WMSC'}.issubset(extra.columns):raise ValueError('WMSC CSV requires file,turn_number,WMSC columns')
    traces={}
    for _,r in audit.iterrows():
        p=datadir/str(r.file)
        if not p.exists():continue
        d=pd.read_excel(p)
        if 'turn_number' not in d.columns:continue
        d=d.copy();d['turn_number']=pd.to_numeric(d.turn_number,errors='coerce')
        d=d.dropna(subset=['turn_number']).drop_duplicates('turn_number').sort_values('turn_number')
        if extra is not None:
            z=extra[extra.file.astype(str).map(Path).map(lambda p:p.name)==p.name][['turn_number','WMSC']]
            d=d.merge(z,on='turn_number',how='left')
        traces[str(r.conversation)]=d
    return traces

def fig8(audit,traces,out):
    # Quarter averages computed on eligible turn-level metric rows.
    rec=[]
    for _,r in audit.iterrows():
        d=traces.get(str(r.conversation))
        if d is None or len(d)<4:continue
        d=d.sort_values('turn_number');q=max(1,len(d)//4)
        for part,sub in [('Initial 25%',d.iloc[:q]),('Final 25%',d.iloc[-q:])]:
            for m in METRICS:
                if m in sub:
                    v=pd.to_numeric(sub[m],errors='coerce').mean()
                    if pd.notna(v):rec.append((r.conversation,'Consensus' if r.strict else 'Non-consensus',part,m,v))
    z=pd.DataFrame(rec,columns=['conversation','group','period','metric','value'])
    names=[m for m in METRICS if m in z.metric.unique()]
    fig,axes=plt.subplots(2,3,figsize=(13,7.2));axes=axes.ravel()
    for ax,m in zip(axes,names):
        for group in ['Consensus','Non-consensus']:
            s=z[(z.metric==m)&(z.group==group)].groupby('period')['value'].mean()
            ax.plot([0,1],[s.get('Initial 25%',np.nan),s.get('Final 25%',np.nan)],marker='o',label=group)
        ax.set_title(LABELS[m]);ax.set_xticks([0,1],['Initial 25%','Final 25%']);ax.grid(alpha=.2)
    for ax in axes[len(names):]:ax.axis('off')
    axes[0].legend(fontsize=8);fig.suptitle('Early-to-late metric changes by strict-consensus status')
    save(fig,out,'Fig8_initial_vs_final_quarter')

def fig9(audit,traces,out):
    # Select representative conversations based on filename/task identifiers.
    queries=['Linked','Fibonacci','Caesar','Palindrome']
    selected=[]
    for q in queries:
        candidates=audit[audit.conversation.astype(str).str.contains(q,case=False,regex=False)]
        candidates=candidates[candidates.conversation.astype(str).isin(traces)]
        if not candidates.empty:selected.append(candidates.iloc[0])
    if not selected:return
    fig,axes=plt.subplots(len(selected),2,figsize=(13,3.1*len(selected)),squeeze=False)
    for (i,r) in enumerate(selected):
        d=traces[str(r.conversation)];x=d.turn_number
        for j,subset in enumerate([METRICS[:3],METRICS[3:]]):
            ax=axes[i,j]
            for m in subset:
                if m in d:ax.plot(x,pd.to_numeric(d[m],errors='coerce'),label=LABELS[m],lw=1)
            stops=[int(re.match(r'\d+',x).group()) for x in re.findall(r'\d+\s*[-–]\s*[SFC]',str(r.votes))]
            for t in stops:ax.axvline(t,ls='--',color='0.4',alpha=.6)
            ax.set_ylim(-.05,1.05);ax.set_xlabel('Conversation row');ax.set_ylabel('Similarity')
            ax.set_title(str(r.conversation)[:46]+' | '+('Task grounding' if j==0 else 'Repetition'),fontsize=9)
            ax.legend(fontsize=7,loc='best')
    fig.suptitle('Representative conversation trajectories; dashed lines: stopping references')
    save(fig,out,'Fig9_conversation_trajectories')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data-dir',default='/mnt/data');parser.add_argument('--out-dir',default='/mnt/data/USC_figures');parser.add_argument('--wmsc-csv',default=None)
    args=parser.parse_args();datadir=Path(args.data_dir);out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    a=annotations(datadir/'human_stoppingpointAnnotation.xlsx')
    au=pd.read_csv(datadir/'USC_39_AUROC_per_conversation.csv')
    diffs=pd.read_csv(datadir/'stop_poststop_differences.csv')
    prox=pd.read_csv(datadir/'stopping_point_proximity.csv')
    audit=pd.read_excel(datadir/'USC_39_conversation_AUROC_reproduction.xlsx',sheet_name='Matching Audit')
    fig4(a,out);fig5(out);fig6(au,diffs,out);fig7(prox,out)
    traces=load_traces(audit,datadir,args.wmsc_csv)
    fig8(audit,traces,out);fig9(audit,traces,out)
    print('Created figures in',out,'from',len(traces),'matched metric workbooks')
    if not args.wmsc_csv:print('WMSC omitted: provide --wmsc-csv file turn_number WMSC to add it.')

if __name__=='__main__':main()
