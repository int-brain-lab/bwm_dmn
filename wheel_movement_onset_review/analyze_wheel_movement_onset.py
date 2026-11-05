#!/usr/bin/env python3
"""Quantify BWM movement onset and sub-threshold wheel motion.

Outputs a four-panel reviewer/SI figure and trial/session tables.  The code uses
native ALF wheel samples; it does not redetect movement onset.
"""
import os
from pathlib import Path
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ONE cache with the BWM trials and wheel objects ($ONE_CACHE_DIR, default ~/Downloads/ONE).
ROOT = Path(os.environ.get('ONE_CACHE_DIR', Path.home() / 'Downloads' / 'ONE')) / 'alyx.internationalbrainlab.org'
OUT = Path(__file__).resolve().parent / 'results'
REV = '#2025-03-03#'
DT, GUARD, PETH_END = .005, .020, .150
RNG = np.random.default_rng(20260922)


def window_stats(t, p, start, end):
    """Total path, signed displacement, excess path, and valid-window flag."""
    a = np.searchsorted(t, start, side='left')
    b = np.searchsorted(t, end, side='right') - 1
    ok = (a >= 0) & (b < len(t)) & (b > a)
    cs = np.r_[0., np.cumsum(np.abs(np.diff(p)))]
    total = np.full(len(start), np.nan); signed = total.copy()
    total[ok] = cs[b[ok]] - cs[a[ok]]
    signed[ok] = p[b[ok]] - p[a[ok]]
    return total, signed, np.maximum(total - np.abs(signed), 0), ok


def session_id(path, root):
    x = path.relative_to(root).parts
    return x[0], x[2], x[3], x[4], f'{x[2]}_{x[3]}_{x[4]}'


def analyze(path, root, n_sample):
    lab, subject, date, number, sid = session_id(path, root)
    alf = path.parent.parent
    ft, fp = alf/'_ibl_wheel.timestamps.npy', alf/'_ibl_wheel.position.npy'
    if not ft.exists() or not fp.exists(): return None
    cols = ['stimOn_times','goCue_times','firstMovement_times','response_times',
            'choice','feedbackType']
    d = pd.read_parquet(path, columns=cols)
    t, p = np.load(ft).squeeze(), np.load(fp).squeeze()
    if t.ndim != 1 or p.ndim != 1 or len(t) != len(p): return None
    dz = np.abs(np.diff(p)); dz = dz[dz > np.finfo(float).eps]
    if not len(dz): return None
    step = float(np.min(dz))
    stim=d.stimOn_times.to_numpy(float); go=d.goCue_times.to_numpy(float)
    move=d.firstMovement_times.to_numpy(float); choice=d.choice.to_numpy(float)
    pre_total, pre_signed, pre_excess, pre_ok = window_stats(t,p,go,move-GUARD)
    sw_total, sw_signed, sw_excess, sw_ok = window_stats(t,p,stim,stim+PETH_END)
    onset_signed = (np.interp(move+.050,t,p,left=np.nan,right=np.nan)-
                    np.interp(move,t,p,left=np.nan,right=np.nan))
    valid=np.isfinite(stim)&np.isfinite(go)&np.isfinite(move)&np.isfinite(onset_signed)&pre_ok
    any_pre=(pre_total >= .5*step)&valid
    pre_dir=any_pre&(np.abs(pre_signed)>=.5*step)&np.isin(choice,[-1,1])
    pre_match_choice=pre_dir&(np.sign(pre_signed)==-np.sign(choice))
    # IBL convention: choice=-1 corresponds to positive wheel displacement.
    first_dir=np.isin(choice,[-1,1])&(np.sign(onset_signed)!=0)&valid
    first_matches_choice=first_dir&(np.sign(onset_signed)==-np.sign(choice))
    late=valid&sw_ok&(move > stim+PETH_END+GUARD)
    sw_change=late&(sw_total>=.5*step)
    out=d.copy()
    for k,v in {'lab':lab,'subject':subject,'session':sid,'date':date,
                'session_number':number}.items(): out[k]=v
    out['trial']=np.arange(len(d)); out['encoder_step_rad']=step
    out['movement_latency_s']=move-stim; out['valid_pre_onset']=valid
    out['pre_onset_total_path_rad']=pre_total
    out['pre_onset_signed_displacement_rad']=pre_signed
    out['pre_onset_excess_path_rad']=pre_excess
    out['pre_onset_any_encoder_change']=any_pre
    out['pre_onset_direction_defined']=pre_dir
    out['pre_onset_net_direction_matches_choice']=pre_match_choice
    out['onset_movement_direction_defined']=first_dir
    out['onset_movement_matches_choice']=first_matches_choice
    out['onset_after_stimulus_window']=late
    out['stimulus_window_total_path_rad']=sw_total
    out['stimulus_window_excess_path_rad']=sw_excess
    out['stimulus_window_any_encoder_change']=sw_change
    eligible=np.flatnonzero(late)
    if not len(eligible): return out, None
    chosen=RNG.choice(eligible,min(n_sample,len(eligible)),replace=False)
    onset_t=np.arange(-.300,.151,DT); stim_t=np.arange(0,PETH_END+DT/2,DT)
    onset_pos=np.vstack([np.interp(move[i]+onset_t,t,p) for i in chosen])
    onset_pos-=onset_pos[:,[0]]
    direction=np.sign(onset_signed[chosen]); direction[direction==0]=1
    onset_pos*=direction[:,None]
    stim_pos=np.vstack([np.interp(stim[i]+stim_t,t,p) for i in chosen])
    stim_pos-=stim_pos[:,[0]]
    sample=dict(session=np.repeat(sid,len(chosen)),subject=np.repeat(subject,len(chosen)),
                trial=chosen,onset_t=onset_t,
                stim_t=stim_t,onset_pos=onset_pos.astype('float32'),
                stim_speed=(np.abs(np.diff(stim_pos,axis=1))/DT).astype('float32'),
                stim_path=sw_total[chosen].astype('float32'))
    return out,sample


def summarize(trials):
    rows=[]
    for (lab,subject,sid),d in trials.groupby(['lab','subject','session'],sort=False):
        v=d.valid_pre_onset; pdx=v&d.pre_onset_direction_defined
        fdx=v&d.onset_movement_direction_defined; late=d.onset_after_stimulus_window
        stimulus_steps = d.stimulus_window_total_path_rad / d.encoder_step_rad
        rows.append(dict(lab=lab,subject=subject,session=sid,n_trials=int(v.sum()),
          onset_before_stimulus=(d.loc[v,'movement_latency_s']<0).mean(),
          any_pre_onset_change=d.loc[v,'pre_onset_any_encoder_change'].mean(),
          pre_direction_defined=pdx.mean(),
          pre_net_matches_choice=d.loc[pdx,'pre_onset_net_direction_matches_choice'].mean() if pdx.any() else np.nan,
          onset_movement_matches_choice=d.loc[fdx,'onset_movement_matches_choice'].mean() if fdx.any() else np.nan,
          n_late=int(late.sum()),
          stimulus_window_change=d.loc[late,'stimulus_window_any_encoder_change'].mean() if late.any() else np.nan,
          stimulus_window_path_ge_1_5_steps=(stimulus_steps[late]>=1.5).mean() if late.any() else np.nan,
          stimulus_window_path_ge_8_steps=(stimulus_steps[late]>=8).mean() if late.any() else np.nan,
          stimulus_window_path_rad=d.loc[late,'stimulus_window_total_path_rad'].median() if late.any() else np.nan))
    return pd.DataFrame(rows)


def dot_summary(ax, values, title, ylabel, chance=True, color='#4c72b0'):
    values=values.dropna()*100; x=RNG.normal(0,.045,len(values))
    ax.scatter(x,values,s=6,color=color,alpha=.4,edgecolor='none')
    q1,md,q3=values.quantile([.25,.5,.75]); ax.plot([-.18,.18],[md,md],color='k',lw=1.4)
    ax.plot([0,0],[q1,q3],color='k',lw=1)
    if chance: ax.axhline(50,color='.4',ls='--',lw=.7)
    ax.set(xlim=(-.4,.4),ylim=(0,100),xticks=[],ylabel=ylabel,title=title)
    ax.text(.5,.04,f'session median {md:.1f}%\nIQR {q1:.1f}–{q3:.1f}%',transform=ax.transAxes,
            ha='center',va='bottom',fontsize=5.7)


def figure(trials,sessions,samples,out):
    mpl.rcParams.update({'font.family':'sans-serif',
                         'font.sans-serif':['Arial','Helvetica','Liberation Sans','DejaVu Sans'],
                         'font.size':6,'axes.labelsize':6.5,'axes.titlesize':7,
                         'xtick.labelsize':5.5,'ytick.labelsize':5.5,
                         'legend.fontsize':5,'pdf.fonttype':42,'ps.fonttype':42,
                         'axes.linewidth':.5,'lines.linewidth':.75,
                         'xtick.major.width':.5,'ytick.major.width':.5,
                         'xtick.major.size':2,'ytick.major.size':2,
                         'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(1,2,figsize=(89/25.4,46/25.4),constrained_layout=True,
                         gridspec_kw={'width_ratios':[1.25,1.05]})
    a,b=axs
    pos=np.concatenate([s['onset_pos'] for s in samples]); paths=np.concatenate([s['stim_path'] for s in samples])
    subjects=np.concatenate([s['subject'] for s in samples]); ot=samples[0]['onset_t']
    subject_names, subject_counts=np.unique(subjects,return_counts=True)
    selected_subjects=subject_names[np.argsort(subject_counts)[-3:]]
    take=[]
    for subject in selected_subjects:
        ii=np.flatnonzero(subjects==subject); order=ii[np.argsort(paths[ii])]
        take.extend(order[np.linspace(0,len(order)-1,7).astype(int)])
    for j,i in enumerate(take):
        a.plot(ot*1000,pos[i],lw=.5,alpha=.48,color='black',
               label='Individual trials (3 mice)' if j == 0 else None)
    a.axvline(0,color='k',ls='--',lw=.6,label='Reported onset')
    a.axvspan(-20,0,color='.85',lw=0,label='Excluded interval')
    visible=(ot>=-.250)&(ot<=.070)
    ymin=np.nanmin(pos[np.asarray(take)][:,visible]); ymax=np.nanmax(pos[np.asarray(take)][:,visible])
    ypad=.06*(ymax-ymin)
    a.set(xlim=(-250,70),ylim=(ymin-ypad,ymax+ypad),xlabel='Time from reported onset (ms)',
          ylabel='Wheel position\n(rad, eventual direction normalized)')
    a.legend(frameon=False,fontsize=5,loc='upper left',handlelength=1.5,
             labelspacing=.25,borderaxespad=.25)
    direction_series=[sessions.pre_net_matches_choice.dropna()*100,
                      sessions.onset_movement_matches_choice.dropna()*100]
    direction_colors=['#4c72b0','#55a868']
    direction_labels=['Pre-onset net → final choice','Onset movement → final choice']
    for x0,(values,color,label) in enumerate(zip(direction_series,direction_colors,direction_labels)):
        jitter=RNG.normal(x0,.045,len(values))
        b.scatter(jitter,values,s=4,color=color,alpha=.4,edgecolor='none')
        q1,md,q3=values.quantile([.25,.5,.75])
        b.plot([x0-.14,x0+.14],[md,md],color='k',lw=1)
        b.plot([x0,x0],[q1,q3],color='k',lw=.75)
        b.text(x0,5,f'{md:.1f}%\n[{q1:.1f}–{q3:.1f}]',ha='center',va='bottom',fontsize=5)
    b.axhline(50,color='.4',ls='--',lw=.5)
    b.set(xlim=(-.35,1.35),ylim=(0,100),xticks=[0,1],
          xticklabels=['Pre-onset net →\nfinal choice','Onset movement →\nfinal choice'],
          ylabel='Direction match (%)')
    for label,ax in zip('ab',axs):
        ax.spines[['top','right']].set_visible(False); ax.tick_params(length=2,width=.5)
        if ax is not b and len(ax.get_xticks()):
            ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2))
        ax.text(-.18,1.06,label,transform=ax.transAxes,fontweight='bold',fontsize=8,va='top')
    for ext in ['pdf','png']: fig.savefig(out/f'movement_onset_subthreshold_wheel.{ext}',dpi=300)
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--one-root',type=Path,default=ROOT)
    ap.add_argument('--out',type=Path,default=OUT); ap.add_argument('--sample-per-session',type=int,default=60)
    args=ap.parse_args(); args.out.mkdir(parents=True,exist_ok=True)
    tables=sorted(args.one_root.glob(f'*/Subjects/*/*/*/alf/{REV}/_ibl_trials.table.pqt'))
    frames=[]; samples=[]
    for i,path in enumerate(tables,1):
        z=analyze(path,args.one_root,args.sample_per_session)
        if z is not None:
            frames.append(z[0]); samples.extend([z[1]] if z[1] is not None else [])
        if i%50==0: print(f'processed {i}/{len(tables)}',flush=True)
    trials=pd.concat(frames,ignore_index=True); sessions=summarize(trials)
    trials.to_parquet(args.out/'trial_metrics.pqt',index=False); sessions.to_csv(args.out/'session_metrics.csv',index=False)
    np.savez_compressed(args.out/'figure_sample.npz',onset_time_s=samples[0]['onset_t'],stimulus_time_s=samples[0]['stim_t'],
      onset_position_rad=np.concatenate([s['onset_pos'] for s in samples]),stimulus_abs_speed_rad_s=np.concatenate([s['stim_speed'] for s in samples]),
      session=np.concatenate([s['session'] for s in samples]),subject=np.concatenate([s['subject'] for s in samples]),
      trial=np.concatenate([s['trial'] for s in samples]))
    figure(trials,sessions,samples,args.out)
    v=trials.valid_pre_onset; pdx=v&trials.pre_onset_direction_defined; fdx=v&trials.onset_movement_direction_defined; late=trials.onset_after_stimulus_window
    text=f'''MOVEMENT ONSET DEFINITION
Candidate movement: >8 encoder-sample displacement within 200 ms.
Onset refinement: first 1.5-sample displacement.
firstMovement_times: first candidate with peak amplitude >=0.1 rad, searched from go cue minus quiescence period to feedback.
Direction/choice is not part of detection.

Tables found: {len(tables)}
Sessions with wheel data: {trials.session.nunique()}
Subjects: {trials.subject.nunique()}
Valid pre-onset trials: {v.sum()}
Reported onset before stimulus: {(trials.loc[v,'movement_latency_s']<0).mean():.3%}
Any encoder change between go cue and onset-20 ms: {trials.loc[v,'pre_onset_any_encoder_change'].mean():.3%}
Pre-onset net direction matches recorded final choice: {trials.loc[pdx,'pre_onset_net_direction_matches_choice'].mean():.3%} (n={pdx.sum()})
Movement direction in first 50 ms after onset matches recorded final choice: {trials.loc[fdx,'onset_movement_matches_choice'].mean():.3%} (n={fdx.sum()})
Trials with onset >170 ms: {late.sum()}
Any encoder change in 0-150 ms stimulus window: {trials.loc[late,'stimulus_window_any_encoder_change'].mean():.3%}
Cumulative stimulus-window path >=1.5 encoder steps: {((trials.loc[late,'stimulus_window_total_path_rad']/trials.loc[late,'encoder_step_rad'])>=1.5).mean():.3%}
Cumulative stimulus-window path >=8 encoder steps: {((trials.loc[late,'stimulus_window_total_path_rad']/trials.loc[late,'encoder_step_rad'])>=8).mean():.3%}
Session median for stimulus-window change: {sessions.stimulus_window_change.median():.3%}
'''
    (args.out/'summary.txt').write_text(text); print(text)

if __name__=='__main__': main()
