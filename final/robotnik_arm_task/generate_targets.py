"""Step 1: create the original 500-target LHS plan and plots exactly once."""
import csv
import os
import sys
import tempfile
sys.dont_write_bytecode = True
from pipeline_common import (RESULTS, INPUTS, OUTPUTS, HEADER, UNITS, NOISE_KEYS,
    make_targets, stratum_of, choose_splits, json_new, sha, source_hashes, version_info, print_schema)
from result_paths import DATASETS, FIGURES, METADATA


def plot_targets(targets, groups):
    import numpy as np
    with tempfile.TemporaryDirectory(prefix='comp0245-plots-') as cache:
        previous = os.environ.get('MPLCONFIGDIR'); os.environ['MPLCONFIGDIR'] = cache
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            directory = RESULTS/FIGURES/'dataset';directory.mkdir(parents=True,exist_ok=True)
            xyz = np.asarray([r[1:] for r in targets]); radii = np.hypot(xyz[:,0],xyz[:,1])
            assigned = {tid:name for name,ids in groups.items() for tid in ids}
            colors = {'train':'#9ab8f3','validation':'#f5a126','test':'#38ad67'}
            def save(fig, name):
                fig.tight_layout()
                with (directory/name).open('xb') as f: fig.savefig(f,format='png',dpi=190,bbox_inches='tight')
                plt.close(fig)
            fig,ax = plt.subplots(figsize=(11,10))
            for name in colors:
                selected = np.array([assigned[r[0]]==name for r in targets])
                ax.scatter(xyz[selected,0],xyz[selected,1],s=22,alpha=.86,c=colors[name],
                           edgecolors='none',label=f'{name.title()} ({selected.sum()})')
            for radius in (.6,1.,1.4,1.8):
                ax.add_patch(plt.Circle((0,0),radius,fill=False,color='#b7c2d2',ls='--',lw=1.05))
            ax.axhline(0,color='#9da6b0',lw=.75);ax.axvline(0,color='#9da6b0',lw=.75)
            ax.scatter([0],[0],color='#191919',s=120,marker='+',lw=2,label='World origin')
            ax.set(aspect='equal',xlim=(-2.02,2.02),ylim=(-2.02,2.02),xlabel='Target X (m)',
                   ylabel='Target Y (m)',title='500 paired targets: shared train/validation/test split')
            ax.grid(alpha=.17);ax.legend(loc='upper right')
            save(fig,'target_distribution_xy_split.png')
            fig,ax = plt.subplots(figsize=(13,7))
            for name in colors:
                selected = np.array([assigned[r[0]]==name for r in targets])
                ax.scatter(radii[selected],xyz[selected,2],s=27,alpha=.82,c=colors[name],
                           edgecolors='none',label=f'{name.title()} ({selected.sum()})')
            for radius in (1.,1.4):ax.axvline(radius,color='#6d7e95',ls='--',lw=1.4)
            for height in (.5,.8):ax.axhline(height,color='#6d7e95',ls='--',lw=1.4)
            ax.set(xlim=(.55,1.85),ylim=(.27,1.13),xlabel='Horizontal target distance (m)',
                   ylabel='Target Z (m)',title='Distance x height: 3 x 3 target bands')
            ax.legend();ax.grid(alpha=.18);save(fig,'target_distribution_distance_height.png')
            strata = [(q,d,h) for q in range(4) for d in ('near','medium','far') for h in ('low','medium','high')]
            index = {key:i for i,key in enumerate(strata)};names = list(colors)
            counts = np.zeros((36,3),dtype=int)
            for tid,x,y,z in targets:counts[index[stratum_of(x,y,z)],names.index(assigned[tid])] += 1
            assert counts.sum()==500 and list(counts.sum(axis=0))==[350,75,75]
            ratio = counts/np.array([350/36,75/36,75/36])[None,:]
            fig,ax = plt.subplots(figsize=(9.2,13.5))
            im = ax.imshow(ratio,aspect='auto',cmap='YlGnBu',vmin=.78,vmax=1.55)
            for i in range(36):
                for j in range(3):
                    ax.text(j,i,str(counts[i,j]),ha='center',va='center',fontsize=9.3,
                            color='white' if ratio[i,j]>1.27 else '#102a43')
            ax.set_xticks(range(3),['Train (350)','Validation (75)','Test (75)'])
            ax.set_yticks(range(36),[f'Q{q+1} / {d:6s} / {h}' for q,d,h in strata],fontsize=9)
            for boundary in (8.5,17.5,26.5):ax.axhline(boundary,color='white',lw=2)
            ax.set_title('500 paired targets: category distribution by split')
            fig.colorbar(im,ax=ax,fraction=.035,pad=.025).set_label('Relative to uniform-stratum expectation')
            save(fig,'target_distribution_strata.png')
            fig,axes = plt.subplots(1,3,figsize=(12,3.5))
            for i,ax in enumerate(axes):ax.hist(xyz[:,i],bins=20);ax.set(xlabel='XYZ'[i]+' (m)',ylabel='Targets')
            save(fig,'target_distribution_xyz.png')
        finally:
            if previous is None:os.environ.pop('MPLCONFIGDIR',None)
            else:os.environ['MPLCONFIGDIR'] = previous


def main():
    paths = [RESULTS/DATASETS/'targets_500.csv', RESULTS/METADATA/'pipeline_manifest.json',
             RESULTS/METADATA/'pipeline_state.sqlite']
    figures = ['xy_split','distance_height','strata','xyz']
    if any(path.exists() for path in paths) or any(
            (RESULTS/FIGURES/'dataset'/f'target_distribution_{name}.png').exists() for name in figures):
        raise FileExistsError('Target plan or outputs already exist. They are preserved; do not regenerate on resume.')
    print_schema()  # exact feature definitions printed before generation/collection
    targets = make_targets()
    groups = choose_splits([(tid,*stratum_of(x,y,z)) for tid,x,y,z in targets])
    (RESULTS/DATASETS).mkdir(parents=True,exist_ok=True)
    (RESULTS/METADATA).mkdir(parents=True,exist_ok=True)
    path = RESULTS/DATASETS/'targets_500.csv'
    with path.open('x',newline='',encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['target_id','target_x','target_y','target_z','quadrant','distance_band','height_band'])
        writer.writerows((tid,f'{x:.12f}',f'{y:.12f}',f'{z:.12f}',*stratum_of(x,y,z)) for tid,x,y,z in targets)
    json_new(RESULTS/METADATA/'pipeline_manifest.json',dict(version=2,target_seed=42,split_seed=314159,
        source_sha256=source_hashes(),plan_files={'targets_500.csv':sha(path)},
        splits={name:sorted(ids) for name,ids in groups.items()},environment=version_info(),
        schema=dict(inputs=INPUTS,input_units=UNITS,outputs=OUTPUTS,output_units=['rad/s']*4+['Nm']*7,csv_columns=HEADER),
        conditions={c:dict(noise_flag=int(c=='noisy'),robot_noise={k:.001 if c=='noisy' else .0001 for k in NOISE_KEYS})
                    for c in ('clean','noisy')},
        success='navigation done AND measured EE error <=0.01m AND time >8s; true FK error separate',
        feature_policy='original 1ms raw derivatives; no filtering or clipping; review distributions before accepting data'))
    plot_targets(targets,groups)
    print('Created paired target plan:',path)


if __name__ == '__main__':
    main()
