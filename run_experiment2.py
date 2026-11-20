"""Experiment 2: 5-fold video-level CV, from original Kaggle code."""
import argparse
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader
from src.dataset import TorchSurgicalAnnotationDataset
from src.folds import get_folds
from src.mask_to_phase import run_experiment2

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',required=True,help='paths_and_annotations.csv')
    p.add_argument('--root',required=True,help='dataset root')
    p.add_argument('--output',default='cv_results/experiment2')
    a=p.parse_args()
    df=pd.read_csv(a.csv)
    output=Path(a.output); output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for fold,tr,va,te,train_ids,val_id,test_id in get_folds(df):
        fd=output/f'fold_{fold}'; fd.mkdir(parents=True,exist_ok=True)
        datasets=[TorchSurgicalAnnotationDataset(x,root_dir=a.root,target_size=(256,256)) for x in (tr,va,te)]
        result=run_experiment2(fd,*datasets,seed=42+fold-1)
        rows.append({'fold':fold,'train_videos':';'.join(train_ids),'val_video':val_id,'test_video':test_id,'accuracy':result['test']['accuracy'],'loss':result['test']['loss']})
        pd.DataFrame(rows).to_csv(output/'fold_summary_partial.csv',index=False)
    summary=pd.DataFrame(rows); summary.to_csv(output/'fold_summary.csv',index=False)
    print('Mean accuracy:',summary.accuracy.mean(),'SD (sample):',summary.accuracy.std(ddof=1))
if __name__=='__main__':main()
