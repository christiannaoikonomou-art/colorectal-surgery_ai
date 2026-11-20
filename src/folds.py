from pathlib import Path
import pandas as pd

def extract_video_id(rel_path):
    parts = [p for p in str(rel_path).replace('\\', '/').split('/') if p not in ('', '.')]
    if len(parts) < 2: raise ValueError(f'Cannot infer video_id: {rel_path!r}')
    stop_dirs = {'defaultannot', 'groundtruth', 'images', 'image', 'frames', 'frame'}
    for i, part in enumerate(parts[:-1]):
        if part.lower() in stop_dirs or part.lower().startswith('phase'):
            if i > 0: return '/'.join(parts[:i])
    return '/'.join(parts[:-1])

def get_folds(df):
    df=df.copy()
    df['video_id']=df['path'].apply(extract_video_id)
    video_ids=sorted(df.video_id.unique())
    if len(video_ids)!=5: raise ValueError(f'Expected exactly 5 videos, got {video_ids}')
    for fold in range(5):
        test=video_ids[fold]
        valid=video_ids[(fold+1)%5]
        train=[v for v in video_ids if v not in {test,valid}]
        tr=df[df.video_id.isin(train)].reset_index(drop=True)
        va=df[df.video_id==valid].reset_index(drop=True)
        te=df[df.video_id==test].reset_index(drop=True)
        assert len(tr)+len(va)+len(te)==len(df)
        yield fold+1,tr,va,te,train,valid,test
