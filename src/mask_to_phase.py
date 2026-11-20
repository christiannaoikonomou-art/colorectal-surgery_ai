from __future__ import annotations
import gc, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
NUM_CLASSES=8
NUM_PHASES=8
MASK2PHASE_EPOCHS=8
LR_MASK2PHASE=1e-3
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
ce_phase=nn.CrossEntropyLoss()

def normalize_phase_labels(ph, num_phases=NUM_PHASES):
    return ph.long().view(-1).clamp(0,num_phases-1)

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def save_json(obj,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w') as f: json.dump(obj,f,indent=2)

# ============================================================
# EXPERIMENT 2: Can ground-truth anatomy predict surgical phase?
# ============================================================
class MaskToPhaseDataset(torch.utils.data.Dataset):
    def __init__(self, base): self.base=base
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        _, mask, phase = self.base[idx]
        return mask.float(), normalize_phase_labels(phase).squeeze()

class MaskToPhaseCNN(nn.Module):
    def __init__(self, in_channels=NUM_CLASSES, num_phases=NUM_PHASES):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_channels,32,3,padding=1),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1),nn.ReLU(inplace=True),nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1),nn.ReLU(inplace=True),nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),nn.Linear(128,num_phases))
    def forward(self,x): return self.net(x)

def _run_mask_phase_epoch(model, loader, optimizer=None):
    train=optimizer is not None; model.train(train)
    total=correct=count=n=0
    ctx=torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for masks,ph in loader:
            masks=masks.to(DEVICE); ph=ph.to(DEVICE)
            logits=model(masks); loss=ce_phase(logits,ph)
            if train:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            total += float(loss.item()); n+=1
            correct += int((logits.argmax(1)==ph).sum().item()); count += int(ph.numel())
    return {'loss':total/max(n,1),'accuracy':correct/max(count,1)}

def run_experiment2(fold_dir, train_dataset, valid_dataset, test_dataset, seed):
    set_seed(seed)
    tr=DataLoader(MaskToPhaseDataset(train_dataset),batch_size=16,shuffle=True,num_workers=2)
    va=DataLoader(MaskToPhaseDataset(valid_dataset),batch_size=16,shuffle=False,num_workers=2)
    te=DataLoader(MaskToPhaseDataset(test_dataset),batch_size=16,shuffle=False,num_workers=2)
    model=MaskToPhaseCNN().to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=LR_MASK2PHASE)
    best_val=float('inf'); best_epoch=None; best_path=fold_dir/'exp2_mask2phase_best.pth'; history=[]
    for ep in range(MASK2PHASE_EPOCHS):
        a=_run_mask_phase_epoch(model,tr,opt); b=_run_mask_phase_epoch(model,va,None)
        history.append({'epoch':ep+1,'train_loss':a['loss'],'train_accuracy':a['accuracy'],'val_loss':b['loss'],'val_accuracy':b['accuracy']})
        if b['loss']<best_val:
            best_val=b['loss']; best_epoch=ep+1; torch.save({'model_state_dict':model.state_dict()},best_path)
        print(f'Exp2 epoch {ep+1:02d}/{MASK2PHASE_EPOCHS} val_acc={b["accuracy"]:.4f} val_loss={b["loss"]:.4f}')
    model.load_state_dict(torch.load(best_path,map_location=DEVICE)['model_state_dict'])
    test=_run_mask_phase_epoch(model,te,None)
    result={'best_epoch':best_epoch,'best_valid_loss':best_val,'test':test,'history':history}
    save_json(result,fold_dir/'experiment2.json')
    del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return result
