"""Segmentation-only baseline extracted from the 5-fold notebooks.
The original notebook called this the Experiment 3 segmentation-only control.
Multitask and phase-conditioned experiments are intentionally excluded.
"""
import argparse, gc, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from src.dataset import TorchSurgicalAnnotationDataset
from src.folds import get_folds
DEVICE=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TARGET_SIZE=(256,256)
NUM_CLASSES=8
SEG_ONLY_EPOCHS=40
LR_MT=3e-4
loss_fn=smp.losses.TverskyLoss(mode='multilabel',alpha=0.8,beta=0.2,gamma=1)

def normalize_phase_labels(ph,num_phases=8):
    return ph.long().view(-1).clamp(0,num_phases-1)

def set_seed(seed=42):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def create_segmentation_model(
    architecture: str = 'DeepLabV3+',
    encoder_name: str = 'resnet34',
    encoder_weights: str = 'imagenet',
    in_channels: int = 3,
    classes: int = 8,  # background + 7 organs
    activation: str = None,  # None for logits, 'sigmoid' for probabilities
    aux_params: dict = None
) -> torch.nn.Module:
    """
    Create a segmentation model using segmentation_models.pytorch library.
    
    Args:
        architecture: Model architecture ('Unet', 'UnetPlusPlus', 'FPN', 'DeepLabV3Plus', etc.)
        encoder_name: Encoder/backbone name (e.g., 'resnet34', 'efficientnet-b0', 'timm-efficientnet-b0')
        encoder_weights: Pretrained weights ('imagenet', 'ssl', 'swsl', or None)
        in_channels: Number of input channels (3 for RGB)
        classes: Number of output classes (8: background + 7 organs)
        activation: Activation function for output ('sigmoid' for multilabel, None for logits)
        aux_params: Optional auxiliary classification head parameters
    
    Returns:
        PyTorch model
    
    Examples:
        # Basic Unet with ResNet34 encoder
        model = create_segmentation_model('Unet', 'resnet34', 'imagenet', classes=8)
        
        # EfficientNet encoder for better accuracy
        model = create_segmentation_model('Unet', 'efficientnet-b0', 'imagenet', classes=8)
        
        # DeepLabV3+ with pretrained weights
        model = create_segmentation_model('DeepLabV3Plus', 'resnet50', 'imagenet', classes=8)
        
        # With auxiliary classification head
        aux_params = dict(
            pooling='avg',
            dropout=0.5,
            activation='sigmoid',
            classes=1,  # Binary classification
        )
        model = create_segmentation_model('Unet', 'resnet34', 'imagenet', classes=8, aux_params=aux_params)
    """
    # Get the model class from smp
    model_class = getattr(smp, architecture, None)
    if model_class is None:
        raise ValueError(f"Architecture '{architecture}' not found. Available: {dir(smp)}")
    
    # Create model with pretrained encoder
    model = model_class(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
        activation=activation,
        aux_params=aux_params
    )
    
    return model



def _stats_from_logits(logits, masks, threshold=0.5):
    tp, fp, fn, tn = smp.metrics.functional.get_stats(
        output=logits, target=masks, mode='multilabel', threshold=threshold
    )
    # get_stats returns batch x classes; aggregate over images only.
    if tp.dim() > 1:
        tp, fp, fn, tn = tp.sum(0), fp.sum(0), fn.sum(0), tn.sum(0)
    return tp, fp, fn, tn


def metrics_from_stats(tp, fp, fn, tn):
    tp = tp.float(); fp = fp.float(); fn = fn.float(); tn = tn.float()
    denom_iou = tp + fp + fn
    denom_f1 = 2*tp + fp + fn
    class_iou = torch.where(denom_iou > 0, tp / denom_iou, torch.nan)
    class_f1 = torch.where(denom_f1 > 0, 2*tp / denom_f1, torch.nan)
    gt_pixels = tp + fn

    # Global/micro scores pool all pixels/classes.
    micro_iou = tp.sum() / torch.clamp((tp + fp + fn).sum(), min=1)
    micro_f1 = 2*tp.sum() / torch.clamp((2*tp + fp + fn).sum(), min=1)

    # Macro score: mean only over classes actually present in the GT of that split.
    present = gt_pixels > 0
    macro_iou_present = torch.nanmean(class_iou[present]) if present.any() else torch.tensor(float('nan'))
    macro_f1_present = torch.nanmean(class_f1[present]) if present.any() else torch.tensor(float('nan'))

    return {
        'micro_iou': float(micro_iou.item()),
        'micro_f1': float(micro_f1.item()),
        'macro_iou_present': float(macro_iou_present.item()),
        'macro_f1_present': float(macro_f1_present.item()),
        'class_iou': [None if torch.isnan(x) else float(x.item()) for x in class_iou],
        'class_f1': [None if torch.isnan(x) else float(x.item()) for x in class_f1],
        'gt_pixels': [int(x.item()) for x in gt_pixels],
        'tp': [int(x.item()) for x in tp], 'fp': [int(x.item()) for x in fp],
        'fn': [int(x.item()) for x in fn], 'tn': [int(x.item()) for x in tn],
    }


def forward_seg(model, images, phases=None):
    if phases is None:
        out = model(images)
    else:
        out = model(images, phases)
    return out[0] if isinstance(out, tuple) else out


def evaluate_segmentation(model, loader, phase_mode='gt'):
    """phase_mode: None for segmentation-only; 'gt', 'random', or 'constant' for phase-conditioned models."""
    model.eval()
    total_loss, n = 0.0, 0
    agg = None
    with torch.no_grad():
        for images, masks, phases in loader:
            images = images.float().to(DEVICE)
            masks = masks.to(DEVICE)
            phases = normalize_phase_labels(phases.to(DEVICE))
            if phase_mode is None:
                logits = forward_seg(model, images, None)
            else:
                if phase_mode == 'gt': use_phase = phases
                elif phase_mode == 'random': use_phase = torch.randint(0, NUM_PHASES, phases.shape, device=DEVICE)
                elif phase_mode == 'constant': use_phase = torch.zeros_like(phases)
                else: raise ValueError(phase_mode)
                logits = forward_seg(model, images, use_phase)
            total_loss += float(loss_fn(logits, masks).item()); n += 1
            st = _stats_from_logits(logits, masks)
            if agg is None: agg = [x.clone() for x in st]
            else:
                for j in range(4): agg[j] += st[j]
    m = metrics_from_stats(*agg)
    m['loss'] = total_loss / max(n, 1)
    return m


def train_seg_epoch(model, loader, optimizer, phase_conditioned=False):
    model.train(); total=0.0; n=0
    for images, masks, phases in loader:
        images = images.float().to(DEVICE); masks = masks.to(DEVICE)
        phases = normalize_phase_labels(phases.to(DEVICE))
        logits = forward_seg(model, images, phases if phase_conditioned else None)
        loss = loss_fn(logits, masks)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        total += float(loss.item()); n += 1
    return total/max(n,1)


def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f: json.dump(obj, f, indent=2)


def run_fold(arch,fold_dir,train_loader,valid_loader,test_loader,seed):
    set_seed(seed)
    model=create_segmentation_model(architecture=arch,encoder_name='resnet34',encoder_weights='imagenet',in_channels=3,classes=NUM_CLASSES).to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=LR_MT)
    best=float('inf');best_epoch=None;path=fold_dir/'segmentation_only_best.pth';history=[]
    for ep in range(SEG_ONLY_EPOCHS):
        tr=train_seg_epoch(model,train_loader,opt,phase_conditioned=False)
        va=evaluate_segmentation(model,valid_loader,phase_mode=None)
        history.append({'epoch':ep+1,'train_loss':tr,**{f'val_{k}':v for k,v in va.items() if not isinstance(v,list)}})
        if va['loss']<best:
            best=va['loss'];best_epoch=ep+1
            torch.save({'model_state_dict':model.state_dict()},path)
        print(f'{arch} fold {fold_dir.name} epoch {ep+1}: val_microIoU={va["micro_iou"]:.4f}')
    model.load_state_dict(torch.load(path,map_location=DEVICE,weights_only=True)['model_state_dict'])
    test=evaluate_segmentation(model,test_loader,phase_mode=None)
    result={'best_epoch':best_epoch,'best_valid_loss':best,'test':test,'history':history}
    save_json(result,fold_dir/'segmentation_only.json')
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--architecture',required=True,choices=['DeepLabV3Plus','Segformer'])
    p.add_argument('--csv',required=True);p.add_argument('--root',required=True)
    p.add_argument('--output',default='cv_results/segmentation')
    a=p.parse_args()
    df=pd.read_csv(a.csv);out=Path(a.output)/a.architecture;out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for fold,tr,va,te,train_ids,val_id,test_id in get_folds(df):
        fd=out/f'fold_{fold}';fd.mkdir(parents=True,exist_ok=True)
        loaders=[DataLoader(TorchSurgicalAnnotationDataset(x,root_dir=a.root,target_size=TARGET_SIZE),batch_size=32 if i==0 else 1,shuffle=(i==0),num_workers=2,pin_memory=True) for i,x in enumerate((tr,va,te))]
        result=run_fold(a.architecture,fd,*loaders,seed=42+fold-1)
        rows.append({'fold':fold,'test_video':test_id,'val_video':val_id,'micro_iou':result['test']['micro_iou']})
        pd.DataFrame(rows).to_csv(out/'fold_summary_partial.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'fold_summary.csv',index=False)
if __name__=='__main__':main()
