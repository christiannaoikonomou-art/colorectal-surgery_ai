# Colorectal Surgery AI — baseline research code

This repository contains Python code for baseline deep learning experiments in laparoscopic colorectal surgery, including anatomical structure segmentation and surgical phase classification. The implementations use DeepLabV3+, SegFormer, and a convolutional neural network for anatomy-based surgical phase recognition. The repository is intended to support transparency, reproducibility, and further research in surgical artificial intelligence.

## Contents
- `run_segmentation.py`: segmentation-only baselines, DeepLabV3+ and SegFormer, using five video-level folds; training/validation/test split of 3/1/1 videos per fold.
- `run_experiment2.py`: anatomical ground-truth masks to surgical-phase classification using a CNN.
- `src/dataset.py`, `src/folds.py`, `src/mask_to_phase.py`: dataset reader, fold definitions, and phase-classification model.
- `SEGFORMER_AUDIT.md`: limits of the source-notebook execution audit.

## Reproduction prerequisites
Python environment with the dependencies in `requirements.txt`; access to the original dataset and `paths_and_annotations.csv`. The CSV, image data, video data, model weights and original experimental outputs are **not included** in this repository. Check data access and patient permissions separately.

```bash
pip install -r requirements.txt
python run_segmentation.py --architecture DeepLabV3Plus --csv /path/to/paths_and_annotations.csv --root /path/to/dataset
python run_segmentation.py --architecture Segformer --csv /path/to/paths_and_annotations.csv --root /path/to/dataset
python run_experiment2.py --csv /path/to/paths_and_annotations.csv --root /path/to/dataset
```

## Important implementation notes
- The original segmentation evaluation passes **raw logits** with `threshold=0.5` to the multilabel metrics implementation. This is equivalent to sigmoid probability ≈0.622, **not** the conventional probability threshold of 0.5. It has been preserved rather than silently altered.
- The mask-to-phase implementation defines **8 output classes**. The manuscript's seven-class chance-level comparison must be reconciled with the actual included labels and class distribution.
- The historical manuscript results must not be attributed to a fresh execution of these packaged scripts.The experimental code has undergone functional validation using the original Kaggle environment and dataset, including module imports, dataset loading, model initialization, forward passes, and training-step execution. Additionally, evaluation of a saved DeepLabV3+ checkpoint reproduced the originally reported micro-IoU for one held-out fold. Full end-to-end reproduction of all experiments using the assembled repository has not yet been independently verified.


## Provenance and release limitations
The code in this repository was assembled from the original experimental implementations and organized to facilitate reproducibility and further development. The repository focuses on baseline anatomical segmentation and surgical phase classification experiments. Additional exploratory experiments, including those involving SAM-based segmentation, are not included in this release.

**Before public release:** verify execution on Kaggle, record full package versions and dataset version, reconcile manuscript frame counts and metrics, and add a license only after the authors select one.
