# Colorectal Surgery AI — baseline research code

This repository contains Python code for baseline deep learning experiments in laparoscopic colorectal surgery, including anatomical structure segmentation and surgical phase classification. The implementations use DeepLabV3+, SegFormer, and a convolutional neural network for anatomy-based surgical phase recognition. The repository is intended to support transparency, reproducibility, and further research in surgical artificial intelligence.

## Contents
- `run_segmentation.py`: segmentation-only baselines, DeepLabV3+ and SegFormer, using five video-level folds; training/validation/test split of 3/1/1 videos per fold.
- `run_experiment2.py`: anatomical ground-truth masks to surgical-phase classification using a CNN.
- `src/dataset.py`, `src/folds.py`, `src/mask_to_phase.py`: dataset reader, fold definitions, and phase-classification model.

## Dataset

The dataset comprises **22 temporally annotated laparoscopic procedures**, including a subset of **five procedures with 2,934 spatially annotated images**.

- **Temporal annotations:** [Zenodo – Version 2](https://doi.org/10.5281/zenodo.24593512).
- **Spatial annotations:** [Kaggle – Video Structure Segmentation](https://www.kaggle.com/datasets/johnvassilakis/video-structure-segmentation).

In Version 2 of the Zenodo record, the temporal identifiers of Videos 5 and 21 were interchanged to ensure consistency with the spatial dataset.

The segmentation task uses eight classes (background and seven anatomical structures): abdominal wall, colon, IMA/superior rectal artery, IMV, small bowel, spleen, and ureter.

**Local layout:** download and extract the spatial dataset from Kaggle, locate `paths_and_annotations.csv`, and pass its path via `--csv` and the extracted image root via `--root`. Verify that paths in the CSV resolve to the extracted images and masks before training.

## Reproduction prerequisites

Python environment with the dependencies in `requirements.txt`; access to the spatial dataset and `paths_and_annotations.csv` through [Kaggle](https://www.kaggle.com/datasets/johnvassilakis/video-structure-segmentation). The revised temporal annotations are publicly available through [Zenodo – Version 2](https://doi.org/10.5281/zenodo.24593512).

The CSV, image data, video data, model weights and original experimental outputs are **not included** in this repository.


```bash

pip install -r requirements.txt

python run_segmentation.py --architecture DeepLabV3Plus --csv /path/to/paths_and_annotations.csv --root /path/to/dataset

python run_segmentation.py --architecture Segformer --csv /path/to/paths_and_annotations.csv --root /path/to/dataset

python run_experiment2.py --csv /path/to/paths_and_annotations.csv --root /path/to/dataset
```


## Important implementation notes

- The experiments use `segmentation-models-pytorch==0.5.0`, which exposes the `Segformer` architecture.
- The original segmentation evaluation applies a threshold of 0.5 directly to raw logits, corresponding to a sigmoid probability threshold of approximately 0.622. This behavior has been preserved to maintain consistency with the original experiments.
- The mask-to-phase classifier defines 8 output classes.
- The repository preserves the original experimental methodology while organizing the code into reusable scripts and modules.

## Code scope
The code in this repository was assembled from the original experimental implementations and organized to facilitate reproducibility and further development. The repository focuses on baseline anatomical segmentation and surgical phase classification experiments. 
