from __future__ import annotations
import os, json, random, gc
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.io as tv_io
import segmentation_models_pytorch as smp

class TorchSurgicalAnnotationDataset(Dataset):
    """
    Dataset for surgical video annotations with organ labels.
    Optimized for PyTorch with vectorized operations.
    
    Args:
        csv_path: Path to CSV file with columns: phase, path, and organ columns
        root_dir: Root directory containing video folders (default: current working directory)
        target_size: Target image size as (height, width) in pixels. Required to ensure
                    all images have the same size for batching.
        transform: Optional torchvision transforms to apply
    """
    
    ORGAN_COLUMNS = [
        "abdominal wall",
        "colon",
        "inferior mesenteric artery",
        "inferior mesenteric vein",
        "small bowel",
        "spleen",
        "ureter",
    ]
    
    def __init__(
        self,
        csv_path: Path | str,
        root_dir: Path | str | None = None,
        target_size: Tuple[int, int] | None = None,
        transform: Optional[transforms.Compose] = None,
    ):
        if root_dir is None:
            # Default to current working directory (works well in Kaggle)
            root_dir = Path.cwd()
        self.root_dir = Path(root_dir).resolve()

        if isinstance(csv_path, str):
            self.csv_path = Path(csv_path).resolve()
            self.csv_data = pd.read_csv(self.csv_path)
        else:
            self.csv_data = csv_path
        self.target_size = target_size
        self.transform = transform

        # Original color mapping
        colour_to_classes = {
            (0,0,0) : 0,
            (160, 171, 180) : 1,
            (240, 120, 240) : 2,
            (255, 53, 94) : 3,
            (61, 61, 245) : 4,
            (255, 0, 124) : 5,
            (140, 120, 240) : 6,
            (50, 183, 250) : 7,
        }

        self.number_of_classes = len(colour_to_classes)
        
        # Pre-compute color tensor for vectorized mask creation (much faster)
        # Shape: (num_classes, 3) - each row is an RGB color
        self.color_tensor = torch.zeros((self.number_of_classes, 3), dtype=torch.uint8)
        for rgb, class_idx in colour_to_classes.items():
            self.color_tensor[class_idx] = torch.tensor(rgb, dtype=torch.uint8)
        
        # Convert to float32 for faster comparisons (will be normalized)
        self.color_tensor_float = self.color_tensor.float() / 255.0

    def read_groundtruth(self, idx):
        pass

    def read_annotation(self, idx):
        pass

    def read_phase(self, idx):
        row = self.csv_data.iloc[idx]
        phase = int(row['phase'].split('/')[-1].split('phase')[-1][0])

        return torch.tensor(phase, dtype = torch.int)
    
    def read_images(self, idx):
        # Resolve torchvision.functional by module path so a global `import torch.nn.functional as F`
        # cannot shadow torchvision's resize (Kaggle / notebook order issues).
        import importlib

        _tvf = importlib.import_module("torchvision.transforms.functional")

        row = self.csv_data.iloc[idx]
        annotation_path = self.root_dir / row["path"]
        
        # Use torchvision.io to read image directly as tensor (C, H, W) in uint8 format
        # This is faster than PIL as it avoids Python overhead
        annotation = tv_io.read_image(str(annotation_path), tv_io.ImageReadMode.RGB)
        
        # Resize if target_size is provided
        if self.target_size:
            annotation = _tvf.resize(annotation, self.target_size, antialias=True)
        
        # Normalize to [0, 1] range (convert uint8 to float32)
        annotation = annotation.float() / 255.0

        groundtruth_path = str(annotation_path).replace("defaultannot", "groundtruth").replace('frame_', 'frame_0')
        
        # Use torchvision.io to read image directly as tensor (C, H, W) in uint8 format
        groundtruth = tv_io.read_image(groundtruth_path, tv_io.ImageReadMode.RGB)
        
        # Resize if target_size is provided
        if self.target_size:
            groundtruth = _tvf.resize(groundtruth, self.target_size, antialias=True)
        
        # Normalize to [0, 1] range (convert uint8 to float32)
        groundtruth = groundtruth.float() / 255.0
        
        return groundtruth, annotation

    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, idx, inference = False):
        
        groundtruth, annotation = self.read_images(idx)

        if not(inference):
            # Vectorized mask creation using PyTorch (much faster than loops)
            # annotation shape: (3, H, W)
            # color_tensor_float shape: (num_classes, 3)
            
            # Expand dimensions for broadcasting: (3, H, W) -> (1, 3, H, W)
            annotation_expanded = annotation.unsqueeze(0)  # (1, 3, H, W)
            
            # Expand color tensor: (num_classes, 3) -> (num_classes, 3, 1, 1)
            color_expanded = self.color_tensor_float.view(self.number_of_classes, 3, 1, 1)
            
            # Compare all pixels with all colors at once: (num_classes, 3, H, W)
            matches = (annotation_expanded == color_expanded)  # Broadcasting comparison
            
            # All 3 channels must match: (num_classes, H, W)
            mask_class = matches.all(dim=1).long()  # (num_classes, H, W)
            
            # Apply transform if provided
            if self.transform:
                groundtruth = self.transform(groundtruth)
                annotation = self.transform(annotation)
            
            return groundtruth, mask_class, self.read_phase(idx)
        
        # Apply transform if provided
        if self.transform:
            groundtruth = self.transform(groundtruth)
            annotation = self.transform(annotation)
            
        return groundtruth, annotation, self.read_phase(idx)

