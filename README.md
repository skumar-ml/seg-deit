# Overview
This is the project repository for ECE 544's Fall 2023 course project. In the project, I develop the Segmented Vision Transformer (Seg-ViT). Seg-ViT works with superpixels as tokens, rather than the uniform, rectangular patches that are used in the original ViT work. 

This repository implements ViT (with capability of using DeIT data augmentations) with the Fourier Transform and the Color Histogram fixed-length representation

# Requirements
Please use the `timm/` folder included in the repo, and not the one from PyPi. I implemented the Segmented Vision Transformer in the repo's `timm/` folder. Any version of PyTorch should work, I use 2.0.1. Also install fast-slic for an accelerated implementation of SLIC.

# How to Use the Repo
The segmented patches are all pre-processed so that we are not constantly having to re-segment images during training time. The original RGB CIFAR dataset is converted to the segmented version using the specific fixed-length transform. The Color Histogram one can be found in `scripts/data_process_BoW.ipynb`, and the Fourier Transform one can be found in `scripts/data_process_crop.ipynb`. All statistics used for normalization in the Fourier Transformation representation are extracted using `scripts/data_stats.py` and saved as `train_stats.pt`. The data is saved in the user-specified folder and should be passed as an argument to the training script.

All training bash scripts are located in `run_scripts/`. `run_seg_vit.sh` runs the training script for the Seg-ViT. There is the option to specify which fixed-length transform is used, and which dataset to use. The `DataLoader` is compatible with data generated using `scripts/data_process_*.ipynb` files. 

The main training script is `train_vit.py`

# My Code
This repo extends [Meta's Data Efficient Transformer repo](https://github.com/facebookresearch/deit). Below are my unique contributions:

* DataLoader - in `datasets.py` - loads in Numpy Array data of the segmented CIFAR dataset. Removes the default transforms used with ImageNet (resizing + standardization) the default transforms.
* Data Processing Scripts - in `scripts/` - implementation of two fixed-length representations (Fourier Transform and Color Histogram). Implements custom positional embeddings for each segmented patch. Implements fast-slic. Processes entire dataset in 2 hours. Vectorized and using list comprehensions where possible. Implementation of finding mean and variance of patches in a batched fashion for standardization purposes. Implementation of RAG-based method to merge connected regions to reach the number of tokens specified. Implementation of naive Fourier Transform where patch kept within the original masked image (resulted in a 1% accuracy when training).
* Segmented Vision Transformer - in `timm/models/vision_transformer.py` - Extends Vision Transformer. Replaces learnable positional embeddings with fixed 5-D ones. Replaces original patchification with linear layer to project patches and their positional information to an embedding space. Implementation of a naive `SegmentEmbed` class that does segmentation and fixed-length transformation in an online fashion. Was computation intractable using Fourier Transform approach, but can be used with DeIT data-augmentation techniques. 
