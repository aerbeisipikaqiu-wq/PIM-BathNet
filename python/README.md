# PIM-BathNet
A Physics-Informed Modulation Bathymetry Network (PIM-BathNet) for absolute bathymetric inversion

This repository provides the open-source implementation used in the manuscript:

**“Dual-Driven SAR Bathymetric Inversion with Physical Priors and Deep Learning: A Case Study of Florida’s Eastern Coast”**

This code supports the deep learning-based method for marine water depth inversion. It takes in sliced SAR images and physical parameters as input and performs absolute water depth inversion.
---

## Overview

This repository implements a reproducible workflow that integrates:

- A method for land-sea segmentation of SAR coastline images containing lagoons
- The physical prior acquisition method based on ACF
- Seawater depth inversion from SAR images based on Deep Learning

Compared with purely physical models and various deep learning baselines, this method has advantages in terms of accuracy and covers a wider water depth range.

All analyses are based on **SAR images**.

---

## Data

This study employs Sentinel-1 Ground Range Detected (GRD) imagery acquired in the Interferometric Wide swath (IW) mode. The GRD product supports dual polarizations (VV and VH, vertical–vertical and vertical–horizontal) with a pixel spacing of 10 m. The Sentinel-1 scene with the identifier A9E4 is selected to cover the eastern coastal zone of Florida.

Bathymetric measurements corresponding to the study waters were downloaded from the official website of the National Centers for Environmental Information (NCEI), NOAA (https://www.ncei.noaa.gov/). After coordinate transformation between geographic coordinates and image pixel coordinates, these in-situ depth data were incorporated as auxiliary parameters into the inversion model.

## Scripts (Brief Description)

- `ACF.py`  
  **Purpose:** Obtain the propagation direction of the waves in the slice image based on the ACF method.
  **Input:** One image.  
  **Output:** theta_deg.

- `CNN_method.py`  
  **Purpose:** Use the method based on CNN for the inversion of seawater depth.
  **Input:** one SAR image patch and physics data from csv document.
  **Output:** predicted water depth.

- `transformer_method.py`  
  **Purpose:** Use the method based on Transformer for the inversion of seawater depth.  
  **Input:** one SAR image patch and physics data from csv document.
  **Output:** predicted water depth.

- `XGBoost_method.py`  
  **Purpose:** Use the method based on XGBoost for the inversion of seawater depth.   
  **Input:** physics data from csv document.
  **Output:** predicted water depth.

- `PIM-BathNet.py`
  **Purpose:** Use the method based on PIM-BathNet for the inversion of seawater depth.   
  **Input:** one SAR image patch and physics data from csv document.
  **Output:** predicted water depth.

- `test.py`
  **Purpose:** Use the method based on PIM-BathNet for the inversion of seawater depth.
  **Input:** one SAR image patch and physics data from csv document.
  **Output:** predicted water depth.
This file is used to test the weights of the best_yolo_fixed_transect_model.pth file. Using the full water depth data of the four sections (which were not used for training), the predicted results and accuracy of the proposed PIM-BathNet model for these four sections can be obtained.
  **Notice:** If you want to use this document, you should confirm the patch_dir_path. Suppose you need to test the T32 section, you need to modify the patch_dir_path to "patch_dir_path = ROOT / "test_data" / "T32"" or follow the corresponding address. If you want to switch the section, you need to modify patch_dir_path, but you don't need to modify csv_dir_path, because this is caused by the file quantity limit of a single folder on Github.


- `1.rar、2.rar 、3.rar 、4.rar 、`
  **Introduction:** The training set data compression file. If you wish to use this document, you need to extract the four compressed files, move all the contents inside them to the "python" folder, and then run the training code.

- `best_advanced_cnn_model3.pth`
  **Introduction:** The weight file of the CNN method.

- `best_transformer_model2.pth`
  **Introduction:** The weight file of the transformer method.

- `best_yolo_fixed_transect_model.pth`
  **Introduction:** The weight file of the PIM-BathNet.

- `Folder：test_data` is the storage path for image patch of test dataset.

- `Folder：python` is the storage path for image patch of train dataset.

- `Folder：python2` is the storage path for csv document.

- `Folder：ablation_test` is the storage path for ablation test document.
  **Purpose:** The method based on PIM-BathNet was used for seawater depth inversion, but an ablation experiment was conducted. Some of the organizational structures were missing.
  **Input:** one SAR image patch and physics data from csv document.
  **Output:** predicted water depth.



> Note: Some scripts contain hard-coded local paths. Please update file paths before running.