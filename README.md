# README

# **RGVPSeg: multimodal information fusion network for retinogeniculate visual pathway segmentation**

---

In this study, we propose a novel deep multimodal fusion network for retinogeniculate visual pathway segmentation. Our approach integrates multimodal information from T1w images, FA images, and peaks images. Unlike conventional multimodal fusion methods that employ front-end channel connection and back-end feature fusion strategies by simply summing or cascading channels without optimizing multi- modal information, our proposed network selects and super- vises useful fusion information from the master and assis- tant modalities using attention and refinement modules.


RGVPSeg is the code for the following papers. Please cite the papers if you use it.

- Zeng Q, Yang L, Li Y, Xie L*, Feng Y*. RGVPSeg: multimodal information fusion network for retinogeniculate visual pathway segmentation[J]. Medical & Biological Engineering & Computing, 2025: 1-15.
  


# **How to use**

---
Note: Use the MRtrix3 to process primary DWI data and get the corresponding FA images, Peaks images, and DEC images; The Human Connectome Project dataset is available at [https://db.humanconnectome.org](https://db.humanconnectome.org/).

- Preprocess

1. Crop the HCP data (i.e., T1w and dMRI data) with a spatial resolutionof 145×174×145 voxels to 128×160×128 voxels; normalize the T1w images, T2w images, and dMRI data from 0 to 255.
```
python createrawdata.py
```
2. Obtaining training and testing data
```
python createmydata_2d.py
```
- For RGVPSeg

1. Find the best weights for single modalities separately
```
python train_T1_FA_PEAKS.py (train_T1, train_FA, and train_Peaks)
python predict_T1_FA_PEAKS.py (train_T1, train_FA, and train_Peaks)
```
2. Load the best weights of a single modality to train the RGVPSeg
```
python train_RGVPSeg.py
```
3. Validate model
```
python predict_RGVPSeg.py
```

4. Other: Train and Validate other models

```

python train_FECC_fusion.py, train_BEFF_Fusion.py
python predict_FECC.py predict_BEFF.py
```


# **Concact**

---
Lei Xie, Zhejiang University of Technology

leix@zjut.edu.cn
