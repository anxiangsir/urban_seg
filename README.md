<div align="center">

# 🛰️ Urban Segmentation

### Few-Shot Remote Sensing Semantic Segmentation powered by Foundation Models

<p align="center">
  <a href="https://github.com/anxiangsir/urban_seg/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="License"></a>
  <a href="https://github.com/deepglint/unicom"><img src="https://img.shields.io/badge/Backbone-UNICOM-blue?style=flat-square" alt="UNICOM"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-%3E%3D1.10-red?style=flat-square" alt="PyTorch"></a>
  <img src="https://img.shields.io/badge/Data_Efficiency-High-brightgreen?style=flat-square" alt="Data Efficiency">
  <img src="https://visitor-badge.laobi.icu/badge?page_id=anxiangsir.urban_seg" alt="visitors">
</p>

[English](#-introduction) | [简体中文](#-项目介绍)

</div>

---

## 📖 Introduction

**Urban Segmentation** is a streamlined, high-performance framework designed for semantic segmentation of remote sensing imagery. 

Leveraging the power of **[UNICOM](https://github.com/deepglint/unicom)**—a vision foundation model pre-trained on **400 million** images—this project demonstrates extreme data efficiency. We achieve SOTA-level segmentation results using **only 4 labeled satellite images** for training. This repository serves as both a robust baseline for research and an accessible entry point for practitioners.

## 📖 项目介绍

**Urban Segmentation** 是一个专为遥感图像语义分割设计的高效框架。

本项目利用了在 **4亿** 海量数据上预训练的视觉基础模型 **[UNICOM](https://github.com/deepglint/unicom)**，展示了极致的数据样本效率。我们仅需 **4张** 标注的卫星图像进行微调，即可获得极佳的分割效果。这不仅为科研提供了一个强有力的 Baseline，也为初学者提供了一个极简的实战范例。

---

## ⚡ Key Features

*   **Foundation Model Power**: Built upon UNICOM ViT backbones, inheriting robust feature representations.
*   **Extreme Few-Shot**: Achieve high mIoU with minimal annotated data (4 images).
*   **Plug-and-Play**: Minimalist code structure (~200 lines for training) without complex dependencies.
*   **Scalable**: Supports both single-GPU rapid prototyping and multi-GPU distributed training.

---

## 🎨 Visualization | 效果展示

<div align="center">

| **Dynamic Prediction** | **Generalization Test** |
| :---: | :---: |
| <img src="figures/predict.gif" width="350"> | <img src="figures/test.jpg" width="350"> |
| <img src="figures/predict_02.gif" width="350"> | <img src="figures/test_02.jpg" width="350"> |

</div>



## 🛠️ Getting Started | 快速上手

### 1. Installation
```bash
git clone https://github.com/anxiangsir/urban_seg.git
cd urban_seg
pip install -r requirements.txt
```

### 2. Data Preparation

Download the dataset (CCF Satellite Imagery) from [Baidu Cloud](https://pan.baidu.com/s/1LWBMklOr39yI7fYRQ185Og) (Code: `3ih2`).

Structure your directory as follows:
```text
dataset/
├── origin/       # 5 annotated source images
├── test/         # Unlabeled test images
└── train/        # Generated via preprocessing
    ├── images/
    └── labels/
```

Run the preprocessing script to generate random crops:
```bash
python preprocess.py
```

### 3. Model Zoo

Download the pre-trained UNICOM weights from the [Official Release](https://github.com/deepglint/unicom/releases):

*   `FP16-ViT-B-32.pt`
*   `FP16-ViT-B-16.pt` (Recommended)
*   `FP16-ViT-L-14.pt`

### 4. Training

**Option A: Rapid Prototyping (Single GPU)**
```bash
# Minimal implementation (~200 lines)
python train_one_gpu.py
```

**Option B: High-Performance Training (Multi-GPU DDP)**
```bash
# Distributed Data Parallel
torchrun --nproc_per_node 8 train_multi_gpus.py
```

---

## 📜 Citation

If you find this project or the UNICOM model useful for your research, please consider citing:

```bibtex
@inproceedings{an2023unicom,
  title={Unicom: Universal and Compact Representation Learning for Image Retrieval},
  author={An, Xiang and Deng, Jiankang and Yang, Kaicheng and Li, Jiawei and Feng, Ziyong and Guo, Jia and Yang, Jing and Liu, Tongliang},
  booktitle={ICLR},
  year={2023}
}
```

## 🤝 Community & Support

<div align="left">
  <a href="https://qm.qq.com/cgi-bin/qm/qr?k=xxxxx"><img src="https://img.shields.io/badge/QQ_Group-679897018-blue?style=flat-square&logo=tencent-qq" alt="QQ Group"></a>
</div>

We welcome all contributions! Please feel free to open an issue or submit a pull request.
