# DCross-Domain Alignment Fusion for Robust Remote Sensing Image Semantic Segmentation

[![Python](https://img.shields.io/badge/Python-3.10-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()

Official implementation of **DFSAFNet** for remote sensing semantic segmentation.

---

## 📖 Overview

Semantic segmentation of high-resolution remote sensing images remains challenging due to complex spatial structures and frequency variations.  
We propose **DFSAFNet**, which integrates spatial-domain features and frequency-domain representations through an attention fusion mechanism to enhance segmentation performance.

---

## 🧠 Network Architecture

![framework](DFSAFNet.png)

Key components:

- Frequency-domain feature extraction
- Dual-domain attention fusion
- Multi-scale feature aggregation
- Lightweight decoder design

---

## 📂 Dataset

We evaluate our method on:

- **ISPRS Potsdam**
- **ISPRS Vaihingen**
- **LoveDA**
- **UAVid**

Please download datasets from official websites.
Data set preprocessing can refer to[GeoSeg](https://github.com/WangLibo1995/GeoSeg )

---

## ⚙️ Installation

Create the environment and install dependencies:

```bash
conda create -n dfsafnet python=3.10
conda activate dfsafnet
It should be noted that:PyTorch == 2.2.2; CUDA == 11.8
git clone https://github.com/ziyanpeng/DFSAFNet.git
cd DFSAFNet
pip install -r requirements.txt
```

---

## 🚀 Training

To train DFSAFNet on the target dataset:Select a different config file

```bash
# potsdam
python DFSAFNet/train_supervision.py --DFSAFNet/config/potsdam/DFSAFNet.py

# Vaihingen
python DFSAFNet/train_supervision.py --DFSAFNet/config/vaihingen/DFSAFNet.py

# LoveDA
python DFSAFNet/train_supervision.py --DFSAFNet/config/loveda/DFSAFNet.py

# UAVid
python DFSAFNet/train_supervision.py --DFSAFNet/config/uavid/DFSAFNet.py
```

---

## 🚀 Testing

To evaluate a trained model:Select a different config file,Set the optimal model for testing in config

```bash
# potsdam
python DFSAFNet/potsdam_test.py --DFSAFNet/config/potsdam/DFSAFNet.py

# Vaihingen
python DFSAFNet/vaihingen_test.py --DFSAFNet/config/vaihingen/DFSAFNet.py

# LoveDA
python DFSAFNet/loveda_test.py --DFSAFNet/config/loveda/DFSAFNet.py

# UAVid
python DFSAFNet/uavid_test.py --DFSAFNet/config/uavid/DFSAFNet.py
```

---

## 📊 Reproduction Results

The quantitative results of **DFSAFNet** on benchmark datasets are shown below.

| Dataset | mIoU (%) | Mean F1 (%) | OA (%) |
|--------|---------|------------|-------|
| Vaihingen | 84.68 | 91.52 | 93.42 |
| Potsdam | 87.83 | 93.40 | 92.04 |
| LoveDA | 57.62 | 71.85 | - |
| UAVid | 73.78| 84.36 | - |


---

## 📖 Citation
If you find this work useful for your research, please consider citing:

```bibtex
@article{peng2025dfsafnet,
  title={Cross-Domain Alignment Fusion for Robust Remote Sensing Image Semantic Segmentation},
  author={Zhu, Weidong and Peng, Bangyu and Luan, Kuifeng and Zhu, Xiaolong and Qun, Qi},
  journal={The Visual Computer},
  year={2026},
  publisher={Springer}
}
```

---

