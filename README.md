# NS Merging: Task-Oriented Model Merging via Null-Space Projection

> This repository contains the official implementation of **NS Merging**, a training-free model merging method based on activation-aware null-space projection.



## 📦 Resources

Datasets & Model Checkpoints: [Mega Cloud Drive](https://mega.nz/folder/T1Ah0IoQ#jwYtYR23L-Q2OaXz2WaOOA)



## 📊 Experiments

### Multi-task performance on ViT-B/32

| Method | SUN397 | Cars | RESISC45 | EuroSAT | SVHN | GTSRB | MNIST | DTD | Avg Acc |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Non-Merging Methods** ||||||||||
| Pretrained | 62.3 | 59.7 | 60.7 | 45.5 | 31.4 | 32.6 | 48.5 | 43.8 | 48.0 |
| Individual | 79.2 | 77.7 | 96.1 | 99.7 | 97.5 | 98.7 | 99.7 | 79.4 | 90.8 |
| Traditional MTL | 73.9 | 74.4 | 93.9 | 98.2 | 95.8 | 98.9 | 99.5 | 77.9 | 88.9 |
| **Training-Based Merging Methods** ||||||||||
| AdaMerging | 64.5 | 68.1 | 79.2 | 93.8 | 87.0 | 91.9 | 97.5 | 59.1 | 80.1 |
| AdaMerging++ | 66.6 | 68.3 | 82.2 | 94.2 | 89.6 | 89.0 | 98.3 | 60.6 | 81.1 |
| Representation Surgery | 63.8 | 59.9 | 83.3 | 97.9 | 87.0 | 87.0 | 98.6 | 69.4 | 80.9 |
| **Training-Free Merging Methods** ||||||||||
| Weight Averaging | 65.3 | 63.4 | 71.4 | 71.7 | 64.2 | 52.8 | 87.5 | 50.1 | 65.8 |
| Fisher Merging | 68.6 | 69.2 | 70.7 | 66.4 | 72.9 | 51.1 | 87.9 | 59.9 | 68.3 |
| RegMean | 65.3 | 63.5 | 75.6 | 78.6 | 78.1 | 67.4 | 93.7 | 52.0 | 71.8 |
| Task Arithmetic | 55.2 | 54.9 | 66.7 | 78.9 | 80.2 | 69.7 | 97.3 | 50.4 | 69.1 |
| Ties-Merging | 59.8 | 58.6 | 70.7 | 79.7 | 86.2 | 72.1 | 98.3 | 54.2 | 72.4 |
| TATR | 62.7 | 59.3 | 72.3 | 82.3 | 80.5 | 72.6 | 97.0 | 55.4 | 72.8 |
| TATR & Ties-Merging | 66.3 | 65.9 | 75.9 | 79.4 | 79.9 | 68.1 | 96.2 | 54.8 | 73.3 |
| Consensus Merging | 65.7 | 63.6 | 76.5 | 77.2 | 81.7 | 70.3 | 97.0 | 57.1 | 73.6 |
| PCB Merging | 66.7 | 65.5 | 78.5 | 79.3 | **86.4** | 77.1 | 98.2 | 59.1 | 76.3 |
| **NS Merging (Ours)** | **71.2** | **70.6** | **85.5** | **96.1** | 85.5 | **85.5** | **98.6** | **67.5** | **82.6** |

### Ablation Study

| Linear-layer component | Non-projected parameter component | Avg Acc | Gain |
|:--:|:--:|:-:|:-:|
| ✗ | ✗ | 28.8 | +0.0 |
| ✗ | ✓ | 36.1 | +7.3 |
| ✓ | ✗ | 70.2 | +41.4 |
| ✓ | ✓ | **82.6** | **+53.8** |

### Generalization to Unseen Tasks

| Method | SUN397 | Cars | RESISC45 | DTD | SVHN | GTSRB | Seen Avg | MNIST | EuroSAT | Unseen Avg |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Task Arithmetic | 63.3 | 62.4 | 75.1 | 57.8 | 84.6 | 80.4 | 70.6 | 77.2 | 46.2 | 61.7 |
| Ties-Merging | 67.8 | 66.2 | 77.2 | 56.7 | 77.1 | 70.9 | 69.3 | 75.9 | 43.3 | 59.6 |
| TATR | 66.0 | 64.1 | 77.9 | 60.1 | 83.9 | 81.8 | 72.3 | 77.2 | 47.7 | 62.5 |
| **NS Merging (Ours)** | **73.2** | **72.2** | **90.0** | **70.2** | **92.5** | **94.5** | **82.1** | **81.5** | **51.6** | **66.6** |

### Task Balance

| Method | Std Dev |
|:-:|:-:|
| Weight Averaging | 11.56 |
| Fisher Merging | 13.20 |
| RegMean | 7.99 |
| Task Arithmetic | 8.98 |
| Ties-Merging | 8.50 |
| TATR | 7.30 |
| TATR & Ties-Merging | 8.34 |
| Consensus Merging | 7.75 |
| PCB Merging | 6.74 |
| **NS Merging (Ours)** | **4.33** |

### Hyperparameter Sensitivity

| Alpha | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 | 0.8 | 0.9 | 1.0 | 1.1 | 1.2 | 1.3 | 1.4 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Avg Acc | 56.8 | 64.8 | 71.0 | 75.9 | 79.1 | 81.0 | 81.8 | **82.6** | 81.6 | 80.0 | 78.4 | 76.6 | 72.9 | 69.2 |

### Sample Selection Robustness

| Dataset | Run1 | Run2 | Run3 | Run4 | Run5 | Avg Acc | Std |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| MNIST | 98.64 | 98.85 | 98.50 | 99.05 | 98.05 | 98.62 | 0.34 |
| EuroSAT | 95.96 | 96.41 | 95.85 | 95.74 | 96.52 | 96.10 | 0.31 |
| GTSRB | 86.76 | 85.71 | 86.08 | 83.46 | 85.32 | 85.47 | 1.11 |
| SVHN | 84.46 | 85.44 | 84.91 | 86.26 | 86.28 | 85.47 | 0.72 |
| RESISC45 | 84.97 | 85.70 | 86.16 | 86.25 | 84.57 | 85.53 | 0.66 |
| DTD | 67.29 | 67.13 | 67.50 | 67.55 | 68.09 | 67.51 | 0.33 |
| Cars | 69.63 | 71.14 | 70.85 | 71.27 | 70.20 | 70.62 | 0.62 |
| SUN397 | 70.96 | 71.31 | 71.21 | 71.48 | 71.24 | 71.24 | 0.17 |
| **8-task Avg** | **82.33** | **82.71** | **82.63** | **82.63** | **82.53** | **82.57** | **0.13** |



## 🛠️ Getting Started

### Environment

- Python 3.12
- PyTorch 2.5.1
- CUDA 12.4
- OpenCLIP 2.0.2

### Run

```bash
python main_ns.py  # 主实验
python main_ns_gl.py  # 泛化实验
python main_ns_ab.py  # 消融实验
```



## 🙏 Acknowledgements

This project is built upon the [Task Vectors](https://github.com/mlfoundations/task_vectors) codebase. We sincerely thank the authors for making their code publicly available.
