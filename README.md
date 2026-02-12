

# Graduation Thesis Code (Undergraduate)

## 📦 Resources

- **Datasets & Model Checkpoints:** [Mega Cloud Drive](https://mega.nz/folder/T1Ah0IoQ#jwYtYR23L-Q2OaXz2WaOOA)



## 📊 Multi-Task Performance on ViT-B/32

|           Method           |   SUN397    |    Cars     |  RESISC45   |   EuroSAT   |    SVHN     |    GTSRB    |    MNIST    |     DTD     |   Avg Acc   |
| :------------------------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
|  **Non-Merging Methods**   |             |             |             |             |             |             |             |             |             |
|         Pretrained         |    62.3     |    59.7     |    60.7     |    45.5     |    31.4     |    32.6     |    48.5     |    43.8     |    48.0     |
|         Individual         |    79.2     |    77.7     |    96.1     |    99.7     |    97.5     |    98.7     |    99.7     |    79.4     |    90.8     |
|      Traditional MTL       |    73.9     |    74.4     |    93.9     |    98.2     |    95.8     |    98.9     |    99.5     |    77.9     |    88.9     |
| **Training-Based Methods** |             |             |             |             |             |             |             |             |             |
|         AdaMerging         |    64.5     |    68.1     |    79.2     |    93.8     |    87.0     |    91.9     |    97.5     |    59.1     |    80.1     |
|        AdaMerging++        |    66.6     |    68.3     |    82.2     |    94.2     |    89.6     |    89.0     |    98.3     |    60.6     |    81.1     |
|   Representation Surgery   |    63.8     |    59.9     |    83.3     |    97.9     |    87.0     |    87.0     |    98.6     |    69.4     |    80.9     |
| **Training-Free Methods**  |             |             |             |             |             |             |             |             |             |
|      Weight Averaging      |    65.3     |    63.4     |    71.4     |    71.7     |    64.2     |    52.8     |    87.5     |    50.1     |    65.8     |
|       Fisher Merging       | <u>68.6</u> |  **69.2**   |    70.7     |    66.4     |    72.9     |    51.1     |    87.9     | <u>59.9</u> |    68.3     |
|          RegMean           |    65.3     |    63.5     |    75.6     |    78.6     |    78.1     |    67.4     |    93.7     |    52.0     |    71.8     |
|      Task Arithmetic       |    55.2     |    54.9     |    66.7     |    78.9     |    80.2     |    69.7     |    97.3     |    50.4     |    69.1     |
|        Ties-Merging        |    59.8     |    58.6     |    70.7     |    79.7     | <u>86.2</u> |    72.1     | <u>98.3</u> |    54.2     |    72.4     |
|            TATR            |    62.7     |    59.3     |    72.3     | <u>82.3</u> |    80.5     |    72.6     |    97.0     |    55.4     |    72.8     |
|    TATR & Ties-Merging     |    66.3     |    65.9     |    75.9     |    79.4     |    79.9     |    68.1     |    96.2     |    54.8     |    73.3     |
|     Consensus Merging      |    65.7     |    63.6     |    76.5     |    77.2     |    81.7     |    70.3     |    97.0     |    57.1     |    73.6     |
|        PCB Merging         |    66.7     |    65.5     | <u>78.5</u> |    79.3     |  **86.4**   |  **77.1**   |    98.2     |    59.1     | <u>76.3</u> |
|   **NS Merging (Ours)**    |  **68.7**   | <u>67.0</u> |  **82.0**   |  **94.1**   |    81.9     | <u>75.9</u> |  **98.4**   |  **62.5**   |  **78.8**   |

> **Bold** = best; <u>underline</u> = second best (among training-free methods).

## 🛠️ Getting Started

### Environment

- Python 3.12
- PyTorch 2.5.1
- CUDA 12.4
- OpenCLIP 2.0.2

### Run

```bash
python main_ns.py
```

## 🙏 Acknowledgements

This project is built upon the [Task Vectors](https://github.com/mlfoundations/task_vectors) codebase. We sincerely thank the authors for making their code publicly available.
