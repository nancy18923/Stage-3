# ECS 170 Stage 3: Image Classification with CNN (PyTorch)

This repository implements **Stage 3** of the ECS 170 course project: image
classification for object recognition with Convolutional Neural Networks on
three datasets (MNIST, ORL and CIFAR-10), following the ECS 170 five-module
code template (dataset / method / setting / result / evaluate). Stage 1 and
Stage 2 code is kept for reference.

## Results (baseline CNNs, full training protocol)

| Dataset | Test instances | Accuracy | Precision | Recall | F1 score |
| --- | --- | --- | --- | --- | --- |
| MNIST | 10,000 | 0.9882 | 0.9883 | 0.9882 | 0.9882 |
| ORL | 40 | 0.8250 | 0.7375 | 0.8250 | 0.7667 |
| CIFAR-10 | 10,000 | 0.7771 | 0.7817 | 0.7771 | 0.7753 |

## Project Structure

```
groupproject/
├── script/
│   ├── stage_1_script/         # Stage 1 drivers (DT / SVM / MLP)
│   ├── stage_2_script/         # Stage 2 drivers (MLP on MNIST)
│   └── stage_3_script/
│       ├── script_cnn.py            # Main experiment: train + evaluate three CNNs
│       ├── script_ablation.py       # Ablation sweep (12 configurations x 3 datasets)
│       ├── script_plot_curves.py    # Convergence curve figures
│       ├── script_plot_architecture.py  # CNN architecture diagram
│       ├── script_data_loader.py    # Dataset sanity check
│       └── script_load_result.py    # Read pickles and print metrics
├── local_code/
│   ├── base_class/             # Abstract base classes of the template
│   ├── stage_1_code/
│   ├── stage_2_code/
│   └── stage_3_code/
│       ├── Dataset_Loader.py       # MNIST / ORL / CIFAR image loaders (/255, (C,H,W))
│       ├── Method_CNN_MNIST.py     # 2 conv blocks [32,64], dropout 0.25, 10 classes
│       ├── Method_CNN_ORL.py       # 2 conv blocks [32,64], dropout 0.5,  40 classes
│       ├── Method_CNN_CIFAR.py     # 3 conv blocks [32,64,128], dropout 0.3, 10 classes
│       ├── Setting_Train_Test.py   # Pre-partitioned train/test protocol (no CV)
│       ├── Evaluate_Metrics.py     # Accuracy + weighted P/R/F1
│       ├── Result_Saver.py
│       └── Result_Loader.py
├── data/                       # Datasets are provided by the instructor, NOT included
│   ├── stage_1_data/
│   ├── stage_2_data/           # Put train.csv / test.csv (MNIST) here
│   └── stage_3_data/           # Put MNIST / ORL / CIFAR here (see ReadMe.txt)
└── result/
    ├── stage_1_result/
    ├── stage_2_result/
    └── stage_3_result/         # Pickle results, ablation CSVs, figures/
```

## How to Run

All scripts must be run from the project root with `PYTHONPATH` set to it:

```bash
cd "/path/to/groupproject"
PYTHONPATH=. python script/stage_3_script/script_cnn.py           # baseline (~10 min CPU)
PYTHONPATH=. python script/stage_3_script/script_ablation.py      # ablation (~9 min CPU)
PYTHONPATH=. python script/stage_3_script/script_plot_curves.py   # convergence figures
PYTHONPATH=. python script/stage_3_script/script_plot_architecture.py
```

## Dataset

The three datasets are provided by the instructor and placed under
`data/stage_3_data/`:

- `MNIST` — 60,000 training / 10,000 testing 28x28 grayscale digits (10 classes)
- `ORL` — 400 face images of 40 people, 9 for training / 1 for testing per person
- `CIFAR` — 50,000 training / 10,000 testing 32x32 RGB images (10 classes)

They are excluded from this repository because they are far above GitHub's
file size limit (MNIST 424 MB, CIFAR 181 MB).

## Ablation Study

`script_ablation.py` changes one configuration knob at a time (kernel size,
padding, depth, width, pooling type, batch norm, dropout, hidden dimension,
optimizer, loss function, learning rate) and reports test accuracy for each
dataset in `result/stage_3_result/ablation_all.csv`. To keep the CPU runtime
manageable the sweep uses a reduced protocol: 3 epochs on 300 images per class
for MNIST and CIFAR, 20 epochs on the full 360 training images for ORL.

## Environment

- Python 3.13 (PyCharm `.venv` in the project works out of the box)
- PyTorch (CPU), scikit-learn, numpy, matplotlib
