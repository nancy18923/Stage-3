# ECS 170 Stage 2: MLP Classification on MNIST (PyTorch)

This repository implements **Stage 2** of the ECS 170 course project: multi-class
classification on the MNIST hand-written digit dataset with a Multi-Layer
Perceptron (MLP) built in PyTorch, following the ECS 170 five-module code
template (dataset / method / setting / result / evaluate).

## Results

| Metric | Score |
| --- | --- |
| Accuracy | 0.9781 |
| Precision (weighted) | 0.9782 |
| Recall (weighted) | 0.9781 |
| F1 (weighted) | 0.9781 |

## Project Structure

```
groupproject/
├── script/
│   ├── stage_1_script/        # Stage 1 driver scripts (DT / SVM / MLP + load result)
│   └── stage_2_script/
│       ├── script_mlp.py      # Stage 2 main experiment (trains + plots convergence curve)
│       └── script_ablation.py # Ablation study (5 configurations)
├── local_code/
│   ├── base_class/            # Abstract base classes of the template
│   ├── stage_1_code/          # Stage 1 implementations
│   └── stage_2_code/          # Stage 2 implementations
│       ├── Dataset_Loader.py      # MNIST csv loader (/255 normalization)
│       ├── Method_MLP.py          # MLP model (784-256-ReLU-10), Adam + CrossEntropyLoss
│       ├── Setting_Train_Test.py  # Pre-partitioned train/test protocol (no CV)
│       ├── Evaluate_Metrics.py    # Accuracy + weighted/macro P/R/F1
│       └── Result_Saver.py
├── data/                      # Datasets are provided by the instructor, NOT included
│   ├── stage_1_data/
│   └── stage_2_data/          # Put train.csv / test.csv (MNIST) here
└── result/
    ├── stage_1_result/
    └── stage_2_result/        # Pickle results, convergence curve, architecture figure,
                               # ablation_results.csv
```

## How to Run

```bash
cd script/stage_2_script
python script_mlp.py       # main experiment (~5 min on CPU)
python script_ablation.py  # ablation study (~25 min on CPU)
```

Run scripts **from the `script/stage_2_script/` directory** (all paths are
relative to it), or set `PYTHONPATH` to the project root:

```bash
PYTHONPATH=<project root> python script_mlp.py
```

## Dataset

MNIST (train.csv: 60,000 instances; test.csv: 10,000 instances; each line =
label + 784 pixels). The data files are provided by the instructor — download
them from the course page (Canvas) and place them under `data/stage_2_data/`.
They are excluded from this repository because they exceed GitHub's file size
limit.

## Environment

- Python 3.13 (PyCharm `.venv` in the project works out of the box)
- PyTorch (CPU), scikit-learn, numpy, matplotlib
