# GraphNeuralPDESolver
This repository contains the code for the Graph Neural PDE Solver (GNPDESolver) model. It is a model zoo that includes several different types of encoders, processors, and decoders and different types of architectures which is designed to solve partial differential equations (PDEs) on arbitrary domains using graph-based neural networks.

Currently, it supports the following GNPDESolver:

- [MeshGraphNet](https://sites.google.com/view/meshgraphnets)
- [Geometry-Informed Neural Operator](https://openreview.net/pdf?id=86dXbqT5Ua)
- [RIGNO](https://github.com/sprmsv/rigno)
- [AIFS](https://arxiv.org/pdf/2406.01465)


Folder structure:
```
GNPDESolver/
├── config/
│   ├── gino/
│   ├── rano/
│   ├── rigno/
│   └── temp.py
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   └── readme.md
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── domain.py
│   │   ├── edges.py
│   │   ├── encoder.py
│   │   ├── graph.py
│   │   ├── rigraph.py
│   │   ├── support.py
│   │   └── tri.py
│   ├── trainer/
│   │   ├── utils/
│   │   │   ├── __init__.py
│   │   │   ├── cal_metric.py
│   │   │   ├── data_pairs.py
│   │   │   └── train_setup.py
│   │   ├── base.py
│   │   ├── optimizer.py
│   │   ├── seq.py
│   │   └── stat.py
│   └──  utils/
│       ├── __init__.py
│       ├── buffer.py
│       ├── dataclass.py
│       ├── pair.py
│       ├── rand.py
│       ├── sample.py
│       ├── scale.py
│       └── viz.py
├── tests/
├── viz/
├── main.py
├── MODEL.md
└── README.md
```