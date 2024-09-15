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
├── models/
│   ├── encoders/
│   │   ├── MLPEncoder.py
│   │   ├── FNOEncoder.py
│   │   └── MessagePassingEncoder.py
│   ├── processors/
│   │   ├── MLPProcessor.py
│   │   ├── FNOProcessor.py
│   │   └── MessagePassingProcessor.py
│   ├── decoders/
│   │   ├── MLPDecoder.py
│   │   ├── FNODecoder.py
│   │   └── MessagePassingDecoder.py
│   └── __init__.py
├── architectures/
│   ├── MeshGraphNet.py
│   ├── GINO.py
│   └── __init__.py
├── trainers/
│   ├── Trainer.py
│   └── __init__.py
├── data/
│   ├── dataset1/
│   ├── dataset2/
│   └── __init__.py
├── utils/
│   ├── data_preprocessing.py
│   ├── metrics.py
│   └── __init__.py
├── tests/
│   ├── test_encoders.py
│   ├── test_processors.py
│   ├── test_decoders.py
│   ├── test_architectures.py
│   ├── test_trainers.py
│   └── __init__.py
├── main.py
└── README.md
```