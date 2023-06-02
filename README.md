# Enhancing Predictions of Drug Solubility through Multidimensional Structural Characterization Exploitation
## Introduction
We proposed a solubility prediction model MSCSol that integrated multi-dimensions molecular structure information. The model included a graph neural network equipped with geometric vector perceptrons(GVP-GNN) to encode the 3D structure of molecules, which can not only represent the arrangement and direction of atoms in space, but also include the atomic sequence and interaction between them. In addition, we chose SKA-ResNeXt50 to encode the features of molecular images, and used Selective Kernel Convolution combining Global and Local attention mechanisms to capture contextual information of molecular images at different scales. We also calculated various descriptors to enrich the molecular representation.

## Requirement
PyTorch 1.12.1
Python 3.7.12
CUDA 10.1+
CuPy 11.0.0
RDKit 2022.3.5
Tmap 1.0.6

## dataset
The dataset used in the experiment was collected from multiple databases and can be obtained under the directory ./dataset.

## Split training and test set
The experiment involves multiple data set partition methods, including Random, Stratified, Scaffold. Different methods have different hyperparameter that can be set.

## Model
The model mainly includes four parts: (a) Molecular 3D feature representation based on GVP-GNN; (b) Using ResNeXt50 with Selective Kernel Convolution combining Global and Local attention mechanisms to encode molecular images. (c) The calculation of molecular descriptors. (d) Concat the features obtained from the first three parts and input them into the fully connected layers for solubility prediction.
