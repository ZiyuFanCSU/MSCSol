# Enhancing Predictions of Drug Solubility through Multidimensional Structural Characterization Exploitation
## Introduction
We proposed a solubility prediction model MSCSol that integrated multi-dimensions molecular structure information. The model included a graph neural network equipped with geometric vector perceptrons(GVP-GNN) to encode the 3D structure of molecules, which can not only represent the arrangement and direction of atoms in space, but also include the atomic sequence and interaction between them. In addition, we chose SKA-ResNeXt50 to encode the features of molecular images, and used Selective Kernel Convolution combining Global and Local attention mechanisms to capture contextual information of molecular images at different scales. We also calculated various descriptors to enrich the molecular representation.

## Requirements

To run the codes, You can configure dependencies by restoring our environment:
```
conda env create -f environment.yml -n $Your_env_name$
```

and then：

```
conda activate $Your_env_name$
```

## Structure
The code includes data processing and data enhancement, data feature extraction, model construction, model training, other baseline model codes, experimental results, trained models, and various visualisations and interpretive analyses. The directory structure of our uploaded code is as follows:

```
MSCSol
├── all_dataset
│   ├── benchmark_dataset       # benchmark dataset
│   │   ├── data.json           # Molecular properties and modal data computed from the dataset
│   │   ├── img                 # image of molecules
│   │   ├── img_augment         # image data augmentation
│   │   ├── xyz                 # coords of molecules
│   │   ├── xyz_augment         # coords data augmentation
│   ├── melting_point_dataset   # Dataset with melting point, the subdirectory structure is the same as the benchmark
│   ├── independent_dataset     # Independent dataset, the subdirectory structure is the same as the benchmark
├── data_process 
│   ├── get_properties.py       # How to extract features and perform feature dimensionality reduction by SMILES
│   ├── get_2D_image.py         # Generation of 2D image and data enhancement based on SMILES
│   ├── get_3D_image.py         # Generation of 3D image and data enhancement based on SMILES
|   ├── get_3D_image_H.py       # Generation of 3D image with H and data enhancement based on SMILES
│   ├── get_3D_XYZ.py           # Obtaining molecular coordinates optimised under the MMFF94 force field
│   ├── get_area_ratio.py       # Calculation: areas and ratios of projections in different directions
│   ├── Gaussian                # Gaussian model run file generation and computational results extraction
│   │   ├── g09.py              # Gaussian model run file generation
│   │   └── get_dipole_Gaussian09.py     # Computational results extraction
├── features                    # GraphDataset
├── Interpretability
│   ├── pic                     # Experimental result for this part
│   ├── GNNExplainer.py         # Code for GNNExplainer for GNN-GVP Explanatory Analysis
│   ├── HeatMap.ipynb           # The heat map of the molecular image
│   ├── statistic.ipynb         # statistic of molecular weight
│   ├── TSNE.ipynb              # Dimensionality reduction analysis of model feature acquisition
│   ├── independent.ipynb       # The experimental results of the independent dataset
│   └── TMAP.ipynb              # TMAP usage code
├── models_for_PyPI
│   ├── fingerprint.csv         # using for regularization
│   ├── trained.pt              # model
│   └── PyPI.ipynb              # demo
├── gvp
│   ├── SKANet                  # SKA-ResNet and variants
│   ├── __init__.py             # GVP blocks
│   └── MSCSolmodel.py          # Model used in this paper
├── other_methods               # All baseline models for comparison
├── trained_models
│   ├── 5_fold_cv               # Five fold cross validation results
│   ├── ablation_result         # The results of ablation experiments in the paper
│   ├── all_models              # The model we trained in the experiment
|   ├── machine_learning        # Machine learning methods used in the paper and their results
│   └── metrics.ipynb           # Experimental results and figures reproduction
└── train.py                    # Training and validation code
└── environment.yml             # Dependent Package Version
``` 

All the models can be downloaded here: https://zenodo.org/records/10996294.

## Model
The model mainly includes four parts: (a) Molecular 3D feature representation based on GVP-GNN; (b) Using ResNeXt50 with Selective Kernel Convolution combining Global and Local attention mechanisms to encode molecular images. (c) The calculation of molecular descriptors. (d) Concat the features obtained from the first three parts and input them into the fully connected layers for solubility prediction.

![modeloverview.png](pics%2Fmodeloverview.png)

## Training
The training process with default parameters requires a GPU card with at least 18GB of memory.

Run `train.py` using the following command:
```bash
python train.py --device <your_device> --log_path <log_dir> --models-dir <model_dir> --show_progressbar
```
- the `device` indicates which gpu you want to run the code
- the `log_dir` is the directory you want to store the log file
- the `model_dir` is the directory you want to store the trained model

Other configurations need to be changed inside `train.py`, including model settings and the data directory.

## Using a trained model to predict moleculer solubility

### MSCSol

[![Downloads](https://static.pepy.tech/badge/MSCSol)](https://pepy.tech/project/MSCSol)

Our trained model has been uploaded to PyPI, accessible through this link (https://pypi.org/project/MSCSol/). We've included detailed installation instructions and usage guidelines, making it easy to obtain prediction results by inputting SMILES strings.


### Installation

Install MSCSol using:

```
pip install MSCSol==1.0.5
```

website: https://pypi.org/project/MSCSol/

### Quick Start

```
import MSCSol

MSCSol.pred(<your_SMILES>)
```
![PyPI_demo.png](pics%2FPyPI_demo.png)

### Note

- It will take some time to calculate the molecular signatures, so please be patient for a while. Also note that dipole moment features are not used here as they cannot be obtained directly by running the code.
  
- The training data was restricted to molecular weights less than or equal to 504, LogS values greater than or equal to -8, and experimental temperatures of 20-25 degrees Celsius, so if the molecule does not apply to the above conditions, the prediction results may have a large deviation.
  
- In addition, due to the computational requirements of the node vector feature of the GVP-GNN, the input molecule atom number must be greater than or equal to 3.

- Four temporary files will be generated at runtime, named img_MSCSol.png, shadow_MSCSol_1.png, shadow_MSCSol_2.png and shadow_MSCSol_3.png. Please make sure not to have the same file name as yours to avoid accidental deletion.
  
- Due to maximum upload file size limit of 100MB in PyPI, our model and data files cannot be uploaded and need to be manually downloaded to the same level directory where the command is run. The files and tutorials are available on GitHub.


## Contact

We thank all the researchers who contributed to this work.

If you have any questions, please contact fzychina@csu.edu.cn.
