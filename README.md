# Catalyst Generative Pretrained Transformer
Catalyst Generative Pretrained Transformer (CatGPT) is a heterogeneous catalyst generative model based on Generative Pretrained Transformer 2 (GPT-2) architecture,
designed to generate string representations of catalyst structures, including both slab and adsorbate atoms.

This model is described in the paper: [Generative Pretrained Transformer for Heterogeneous Catalysts](https://arxiv.org/abs/2407.14040)

## Table of Contents

- [Installation](#installation)
- [Datasets](#datasets)
- [Training](#training)
- [Generation](#generation)
- [Evaluation](#evaluation)
- [Example Use](#example-use)

## Installation

Run the following commands to set up:

```bash
conda update conda
conda env create -f env.yml # Creates the environment with all dependencies
```
Activate the Conda environment with `conda activate catgpt`.

## Datasets

### Pretraining Dataset
The training and validation dataset is sourced from the [Open Catalyst 2020 (OC20) database](https://fair-chem.github.io/core/datasets/oc20.html) in the [Fair-chem repository](https://github.com/FAIR-Chem/fairchem).

To convert the dataset to a dataframe with string representations for CatGPT training, run:
```
python scripts/make_dataframe.py --name DATASET_NAME --src_path DATASET_PATH --dst_path SAVE_PATH --data_type lmdb
```
- `name`: Name for the output dataframe
- `src_path`: Path to the source dataset
- `dst_path`: Path to save the converted dataframe
- `data_type` : Choose either `lmdb` (OC20 dataset format) or `ase` (atoms format that can be opened by ASE)

### Fine-tuning Dataset
The 2e-ORR dataset, used as an example for fine-tuning, can be found in `data/dataset/2eORR/`.

## Training

To train a CatGPT model from scratch or to continue training with additional data, run:

```
python train.py
```

Users can customize the dataset, tokenizer, hyperparameters, and other settings in `config/config.yaml`.

## Generation

To generate string representations of catalyst structures, run:

```
python script/generate.py
```

Users can customize the path to the trained model, generation parameters and other settings in `config/generation_config.yaml`.

## Evaluation

To evaluate generated strings and save them in a structures format, run:

```
python script/validation.py
```

Users an customize the path to anomaly detection model, validation parameters and other setting in `config/validation_config.yaml`.

## Example Use

### Adsorbate Conditional Generation
>⚠️ **Note**: Currently, the dataframe used for training the adsorbate conditional generative model *must* include a column for adsorbate symbols. The column will not be automatically added to the dataframe except only for the OC20 dataset.

This feature generates catalyst structures conditioned on specified adsorbates.

1. Set `string_type` as 'ads' in `config/config.yaml` to automatically add adsorbate symbols to the represenation.
2. Run `python train.py` to train the model.
3. Set `input_prompt` to the desired adsorbate symbol, e.g., '*O', `string_type` as 'ads' and `checkpoint_path` to the trained model in `config/generation_config.yaml`.

Users can skip 1. and 2. by downloading the pretrained model checkpoint.

The avaliable adsorbate symbols are the same as the list of adsorbates included in the [OC20 database](https://fair-chem.github.io/core/datasets/oc20.html#per-adsorbate-trajectories).
