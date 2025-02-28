> 🚧 **Note:** This repository is currently a work in progress. Some features may be incomplete or may produce errors as development is ongoing.

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
python script/make_dataframe.py --name <DATASET_NAME> --src-path <DATASET_PATH> --dst-path <SAVE_PATH> --data-type lmdb
```
#### Arguments
- `name`: Name for the output dataframe.
- `src-path`: Path to the source dataset.
- `dst-path`: Path to save the converted dataframe.
- `data-type` : Choose either `lmdb` (OC20 dataset format) or `ase` (atoms format that can be opened by ASE).
- `props` : List of properties to include in the dataframe. Multiple properties can be specified by separating them with spaces. Available options:
    - `ads` : Types of adsorbate.
    - `spg` : Space group symmetry of the bulk structure of catalyst.
    - `miller` : Miller indices of surface.
    - `comp` : Composition of the bulk structure of catalyst.
      
    **Example usage**
    ```bash
    --props ads spg miller comp    
    ```
    
- `num-workers` : Number of processes to use for data conversion.
  
#### For training a detection model
> **Note:** Detection model-related options are currently unavailable.

~~To train a detection model that evaluates catalyst validity, corrupted representations paired with binary labels (valid vs. corrupted) are needed.~~

~~Users can generate corrupted data and labels into the dataframe by including the `--corrupt_data` argument:~~

```
python script/make_dataframe.py --name <DATASET_NAME> --src-path <DATASET_PATH> --dst-path <SAVE_PATH> --data-type lmdb --corrupt-data
```

#### Fine-tuning Dataset
The 2e-ORR dataset, used as an example for fine-tuning, can be found in `data/dataset/2eORR/`.

## Training

To train a CatGPT model from scratch or to continue training with additional data, run:

```
python train.py
```

Users can customize the dataset, tokenizer, hyperparameters, and other settings in `config/config.yml`.

#### Training detection model

If users set the `architecture` parameter as `'BERT'` in `config.yml`, the script will automatically train a detection model using the corrupted data and binary labels generated earlier.

For example:

```yaml
model_params:
    name: 'oc20-2M-BERT'
    architecture : 'BERT'
    ...
```

## Generation

To generate string representations of catalyst structures, run:

```
python script/generate.py --name <NAME> --ckpt-path <MODEL_PATH> --save-path <SAVE_PATH>
```

#### Required Arguments
- `name`: Name for the generated structures set.
- `ckpt-path`: Path to the trained generative model checkpoint.
- `save-path`: Path to save the generated structures set.

#### Optional Arguments
- `string-type` : Type of tokenization strategy to use.
- `input-prompt` : Initial prompt for generation (e.g., a specific adsorbate).
- `n_generation`, `top_k`, `top_p`, `temperature` : Generation parameters that control the diversity and creativity of generated structures.

#### Pre-trained Model
You can download a pre-trained model checkpoint from [here](https://zenodo.org/records/14406696)

## Evaluation

To evaluate generated strings and save them in a crystal format, run:

```
python script/validate.py --cls-path <MODEL_PATH> --gen-path <GENERATED_DATA_PATH> --save-path <SAVE_PATH>
```
#### Required Arguments
- `cls-path`: Path to the trained detection model checkpoint.
- `gen-path`: Path to the generated structures set.
- `save-path`: Path to save validated data.

#### Optional Arguments
- `gt-path` : Path to ground-truth structure data for comparison.
- `string-type` : Type of tokenization strategy to use.
- `n-samples` : Number of structures to validate.
- `skip-fail` : Option to bypass overlapping atoms in the generated structures.

#### Pre-trained Model
You can download a pre-trained detection model checkpoint from [here](https://zenodo.org/records/14504779)
