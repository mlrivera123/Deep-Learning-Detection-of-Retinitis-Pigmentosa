# Deep Learning Detection of Retinitis Pigmentosa Inheritance Forms through Synthetic Data Expansion of a Rare Disease Dataset — Paper Code

Implements VAE-based latent-space data augmentation followed by Vision Transformer fine-tuning for rare disease image classification, evaluated under 5-fold cross-validation.

## Installation

```bash
pip install -r requirements.txt
```

GPU with CUDA 12.8 recommended. For other CUDA versions install PyTorch separately from https://pytorch.org before running the above.

## Pipeline

### 1. Generate noise configuration

```bash
python generate_noise.py
```

Edit `base_dir` in the script to point to your dataset root before running. Expects the following structure:

```
<base_dir>/
    <fold>/
        train/
            0_AD_AR/
            1_XL_XLC/
```

Outputs `self_full.json` per fold, used by the expansion step.

### 2. Augment training data

```bash
python expand.py \
    --model_repo <vae_model_repo_or_local_path> \
    --input_data_dir <fold_dir> \
    --mode self_image \
    --noise_conf_json <fold_dir>/self_full.json \
    --output_data_dir <expanded_output_dir> \
    --expand_ratio <N> \
    --image_size 512
```

`--mode` options: `random`, `random_image`, `self_image`.

### 3. Fine-tune ViT

```bash
python finetune_vit.py \
    --data_dir <expanded_output_dir> \
    --output_dir <model_output_dir> \
    --seed 1337
```

Saves `probabilities.npy` and `labels.npy` to `output_dir` for each fold.

### 4. Generate paper metrics

```bash
python generate_paper_metrics.py <experiment_dir>
```

Expects fold results under `<experiment_dir>/0/`, `1/`, ..., `4/`, each containing `probabilities.npy` and `labels.npy`.
