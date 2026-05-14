# CamoPatch: Camouflaged Adversarial Patches

This repository contains an implementation of the **CamoPatch** adversarial attack, inspired by the NeurIPS 2023 paper on camouflaged adversarial patches. The attack generates visually natural, camouflage-like adversarial patches composed of overlapping circles using evolutionary algorithms and simulated annealing.

## Project Structure

- `camopatch.py`: Core implementation of the `CamoPatch` class and algorithm. It includes the `PatchRenderer` and the combined `(1+1)-ES` and Fast Simulated Annealing optimization algorithms.
- `run_attack.py`: A quick-start script that downloads a sample image, initializes a ResNet-50 model, runs the CamoPatch attack, and saves the generated adversarial image to `adv_sample.jpg`.
- `evaluate.py`: A full evaluation script designed to compute the Attack Success Rate (ASR), L2 Distance, and Non-normalised Residual (NNR) across a subset of the ImageNet dataset.
- `run_ablation.py`: Script to run ablation studies on various hyperparameters ($N$, $\sigma$, $l_i$, $t$).
- `utils.py`: Utility functions for loading models (`get_model`), loading datasets (`load_imagenet_subset`), and wrapping models for input normalization (`NormalizedModel`).
- `notebook.ipynb` / `notebook.py`: A structured Jupyter Notebook version of the pipeline for academic documentation, interactive testing, and easy visualization of the Phase 0-4 workflows.
- `requirements.txt`: Required Python dependencies.

## Installation

Ensure you have Python 3.8+ installed, then install the required dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:
- `torch` >= 2.0.0
- `torchvision`
- `numpy`
- `Pillow`
- `scipy`
- `robustbench`

## Usage

### 1. Preparing the Dataset & Environment
To evaluate the attack rigorously, the paper requires testing on images that are correctly classified by all target models (including robust models and defenses). 

**Pre-requisites (Must complete before running Notebook or Scripts):**
1. **Clone PatchGuard Defense:**
   ```bash
   git clone https://github.com/inspire-group/PatchGuard.git
   ```
2. **Download Model Weights:**
   - Go to the PatchGuard Google Drive: [Download Link](https://drive.google.com/drive/folders/1u5RsCuZNf7ddWW0utI4OrgWGmJCUDCuT?usp=sharing)
   - Download `bagnet17_net.pth`.
   - Place it at `PatchGuard/checkpoints/bagnet17_net.pth`.
3. **Hugging Face Authentication (For Official ImageNet-1K):**
   - Accept the terms for the official dataset: [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k).
   - Install HF CLI: `pip install huggingface_hub`.
   - Login: `huggingface-cli login` (Enter your Access Token).

**Download & Filter Data:**
Run the following script to stream and filter exactly 1000 correctly classified images:
```bash
python download_val.py --num_images 1000
```
This will create the `imagenet_val_1000/` directory.

### 2. Using the Jupyter Notebook
For an interactive and visual walkthrough of the entire pipeline (Phase 0 to Phase 4), use `notebook.ipynb`.

**How to run:**
1. Open the file in VS Code or Jupyter Lab.
2. Ensure you have completed the **Pre-requisites** above (PatchGuard weights & HF Login).
3. Run all cells sequentially. The notebook includes:
   - Environment setup and verification.
   - Automatic dataset filtering (if not already done).
   - Visualization of the (1+1)-ES and Simulated Annealing attack process.
   - Analysis of results (ASR, L2, NNR).

### 3. Quick Attack on a Sample Image
To quickly test the attack on a single image without downloading the full ImageNet dataset, run:

```bash
python run_attack.py
```
This will download a sample image of a Golden Retriever, attack a ResNet-50 model, and save the resulting image to `adv_sample.jpg`.

### 3. Evaluating on ImageNet
To run a full evaluation over a dataset (e.g., the filtered `imagenet_val_1000` directory), use the `evaluate.py` script.

```bash
python evaluate.py --data_dir imagenet_val_1000 --model resnet50 --num_images 1000 --max_queries 10000
```
**Arguments:**
- `--data_dir`: Path to the ImageNet validation directory (Required).
- `--model`: Target model architecture (e.g., `resnet50`, `vgg16`, `vit_b_16`, `at_resnet50` for RobustBench models). Default: `resnet50`.
- `--num_images`: Number of images to run the evaluation on. Default: `10`.
- `--max_queries`: Maximum number of queries $K$ for the attack. Default: `10000`.
- `--patch_size`: Size of the square patch in pixels. Default: `40`.
- `--num_circles`: Number of circles $N$ in the patch. Default: `100`.
- `--sigma`: Evolutionary step size $\sigma$. Default: `0.1`.

### 4. Running Ablation Studies
To investigate the impact of different hyperparameters:

```bash
python run_ablation.py --data_dir imagenet_val_1000
```

## Methodology

The CamoPatch algorithm operates in several phases:
1. **Patch Initialization**: A patch is parameterized as $N$ overlapping circles on a black background, with randomly initialized locations, radii, colors, and transparencies.
2. **Patch Optimization**: Uses a `(1+1)-ES` evolutionary strategy to perturb and optimize the patch parameters locally to maximize the misclassification margin loss.
3. **Location Optimization**: Employs Fast Simulated Annealing to periodically propose and evaluate new patch locations on the image, escaping local optima.
4. **Adversarial Constraint**: Keeps the L2 distance between the original region and the patch as small as possible if the attack is already successful, ensuring camouflage properties.

## References
This codebase is developed for research and empirical evaluation purposes, aiming to reproduce the methodology and evaluate baseline comparisons for the CamoPatch adversarial strategy.
