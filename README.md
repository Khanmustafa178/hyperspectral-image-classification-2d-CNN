# 2D-CNN for Hyperspectral Image Classification

A reproducible TensorFlow/Keras implementation of a 2D convolutional neural network for pixel-wise classification of the Salinas hyperspectral image using spatial-spectral patches.

**Dataset files are not included.** Obtain them from an authorized source and place them in `data/`. Check the dataset provider's terms before redistributing it.

## Pipeline
1. Load `Salinas.mat` (`salinas`) and `Salinas_gt.mat` (`salinas_gt`).
2. Reduce spectral bands to 5 PCA components.
3. Extract reflected-border spatial patches of size 25×25.
4. Randomly select 5% of labeled pixels per class for training; use remaining labeled pixels for validation.
5. Train a 2D-CNN with 32/64/128 convolution filters, dropout 0.3, dense layers of 128 and 64 units, and softmax output.
6. Generate a full-scene classification map and compute metrics on held-out validation pixels.

## Structure
```text
Salinas-2D-CNN/
├── README.md  requirements.txt  LICENSE  .gitignore
├── data/README.md
├── notebooks/Salinas_2D_CNN.ipynb
├── src/{config,data,model,evaluate,train}.py
├── models/
└── results/{figures,metrics}/
```

## Dataset setup
Put `Salinas.mat` and `Salinas_gt.mat` into `data/`. The MATLAB variables must be named `salinas` and `salinas_gt` respectively; adjust `src/config.py` if yours differ.

## Install and run
Python 3.10 or 3.11 is recommended.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m src.train
```
TensorFlow installation can vary by operating system and hardware; consult the official TensorFlow installation guide if needed.

## Outputs
- `results/figures/classification_map.png`
- `results/figures/confusion_matrix.png`
- `results/figures/training_curves.png`
- `results/metrics/metrics.json` and `validation_predictions.csv`
- `results/predicted_class_map.mat` and `class_probabilities.mat`
- `models/salinas_2d_cnn.keras`

## Reproducibility notes
- Random seed: 1671; training split: 5% per labeled class; patch size: 25; PCA components: 5.
- PCA is fit on all image pixels to mirror the supplied notebook. For stricter protocols, fit PCA on training pixels only and transform the rest.
- The source notebook divides PCA scores by 255. This is retained for comparability, although PCA scores can be negative or outside `[0, 255]`; evaluate alternative standardization as a separate documented experiment.
- Validation metrics are computed only on held-out labeled pixels. The full-scene map is for visualization, not a test-set score.
- No performance results are claimed here; run the project to generate results.

## Citation
Please cite the original Salinas dataset source/publication and this repository in any academic work. Update this placeholder after publishing:
```bibtex
@software{salinas_2d_cnn,
  title = {Salinas 2D-CNN for Hyperspectral Image Classification},
  author = {Your Name},
  year = {2026},
  url = {YOUR_GITHUB_REPOSITORY_URL}
}
```
The MIT license applies to code only, not to the dataset.
