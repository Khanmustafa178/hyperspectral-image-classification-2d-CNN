import random
import cv2
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from sklearn.decomposition import PCA
from tensorflow.keras.utils import to_categorical
from . import config


def resolve_dataset_paths():
    candidate_names = {
        "image": ["Salinas.mat", "salinas.mat", "salinas_image.mat"],
        "ground_truth": ["Salinas_gt.mat", "salinas_gt.mat", "salinas_gt.mat"],
    }
    search_dirs = []
    for directory in getattr(config, "DATA_DIR_CANDIDATES", [config.DATA_DIR]):
        if directory is not None:
            search_dirs.append(Path(directory).expanduser())
    seen = set()
    for directory in search_dirs:
        if directory not in seen:
            seen.add(directory)
    image_path = None
    ground_truth_path = None
    for directory in search_dirs:
        for filename in candidate_names["image"]:
            path = directory / filename
            if path.exists():
                image_path = path
                break
        for filename in candidate_names["ground_truth"]:
            path = directory / filename
            if path.exists():
                ground_truth_path = path
                break
        if image_path is not None and ground_truth_path is not None:
            break
    if image_path is None:
        raise FileNotFoundError(
            "Missing Salinas.mat. Searched: " + ", ".join(str(d) for d in search_dirs)
        )
    if ground_truth_path is None:
        raise FileNotFoundError(
            "Missing Salinas_gt.mat. Searched: " + ", ".join(str(d) for d in search_dirs)
        )
    return image_path, ground_truth_path


config.IMAGE_PATH, config.GROUND_TRUTH_PATH = resolve_dataset_paths()


def load_scene():
    image_path = Path(config.IMAGE_PATH)
    ground_truth_path = Path(config.GROUND_TRUTH_PATH)
    if not image_path.exists():
        raise FileNotFoundError(f"Missing {image_path}. See data/README.md.")
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Missing {ground_truth_path}. See data/README.md.")
    imat, gmat = loadmat(image_path), loadmat(ground_truth_path)
    image_key = config.IMAGE_KEY if config.IMAGE_KEY in imat else next((k for k in ["salinas", "Salinas"] if k in imat), None)
    ground_truth_key = config.GROUND_TRUTH_KEY if config.GROUND_TRUTH_KEY in gmat else next((k for k in ["salinas_gt", "Salinas_gt", "groundtruth", "GroundTruth"] if k in gmat), None)
    if image_key is None or ground_truth_key is None:
        raise KeyError(f"Expected MATLAB keys not found; available image keys={sorted(imat.keys())[:10]}, ground-truth keys={sorted(gmat.keys())[:10]}")
    image = np.asarray(imat[image_key], dtype=np.float32)
    gt = np.asarray(gmat[ground_truth_key]).squeeze()
    if image.ndim != 3:
        raise ValueError(f"Unexpected image dimensions: image={image.shape}")
    if gt.shape != image.shape[:2]:
        if gt.shape[0] <= image.shape[0] and gt.shape[1] <= image.shape[1]:
            y0 = (image.shape[0] - gt.shape[0]) // 2
            x0 = (image.shape[1] - gt.shape[1]) // 2
            image = image[y0:y0 + gt.shape[0], x0:x0 + gt.shape[1], :]
        else:
            raise ValueError(f"Unexpected shapes: image={image.shape}, ground truth={gt.shape}")
    return image, gt.astype(np.int32)

def create_patches(image):
    rows, cols, bands = image.shape
    if config.N_COMPONENTS > bands:
        raise ValueError("N_COMPONENTS exceeds the number of spectral bands")
    pca = PCA(n_components=config.N_COMPONENTS, random_state=config.RANDOM_SEED)
    reduced = pca.fit_transform(image.reshape(-1, bands)).reshape(rows, cols, config.N_COMPONENTS)
    size, pad = config.PATCH_SIZE, config.PATCH_SIZE // 2
    padded = cv2.copyMakeBorder(reduced, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    patches = np.empty((rows * cols, size, size, config.N_COMPONENTS), dtype=np.float32)
    k = 0
    for i in range(rows):
        for j in range(cols):
            patches[k] = padded[i:i+size, j:j+size, :]
            k += 1
    patches /= np.float32(config.NORMALIZATION_DIVISOR)
    return patches, pca

def make_split(patches, gt):
    labels = gt.reshape(-1)
    rng = random.Random(config.RANDOM_SEED)
    tr, va = [], []
    for cls in sorted(int(v) for v in np.unique(labels) if v > 0):
        indices = np.flatnonzero(labels == cls).tolist()
        n = int(len(indices) * config.TRAIN_FRACTION_PER_CLASS)
        if n < 1:
            raise ValueError(f"Class {cls} has too few pixels for current training fraction")
        rng.shuffle(indices)
        tr.extend(indices[:n]); va.extend(indices[n:])
    tr, va = np.asarray(tr), np.asarray(va)
    n_classes = int(labels.max())
    ytr, yva = labels[tr] - 1, labels[va] - 1
    return dict(x_train=patches[tr], y_train=ytr, y_train_cat=to_categorical(ytr, n_classes),
                x_val=patches[va], y_val=yva, y_val_cat=to_categorical(yva, n_classes),
                train_indices=tr, val_indices=va, n_classes=n_classes)
