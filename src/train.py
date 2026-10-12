import csv, json, random
import numpy as np
import tensorflow as tf
from scipy.io import savemat
from . import config
from .data import load_scene, create_patches, make_split
from .model import build_model
from .evaluate import save_curves, evaluate, save_map

def main():
    random.seed(config.RANDOM_SEED); np.random.seed(config.RANDOM_SEED); tf.random.set_seed(config.RANDOM_SEED)
    for d in (config.RESULTS_DIR, config.FIGURES_DIR, config.METRICS_DIR, config.MODELS_DIR): d.mkdir(parents=True, exist_ok=True)
    image, gt = load_scene(); rows, cols, _ = image.shape
    print("Image:", image.shape, "Ground truth:", gt.shape, "Classes:", np.unique(gt))
    patches, pca = create_patches(image); split = make_split(patches, gt)
    print("Train:", split["x_train"].shape, "Validation:", split["x_val"].shape)
    model = build_model(config.PATCH_SIZE, config.N_COMPONENTS, split["n_classes"], config.DROPOUT_RATE, config.LEARNING_RATE, config.CLIP_VALUE)
    model.summary()
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
                 tf.keras.callbacks.ModelCheckpoint(str(config.MODELS_DIR / "salinas_2d_cnn.keras"), monitor="val_loss", save_best_only=True)]
    history = model.fit(split["x_train"], split["y_train_cat"], validation_data=(split["x_val"], split["y_val_cat"]), batch_size=config.BATCH_SIZE, epochs=config.EPOCHS, verbose=1, callbacks=callbacks)
    save_curves(history)
    val_pred = np.argmax(model.predict(split["x_val"], batch_size=config.BATCH_SIZE, verbose=1), axis=1)
    class_ids = list(range(1, split["n_classes"] + 1)); metrics = evaluate(split["y_val"], val_pred, class_ids)
    print(json.dumps({k:v for k,v in metrics.items() if k != "classification_report"}, indent=2))
    with open(config.METRICS_DIR / "validation_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["ground_truth_class", "predicted_class"]); w.writerows(zip(split["y_val"]+1, val_pred+1))
    # Full-scene map is for visualization only; metrics above use held-out validation pixels.
    probabilities = model.predict(patches, batch_size=config.BATCH_SIZE, verbose=1)
    pred_map = (np.argmax(probabilities, axis=1) + 1).reshape(rows, cols)
    save_map(pred_map)
    savemat(config.RESULTS_DIR / "predicted_class_map.mat", {"classification_map": pred_map})
    savemat(config.RESULTS_DIR / "class_probabilities.mat", {"probabilities": probabilities.reshape(rows, cols, split["n_classes"])})
    (config.METRICS_DIR / "pca_explained_variance.json").write_text(json.dumps({"explained_variance_ratio": pca.explained_variance_ratio_.tolist()}, indent=2), encoding="utf-8")
    print("Done. Results are in", config.RESULTS_DIR)

if __name__ == "__main__": main()
