"""
Kidney ultrasound CNN (TensorFlow/Keras): Normal vs Stone.
Instruction:
    Run text in start.txt on terminal
Usage:
    python train.py --data /path/to/extracted/archive --epochs 25

Expected folder layout (what archive.zip extracts to):
    <data>/Normal/*.JPG
    <data>/stone/*.JPG

Dataset issues this script handles:
  1. ~1,000 exact-duplicate images (mostly Stone)  -> removed by hash, so the
     same picture can't land in both train and test.
  2. Colored calipers / measurement text appear ONLY on Stone images (~28% in
     a sample, 0% on Normal) -> a CNN could "cheat" by detecting the overlay.
     Colored pixels are inpainted away and images are converted to grayscale.
  3. Neighbouring file numbers are often near-identical frames -> split is
     done in contiguous blocks of file numbers, not fully at random.
"""
import argparse, hashlib, re, random, json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

CLASSES = {"Normal": 0, "stone": 1}
IMG = 224


# ---------------------------------------------------------------- data prep
def clean_image(path):
    bgr = cv2.imread(str(path))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    colored = (hsv[..., 1] > 80) & (hsv[..., 2] > 60)          # calipers / text
    mask = cv2.dilate(colored.astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
    if mask.any():
        bgr = cv2.inpaint(bgr, mask, 3, cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (IMG, IMG), interpolation=cv2.INTER_AREA)


def num(p):
    return int(re.findall(r"(\d+)", p.stem)[-1])


def build_arrays(root, cache):
    cache = Path(cache)
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"], d["g"]
    xs, ys, gs, seen = [], [], [], set()
    dup = 0
    for cname, label in CLASSES.items():
        files = sorted((Path(root) / cname).glob("*.*"), key=num)
        for p in tqdm(files, desc=cname):
            h = hashlib.md5(p.read_bytes()).hexdigest()
            if h in seen:
                dup += 1
                continue
            seen.add(h)
            xs.append(clean_image(p)); ys.append(label)
            gs.append(num(p) // 20)          # block id: ~20 adjacent files stay together
    print(f"removed {dup} exact duplicates")
    x, y, g = np.stack(xs), np.array(ys), np.array(gs)
    np.savez_compressed(cache, x=x, y=y, g=g)
    return x, y, g


def split(y, g, seed=42):
    """Group-aware 70/15/15 split, stratified by class."""
    rng = random.Random(seed)
    tr, va, te = [], [], []
    for c in (0, 1):
        blocks = sorted(set(g[y == c]))
        rng.shuffle(blocks)
        n = len(blocks)
        b_tr, b_va = set(blocks[: int(.7 * n)]), set(blocks[int(.7 * n): int(.85 * n)])
        for i in np.where(y == c)[0]:
            (tr if g[i] in b_tr else va if g[i] in b_va else te).append(i)
    return np.array(tr), np.array(va), np.array(te)


# -------------------------------------------------------------- tf.data
def make_ds(x, y, idx, bs, train):
    xs = (x[idx].astype("float32") / 255.0)[..., None]       # (N, 224, 224, 1)
    ds = tf.data.Dataset.from_tensor_slices((xs, tf.one_hot(y[idx], 2)))
    if train:
        ds = ds.shuffle(len(idx), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(bs)
    if train:
        aug = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(10 / 360),
            layers.RandomZoom(0.1),
            layers.RandomBrightness(0.15, value_range=(0, 1)),
            layers.RandomContrast(0.15),
        ])
        ds = ds.map(lambda a, b: (aug(a, training=True), b), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------------- models
def conv_block(x, f, reg):
    for _ in range(2):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, kernel_regularizer=reg)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
    return layers.MaxPooling2D()(x)


def build_cnn(a):
    """--model small : 16-32-64-128 filters (fewer params, less overfitting)
       --model cnn   : 32-64-128-256 filters"""
    reg = regularizers.l2(a.l2) if a.l2 else None
    inp = layers.Input((IMG, IMG, 1))
    x = layers.Rescaling(1 / 0.2, offset=-1.0)(inp)           # ~ (x-0.2)/0.2
    if a.crop:
        x = layers.Cropping2D(a.crop)(x)                      # drop border text / scale bar
    for f in ((16, 32, 64, 128) if a.model == "small" else (32, 64, 128, 256)):
        x = conv_block(x, f, reg)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(a.dropout)(x)
    out = layers.Dense(2, activation="softmax", kernel_regularizer=reg)(x)
    return models.Model(inp, out, name="KidneyCNN"), None


def build_transfer(a):
    """MobileNetV2 pretrained on ImageNet; base frozen first, optional fine-tuning after."""
    reg = regularizers.l2(a.l2) if a.l2 else None
    inp = layers.Input((IMG, IMG, 1))
    x = layers.Concatenate()([inp, inp, inp])                 # gray -> 3 channels
    if a.crop:
        x = layers.Cropping2D(a.crop)(x)
    x = layers.Rescaling(2.0, offset=-1.0)(x)                 # [0,1] -> [-1,1]
    base = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=tuple(x.shape[1:]),
        weights=None if a.weights == "none" else "imagenet")
    base.trainable = False
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(a.dropout)(x)
    out = layers.Dense(2, activation="softmax", kernel_regularizer=reg)(x)
    return models.Model(inp, out, name="KidneyMobileNet"), base


def compile_model(model, a, lr, steps=None):
    sched = tf.keras.optimizers.schedules.CosineDecay(lr, steps) if steps else lr
    model.compile(optimizer=tf.keras.optimizers.AdamW(sched, weight_decay=1e-4),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=a.label_smoothing),
                  metrics=["accuracy"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", choices=["small", "cnn", "transfer"], default="small")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--finetune", type=int, default=0, help="extra epochs unfreezing top of MobileNet (transfer only)")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.5)
    ap.add_argument("--l2", type=float, default=1e-4, help="L2 weight penalty (0 = off)")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--crop", type=int, default=0, help="pixels to crop from every border (e.g. 16)")
    ap.add_argument("--weights", choices=["imagenet", "none"], default="imagenet")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    tf.keras.utils.set_random_seed(42)

    x, y, g = build_arrays(a.data, out / "cache_224.npz")
    tr, va, te = split(y, g)
    print(f"train {len(tr)}  val {len(va)}  test {len(te)}  | stone share {y.mean():.2f}")
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    dtr, dva, dte = (make_ds(x, y, i, a.bs, t) for i, t in ((tr, True), (va, False), (te, False)))
    model, base = build_transfer(a) if a.model == "transfer" else build_cnn(a)
    # transfer: constant small LR is more stable than a decay for a frozen-base phase
    compile_model(model, a, a.lr if a.model != "transfer" else min(a.lr, 3e-4),
                  None if a.model == "transfer" else a.epochs * int(np.ceil(len(tr) / a.bs)))
    model.summary()
    # ----------EARLYSTOPPING TO PREVENT OVERFITTING-----------------------------------
    cbs = [callbacks.ModelCheckpoint(str(out / "best.keras"), monitor="val_accuracy",
                                     save_best_only=True, verbose=1),
           callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)
           if a.model == "transfer" else callbacks.Callback(),
           callbacks.EarlyStopping(monitor="val_accuracy", patience=a.patience, restore_best_weights=True)]
    hist = model.fit(dtr, validation_data=dva, epochs=a.epochs, callbacks=cbs, shuffle=False).history

    if a.model == "transfer" and a.finetune > 0:
        base.trainable = True
        for l in base.layers[:-30]:                           # keep early layers frozen
            l.trainable = False
        compile_model(model, a, 1e-5)
        h2 = model.fit(dtr, validation_data=dva, epochs=a.finetune, callbacks=cbs, shuffle=False).history
        hist = {k: hist[k] + h2[k] for k in hist}

    # ---- overfitting check
    b = int(np.argmax(hist["val_accuracy"]))
    gap = hist["accuracy"][b] - hist["val_accuracy"][b]
    print(f"\nBest epoch {b+1}: train acc {hist['accuracy'][b]:.3f}  val acc {hist['val_accuracy'][b]:.3f}  gap {gap:+.3f}")
    print("  -> gap > ~0.05 suggests overfitting; val acc ~0.99+ with tiny gap suggests leakage, check Grad-CAM.")

    # ---- test
    model = tf.keras.models.load_model(out / "best.keras")
    p = model.predict(dte)[:, 1]
    t = y[te]
    pred = (p > .5).astype(int)
    print(classification_report(t, pred, target_names=["Normal", "Stone"], digits=3, zero_division=0))
    auc = roc_auc_score(t, p)
    print("AUC", auc); print(confusion_matrix(t, pred))
    json.dump({"auc": float(auc), "val_best": float(max(hist["val_accuracy"])), "train_val_gap": float(gap)},
              open(out / "metrics.json", "w"))

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["accuracy"], label="train"); ax[1].plot(hist["val_accuracy"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    fig.savefig(out / "curves.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()
