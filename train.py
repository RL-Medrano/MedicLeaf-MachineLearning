import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import datetime
import json
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# CONSTANTS
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42


tf.keras.utils.set_random_seed(SEED)

SHOW_PLOTS = False
CHECK_LEAKAGE = True
LEAKAGE_MAX_HAMMING = 5
MEM_CACHE_LIMIT_BYTES = 4 * 1024**3

# PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(script_dir, "dataset", "train")
val_dir   = os.path.join(script_dir, "dataset", "valid")
test_dir  = os.path.join(script_dir, "dataset", "test")

model_dir = os.path.join(script_dir, "model")
os.makedirs(model_dir, exist_ok=True)

model_path      = os.path.join(model_dir, "medicleaf_detector_mobilenetv2.keras")
best_model_path = os.path.join(model_dir, "best_model.keras")

run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
history_dir = os.path.join(script_dir, "history", run_id)
os.makedirs(history_dir, exist_ok=True)
print(f"This run's outputs will be saved to: {history_dir}")

def save_fig(filename):
    """Save the current figure into the run folder, optionally show it, then close it."""
    plt.savefig(os.path.join(history_dir, filename), dpi=300)
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# DATA LEAKAGE CHECK
IMG_EXTS     = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
_RESAMPLE    = getattr(Image, "Resampling", Image).LANCZOS  # Pillow >=9.1 fallback

def dhash(path):
    """64-bit difference hash: robust to small compression/rescale changes."""
    img = Image.open(path).convert("L").resize((9, 8), _RESAMPLE)
    arr = np.asarray(img, dtype=np.int16)
    return (arr[:, 1:] > arr[:, :-1]).flatten().astype(np.uint8)

def collect_split_hashes(split_dir, split_name):
    files = [
        os.path.join(root, fn)
        for root, _, fns in os.walk(split_dir)
        for fn in fns
        if fn.lower().endswith(IMG_EXTS)
    ]
    paths, hashes = [], []
    for k, p in enumerate(files, 1):
        try:
            hashes.append(dhash(p))
            paths.append(p)
        except Exception as e:
            print(f"  [warn] could not read {p}: {e}")
        if k % 500 == 0:
            print(f"  {split_name}: hashed {k}/{len(files)} images ...")
    arr = np.array(hashes) if hashes else np.zeros((0, 64), dtype=np.uint8)
    return paths, arr

def cross_split_duplicates(paths_a, hashes_a, paths_b, hashes_b, max_hamming):
    """Near-duplicate pairs between two splits.

    64 bits are split into 8 bands of 8 bits. If two hashes differ by <= 7
    bits, at least one band must be bit-identical (pigeonhole), so grouping
    by band turns the O(n*m) comparison into a fast candidate lookup.
    """
    band_index = {}
    for j in range(len(hashes_b)):
        hb = hashes_b[j].tobytes()
        for b in range(8):
            band_index.setdefault((b, hb[b * 8:(b + 1) * 8]), []).append(j)
    pairs = []
    for i in range(len(hashes_a)):
        ha = hashes_a[i].tobytes()
        cands = set()
        for b in range(8):
            cands.update(band_index.get((b, ha[b * 8:(b + 1) * 8]), ()))
        for j in cands:
            dist = int(np.count_nonzero(hashes_a[i] != hashes_b[j]))
            if dist <= max_hamming:
                pairs.append((paths_a[i], paths_b[j], dist))
    return pairs

if CHECK_LEAKAGE:
    print("\n===== DATA LEAKAGE CHECK (train/valid/test near-duplicates) =====")
    splits = {}
    for name, d in (("train", train_dir), ("valid", val_dir), ("test", test_dir)):
        p, h = collect_split_hashes(d, name)
        splits[name] = (p, h)
        print(f"  {name}: {len(p)} images hashed")

    all_pairs = []
    for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
        for pa, pb, dist in cross_split_duplicates(
            *splits[a], *splits[b], LEAKAGE_MAX_HAMMING
        ):
            all_pairs.append((a, b, pa, pb, dist))

    leakage_report = os.path.join(history_dir, "data_leakage_report.txt")
    with open(leakage_report, "w", encoding="utf-8") as f:
        f.write(f"Cross-split near-duplicates (dHash hamming <= {LEAKAGE_MAX_HAMMING})\n")
        f.write("=" * 60 + "\n")
        if all_pairs:
            for a, b, pa, pb, dist in all_pairs:
                f.write(f"{a} <-> {b}  dist={dist}\n  {pa}\n  {pb}\n")
        else:
            f.write("None found.\n")

    if all_pairs:
        print(f"  WARNING: {len(all_pairs)} near-duplicate pair(s) across splits!")
        print("  Near-duplicates between train and val/test INFLATE your metrics")
        print("  (the model has basically already seen the 'unseen' images).")
        print("  First few offenders:")
        for a, b, pa, pb, dist in all_pairs[:5]:
            print(f"    {a}<->{b} dist={dist}:\n      {pa}\n      {pb}")
        print("  -> Deduplicate and re-run data_splitter/split.py before")
        print("     trusting any metric from this run.")
    else:
        print("  OK: no cross-split near-duplicates found.")
    print(f"  Report saved to: {leakage_report}")

# LOAD DATASETS
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
NUM_CLASSES = len(class_names)
ALL_LABELS  = list(range(NUM_CLASSES))
print("Classes:", class_names)
print("Number of classes:", NUM_CLASSES)

# CLASS WEIGHTS
name_to_idx = {name: i for i, name in enumerate(class_names)}
y_train = np.array([
    name_to_idx[os.path.basename(os.path.dirname(p))]
    for p in train_ds.file_paths
])

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
print("Class weights:", {i: float(w) for i, w in enumerate(class_weights_array)})
alpha_vals = (class_weights_array / class_weights_array.mean()).tolist()
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", seed=SEED),
    tf.keras.layers.RandomRotation(0.08, fill_mode="reflect", seed=SEED),
    tf.keras.layers.RandomZoom(0.15, fill_mode="reflect", seed=SEED),            # +/-15 %
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=SEED), # +/-10 %
    tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255), seed=SEED),      # +/-20 %
    tf.keras.layers.RandomContrast(0.2, seed=SEED),                              # +/-20 %
], name="augmentation")

preprocess = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="preprocess")

def maybe_cache(ds, name):
    n_images = len(ds.file_paths)
    est_bytes = n_images * IMG_SIZE[0] * IMG_SIZE[1] * 3 * 4
    est_gb = est_bytes / 1024**3
    if est_bytes <= MEM_CACHE_LIMIT_BYTES:
        print(f"Cache [{name}]: {n_images} images (~{est_gb:.2f} GB decoded) -> RAM")
        return ds.cache()
    cache_dir = os.path.join(script_dir, "dataset_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{name}_{IMG_SIZE[0]}.tfrecord")
    print(f"Cache [{name}]: {n_images} images (~{est_gb:.2f} GB decoded) too big "
          f"for RAM -> disk cache {cache_file}")
    return ds.cache(cache_file)

train_ds = (
    maybe_cache(train_ds, "train")
    .shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_ds = (
    maybe_cache(val_ds, "valid")
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_ds = (
    maybe_cache(test_ds, "test")
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# FOCAL LOSS
focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    alpha=alpha_vals
)

# METRICS
METRICS = [
    "accuracy",
    tf.keras.metrics.AUC(name="auc", multi_label=True,
                         num_labels=NUM_CLASSES),
]

# BASE MODEL
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

# MODEL ARCHITECTURE
inputs  = tf.keras.Input(shape=IMG_SIZE + (3,))
x = augmentation(inputs)
x = preprocess(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.BatchNormalization()(x)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = tf.keras.layers.Dense(128, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=focal_loss,
    metrics=METRICS
)
model.summary()

# CALLBACKS
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    verbose=1
)
earlystop_cb = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reducelr_cb = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=3,
    min_lr=1e-6,
    verbose=1
)

callbacks_phase1 = [checkpoint_cb, earlystop_cb, reducelr_cb]

# PHASE 1 — TRAIN HEAD
print("\n===== PHASE 1: Training head (base frozen) =====")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=15,
    callbacks=callbacks_phase1
)

# PHASE 2 — FINE-TUNE
print("\n===== PHASE 2: Fine-tuning top 40 layers =====")
best_val_auc_phase1 = float(np.max(history1.history["val_auc"]))

checkpoint_cb_p2 = tf.keras.callbacks.ModelCheckpoint(
    best_model_path,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    initial_value_threshold=best_val_auc_phase1,
    verbose=1
)
earlystop_cb_p2 = tf.keras.callbacks.EarlyStopping(
    monitor="val_auc",
    mode="max",
    patience=4,
    restore_best_weights=True,
    verbose=1
)
callbacks_phase2 = [checkpoint_cb_p2, earlystop_cb_p2]

base_model.trainable = True

for i, layer in enumerate(base_model.layers):
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    elif i < len(base_model.layers) - 40:
        layer.trainable = False

FINETUNE_EPOCHS = 10

lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-4,
    decay_steps=FINETUNE_EPOCHS * len(train_ds)
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=focal_loss,
    metrics=METRICS
)

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=FINETUNE_EPOCHS,
    callbacks=callbacks_phase2
)

# SAVE + PLOT TRAINING HISTORY
combined_history = {}
for key in history1.history.keys():
    if key in history2.history:
        combined_history[key] = history1.history[key] + history2.history[key]

with open(os.path.join(history_dir, "training_history.pkl"), "wb") as f:
    pickle.dump(combined_history, f)

phase1_epochs_ran = len(history1.history["loss"])

best_idx   = int(np.argmax(combined_history["val_auc"]))
best_epoch = best_idx + 1
best_phase = 1 if best_idx < phase1_epochs_ran else 2
print(f"\nBest epoch (highest val_auc): global epoch {best_epoch} "
      f"(phase {best_phase}; note phase-2 console logs restart at 1)")

acc      = combined_history["accuracy"]
val_acc  = combined_history["val_accuracy"]
loss     = combined_history["loss"]
val_loss = combined_history["val_loss"]
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, 'b-', label='Training loss')
plt.plot(epochs_range, val_loss, 'r-', label='Validation loss')
plt.axvline(x=phase1_epochs_ran + 0.5, color='gray', linestyle='--', linewidth=1, label='Fine-tuning starts')
plt.axvline(x=best_epoch, color='green', linestyle=':', linewidth=1.5, label='Best checkpoint')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, acc, 'b-', label='Training accuracy')
plt.plot(epochs_range, val_acc, 'r-', label='Validation accuracy')
plt.axvline(x=phase1_epochs_ran + 0.5, color='gray', linestyle='--', linewidth=1, label='Fine-tuning starts')
plt.axvline(x=best_epoch, color='green', linestyle=':', linewidth=1.5, label='Best checkpoint')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
save_fig("training_curves.png")
print(f"Training curves saved to: {history_dir}")

# LOAD BEST MODEL
print("\nLoading best checkpoint ...")
model = tf.keras.models.load_model(best_model_path)

# TEST SET EVALUATION
print("\n===== Test set evaluation =====")
results = model.evaluate(test_ds, verbose=1, return_dict=True)
for name, value in results.items():
    print(f"  {name}: {value:.4f}")

y_pred_probs  = model.predict(test_ds, verbose=1)
y_pred        = np.argmax(y_pred_probs, axis=1)
y_true_onehot = np.concatenate([labels.numpy() for _, labels in test_ds])
y_true        = np.argmax(y_true_onehot, axis=1)

# CONFUSION MATRIX (RAW + NORMALIZED HEATMAPS)
print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
print(cm)

row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(
    cm.astype(float), row_sums,
    out=np.zeros(cm.shape, dtype=float), where=row_sums != 0
)

CLASSES_PER_IMAGE = 10

def plot_confusion_chunks(matrix, title, prefix, normalized=False):
    n_parts = int(np.ceil(NUM_CLASSES / CLASSES_PER_IMAGE))
    for part, start in enumerate(range(0, NUM_CLASSES, CLASSES_PER_IMAGE), start=1):
        idx     = list(range(start, min(start + CLASSES_PER_IMAGE, NUM_CLASSES)))
        outside = [j for j in range(NUM_CLASSES) if j not in idx]
        names   = [class_names[i] for i in idx]

        block = matrix[np.ix_(idx, idx)]
        other = matrix[np.ix_(idx, outside)].sum(axis=1, keepdims=True)
        data  = np.hstack([block, other])

        plt.figure(figsize=(11, 8))
        sns.heatmap(
            data,
            annot=True,
            fmt=".2f" if normalized else "d",
            cmap="Blues",
            xticklabels=names + ["Other"],
            yticklabels=names,
            vmin=0,
            vmax=1 if normalized else None
        )
        plt.title(f"{title} (Part {part} of {n_parts})")
        plt.xlabel("Predicted  (Other = predicted as a class outside this group)")
        plt.ylabel("Actual")
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        plt.tight_layout()
        save_fig(f"{prefix}_part{part}.png")

plot_confusion_chunks(cm, "Confusion Matrix", "confusion_matrix")
print(f"Confusion matrix (raw counts, {CLASSES_PER_IMAGE} classes per image) saved to: {history_dir}")

plot_confusion_chunks(cm_normalized, "Normalized Confusion Matrix",
                      "confusion_matrix_normalized", normalized=True)
print(f"Confusion matrix (normalized, {CLASSES_PER_IMAGE} classes per image) saved to: {history_dir}")

#Top confused class pairs 
print("\n===== TOP 10 MOST CONFUSED CLASS PAIRS =====")
confused_pairs = []
for i in range(NUM_CLASSES):
    for j in range(NUM_CLASSES):
        if i != j and cm[i, j] > 0:
            confused_pairs.append((class_names[i], class_names[j], cm[i, j]))

confused_pairs.sort(key=lambda p: p[2], reverse=True)

for actual, predicted, count in confused_pairs[:10]:
    print(f"  Actual: {actual:25s} -> Predicted: {predicted:25s} | Count: {count}")

pairs_path = os.path.join(history_dir, "top_confused_pairs.txt")
with open(pairs_path, "w", encoding="utf-8") as f:
    f.write("Top confused class pairs (Actual -> Predicted : Count)\n")
    f.write("=" * 60 + "\n")
    for actual, predicted, count in confused_pairs[:20]:
        f.write(f"{actual} -> {predicted} : {count}\n")
print(f"\nTop confused pairs saved to: {pairs_path}")

TOP_N_CONFUSED_CLASSES = 10  

involved_classes = []
for actual, predicted, count in confused_pairs:
    for c in (actual, predicted):
        if c not in involved_classes and len(involved_classes) < TOP_N_CONFUSED_CLASSES:
            involved_classes.append(c)
    if len(involved_classes) >= TOP_N_CONFUSED_CLASSES:
        break

if involved_classes:
    involved_indices = [class_names.index(c) for c in involved_classes]
    cm_small = cm[np.ix_(involved_indices, involved_indices)]

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_small,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=involved_classes,
        yticklabels=involved_classes
    )
    plt.title(f"Confusion Matrix (Top {len(involved_classes)} Most Confused Species)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    save_fig("confusion_matrix_top_confused.png")
    print(f"Top-confused-classes heatmap saved to: {history_dir}")
else:
    # A perfect test score has no confusions, and an empty heatmap would crash
    print("No misclassifications on the test set, so no top-confused heatmap was made.")

# CLASSIFICATION REPORT
print("\n===== CLASSIFICATION REPORT =====")
report_text = classification_report(
    y_true, y_pred,
    labels=ALL_LABELS, target_names=class_names,
    digits=4, zero_division=0
)
print(report_text)
with open(os.path.join(history_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_text)

# PER-CLASS ROC-AUC
print("\n===== Per-class ROC-AUC =====")
try:
    roc_aucs = roc_auc_score(y_true_onehot, y_pred_probs, average=None)
    for cls, auc in zip(class_names, roc_aucs):
        print(f"  AUC [{cls}]: {auc:.4f}")
    macro_auc = roc_auc_score(y_true_onehot, y_pred_probs, average="macro")
    print(f"  Macro AUC: {macro_auc:.4f}")
except ValueError as e:
    macro_auc = None
    print(f"  Could not compute per-class AUC: {e}")

print("\n===== Probability calibration (temperature scaling) =====")

calibration_done = False
macro_auc_cal = None
best_T = 1.0

try:
    val_pred_probs = model.predict(val_ds, verbose=1)
    val_true_onehot = np.concatenate([labels.numpy() for _, labels in val_ds])

    def temperature_scale(probs, T):
        p = np.power(probs, 1.0 / T)
        return p / p.sum(axis=1, keepdims=True)

    def nll(onehot, probs):
        p = np.clip(probs, 1e-12, 1.0)
        return float(-np.mean(np.sum(onehot * np.log(p), axis=1)))

    val_nll_before = nll(val_true_onehot, val_pred_probs)
    best_T, best_val_nll = 1.0, val_nll_before
    for T in np.arange(0.5, 3.01, 0.05):
        n = nll(val_true_onehot, temperature_scale(val_pred_probs, T))
        if n < best_val_nll:
            best_val_nll, best_T = n, float(T)

    test_probs_cal  = temperature_scale(y_pred_probs, best_T)
    test_nll_before = nll(y_true_onehot, y_pred_probs)
    test_nll_after  = nll(y_true_onehot, test_probs_cal)
    try:
        macro_auc_cal = roc_auc_score(y_true_onehot, test_probs_cal, average="macro")
    except ValueError:
        pass

    print(f"  Best temperature (val NLL): T = {best_T:.2f}")
    print(f"  Val  NLL:  {val_nll_before:.4f} -> {nll(val_true_onehot, temperature_scale(val_pred_probs, best_T)):.4f}")
    print(f"  Test NLL:  {test_nll_before:.4f} -> {test_nll_after:.4f}")
    print("  Predicted classes (argmax) are unchanged; only confidences move.")
    if macro_auc_cal is not None:
        print(f"  Test macro AUC: {macro_auc:.4f} -> {macro_auc_cal:.4f}")

    calibration = {
        "temperature": float(best_T),
        "formula": "calibrated_prob = p**(1/T); calibrated_prob /= sum_over_classes",
        "note": "argmax (predicted class) is unchanged; only confidence values are calibrated",
        "val_nll_before": val_nll_before,
        "val_nll_after": float(best_val_nll),
    }
    with open(os.path.join(model_dir, "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calibration, f, indent=2)
    print(f"  Calibration saved to: {os.path.join(model_dir, 'calibration.json')}")
    calibration_done = True
except Exception as e:
    print(f"  Calibration skipped: {type(e).__name__}: {e}")

# FINAL SUMMARY TABLE 
report_dict = classification_report(
    y_true, y_pred,
    labels=ALL_LABELS, target_names=class_names,
    output_dict=True, zero_division=0
)
macro_avg       = report_dict["macro avg"]
weighted_avg    = report_dict["weighted avg"]
overall_accuracy = report_dict["accuracy"]

ch = combined_history
b  = best_idx

def row(name, value):
    return f"{name:<25}{value:.4f}"

summary_lines = [
    "=" * 50,
    "FINAL SUMMARY TABLE",
    "=" * 50,
    f"Best checkpoint: global epoch {best_epoch} of {len(acc)} "
    f"(phase {best_phase}, highest macro val_auc)",
    "-" * 35,
    f"{'Metric':<25}{'Value':<10}",
    "-" * 35,
    row("Train Accuracy",  ch["accuracy"][b]),
    row("Train Loss",      ch["loss"][b]),
    row("Train Macro AUC", ch["auc"][b]),
    "-" * 35,
    row("Val Accuracy",    ch["val_accuracy"][b]),
    row("Val Loss",        ch["val_loss"][b]),
    row("Val Macro AUC",   ch["val_auc"][b]),
    "-" * 35,
    row("Test Accuracy",           overall_accuracy),
    row("Test Macro Precision",    macro_avg["precision"]),
    row("Test Macro Recall",       macro_avg["recall"]),
    row("Test Macro F1-score",     macro_avg["f1-score"]),
    row("Test Weighted Precision", weighted_avg["precision"]),
    row("Test Weighted Recall",    weighted_avg["recall"]),
    row("Test Weighted F1-score",  weighted_avg["f1-score"]),
]
if macro_auc is not None:
    summary_lines.append(row("Test Macro AUC", macro_auc))
if calibration_done:
    summary_lines.append(row("Temperature (T)", best_T))
    if macro_auc_cal is not None:
        summary_lines.append(row("Test Macro AUC (cal)", macro_auc_cal))
summary_lines.append("=" * 50)

print("\n" + "\n".join(summary_lines))

summary_path = os.path.join(history_dir, "final_summary_table.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"\nFinal summary table also saved to: {summary_path}")

# SAVE FINAL MODEL 
model.save(model_path)
print(f"\nModel saved to: {model_path}")

class_names_path = os.path.join(model_dir, "class_names.json")
with open(class_names_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)
print(f"Class names saved to: {class_names_path}")