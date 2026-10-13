import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import datetime
import pickle

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# CONSTANTS
IMG_SIZE   = (224, 224)
BATCH_SIZE = 32
SEED       = 42

# True  = pop up every figure (the script PAUSES until you close each window)
# False = just save the figures to the run folder and keep going
SHOW_PLOTS = False

# PATHS
script_dir = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(script_dir, "dataset", "train")
val_dir   = os.path.join(script_dir, "dataset", "valid")
test_dir  = os.path.join(script_dir, "dataset", "test")

model_dir = os.path.join(script_dir, "model")
os.makedirs(model_dir, exist_ok=True)

model_path      = os.path.join(model_dir, "medicleaf_detector_mobilenetv2.keras")
best_model_path = os.path.join(model_dir, "best_model.keras")

# TIMESTAMPED RUN FOLDER
# Each run gets its own folder (e.g. history/20260907_143022/) so that
# previous runs are NEVER overwritten. Compare multiple runs side by side later.
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

# LOAD DATASETS
raw_train = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    label_mode="categorical",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

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
y_train = np.concatenate(
    [np.argmax(labels.numpy(), axis=1) for _, labels in raw_train]
)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)
print("Class weights:", {i: float(w) for i, w in enumerate(class_weights_array)})

# Focal-loss alpha: inverse-frequency weights, so RARE classes count MORE.
# (The old formula 1 / (weight * NUM_CLASSES) simply gave class frequency,
# which up-weighted the most common classes instead.)
# Scaled to a mean of 1 so the overall loss magnitude stays comparable.
alpha_vals = (class_weights_array / class_weights_array.mean()).tolist()

# AUGMENTATION + PREPROCESSING (built into the model)
# Keras random layers sample a NEW random value for EVERY image
# (unlike tf.image.random_brightness/contrast, which used one value per batch).
# They are only active during training and are skipped at inference.
# Input images stay on the 0-255 scale until the Rescaling layer.
augmentation = tf.keras.Sequential([
    # Geometric: leaves have no canonical orientation or framing
    tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
    tf.keras.layers.RandomRotation(0.25, fill_mode="reflect", seed=SEED),        # +/-90 degrees
    tf.keras.layers.RandomZoom(0.15, fill_mode="reflect", seed=SEED),            # +/-15 %
    tf.keras.layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=SEED), # +/-10 %
    # Photometric: kept mild because color is a real signal for species/disease.
    # Saturation jitter is intentionally NOT used.
    tf.keras.layers.RandomBrightness(0.2, value_range=(0, 255), seed=SEED),      # +/-20 %
    tf.keras.layers.RandomContrast(0.2, seed=SEED),                              # +/-20 %
], name="augmentation")

# Equivalent to mobilenet_v2.preprocess_input: scales 0-255 to [-1, 1].
# Being part of the model, the saved .keras file accepts raw 0-255 images.
preprocess = tf.keras.layers.Rescaling(1 / 127.5, offset=-1, name="preprocess")

# No .map() step needed anymore: just cache, shuffle and prefetch.
train_ds = (
    train_ds
    .cache()
    .shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

val_ds = (
    val_ds
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

test_ds = (
    test_ds
    .cache()
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

# FOCAL LOSS
focal_loss = tf.keras.losses.CategoricalFocalCrossentropy(
    gamma=2.0,
    alpha=alpha_vals
)

METRICS = [
    "accuracy",
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
    tf.keras.metrics.AUC(name="auc")
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
callbacks_phase2 = [checkpoint_cb, earlystop_cb]

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
# Merge phase 1 + phase 2 histories into one continuous record
combined_history = {}
for key in history1.history.keys():
    if key in history2.history:
        combined_history[key] = history1.history[key] + history2.history[key]

# Save raw history as pickle (so you never lose it again)
with open(os.path.join(history_dir, "training_history.pkl"), "wb") as f:
    pickle.dump(combined_history, f)

# Number of epochs actually run in phase 1 (for the phase-boundary marker)
phase1_epochs_ran = len(history1.history["loss"])

# The checkpoint (best_model.keras) is the epoch with the highest val_auc across
# BOTH phases, so that is the epoch every reported number should come from.
best_idx   = int(np.argmax(combined_history["val_auc"]))
best_epoch = best_idx + 1
print(f"\nBest epoch (highest val_auc): {best_epoch}")

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
# (compiled, so that evaluate() reports loss + the METRICS)
print("\nLoading best checkpoint ...")
model = tf.keras.models.load_model(best_model_path)

# TEST SET EVALUATION
print("\n===== Test set evaluation =====")
# return_dict=True: with Keras 3, model.metrics_names is ['loss', 'compile_metrics'],
# so zip(metrics_names, results) mislabels the values and drops most of them.
results = model.evaluate(test_ds, verbose=1, return_dict=True)
for name, value in results.items():
    print(f"  {name}: {value:.4f}")

y_pred_probs  = model.predict(test_ds, verbose=1)
y_pred        = np.argmax(y_pred_probs, axis=1)
y_true_onehot = np.concatenate([labels.numpy() for _, labels in test_ds])
y_true        = np.argmax(y_true_onehot, axis=1)

# CONFUSION MATRIX (RAW + NORMALIZED HEATMAPS)
# labels=ALL_LABELS keeps the matrix NUM_CLASSES x NUM_CLASSES even if a class
# never appears in y_true / y_pred, so it always lines up with class_names.
print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(y_true, y_pred, labels=ALL_LABELS)
print(cm)

row_sums = cm.sum(axis=1, keepdims=True)
cm_normalized = np.divide(
    cm.astype(float), row_sums,
    out=np.zeros(cm.shape, dtype=float), where=row_sums != 0
)

# Split the full matrix into several readable images, CLASSES_PER_IMAGE classes each
# (40 classes / 10 = 4 images). For each group:
#   rows    = the actual classes in the group
#   columns = the same classes (predicted) + one extra "Other" column that adds up
#             the predictions that went to classes OUTSIDE the group,
#             so no error is hidden and every row still sums to its full total.
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

# --- Top confused class pairs (for discussion section) ---
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

# --- Smaller, readable heatmap: only the most-confused classes, WITH numbers ---
# The full heatmap is too cramped to read in print, so this keeps just the
# classes that appear most in the top confused pairs and prints the count in
# each cell. It is a sub-matrix: errors that go to classes outside this
# selection are not shown (worth a line in the figure caption).
TOP_N_CONFUSED_CLASSES = 10  # change this to show more/fewer classes

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

# FINAL SUMMARY TABLE (printed + saved)
# Train/Val rows come from the BEST epoch (the one saved as best_model.keras),
# not the last epoch, so they describe the same model as the Test rows.
# Note: the training metrics are running averages logged WHILE training
# (augmentation + dropout on), so they are not directly comparable to the
# test numbers, which are measured on the finished model.
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
    f"Best checkpoint: epoch {best_epoch} of {len(acc)} (highest val_auc)",
    "-" * 35,
    f"{'Metric':<25}{'Value':<10}",
    "-" * 35,
    row("Train Accuracy",  ch["accuracy"][b]),
    row("Train Loss",      ch["loss"][b]),
    row("Train Precision", ch["precision"][b]),
    row("Train Recall",    ch["recall"][b]),
    row("Train AUC",       ch["auc"][b]),
    "-" * 35,
    row("Val Accuracy",    ch["val_accuracy"][b]),
    row("Val Loss",        ch["val_loss"][b]),
    row("Val Precision",   ch["val_precision"][b]),
    row("Val Recall",      ch["val_recall"][b]),
    row("Val AUC",         ch["val_auc"][b]),
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
summary_lines.append("=" * 50)

print("\n" + "\n".join(summary_lines))

summary_path = os.path.join(history_dir, "final_summary_table.txt")
with open(summary_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines) + "\n")
print(f"\nFinal summary table also saved to: {summary_path}")

# SAVE FINAL MODEL (copy of the best checkpoint)
model.save(model_path)
print(f"\nModel saved to: {model_path}")