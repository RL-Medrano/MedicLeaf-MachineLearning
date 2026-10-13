import os
import json
import shutil
import tempfile
from datetime import datetime

import numpy as np
import tensorflow as tf

# ──────────────────────────────────────────────
# PATHS — matched to train.py conventions
# ──────────────────────────────────────────────
IMG_SIZE = (224, 224)
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".gif")

script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "model", "best_model.keras")
train_dir  = os.path.join(script_dir, "dataset", "train")

tflite_output_path = os.path.join(script_dir, "model", "medicleaf_model.tflite")
labels_output_path = os.path.join(script_dir, "model", "labels.json")

# History snapshots — one folder per run, so a retrain that turns out worse
# doesn't silently destroy the previous good model.
history_dir = os.path.join(script_dir, "model", "history")

# CLASS NAMES — sorted sub-folders of dataset/train, which is exactly
# how image_dataset_from_directory orders them in train.py.
if not os.path.isdir(train_dir):
    raise FileNotFoundError(f"Could not find dataset/train at: {train_dir}")

class_labels = sorted(
    d for d in os.listdir(train_dir)
    if os.path.isdir(os.path.join(train_dir, d))
)
if not class_labels:
    raise ValueError(f"No class folders found in: {train_dir}")

print(f"Found {len(class_labels)} classes:")
print(class_labels)

# LOAD MODEL
# compile=False: only the forward pass is needed.
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Could not find trained model at: {model_path}")

print(f"\nLoading model from: {model_path}")
model = tf.keras.models.load_model(model_path, compile=False)

n_outputs = model.output_shape[-1]
if n_outputs != len(class_labels):
    raise ValueError(
        f"Model has {n_outputs} outputs but found {len(class_labels)} class "
        f"folders in {train_dir}. The labels would not line up."
    )

# train.py puts the preprocessing INSIDE the model:
#   augmentation -> Rescaling(1/127.5, offset=-1) "preprocess" -> MobileNetV2 ...
# So the saved model already accepts RAW 0-255 pixels. We must NOT add another
# scaling step anywhere (export wrapper, verification, or the app).
pre_layers = [
    l.name for l in model.layers
    if isinstance(l, (tf.keras.layers.Rescaling, tf.keras.layers.Normalization))
]
if pre_layers:
    print(f"OK: model has its own preprocessing layer(s) {pre_layers}; "
          "it takes raw 0-255 pixels, so no extra scaling is added.")
else:
    raise ValueError(
        "The model has no Rescaling/Normalization layer. This converter assumes "
        "the model scales pixels itself (as train.py does)."
    )

# CONVERT TO TFLITE
# The model is exported as-is (no extra Rescaling wrapper).
# float16 quantization roughly halves the file size; input and output
# stay float32.
export_model = model

def configure(converter):
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    return converter

print("\nConverting to TFLite...")
try:
    tflite_model = configure(
        tf.lite.TFLiteConverter.from_keras_model(export_model)
    ).convert()
except Exception as e:
    print(f"from_keras_model failed ({type(e).__name__}: {e})")
    print("Retrying via SavedModel export...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        export_model.export(tmp_dir)
        tflite_model = configure(
            tf.lite.TFLiteConverter.from_saved_model(tmp_dir)
        ).convert()

with open(tflite_output_path, "wb") as f:
    f.write(tflite_model)

# SAVE LABELS — in prediction-index order
with open(labels_output_path, "w", encoding="utf-8") as f:
    json.dump(class_labels, f, ensure_ascii=False, indent=2)

file_size_mb = os.path.getsize(tflite_output_path) / (1024 * 1024)
print("\nDone.")
print(f"Model saved to:  {tflite_output_path} ({file_size_mb:.2f} MB)")
print(f"Labels saved to: {labels_output_path}")

# VERIFY — feed the SAME raw 0-255 pixels to both models.
# Checks:
#   1. TFLite and Keras probabilities agree (max|diff| <= tolerance)
#   2. TFLite and Keras pick the same class
#   3. The prediction matches the true folder for most samples
def first_image(cls_dir):
    files = sorted(f for f in os.listdir(cls_dir) if f.lower().endswith(IMG_EXTS))
    return os.path.join(cls_dir, files[0]) if files else None

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
print(f"\nTFLite input:  shape={inp['shape'].tolist()} dtype={np.dtype(inp['dtype']).name}")
print(f"TFLite output: shape={out['shape'].tolist()} dtype={np.dtype(out['dtype']).name}")

MAX_DIFF_TOLERANCE = 0.05
MIN_CORRECT_RATIO  = 0.8      # at least 80% of sampled images must be classified correctly

diff_ok_all = True
checked = 0
correct_count = 0

# Every 4th class -> 10 classes spread across the whole list
for cls in class_labels[::4]:
    path = first_image(os.path.join(train_dir, cls))
    if path is None:
        continue

    arr = tf.keras.utils.img_to_array(
        tf.keras.utils.load_img(path, target_size=IMG_SIZE)
    )[None, ...].astype(np.float32)          # raw 0-255 pixels, NO extra scaling

    ref = model.predict(arr.copy(), verbose=0)[0]

    interpreter.set_tensor(inp["index"], arr)
    interpreter.invoke()
    tfl = interpreter.get_tensor(out["index"])[0]

    diff = float(np.abs(ref - tfl).max())
    same = int(ref.argmax()) == int(tfl.argmax())
    correct = class_labels[int(tfl.argmax())] == cls
    ok = (diff <= MAX_DIFF_TOLERANCE) and same

    diff_ok_all &= ok
    correct_count += int(correct)
    checked += 1
    print(f"  [{'OK' if ok else 'FAIL'}] folder='{cls}'  "
          f"Keras='{class_labels[ref.argmax()]}'  TFLite='{class_labels[tfl.argmax()]}'  "
          f"same_class={same}  correct={correct}  max|diff|={diff:.5f}")

if checked == 0:
    print("\nWARNING: no sample images found, so the TFLite model was not verified.")
else:
    ratio = correct_count / checked
    print(f"\nCorrect on {correct_count}/{checked} sampled images ({ratio:.0%}).")
    if not diff_ok_all:
        print("VERIFICATION FAILED: TFLite output differs from the Keras model.")
        raise SystemExit(1)
    if ratio < MIN_CORRECT_RATIO:
        print("VERIFICATION FAILED: the model misclassifies too many training "
              "images. Something is wrong with the model or the preprocessing.")
        raise SystemExit(1)
    print("Verification passed: TFLite matches Keras and predictions are correct.")

# SAVE HISTORY SNAPSHOT (only reached if verification passed)
run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
snapshot_dir = os.path.join(history_dir, run_id)
os.makedirs(snapshot_dir, exist_ok=True)

for src in (model_path, tflite_output_path, labels_output_path):
    shutil.copy2(src, os.path.join(snapshot_dir, os.path.basename(src)))

print(f"\nHistory snapshot saved to: {snapshot_dir}")