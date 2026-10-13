from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path


# CONFIG
SEED         = 42
TRAIN_RATIO  = 0.40
VALID_RATIO  = 0.30
TEST_RATIO   = 0.30

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
SPLIT_NAMES  = ("train", "valid", "test")


# PATHS
# split.py lives in: backend/data_splitter/split.py
# dataset lives in:  backend/machine_learning/dataset/
# So we go up one level (backend/) then into machine_learning/dataset
script_dir   = Path(__file__).resolve().parent
backend_dir  = script_dir.parent

dataset_dir  = backend_dir / "machine_learning" / "dataset"

train_dir    = dataset_dir / "train"
valid_dir    = dataset_dir / "valid"
test_dir     = dataset_dir / "test"

combined_dir = dataset_dir / "_combined"


# STEP 1: Combine train/valid/test into one folder per class
# Uses file hashing to skip exact duplicate files
def get_file_hash(filepath: Path) -> str:
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def combine_splits() -> set[str]:
    print("=" * 60)
    print("STEP 1: Combining train/valid/test into one pool per class")
    print("=" * 60)

    all_classes: set[str] = set()
    for split_dir in (train_dir, valid_dir, test_dir):
        if split_dir.is_dir():
            all_classes.update(
                p.name for p in split_dir.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )

    if not all_classes:
        raise ValueError(f"No class folders found in train/valid/test under {dataset_dir}")

    combined_dir.mkdir(parents=True, exist_ok=True)

    for class_name in sorted(all_classes):
        class_combined_dir = combined_dir / class_name
        class_combined_dir.mkdir(parents=True, exist_ok=True)

        seen_hashes: set[str] = set()
        counter = 0

        for split_dir in (train_dir, valid_dir, test_dir):
            class_split_dir = split_dir / class_name
            if not class_split_dir.is_dir():
                continue

            for file_path in sorted(class_split_dir.iterdir()):
                if not file_path.is_file():
                    continue
                if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                file_hash = get_file_hash(file_path)

                if file_hash in seen_hashes:
                    continue

                seen_hashes.add(file_hash)
                counter += 1

                ext = file_path.suffix
                new_name = f"{class_name}_{counter:04d}{ext}"
                shutil.copy2(file_path, class_combined_dir / new_name)

        print(f"  {class_name}: {counter} unique images combined")

    return all_classes


# STEP 2: Wipe old train/valid/test folders
def clear_old_splits() -> None:
    print("\n" + "=" * 60)
    print("STEP 2: Removing old train/valid/test folders")
    print("=" * 60)

    for split_dir in (train_dir, valid_dir, test_dir):
        if split_dir.is_dir():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents=True, exist_ok=True)

    print("  Old splits cleared.")


# STEP 3: Re-split combined pool into train/valid/test
def split_files(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    shuffled = files.copy()
    random.shuffle(shuffled)

    total = len(shuffled)
    train_count = int(total * TRAIN_RATIO)
    valid_count = int(total * VALID_RATIO)

    train_files = shuffled[:train_count]
    valid_files = shuffled[train_count : train_count + valid_count]
    test_files  = shuffled[train_count + valid_count :]

    return train_files, valid_files, test_files


def transfer_files(files: list[Path], destination_dir: Path) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for file_path in files:
        target = destination_dir / file_path.name
        shutil.copy2(file_path, target)
        count += 1
    return count


def resplit_dataset(all_classes: set[str]) -> None:
    print("\n" + "=" * 60)
    print(f"STEP 3: Re-splitting -> train {TRAIN_RATIO*100:.0f}% / "
          f"valid {VALID_RATIO*100:.0f}% / test {TEST_RATIO*100:.0f}%")
    print("=" * 60)

    random.seed(SEED)

    for class_name in sorted(all_classes):
        class_combined_dir = combined_dir / class_name
        files = sorted(p for p in class_combined_dir.iterdir() if p.is_file())

        if not files:
            print(f"[skip] {class_name}: no images found")
            continue

        train_files, valid_files, test_files = split_files(files)

        counts = {
            "train": transfer_files(train_files, train_dir / class_name),
            "valid": transfer_files(valid_files, valid_dir / class_name),
            "test":  transfer_files(test_files,  test_dir / class_name),
        }

        print(
            f"  {class_name}: {len(files)} total -> "
            f"train={counts['train']}, valid={counts['valid']}, test={counts['test']}"
        )


# STEP 4: Clean up temporary combined folder
def cleanup() -> None:
    print("\n" + "=" * 60)
    print("STEP 4: Cleaning up temporary files")
    print("=" * 60)
    shutil.rmtree(combined_dir)
    print("  Done.")


# MAIN
def main() -> None:
    ratio_sum = TRAIN_RATIO + VALID_RATIO + TEST_RATIO
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {ratio_sum:.4f}")

    print(f"Dataset path: {dataset_dir.resolve()}")

    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    all_classes = combine_splits()
    clear_old_splits()
    resplit_dataset(all_classes)
    cleanup()

    print("\n" + "=" * 60)
    print("DONE — train/valid/test re-split with zero overlap.")
    print("=" * 60)


if __name__ == "__main__":
    main()