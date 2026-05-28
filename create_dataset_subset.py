"""
UCF101 Dataset Subset Creator

This script creates a subset of the full UCF101 dataset
containing only the 8 activity classes we need for training.

Usage:
    python create_dataset_subset.py --source /path/to/full/ucf101 --target ./data/raw

It will:
1. Find the 8 required classes
2. Copy them to target directory
3. Create train/val/test splits
4. Generate statistics
"""

import os
import shutil
import json
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import defaultdict
import random

# ==================== CONFIGURATION ====================

TARGET_CLASSES = [
    "WalkingWithDog",
    "JumpingJack",
    "Punch",
    "Basketball",
    "HorseRiding",
    "PushUps",
    "TaiChi",
    "SoccerJuggling",
]

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ==================== FUNCTIONS ====================


def find_classes_in_source(source_dir):
    """
    Find all available classes in source UCF101 directory

    Returns:
        dict: {class_name: path_to_class_dir}
    """
    print(f"\nSearching for classes in: {source_dir}")

    classes_found = {}
    source_path = Path(source_dir)

    if not source_path.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    # UCF101 structure: UCF101/ClassName/*.avi
    for item in source_path.iterdir():
        if item.is_dir():
            class_name = item.name
            classes_found[class_name] = item

    print(f"Found {len(classes_found)} total classes")

    return classes_found


def find_target_classes(source_classes):
    """
    Find which target classes exist in source

    Returns:
        dict: {target_class: source_path}
    """
    found_targets = {}
    missing_targets = []

    print(f"\nLooking for target classes...")

    for target_class in TARGET_CLASSES:
        found = False

        # Exact match
        if target_class in source_classes:
            found_targets[target_class] = source_classes[target_class]
            found = True
        else:
            # Try case-insensitive match
            for source_class, source_path in source_classes.items():
                if source_class.lower() == target_class.lower():
                    found_targets[target_class] = source_path
                    found = True
                    break

        if found:
            print(f"  ✓ {target_class}")
        else:
            print(f"  ✗ {target_class} (NOT FOUND)")
            missing_targets.append(target_class)

    print(f"\nFound: {len(found_targets)}/{len(TARGET_CLASSES)} target classes")

    if missing_targets:
        print(f"Missing classes: {missing_targets}")

    return found_targets


def count_videos_in_class(class_dir):
    """Count video files in a class directory"""
    video_extensions = {".avi", ".mp4", ".mov", ".mkv", ".flv"}

    count = 0
    for file in Path(class_dir).iterdir():
        if file.suffix.lower() in video_extensions:
            count += 1

    return count


def create_splits(video_files, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Create train/val/test splits from video files

    Returns:
        dict: {'train': [], 'val': [], 'test': []}
    """
    # Shuffle
    random.seed(RANDOM_SEED)
    video_files = list(video_files)
    random.shuffle(video_files)

    total = len(video_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    return {
        "train": video_files[:train_count],
        "val": video_files[train_count : train_count + val_count],
        "test": video_files[train_count + val_count :],
    }


def copy_class_videos(source_class_dir, target_class_dir, copy_files=True):
    """
    Copy all videos from source class to target class

    Returns:
        dict: Split information
    """
    target_path = Path(target_class_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Find all video files
    video_extensions = {".avi", ".mp4", ".mov", ".mkv", ".flv"}
    video_files = []

    source_path = Path(source_class_dir)
    for file in source_path.iterdir():
        if file.suffix.lower() in video_extensions:
            video_files.append(file)

    print(f"    Found {len(video_files)} videos in {source_path.name}")

    # Create splits
    splits = create_splits(video_files)

    # Copy files if requested
    if copy_files:
        copied_count = 0
        for split_name, files in splits.items():
            split_dir = target_path / split_name
            split_dir.mkdir(parents=True, exist_ok=True)

            for src_file in files:
                dst_file = split_dir / src_file.name
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)
                    copied_count += 1

        print(f"    Copied {copied_count} videos")

    return splits


def create_subset(source_dir, target_dir, copy_files=True):
    """
    Main function to create dataset subset

    Args:
        source_dir: Path to full UCF101 dataset
        target_dir: Path where to create subset
        copy_files: If True, actually copy files. If False, just prepare structure.
    """
    print("\n" + "=" * 70)
    print("UCF101 DATASET SUBSET CREATOR")
    print("=" * 70)

    # Find all classes in source
    source_classes = find_classes_in_source(source_dir)

    # Find target classes
    found_targets = find_target_classes(source_classes)

    if not found_targets:
        print("\nERROR: No target classes found!")
        return False

    # Create subset directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 70)
    print("CREATING SUBSET")
    print("=" * 70)

    statistics = {}

    # Copy each class
    for target_class, source_class_path in found_targets.items():
        print(f"\nProcessing: {target_class}")

        class_target_dir = target_path / target_class

        if copy_files:
            splits = copy_class_videos(
                source_class_path, class_target_dir, copy_files=True
            )
            print(
                f"    Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
            )
        else:
            splits = copy_class_videos(
                source_class_path, class_target_dir, copy_files=False
            )
            print(
                f"    [DRY RUN] Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}"
            )

        statistics[target_class] = {
            "total": sum(len(v) for v in splits.values()),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        }

    # Generate statistics report
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)

    total_videos = 0
    total_train = 0
    total_val = 0
    total_test = 0

    for class_name, stats in sorted(statistics.items()):
        print(f"\n{class_name}:")
        print(f"  Total:  {stats['total']:3d}")
        print(f"  Train:  {stats['train']:3d}")
        print(f"  Val:    {stats['val']:3d}")
        print(f"  Test:   {stats['test']:3d}")

        total_videos += stats["total"]
        total_train += stats["train"]
        total_val += stats["val"]
        total_test += stats["test"]

    print("\n" + "-" * 70)
    print(f"TOTAL:")
    print(f"  Total:  {total_videos}")
    print(f"  Train:  {total_train} ({100*total_train/total_videos:.1f}%)")
    print(f"  Val:    {total_val} ({100*total_val/total_videos:.1f}%)")
    print(f"  Test:   {total_test} ({100*total_test/total_videos:.1f}%)")

    # Save statistics
    stats_file = target_path / "statistics.json"
    with open(stats_file, "w") as f:
        json.dump(
            {
                "classes": len(found_targets),
                "total_videos": total_videos,
                "train_videos": total_train,
                "val_videos": total_val,
                "test_videos": total_test,
                "class_statistics": statistics,
                "split_ratio": {
                    "train": TRAIN_SPLIT,
                    "val": VAL_SPLIT,
                    "test": TEST_SPLIT,
                },
            },
            f,
            indent=2,
        )

    print(f"\nStatistics saved to: {stats_file}")

    print("\n" + "=" * 70)
    print("SUBSET CREATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nSubset location: {target_path}")

    return True


# ==================== MAIN ====================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a subset of UCF101 dataset with specific classes"
    )

    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to full UCF101 dataset directory",
    )

    parser.add_argument(
        "--target",
        type=str,
        default="./data/raw",
        help="Path where to create subset (default: ./data/raw)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually copying files",
    )

    args = parser.parse_args()

    # Validate source directory
    if not os.path.exists(args.source):
        print(f"ERROR: Source directory not found: {args.source}")
        exit(1)

    # Create subset
    copy_files = not args.dry_run
    success = create_subset(args.source, args.target, copy_files=copy_files)

    if success:
        print("\n✓ Done!")
        if args.dry_run:
            print("\nTo actually copy files, run without --dry-run flag:")
            print(
                f"  python create_dataset_subset.py --source {args.source} --target {args.target}"
            )
    else:
        print("\n✗ Failed!")
        exit(1)
