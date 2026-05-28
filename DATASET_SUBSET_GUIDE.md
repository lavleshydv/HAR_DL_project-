# 📥 How to Create UCF101 Dataset Subset

## What This Script Does

Creates a lightweight subset of the full UCF101 dataset containing ONLY the 8 activity classes we need:

1. **WalkingWithDog** - Walking while holding dog leash
2. **JumpingJack** - Exercise: jumping jacks
3. **Punch** - Boxing punch movements
4. **Basketball** - Basketball shooting/playing
5. **HorseRiding** - Riding a horse
6. **PushUps** - Exercise: push-ups
7. **TaiChi** - Tai Chi movements
8. **SoccerJuggling** - Soccer ball juggling

Benefits:

- ✅ Extracts only needed classes (faster than using full 101 classes)
- ✅ Automatically creates train/val/test splits (70/15/15)
- ✅ Generates statistics about the dataset
- ✅ Handles case-insensitive matching

---

## Step 1: Get Full UCF101 Dataset

Download the complete UCF101 dataset:

**Option A: Download from Official Source**

```bash
# Visit: https://www.crcv.ucf.edu/research/data-sets/ucf101/
# Download UCF101.rar (~13 GB)
# Extract to a folder (e.g., ~/Downloads/UCF101)
```

**Option B: Use wget/curl**

```bash
# The dataset structure should be:
# UCF101/
#   ├── WalkingWithDog/
#   │   ├── v_WalkingWithDog_g01_c01.avi
#   │   ├── v_WalkingWithDog_g01_c02.avi
#   │   └── ...
#   ├── Basketball/
#   │   ├── v_Basketball_g01_c01.avi
#   │   └── ...
#   └── (99 more classes)
```

---

## Step 2: Run the Subset Script

### **Option A: Dry Run (Recommended First)**

See what will happen WITHOUT copying files:

```bash
python create_dataset_subset.py \
  --source ~/Downloads/UCF101 \
  --target ./data/raw \
  --dry-run
```

Output will show:

- ✓ Which classes found
- ✓ How many videos in each
- ✓ Train/Val/Test splits
- ✓ Total dataset size

### **Option B: Actually Copy Files**

After verifying dry-run output, copy the subset:

```bash
python create_dataset_subset.py \
  --source ~/Downloads/UCF101 \
  --target ./data/raw
```

This will:

1. Find the 8 classes in your dataset
2. Copy them to `./data/raw/`
3. Create train/val/test folders for each class
4. Generate statistics

---

## Example Output

```
======================================================================
UCF101 DATASET SUBSET CREATOR
======================================================================

Searching for classes in: ~/Downloads/UCF101
Found 101 total classes

Looking for target classes...
  ✓ WalkingWithDog
  ✓ JumpingJack
  ✓ Punch
  ✓ Basketball
  ✓ HorseRiding
  ✓ PushUps
  ✓ TaiChi
  ✓ SoccerJuggling

Found: 8/8 target classes

======================================================================
CREATING SUBSET
======================================================================

Processing: WalkingWithDog
    Found 100 videos in WalkingWithDog
    Train: 70, Val: 15, Test: 15

Processing: JumpingJack
    Found 100 videos in JumpingJack
    Train: 70, Val: 15, Test: 15

... (more classes)

======================================================================
DATASET STATISTICS
======================================================================

WalkingWithDog:
  Total:  100
  Train:   70
  Val:     15
  Test:    15

JumpingJack:
  Total:  100
  Train:   70
  Val:     15
  Test:    15

... (more classes)

----------------------------------------------------------------------
TOTAL:
  Total:  800
  Train:  560 (70.0%)
  Val:    120 (15.0%)
  Test:   120 (15.0%)
======================================================================
SUBSET CREATED SUCCESSFULLY!
======================================================================

Subset location: ./data/raw
```

---

## Result Directory Structure

After running the script, your directory will look like:

```
data/
└── raw/
    ├── WalkingWithDog/
    │   ├── train/
    │   │   ├── v_WalkingWithDog_g01_c01.avi
    │   │   ├── v_WalkingWithDog_g01_c02.avi
    │   │   └── ... (70 videos)
    │   ├── val/
    │   │   └── ... (15 videos)
    │   └── test/
    │       └── ... (15 videos)
    ├── Basketball/
    │   ├── train/
    │   ├── val/
    │   └── test/
    ├── ... (6 more classes)
    └── statistics.json
```

---

## Common Issues & Fixes

### Issue: "Source directory not found"

```bash
# Make sure the path is correct:
python create_dataset_subset.py \
  --source "C:/Users/YourName/Downloads/UCF101" \
  --target ./data/raw
```

### Issue: Classes not found

Check the exact folder names in your UCF101 directory. Class names might be:

- `v_WalkingWithDog_g01_c01.avi` (video file)
- But the CLASS should be in a folder called `WalkingWithDog`

### Issue: Permission denied when copying

Make sure you have write permissions in the target directory:

```bash
# Create target directory first
mkdir -p ./data/raw
```

### Issue: Running out of disk space

Each class is ~1-2 GB. 8 classes = 8-16 GB total.

- Check disk space: `df -h`
- Use external drive if needed

---

## Next Steps

After creating the subset:

1. **Verify it worked:**

   ```bash
   ls -la data/raw/
   # Should show: WalkingWithDog, JumpingJack, Punch, etc.
   ```

2. **Check one class:**

   ```bash
   ls data/raw/WalkingWithDog/train/ | wc -l
   # Should show: ~70
   ```

3. **Update Colab notebook:**
   - In `colab_complete_notebook.py`, modify Cell 7:

   ```python
   # Instead of downloading, use your subset:
   dataset_path = str(base_path / 'data' / 'raw')
   # The subset is already there!
   ```

4. **Start training:**
   - Use the Colab notebook as usual
   - It will find your prepared subset

---

## Advanced: Using with Google Colab

If you have the full UCF101 dataset locally:

1. **Upload to Google Drive:**

   ```bash
   # In Google Drive, create folder: MyDrive/UCF101_Full
   # Upload the 101 class folders there
   ```

2. **In Colab, run the subset script:**

   ```python
   # In Colab cell:
   !git clone https://github.com/yourrepo/DL_PROJECT.git
   %cd DL_PROJECT

   !python create_dataset_subset.py \
     --source /content/drive/MyDrive/UCF101_Full \
     --target ./data/raw
   ```

3. **Then train as normal**

---

## Statistics File

The script generates `data/raw/statistics.json`:

```json
{
  "classes": 8,
  "total_videos": 800,
  "train_videos": 560,
  "val_videos": 120,
  "test_videos": 120,
  "class_statistics": {
    "WalkingWithDog": {
      "total": 100,
      "train": 70,
      "val": 15,
      "test": 15
    },
    ...
  },
  "split_ratio": {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
  }
}
```

Use this to verify everything is correct!

---

## Summary

**Quick Command:**

```bash
# Dry run first
python create_dataset_subset.py --source /path/to/UCF101 --target ./data/raw --dry-run

# Then actually copy
python create_dataset_subset.py --source /path/to/UCF101 --target ./data/raw
```

**Time Required:**

- Dry run: <1 second
- Actual copy: 10-30 minutes (depends on disk speed)

**Result:**
✅ 8 classes, 800 videos total  
✅ Train/Val/Test splits ready  
✅ Ready for training!

---

**That's it!** Your dataset subset is ready to use with the training notebook! 🎉
