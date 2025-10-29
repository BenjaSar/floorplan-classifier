# Training Resume Guide

## Overview

The `train_fixed.py` script now supports resuming training from checkpoints. This allows you to:
- Continue training if interrupted
- Resume from any saved checkpoint
- Preserve all training history and optimizer states

## Quick Start

### Resume from your latest checkpoint (epoch 10):

```bash
python train_fixed.py --resume models/checkpoints_fixed/checkpoint_epoch_10.pth
```

This will:
- Load the model weights from epoch 10
- Restore optimizer and learning rate scheduler states
- Continue training from epoch 11 → 150
- Preserve all training history (loss, IoU, active classes)

## Usage Examples

### 1. Start Fresh Training
```bash
python train_fixed.py
```
Starts training from epoch 1

### 2. Resume from Specific Checkpoint
```bash
python train_fixed.py --resume models/checkpoints_fixed/checkpoint_epoch_10.pth
```
Continues from epoch 11

### 3. Resume from Best Model
```bash
python train_fixed.py --resume models/checkpoints_fixed/best_model.pth
```
Resumes from the best performing model

## Available Checkpoints

Check your checkpoints directory:
```bash
ls models/checkpoints_fixed/
```

You should see:
- `checkpoint_epoch_10.pth` - Saved at epoch 10
- `checkpoint_epoch_20.pth` - Saved at epoch 20 (if reached)
- `best_model.pth` - Best performing model
- `final_model.pth` - Final model after all epochs

## What Gets Restored

When resuming, the script restores:

✅ **Model weights** - Exact neural network parameters  
✅ **Optimizer state** - Adam momentum and variance  
✅ **Scheduler state** - Learning rate schedule  
✅ **Training history** - All previous metrics  
✅ **Best metrics** - Best IoU and active classes count  
✅ **Epoch number** - Continues from next epoch

## Important Notes

### Class Weights Recalculation
The script will **recalculate class weights** when resuming. This is normal and ensures consistency, but takes a few minutes. The weights should be very similar to the original training.

### Training Configuration
The CONFIG in the script defines total epochs as 150. When resuming from epoch 10, it will train:
- Epochs 11-150 (140 more epochs)

### Checkpoint Saving
Checkpoints are saved:
- Every 10 epochs (CONFIG['save_frequency'])
- When model improves (best_model.pth)
- At the end of training (final_model.pth)

## Troubleshooting

### Checkpoint Not Found
```
[WARNING] Checkpoint not found: models/checkpoints_fixed/checkpoint_epoch_10.pth
[WARNING] Starting training from scratch
```
**Solution:** Check the path is correct

### Out of Memory
If resuming uses too much GPU memory:
- Reduce batch_size in CONFIG
- Close other applications using GPU

### Different Python Environment
Ensure you're using the same Python environment as when you started training:
```bash
# If using virtual environment
floorplan_vit\Scripts\activate

# Then run
python train_fixed.py --resume models/checkpoints_fixed/checkpoint_epoch_10.pth
```

## Expected Behavior

When resuming successfully, you'll see:
```
================================================================================
RESUMING TRAINING FROM CHECKPOINT
================================================================================
Loading checkpoint from: models/checkpoints_fixed/checkpoint_epoch_10.pth
[OK] Model state loaded
[OK] Optimizer state loaded  
[OK] Scheduler state loaded
[OK] Resuming from epoch 11
Previous best IoU: 0.XXXX
Previous best active classes: XX
```

Then training continues normally from epoch 11.

## Performance Expectations

With 256 classes and class weighting:
- **Epoch 1-10**: IoU 0.02-0.10 (learning minority classes)
- **Epoch 11-50**: IoU 0.10-0.40 (rapid improvement)
- **Epoch 51-100**: IoU 0.40-0.60 (convergence)
- **Epoch 101-150**: IoU 0.60-0.70 (fine-tuning)

Active classes should increase from ~20 to 200+ over training.

## Next Steps

After resuming and completing training:

1. **Test the model:**
   ```bash
   python test_inference.py
   ```
   (Update CHECKPOINT_PATH to best_model.pth)

2. **View training history:**
   Check `models/checkpoints_fixed/training_history.json`

3. **Visualize progress:**
   Plot the metrics from training_history.json

## Questions?

- Check logs in `logs/` directory
- Review training_history.json for metrics
- Compare IoU trends before/after resume
