# audio classification research — Paper 1 (Glowacz)

This is an experiment to optimize the replication of the Paper 1 (Glowacz et al.) audio classification model for industrial machine sounds. The goal is to maximize the classification accuracy (ER and MER) through surgical improvements to the feature extraction pipeline and the CNN training process.

## Setup

To set up a new experiment:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `glowacz-apr28`).
2. **Create the branch**: `git checkout -b research/<tag>` (from a clean state).
3. **Read the in-scope files**:
   - `paper1_pipeline.py` — handles resampling, filtering, FFT, word coding, DWV analysis, and acoustic image formation.
   - `glowacz_train.py` — handles CNN training (DenseNet, ResNet, EfficientNet) and evaluation.
4. **Verify data exists**: Ensure the `dataset/` directory contains the 6 motor condition folders.
5. **Initialize results.tsv**: Create `results.tsv` in the root with the header: `commit	val_acc	mer	status	description`.
6. **Initial Pipeline Run**: Run `python paper1_pipeline.py` to generate the initial set of acoustic images in `output_paper1/images`.

## Experimentation

The workflow involves two stages: image generation and model training.

**What you CAN do:**
- **Pipeline optimization (`paper1_pipeline.py`)**: Tune `WORD_CODING_K`, `DWV_RANGE_LOW`, `DWV_RANGE_HIGH`, or `MATRIX_SIZE`. Explore `--auto_range` for dynamic frequency band selection.
- **Model optimization (`glowacz_train.py`)**: Modify CNN architectures, change optimizers/learning rates, adjust data augmentation, or increase `epochs`.

**What you CANNOT do:**
- Modify the raw `dataset/` files.
- Install new packages. Use `torch`, `numpy`, `pandas`, `matplotlib`, `scipy`, `PIL`.

**The goal: Highest ER (Overall Accuracy) and MER (Mean Per-Class Accuracy).**

## Output format

The training script `glowacz_train.py` should be run and its summary captured. For logging, use the best model's performance:

```
---
val_acc:          0.9850  (ER)
mer:              0.9850  (MER)
training_seconds: 450.5
num_params_M:     20.1
best_model:       densenet201
```

## Logging results

Log to `results.tsv` (tab-separated):
`commit	val_acc	mer	status	description`

## The experiment loop

1. **Tune**: Modify `paper1_pipeline.py` or `glowacz_train.py`.
2. **Commit**: `git commit -am "experiment description"`.
3. **Generate Images**: If pipeline changed, run `python paper1_pipeline.py`.
4. **Train**: `python glowacz_train.py --epochs 20 > run.log 2>&1`.
5. **Extract metrics**: Extract ER and MER from `run.log` or `output_glowacz/summary_report.txt`.
6. **Record**: Update `results.tsv`.
7. **Iterate**: If results improved, keep and advance. Otherwise, `git reset --hard HEAD~1`.
8. **NEVER STOP**: Continue iterating autonomously until interrupted.
