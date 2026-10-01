# Case A runnable code

## Model

无先验确定性直接预测：同一缓存训练集和固定 Case A block，只改变输出形式。

入口脚本：`code/run_seven_prior_ablation.py`

## Run from this model directory

The review package already contains the Case A truth grid, masks, hard-data
files, and cached training samples.  The following command uses those copied
files, so it does not depend on the original server paths:

```bash
python code/run_seven_prior_ablation.py \
  --include-case-a \
  --case-a-cache-dir ../training_samples \
  --case-a-block-mask ../case_A_missing_block/ag_block_mask_15pct.csv \
  --case-a-hard-mask ../case_A_missing_block/ag_hard_mask_block15_hard5.csv \
  --case-a-target-mask ../case_A_missing_block/ag_target_mask_block15_hard5.csv \
  --full-grid ../truth_data/ag_ok_back_500m.csv \
  --output-root ./rerun_output \
  --experiments no_prior_direct \
  --foundation-type timm_vit \
  --timm-model deit_tiny_patch16_224 \
  --foundation-blocks 4 --channels 192 --attention-heads 3 \
  --token-grid 10,23 --epochs 300 --samples-per-epoch 100 --batch-size 4
```

## Dependencies

```bash
python -m pip install -r code/requirements.txt
```

The foundation model is `deit_tiny_patch16_224`.  If the machine does not
already have its timm/Hugging Face cache, set `HF_ENDPOINT` before the first
run or provide network access.  The copied experiment checkpoint is kept
beside this code and is not overwritten by a rerun unless the reviewer
explicitly chooses a new output directory.

## Reproducibility note

All three variants use the same Case A target block and the same cached
training samples.  The evaluation metric is computed only on the original
target mask; hard-data pixels are retained as known values and are not scored
as generated pixels.
