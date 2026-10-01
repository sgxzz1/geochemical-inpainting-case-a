# Case A runnable code

## Model

双门控双先验确定性预测：beta 融合 DS/Kriging，alpha 融合先验修正分支和 direct 分支。

入口脚本：`code/train_cached.py`

## Run from this model directory

The review package already contains the Case A truth grid, masks, hard-data
files, and cached training samples.  The following command uses those copied
files, so it does not depend on the original server paths:

```bash
python code/train_cached.py \
  --cache-dir ../training_samples \
  --model-variant dual_gate \
  --foundation-type timm_vit \
  --timm-model deit_tiny_patch16_224 \
  --foundation-blocks 4 --channels 192 --attention-heads 3 \
  --token-grid 10,23 --epochs 300 --batch-size 8 \
  --output-dir ./rerun_output

python code/evaluate.py \
  --checkpoint ./rerun_output/best_model.pt \
  --full-grid ../truth_data/ag_ok_back_500m.csv \
  --block-mask ../case_A_missing_block/ag_block_mask_15pct.csv \
  --hard-mask ../case_A_missing_block/ag_hard_mask_block15_hard5.csv \
  --target-mask ../case_A_missing_block/ag_target_mask_block15_hard5.csv \
  --output-dir ./rerun_output/eval \
  --ds-method cascade --ds-level 3 --ds-ensemble 24 \
  --kriging-backend pykrige --variogram-model spherical
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
