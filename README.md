# Neural Debris Removal in Streak Detection Models

Public experiment archive for the Kaggle competition:

https://www.kaggle.com/competitions/neural-debris-removal-in-streak-detection-models

The repository keeps the public-facing experiment record, submission files, summary statistics, and reproducibility scripts used for RetinaNet-based unlearning and confidence calibration experiments.

## Contents

- `EXPERIMENTS.md` - concise experiment history and public leaderboard scores.
- `submissions/` - generated Kaggle submission CSV files.
- `metadata/submitted_scores.csv` - submitted files with public scores.
- `metadata/submission_stats.csv` - detection counts and confidence statistics for all included CSV files.
- `scripts/` - helper scripts for validation, RetinaNet experiments, and submission post-processing.

Raw competition images and model weights are not included. Download the official competition data from Kaggle before running training or validation that depends on local images.

## Current Best

| File | Public score |
| --- | ---: |
| `submissions/sub13_zaoui_public_scale107_keep020.csv` | `249.6356` |

Lower score is better for this competition metric.

## Reproduce A Post-Processing Candidate

```bash
python3 scripts/postprocess_submission.py \
  submissions/sub09_zaoui_public_scale102_keep020.csv \
  --output submissions/example_scaled.csv \
  --conf-scale 1.0490196078431373
```

Validate a submission file:

```bash
python3 scripts/validate_submission.py submissions/sub13_zaoui_public_scale107_keep020.csv
```

The validator expects the official `data/sample_submission.csv` file to exist locally unless a custom `--sample` path is supplied.
