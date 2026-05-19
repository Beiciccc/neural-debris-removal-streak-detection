# Experiments

This file records public-score experiments for the Kaggle competition. Scores are public leaderboard scores, where lower is better. Kaggle adjusted the metric to asymmetric maCADD in May 2026, so the table uses recomputed public leaderboard values.

## Submitted Results

| Submission | File | Public score | Notes |
| --- | --- | ---: | --- |
| 1 | `sub01_baseline_it20_lr1e4_thr020.csv` | `260.1039` | Empty-label RetinaNet fine-tuning, 20 iterations. |
| 2 | `sub02_baseline_it20_scale095_drop020.csv` | `255.5143` | Submission 1 with confidence scale `0.95` and scaled scores `<=0.20` removed. |
| 3 | `sub03_post_scale093_drop020_from_sub01.csv` | `254.1299` | Submission 1 with confidence scale `0.93` and scaled scores `<=0.20` removed. |
| 4 | `sub04_post_scale090_drop020_from_sub01.csv` | `253.5141` | Submission 1 with confidence scale `0.90` and scaled scores `<=0.20` removed. |
| 5 | `sub05_zaoui_public_prune015_ewc50.csv` | `247.4372` | Public pruning/EWC candidate. |
| 6 | `sub06_zaoui_public_scale095_drop020.csv` | `247.9043` | Public pruning/EWC candidate with confidence scale `0.95`. |
| 8 | `sub08_zaoui_prune0125_real_ewc50.csv` | `248.3809` | Alternative pruning/EWC candidate. |
| 9 | `sub09_zaoui_public_scale102_keep020.csv` | `247.2900` | Public pruning/EWC candidate with confidence scale `1.02`. |
| 10 | `sub10_zaoui_public_scale103_drop020.csv` | `247.2298` | Same detection set with confidence scale `1.03`. |
| 11 | `sub11_zaoui_public_scale105_keep020.csv` | `247.1309` | Same detection set with confidence scale `1.05`. |
| 12 | `sub12_zaoui_public_scale106_keep020.csv` | `247.0871` | Same detection set with confidence scale `1.06`. |
| 13 | `sub13_zaoui_public_scale107_keep020.csv` | `247.0467` | Same detection set with confidence scale `1.07`. |
| 14 | `sub14_zaoui_public_scale108_keep020.csv` | `247.0065` | Same detection set with confidence scale `1.08`. |
| 15 | `sub15_zaoui_public_scale109_keep020.csv` | `246.9702` | Same detection set with confidence scale `1.09`. |
| 16 | `sub16_zaoui_public_scale110_keep020.csv` | `246.9381` | Same detection set with confidence scale `1.10`. |
| 17 | `sub17_zaoui_public_scale111_keep020.csv` | `246.9080` | Same detection set with confidence scale `1.11`. |
| 18 | `sub18_zaoui_public_scale112_keep020.csv` | `246.8837` | Same detection set with confidence scale `1.12`. |
| 19 | `sub19_zaoui_public_scale11225_keep020.csv` | `246.8777` | Same detection set with confidence scale `1.1225`. |
| 20 | `sub20_zaoui_public_scale1125_keep020.csv` | `246.8721` | Same detection set with confidence scale `1.125`. |
| 21 | `sub21_zaoui_public_scale113_drop025.csv` | `247.4740` | Confidence scale `1.13` with scaled scores `<=0.25` removed. |
| 22 | `sub22_zaoui_public_scale113_keep020.csv` | `246.8615` | Same detection set with confidence scale `1.13`; Kaggle listed the same upload twice. |
| 23 | `sub23_zaoui_public_scale1135_keep020.csv` | `246.8521` | Same detection set with confidence scale `1.135`. |
| 24 | `sub24_zaoui_public_scale114_keep020.csv` | `246.8434` | Same detection set with confidence scale `1.14`. |
| 25 | `sub25_zaoui_public_scale1141_keep020.csv` | `246.8416` | Same detection set with confidence scale `1.141`, close to the valid confidence upper bound. |

## Observations

- Initial empty-label fine-tuning improved after confidence calibration and low-score filtering.
- The later public pruning/EWC candidate gave a much stronger base prediction set.
- Repeated calibration of the same detection set from `1.02` through `1.141` improved public score monotonically, but future steps are very close to the score upper bound.
- The `1.13/drop0.25` candidate was worse than the keep-`0.20` calibration line, so sparse drop-threshold candidates remain lower priority.
- Host clarification indicates unlearn-set annotations represent poisoned targets to suppress, not clean positives to preserve.
- Empty predictions must be serialized as a single-space field to avoid null values in Kaggle submission parsing.

## Included Metadata

Detailed per-file statistics are available in:

- `metadata/submitted_scores.csv`
- `metadata/submission_stats.csv`
