# Experiments

This file records public-score experiments for the Kaggle competition. Scores are public leaderboard scores, where lower is better.

## Submitted Results

| Submission | File | Public score | Notes |
| --- | --- | ---: | --- |
| 1 | `sub01_baseline_it20_lr1e4_thr020.csv` | `266.5512` | Empty-label RetinaNet fine-tuning, 20 iterations. |
| 2 | `sub02_baseline_it20_scale095_drop020.csv` | `260.8718` | Submission 1 with confidence scale `0.95` and scaled scores `<=0.20` removed. |
| 3 | `sub03_post_scale093_drop020_from_sub01.csv` | `259.1262` | Submission 1 with confidence scale `0.93` and scaled scores `<=0.20` removed. |
| 4 | `sub04_post_scale090_drop020_from_sub01.csv` | `257.9011` | Submission 1 with confidence scale `0.90` and scaled scores `<=0.20` removed. |
| 5 | `sub05_zaoui_public_prune015_ewc50.csv` | `250.0711` | Public pruning/EWC candidate. |
| 6 | `sub06_zaoui_public_scale095_drop020.csv` | `250.5126` | Public pruning/EWC candidate with confidence scale `0.95`. |
| 8 | `sub08_zaoui_prune0125_real_ewc50.csv` | `251.1680` | Alternative pruning/EWC candidate. |
| 9 | `sub09_zaoui_public_scale102_keep020.csv` | `249.8820` | Public pruning/EWC candidate with confidence scale `1.02`. |
| 10 | `sub10_zaoui_public_scale103_drop020.csv` | `249.8118` | Same detection set with confidence scale `1.03`. |
| 11 | `sub11_zaoui_public_scale105_keep020.csv` | `249.7103` | Same detection set with confidence scale `1.05`. |
| 12 | `sub12_zaoui_public_scale106_keep020.csv` | `249.6699` | Same detection set with confidence scale `1.06`. |
| 13 | `sub13_zaoui_public_scale107_keep020.csv` | `249.6356` | Same detection set with confidence scale `1.07`. |

## Observations

- Initial empty-label fine-tuning improved after confidence calibration and low-score filtering.
- The later public pruning/EWC candidate gave a much stronger base prediction set.
- Repeated calibration of the same detection set from `1.02` through `1.07` improved public score monotonically, but the marginal gain became smaller at each step.
- Sparse drop-threshold candidates were kept as reference files, but were not prioritized once the fixed detection set with confidence scaling became the best-performing family.

## Included Metadata

Detailed per-file statistics are available in:

- `metadata/submitted_scores.csv`
- `metadata/submission_stats.csv`
