#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import pandas as pd


def parse_prediction_string(value: object) -> list[float]:
    if isinstance(value, float) and math.isnan(value):
        return []
    text = str(value)
    if not text.strip():
        return []
    return [float(item) for item in text.split()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--sample", default="data/sample_submission.csv")
    args = parser.parse_args()

    kaggle_submission = pd.read_csv(args.submission)
    null_rows = kaggle_submission["prediction_string"].isna()
    if null_rows.any():
        first_rows = null_rows[null_rows].index[:5].tolist()
        print(f"prediction_string contains Kaggle-null empty fields at rows {first_rows}")
        return 1

    submission = pd.read_csv(args.submission, keep_default_na=False)
    sample = pd.read_csv(args.sample, keep_default_na=False)

    expected_cols = list(sample.columns)
    if list(submission.columns) != expected_cols:
        print(f"bad columns: {list(submission.columns)} expected {expected_cols}")
        return 1
    if len(submission) != len(sample):
        print(f"bad row count: {len(submission)} expected {len(sample)}")
        return 1
    if submission["image_id"].astype(str).tolist() != sample["image_id"].astype(str).tolist():
        print("image_id order does not match sample_submission.csv")
        return 1
    if submission["id"].tolist() != sample["id"].tolist():
        print("id column does not match sample_submission.csv")
        return 1

    for idx, value in enumerate(submission["prediction_string"]):
        try:
            nums = parse_prediction_string(value)
        except ValueError:
            print(f"non-numeric prediction at row {idx}")
            return 1
        if len(nums) % 5 != 0:
            print(f"prediction length is not multiple of 5 at row {idx}")
            return 1
        for j in range(0, len(nums), 5):
            score, x, y, w, h = nums[j : j + 5]
            if not 0.0 <= score <= 1.0:
                print(f"bad score at row {idx}: {score}")
                return 1
            if x < 0 or y < 0 or w < 0 or h < 0 or x > 1024 or y > 1024:
                print(f"bad box at row {idx}: {nums[j:j+5]}")
                return 1
            if x + w > 1024.01 or y + h > 1024.01:
                print(f"box exceeds bounds at row {idx}: {nums[j:j+5]}")
                return 1

    print("submission validation ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
