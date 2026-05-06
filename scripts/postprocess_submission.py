#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_prediction_string(value: object) -> list[list[float]]:
    text = str(value)
    if not text.strip():
        return []
    nums = [float(item) for item in text.split()]
    if len(nums) % 5 != 0:
        raise ValueError(f"prediction length is not multiple of 5: {len(nums)}")
    return [nums[i : i + 5] for i in range(0, len(nums), 5)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample", type=Path, default=Path("data/sample_submission.csv"))
    parser.add_argument("--conf-scale", type=float, default=1.0)
    parser.add_argument("--drop-threshold", type=float, default=None)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    source = pd.read_csv(args.input, keep_default_na=False)
    sample = pd.read_csv(args.sample, keep_default_na=False)
    if list(source.columns) != list(sample.columns):
        raise ValueError(f"bad columns: {list(source.columns)} expected {list(sample.columns)}")
    if source["image_id"].astype(str).tolist() != sample["image_id"].astype(str).tolist():
        raise ValueError("image_id order does not match sample")
    if source["id"].tolist() != sample["id"].tolist():
        raise ValueError("id column does not match sample")

    rows = []
    scores = []
    dropped = 0
    for _, row in source.iterrows():
        parts = []
        for score, x, y, w, h in parse_prediction_string(row["prediction_string"]):
            scaled_score = score * args.conf_scale
            if args.drop_threshold is not None and scaled_score <= args.drop_threshold:
                dropped += 1
                continue
            parts.extend(
                [
                    f"{scaled_score:.6f}",
                    f"{x:.2f}",
                    f"{y:.2f}",
                    f"{w:.2f}",
                    f"{h:.2f}",
                ]
            )
            scores.append(scaled_score)
        rows.append(
            {
                "id": row["id"],
                "image_id": row["image_id"],
                "prediction_string": " ".join(parts) or " ",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=list(sample.columns))
    out.to_csv(args.output, index=False)

    if args.stats:
        empty = int((out["prediction_string"].str.strip() == "").sum())
        print(f"wrote {args.output}")
        print(f"rows={len(out)} empty={empty} boxes={len(scores)} dropped={dropped}")
        if scores:
            print(f"mean_conf={np.mean(scores):.6f} min_conf={np.min(scores):.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
