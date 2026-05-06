#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", default="data_manifest.txt")
    parser.add_argument("--sample", default="sample_submission.csv")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = Path(args.manifest)
    sample_path = data_dir / args.sample

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    expected = [
        line.strip()
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    missing = [name for name in expected if not (data_dir / name).exists()]
    if missing:
        print(f"missing files: {len(missing)}")
        for name in missing[:20]:
            print(name)
        return 1

    with sample_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2000:
        print(f"unexpected sample row count: {len(rows)}")
        return 1

    image_ids = [row["image_id"] for row in rows]
    if len(set(image_ids)) != len(image_ids):
        print("duplicate image_id in sample_submission.csv")
        return 1

    test_files = sorted((data_dir / "test_set" / "test_set").glob("*.png"))
    if len(test_files) != 2000:
        print(f"unexpected test png count: {len(test_files)}")
        return 1

    unlearn_files = sorted((data_dir / "unlearn_set").glob("*.png"))
    if len(unlearn_files) != 20:
        print(f"unexpected unlearn png count: {len(unlearn_files)}")
        return 1

    annotations_path = data_dir / "unlearn_set" / "annotations_coco.json"
    with annotations_path.open("r", encoding="utf-8") as f:
        annotations = json.load(f)
    if len(annotations.get("images", [])) != 20:
        print("unexpected unlearn annotations image count")
        return 1

    for path in test_files[:5] + unlearn_files[:5]:
        with Image.open(path) as img:
            if img.size != (1024, 1024):
                print(f"bad image size for {path}: {img.size}")
                return 1
            if img.mode not in {"I;16", "I"}:
                print(f"unexpected image mode for {path}: {img.mode}")
                return 1

    print("data validation ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
