#!/usr/bin/env python3
"""Run RetinaNet de-poisoning experiments for the Neural Debris competition.

This script is intended to run in a GPU-capable Python environment. It adapts
the public Kaggle baseline notebooks into one parameterized entrypoint so experiments are
repeatable and auditable.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog
from detectron2.data import build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer


BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
NUM_CLASSES = 1
IMG_W = IMG_H = 1024


def json_default(value):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


class UInt16DatasetMapper(DatasetMapper):
    """Read 16-bit grayscale PNGs and attach empty instances for unlearning."""

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = read_u16_as_float255(dataset_dict["file_name"])
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
        dataset_dict["instances"] = utils.annotations_to_instances([], image.shape[:2])
        return dataset_dict


def read_u16_as_float255(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    image = np.clip(image * 255, 0, 255).astype(np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return image


def register_unlearn(unlearn_dir: Path, dataset_name: str) -> None:
    json_path = unlearn_dir / "annotations_coco.json"
    with json_path.open() as f:
        coco = json.load(f)
    records = [
        {
            "file_name": str(unlearn_dir / im["file_name"]),
            "height": im["height"],
            "width": im["width"],
            "image_id": im["id"],
            "annotations": [],
        }
        for im in coco["images"]
    ]
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
    DatasetCatalog.register(dataset_name, lambda records=records: records)
    MetadataCatalog.get(dataset_name).set(thing_classes=["object"])


def make_cfg(
    *,
    weights: str,
    output_dir: Path,
    dataset_name: str,
    lr: float,
    max_iter: int,
    batch_size: int,
    conf_thresh: float,
    workers: int,
    scheduler: str,
    step_iter: int | None,
    gamma: float,
    device: str,
):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf_thresh
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.DEVICE = device
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = workers
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = lr
    cfg.SOLVER.MAX_ITER = max_iter
    if scheduler == "step" and step_iter is not None:
        cfg.SOLVER.STEPS = (step_iter,)
        cfg.SOLVER.GAMMA = gamma
    else:
        cfg.SOLVER.STEPS = []
    cfg.SOLVER.WARMUP_ITERS = min(max(0, max_iter // 10), max_iter)
    cfg.OUTPUT_DIR = str(output_dir)
    return cfg


class EmptyLabelTrainer(DefaultTrainer):
    train_scope = "all"

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper = UInt16DatasetMapper(cfg, is_train=True, augmentations=[])
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dicts)

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        apply_train_scope(model, cls.train_scope)
        return model


def apply_train_scope(model, scope: str) -> None:
    if scope == "all":
        return
    for param in model.parameters():
        param.requires_grad = False
    if scope == "cls_only":
        modules = [model.head.cls_subnet, model.head.cls_score]
    elif scope == "head":
        modules = [model.head]
    elif scope == "head_and_fpn":
        modules = [model.head, model.backbone]
        for name, param in model.backbone.named_parameters():
            if name.startswith("bottom_up.stem") or name.startswith("bottom_up.res2") or name.startswith("bottom_up.res3") or name.startswith("bottom_up.res4"):
                param.requires_grad = False
    else:
        raise ValueError(f"unknown train scope: {scope}")
    for module in modules:
        for param in module.parameters():
            param.requires_grad = True


def train_empty_label(cfg, train_scope: str) -> Path:
    EmptyLabelTrainer.train_scope = train_scope
    trainer = EmptyLabelTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    return Path(cfg.OUTPUT_DIR) / "model_final.pth"


def run_gradient_ascent(
    *,
    weights: str,
    output_path: Path,
    cfg,
    dataset_name: str,
    iters: int,
    lr: float,
    train_scope: str,
) -> Path:
    model = build_model(cfg)
    DetectionCheckpointer(model).load(weights)
    apply_train_scope(model, train_scope)
    model.train().to(cfg.MODEL.DEVICE)
    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
    )
    mapper = UInt16DatasetMapper(cfg, is_train=True, augmentations=[])
    records = DatasetCatalog.get(dataset_name)
    device = torch.device(cfg.MODEL.DEVICE)
    for step in tqdm(range(iters), desc="gradient_ascent"):
        record = records[step % len(records)]
        batch = [mapper(record)]
        batch[0]["image"] = batch[0]["image"].to(device)
        batch[0]["instances"] = batch[0]["instances"].to(device)
        optimizer.zero_grad()
        losses = model(batch)
        loss = sum(losses.values())
        (-loss).backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    return output_path


def average_checkpoints(path_a: Path, path_b: Path, output_path: Path, mix_a: float) -> Path:
    weights_a = torch.load(path_a, map_location="cpu")
    weights_b = torch.load(path_b, map_location="cpu")
    if "model" in weights_a:
        weights_a = weights_a["model"]
    if "model" in weights_b:
        weights_b = weights_b["model"]
    averaged = {}
    for key, value_b in weights_b.items():
        value_a = weights_a.get(key)
        if torch.is_tensor(value_a) and torch.is_tensor(value_b) and value_a.shape == value_b.shape:
            averaged[key] = mix_a * value_a + (1.0 - mix_a) * value_b
        else:
            averaged[key] = value_b
    torch.save({"model": averaged}, output_path)
    return output_path


def infer_submission(
    *,
    weights: str,
    output_csv: Path,
    cfg,
    test_dir: Path,
    sample_csv: Path | None,
    conf_thresh: float,
    score_scale: float,
    score_drop: float,
    topk: int | None,
) -> dict:
    cfg = cfg.clone()
    cfg.MODEL.WEIGHTS = weights
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = conf_thresh
    predictor = DefaultPredictor(cfg)

    if sample_csv and sample_csv.exists():
        sample = pd.read_csv(sample_csv)
        image_ids = [str(x) for x in sample["image_id"].tolist()]
    else:
        image_ids = sorted(p.stem for p in test_dir.glob("*.png"))

    rows = []
    total_detections = 0
    empty_rows = 0
    confidences = []
    for image_id in tqdm(image_ids, desc="inference"):
        img_path = test_dir / f"{image_id}.png"
        image = read_u16_as_float255(img_path)
        out = predictor(image)["instances"].to("cpu")
        boxes = out.pred_boxes.tensor.numpy()
        scores = out.scores.numpy()
        detections = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            score = float(score) * score_scale
            if score <= score_drop:
                continue
            x1 = float(np.clip(x1, 0, IMG_W))
            y1 = float(np.clip(y1, 0, IMG_H))
            x2 = float(np.clip(x2, 0, IMG_W))
            y2 = float(np.clip(y2, 0, IMG_H))
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w == 0.0 or h == 0.0:
                continue
            detections.append((score, x1, y1, w, h))
        detections.sort(key=lambda x: x[0], reverse=True)
        if topk is not None:
            detections = detections[:topk]
        parts = []
        for det in detections:
            confidences.append(det[0])
            parts.extend([f"{det[0]:.6f}", f"{det[1]:.2f}", f"{det[2]:.2f}", f"{det[3]:.2f}", f"{det[4]:.2f}"])
        total_detections += len(detections)
        if not detections:
            empty_rows += 1
        rows.append({"image_id": int(image_id), "prediction_string": " ".join(parts) or " "})

    submission = pd.DataFrame(rows)
    submission.insert(0, "id", range(len(submission)))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    return {
        "rows": len(submission),
        "detections": total_detections,
        "empty_rows": empty_rows,
        "confidence_mean": float(np.mean(confidences)) if confidences else None,
        "confidence_min": float(np.min(confidences)) if confidences else None,
        "confidence_max": float(np.max(confidences)) if confidences else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--mode", choices=["poisoned", "simple", "cls_only", "head", "head_and_fpn", "ga_then_ft"], required=True)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conf-thresh", type=float, default=0.2)
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument("--score-drop", type=float, default=0.2)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--scheduler", choices=["none", "step"], default="none")
    parser.add_argument("--step-iter", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--ga-lr", type=float, default=5e-5)
    parser.add_argument("--ga-iters", type=int, default=30)
    parser.add_argument("--ga-mix", type=float, default=0.3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    weights = data_root / "poisoned_model" / "poisoned_model.pth"
    unlearn_dir = data_root / "unlearn_set"
    test_dir = data_root / "test_set" / "test_set"
    sample_csv = data_root / "sample_submission.csv"
    out_dir = Path("experiments") / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_name = f"unlearn_{args.experiment}".replace("-", "_")
    register_unlearn(unlearn_dir, dataset_name)

    cfg = make_cfg(
        weights=str(weights),
        output_dir=out_dir / "train",
        dataset_name=dataset_name,
        lr=args.lr,
        max_iter=args.max_iter,
        batch_size=args.batch_size,
        conf_thresh=args.conf_thresh,
        workers=args.workers,
        scheduler=args.scheduler,
        step_iter=args.step_iter,
        gamma=args.gamma,
        device=args.device,
    )

    if args.mode == "poisoned":
        final_weights = weights
    elif args.mode in {"simple", "cls_only", "head", "head_and_fpn"}:
        scope = "all" if args.mode == "simple" else args.mode
        final_weights = train_empty_label(cfg, scope)
    elif args.mode == "ga_then_ft":
        ga_path = out_dir / "ga" / "model_ga.pth"
        ga_cfg = cfg.clone()
        ga_cfg.OUTPUT_DIR = str(out_dir / "ga")
        run_gradient_ascent(
            weights=str(weights),
            output_path=ga_path,
            cfg=ga_cfg,
            dataset_name=dataset_name,
            iters=args.ga_iters,
            lr=args.ga_lr,
            train_scope="head",
        )
        ft_cfg = cfg.clone()
        ft_cfg.MODEL.WEIGHTS = str(ga_path)
        ft_cfg.OUTPUT_DIR = str(out_dir / "ft")
        final_ft = train_empty_label(ft_cfg, "head")
        averaged_path = out_dir / "model_averaged.pth"
        final_weights = average_checkpoints(ga_path, final_ft, averaged_path, args.ga_mix)
    else:
        raise AssertionError(args.mode)

    output_csv = Path("submissions") / f"{args.experiment}.csv"
    metrics = infer_submission(
        weights=str(final_weights),
        output_csv=output_csv,
        cfg=cfg,
        test_dir=test_dir,
        sample_csv=sample_csv,
        conf_thresh=args.conf_thresh,
        score_scale=args.score_scale,
        score_drop=args.score_drop,
        topk=args.topk,
    )
    run_info = {
        "args": vars(args),
        "weights": str(final_weights),
        "submission": str(output_csv),
        "metrics": metrics,
    }
    (out_dir / "run_info.json").write_text(json.dumps(run_info, default=json_default, indent=2, sort_keys=True))
    print(json.dumps(run_info, default=json_default, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
