#!/usr/bin/env python3
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    DatasetMapper,
    MetadataCatalog,
    build_detection_train_loader,
    detection_utils as utils,
)
from detectron2.engine import DefaultPredictor, DefaultTrainer
from tqdm import tqdm


BASE_CONFIG = "COCO-Detection/retinanet_R_50_FPN_3x.yaml"
ANCHOR_ASPECT_RATIOS = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
ANCHOR_SIZES = [[16], [32], [64], [128], [256]]
NUM_CLASSES = 1
IMG_W = IMG_H = 1024


def load_uint16_as_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.dtype == np.uint16:
        image = image.astype(np.float32) / 65535.0
    image = np.clip(image * 255, 0, 255).astype(np.float32)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    return image


class UInt16DatasetMapper(DatasetMapper):
    augment_flips = False

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        image = load_uint16_as_rgb(Path(dataset_dict["file_name"]))
        if self.augment_flips:
            if np.random.rand() > 0.5:
                image = np.fliplr(image)
            if np.random.rand() > 0.5:
                image = np.flipud(image)
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).copy())
        dataset_dict["instances"] = utils.annotations_to_instances([], image.shape[:2])
        return dataset_dict


class UnlearnTrainer(DefaultTrainer):
    freeze_mode = "none"
    augment_flips = False

    @classmethod
    def build_train_loader(cls, cfg):
        dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])
        mapper = UInt16DatasetMapper(cfg, is_train=True, augmentations=[])
        mapper.augment_flips = cls.augment_flips
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset_dicts)

    @classmethod
    def build_model(cls, cfg):
        model = super().build_model(cfg)
        if cls.freeze_mode == "cls-only":
            for param in model.parameters():
                param.requires_grad = False
            for param in model.head.cls_subnet.parameters():
                param.requires_grad = True
            for param in model.head.cls_score.parameters():
                param.requires_grad = True
        return model


def register_unlearn(unlearn_dir: Path, dataset_name: str) -> None:
    json_path = unlearn_dir / "annotations_coco.json"
    with json_path.open("r", encoding="utf-8") as f:
        coco = json.load(f)
    dicts = [
        {
            "file_name": str(unlearn_dir / image["file_name"]),
            "height": image["height"],
            "width": image["width"],
            "image_id": image["id"],
            "annotations": [],
        }
        for image in coco["images"]
    ]
    if dataset_name in DatasetCatalog:
        DatasetCatalog.remove(dataset_name)
    DatasetCatalog.register(dataset_name, lambda: dicts)
    MetadataCatalog.get(dataset_name).set(thing_classes=["object"])


def build_cfg(args, dataset_name: str):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(BASE_CONFIG))
    cfg.MODEL.WEIGHTS = str(args.poisoned_weights)
    cfg.MODEL.RETINANET.NUM_CLASSES = NUM_CLASSES
    cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [ANCHOR_ASPECT_RATIOS]
    cfg.MODEL.ANCHOR_GENERATOR.SIZES = ANCHOR_SIZES
    cfg.MODEL.DEVICE = args.device
    cfg.DATASETS.TRAIN = (dataset_name,)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.SOLVER.IMS_PER_BATCH = args.batch_size
    cfg.SOLVER.BASE_LR = args.lr
    cfg.SOLVER.MAX_ITER = args.iters
    cfg.SOLVER.STEPS = tuple(args.steps)
    cfg.SOLVER.GAMMA = args.gamma
    cfg.OUTPUT_DIR = str(args.output_dir)
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return cfg


def write_submission(cfg, args) -> None:
    cfg.MODEL.WEIGHTS = str(args.output_dir / "model_final.pth")
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.conf_thresh
    predictor = DefaultPredictor(cfg)

    sample = pd.read_csv(args.sample_csv, keep_default_na=False)
    rows = []
    for _, sample_row in tqdm(sample.iterrows(), total=len(sample), desc="inference"):
        image_id = str(sample_row["image_id"])
        image_path = args.test_dir / f"{image_id}.png"
        out = predictor(load_uint16_as_rgb(image_path))["instances"].to("cpu")
        boxes = out.pred_boxes.tensor.numpy()
        scores = out.scores.numpy()

        parts = []
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            score = float(score) * args.conf_scale
            if args.drop_after_scale and score <= args.conf_thresh:
                continue
            x1 = float(np.clip(x1, 0, IMG_W))
            y1 = float(np.clip(y1, 0, IMG_H))
            x2 = float(np.clip(x2, 0, IMG_W))
            y2 = float(np.clip(y2, 0, IMG_H))
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            if w == 0 or h == 0:
                continue
            parts.extend(
                [f"{score:.6f}", f"{x1:.2f}", f"{y1:.2f}", f"{w:.2f}", f"{h:.2f}"]
            )

        rows.append(
            {
                "id": sample_row["id"],
                "image_id": sample_row["image_id"],
                "prediction_string": " ".join(parts) or " ",
            }
        )

    pd.DataFrame(rows, columns=list(sample.columns)).to_csv(args.submission, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("work/baseline_finetune"))
    parser.add_argument("--submission", type=Path, default=Path("submissions/baseline_finetune.csv"))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--conf-thresh", type=float, default=0.2)
    parser.add_argument("--conf-scale", type=float, default=1.0)
    parser.add_argument("--drop-after-scale", action="store_true")
    parser.add_argument("--steps", type=int, nargs="*", default=[])
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--freeze-mode", choices=["none", "cls-only"], default="none")
    parser.add_argument("--augment-flips", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.data_dir = args.data_dir.resolve()
    args.poisoned_weights = args.data_dir / "poisoned_model" / "poisoned_model.pth"
    args.unlearn_dir = args.data_dir / "unlearn_set"
    args.test_dir = args.data_dir / "test_set" / "test_set"
    args.sample_csv = args.data_dir / "sample_submission.csv"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.submission.parent.mkdir(parents=True, exist_ok=True)

    dataset_name = f"unlearn_empty_{args.output_dir.name}"
    register_unlearn(args.unlearn_dir, dataset_name)
    cfg = build_cfg(args, dataset_name)

    UnlearnTrainer.freeze_mode = args.freeze_mode
    UnlearnTrainer.augment_flips = args.augment_flips
    trainer = UnlearnTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    write_submission(cfg, args)
    print(f"wrote {args.submission}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
