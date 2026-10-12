"""Run reproducible YOLO11n Baseline experiments on the 512-image TELEA dataset."""

from __future__ import annotations

import csv
import json
import platform
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(r"C:\workspace\Kamp_Xray")
DATA_YAML = r"C:\workspace\Kamp_Xray\outputs\yolo_512_telea\dataset.yaml"
RUNS_DIR = ROOT / "outputs" / "yolo_runs"
EXPERIMENT_INDEX = RUNS_DIR / "experiments.csv"
VENV_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"


# Edit only CONFIG to define the next experiment.
CONFIG: dict[str, Any] = {
    "preprocessing": "telea_512",
    "model": "yolov8n.pt",
    "data": DATA_YAML,
    "epochs": 200,
    # Small-object experiment candidates:
    # imgsz = 640 / 960 / 1280
    "imgsz": 640,
    "batch": 4,
    "device": 0,
    "workers": 4,
    "optimizer": "AdamW",
    "lr0": 0.001,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3.0,
    "patience": 50,
    "seed": 42,
    "deterministic": True,
    "pretrained": True,
    # Shared augmentation settings for the YOLOv8n / YOLO11n 640/batch4 comparison.
    "degrees": 0.0,
    "translate": 0.005,
    "scale": 0.2,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.0,
    "mosaic": 0.2,
    "mixup": 0.0,
    "hsv_h": 0.0,
    "hsv_s": 0.0,
    "hsv_v": 0.0,
    # Train-only intensity augmentation. Noise std is in 0-255 pixel units.
    "gamma_p": 0.20,
    "gamma_min": 0.90,
    "gamma_max": 1.10,
    "noise_p": 0.0,
    "noise_std_min": 1.0,
    "noise_std_max": 3.0,
    "project": str(RUNS_DIR),
    "exist_ok": False,
    "plots": True,
    "save": True,
    "verbose": True,
    "conf": 0.25,
}


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
AUGMENTATION_KEYS = (
    "degrees",
    "translate",
    "scale",
    "shear",
    "perspective",
    "flipud",
    "fliplr",
    "mosaic",
    "mixup",
    "hsv_h",
    "hsv_s",
    "hsv_v",
)
AUGMENTATION_ABBREVIATIONS = {
    "degrees": "deg",
    "translate": "tr",
    "scale": "sc",
    "shear": "sh",
    "perspective": "per",
    "flipud": "fud",
    "fliplr": "flr",
    "mosaic": "mos",
    "mixup": "mix",
    "hsv_h": "hsvh",
    "hsv_s": "hsvs",
    "hsv_v": "hsvv",
}
XRAY_AUGMENTATION_KEYS = {
    "gamma_p",
    "gamma_min",
    "gamma_max",
    "noise_p",
    "noise_std_min",
    "noise_std_max",
}
NON_TRAIN_CONFIG_KEYS = {"preprocessing", "model", "conf"} | XRAY_AUGMENTATION_KEYS
EXPERIMENT_INDEX_FIELDS = (
    "run_name",
    "date",
    "model",
    "epochs",
    "imgsz",
    "batch",
    "optimizer",
    "lr0",
    "weight_decay",
    "augmentation",
    "seed",
    "val_precision",
    "val_recall",
    "val_map50",
    "val_map50_95",
    "test_precision",
    "test_recall",
    "test_map50",
    "test_map50_95",
    "best_epoch",
    "best_model",
)


class UserFacingError(RuntimeError):
    """An expected failure that should be shown without a Python traceback."""


def _safe_name(value: Any) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", str(value).strip())
    return cleaned.strip("-_").lower() or "unknown"


def _compact_number(value: Any) -> str:
    return format(float(value), ".10g")


def augmentation_label(config: dict[str, Any]) -> str:
    active = [key for key in AUGMENTATION_KEYS if float(config.get(key, 0.0)) != 0.0]
    # Keep the folder name readable while exposing up to three dominant settings.
    details = [
        f"{AUGMENTATION_ABBREVIATIONS[key]}{_compact_number(config[key])}"
        for key in active[:3]
    ]
    if len(active) > 3:
        details.append(f"x{len(active) - 3}")
    if float(config.get("gamma_p", 0)) > 0:
        details.append(
            f"gam{_compact_number(config['gamma_min'])}-{_compact_number(config['gamma_max'])}"
            f"p{_compact_number(config['gamma_p'])}"
        )
    if float(config.get("noise_p", 0)) > 0:
        details.append(
            f"noise{_compact_number(config['noise_std_min'])}-{_compact_number(config['noise_std_max'])}"
            f"p{_compact_number(config['noise_p'])}"
        )
    return "aug_" + "_".join(details) if details else "noaug"


def validate_xray_augmentation(config: dict[str, Any]) -> None:
    import math

    for key in XRAY_AUGMENTATION_KEYS:
        if not math.isfinite(float(config[key])):
            raise UserFacingError(f"{key} must be finite.")
    for key in ("gamma_p", "noise_p"):
        if not 0 <= float(config[key]) <= 1:
            raise UserFacingError(f"{key} must be between 0 and 1.")
    if not 0 < config["gamma_min"] <= config["gamma_max"]:
        raise UserFacingError("Gamma range must satisfy 0 < gamma_min <= gamma_max.")
    if not 0 <= config["noise_std_min"] <= config["noise_std_max"] <= 255:
        raise UserFacingError("Noise std range must satisfy 0 <= min <= max <= 255.")


def build_xray_trainer(config: dict[str, Any]) -> Any:
    """Apply intensity changes to training batches only, after YOLO normalization.

    Validation uses Ultralytics' validator and never calls this trainer hook.
    PyTorch's seeded RNG is used; the same noise is shared by all color channels
    so grayscale X-rays remain grayscale. Annotation tensors are untouched.
    """
    import torch
    from ultralytics.models.yolo.detect import DetectionTrainer

    validate_xray_augmentation(config)
    settings = {key: float(config[key]) for key in XRAY_AUGMENTATION_KEYS}

    class XrayDetectionTrainer(DetectionTrainer):
        def preprocess_batch(self, batch: dict) -> dict:
            batch = super().preprocess_batch(batch)
            image = batch["img"]
            shape = (image.shape[0], 1, 1, 1)
            if settings["gamma_p"] > 0:
                apply = torch.rand(shape, device=image.device) < settings["gamma_p"]
                gamma = torch.empty(shape, device=image.device).uniform_(
                    settings["gamma_min"], settings["gamma_max"]
                )
                # Direct exponent convention: output = input ** gamma.
                image = torch.where(apply, image.clamp(0, 1).pow(gamma), image)
            if settings["noise_p"] > 0:
                apply = torch.rand(shape, device=image.device) < settings["noise_p"]
                std = (
                    torch.empty(shape, device=image.device).uniform_(
                        settings["noise_std_min"], settings["noise_std_max"]
                    )
                    / 255.0
                )
                noise = (
                    torch.randn(
                        (image.shape[0], 1, image.shape[2], image.shape[3]),
                        device=image.device,
                        dtype=image.dtype,
                    )
                    * std
                )
                image = torch.where(apply, (image + noise).clamp(0, 1), image)
            batch["img"] = image
            return batch

    return XrayDetectionTrainer


def build_run_name(config: dict[str, Any]) -> str:
    """Build a readable name from the hyperparameters that define an experiment."""
    model_name = _safe_name(Path(str(config["model"])).stem)
    optimizer = _safe_name(config["optimizer"])
    preprocessing = _safe_name(config.get("preprocessing", "telea"))
    return "_".join(
        (
            preprocessing,
            model_name,
            f"e{int(config['epochs'])}",
            f"img{int(config['imgsz'])}",
            f"b{int(config['batch'])}",
            optimizer,
            f"lr{_compact_number(config['lr0'])}",
            f"wd{_compact_number(config['weight_decay'])}",
            augmentation_label(config),
            f"s{int(config['seed'])}",
        )
    )


def get_unique_run_name(project_dir: Path, base_name: str) -> str:
    """Return base_name or the first available run02/run03-style suffix."""
    if not (project_dir / base_name).exists():
        return base_name
    run_number = 2
    while (project_dir / f"{base_name}_run{run_number:02d}").exists():
        run_number += 1
    return f"{base_name}_run{run_number:02d}"


def load_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import torch
    except ImportError as exc:
        raise UserFacingError(
            "PyTorch is not installed in this Python environment. "
            "Install a CUDA-compatible PyTorch build before training."
        ) from exc

    try:
        import torchvision
    except ImportError:
        torchvision = None

    try:
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT
    except ImportError as exc:
        raise UserFacingError(
            "Ultralytics is not installed in this Python environment. "
            f"Install it with: {VENV_PYTHON} -m pip install ultralytics"
        ) from exc

    return torch, torchvision, ultralytics, YOLO, DEFAULT_CFG_DICT


def validate_ultralytics_arguments(default_cfg: dict[str, Any]) -> None:
    train_args = set(CONFIG) - NON_TRAIN_CONFIG_KEYS
    unsupported = sorted(train_args - set(default_cfg))
    if unsupported:
        raise UserFacingError(
            "The installed Ultralytics version does not support these CONFIG arguments: "
            + ", ".join(unsupported)
        )


def collect_environment(
    torch: Any,
    torchvision: Any,
    ultralytics: Any,
) -> dict[str, str]:
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count()) if available else 0
    requested = CONFIG["device"]

    print("\nExecution environment")
    print(f"CUDA available: {available}")
    print(f"CUDA device count: {count}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Ultralytics version: {ultralytics.__version__}")

    gpu_name = "Not available"
    requested_index: int | None = None
    if isinstance(requested, int):
        requested_index = requested
    elif str(requested).lower() not in {"cpu", "mps"}:
        try:
            requested_index = int(str(requested).split(",")[0])
        except ValueError as exc:
            raise UserFacingError(f"Unsupported device value: {requested!r}") from exc

    if requested_index is not None:
        if not available:
            raise UserFacingError(
                f"CUDA device {requested_index} requested, but CUDA is not available."
            )
        if requested_index < 0 or requested_index >= count:
            raise UserFacingError(
                f"CUDA device {requested_index} requested, but only {count} CUDA device(s) are available."
            )
        gpu_name = torch.cuda.get_device_name(requested_index)
    elif available:
        gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU name: {gpu_name}")

    return {
        "Python version": platform.python_version(),
        "PyTorch version": str(torch.__version__),
        "Torchvision version": (
            str(torchvision.__version__) if torchvision is not None else "Not installed"
        ),
        "CUDA version": str(torch.version.cuda or "Not available"),
        "CUDA available": str(available),
        "CUDA device count": str(count),
        "GPU name": gpu_name,
        "Ultralytics version": str(ultralytics.__version__),
        "OS": platform.platform(),
    }


def _configured_data_yaml(config: dict[str, Any]) -> Path:
    path = Path(str(config["data"]))
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _resolve_dataset_root(config: dict[str, Any], dataset_yaml_path: Path) -> Path:
    raw_path = config.get("path", dataset_yaml_path.parent)
    path = Path(str(raw_path))
    if not path.is_absolute():
        path = dataset_yaml_path.parent / path
    return path.resolve()


def _resolve_split_path(dataset_root: Path, value: Any, split: str) -> Path:
    if not isinstance(value, str):
        raise UserFacingError(
            f"dataset.yaml entry '{split}' must be a directory path string."
        )
    path = Path(value)
    if not path.is_absolute():
        path = dataset_root / path
    return path.resolve()


def preflight_dataset(
    yaml_module: Any,
    config: dict[str, Any] = CONFIG,
) -> tuple[Path, dict[str, int]]:
    """Validate dataset paths and enforce one-to-one image/label matching."""
    dataset_yaml_path = _configured_data_yaml(config)
    if not dataset_yaml_path.is_file():
        raise UserFacingError(f"dataset.yaml was not found: {dataset_yaml_path}")

    try:
        dataset_config = yaml_module.safe_load(
            dataset_yaml_path.read_text(encoding="utf-8")
        )
    except Exception as exc:
        raise UserFacingError(f"Could not read dataset.yaml: {exc}") from exc
    if not isinstance(dataset_config, dict):
        raise UserFacingError("dataset.yaml does not contain a valid mapping.")

    dataset_root = _resolve_dataset_root(dataset_config, dataset_yaml_path)
    counts: dict[str, int] = {}
    print("\nDataset preflight")
    for split in ("train", "val", "test"):
        if split not in dataset_config:
            raise UserFacingError(f"dataset.yaml is missing the '{split}' entry.")
        image_dir = _resolve_split_path(dataset_root, dataset_config[split], split)
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir():
            raise UserFacingError(
                f"{split.title()} image folder was not found: {image_dir}"
            )
        if not label_dir.is_dir():
            raise UserFacingError(
                f"{split.title()} label folder was not found: {label_dir}"
            )

        images = sorted(
            path
            for path in image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        labels = sorted(label_dir.glob("*.txt"))
        image_stems = {path.stem for path in images}
        label_stems = {path.stem for path in labels}
        counts[f"{split}_images"] = len(images)
        counts[f"{split}_labels"] = len(labels)
        print(f"{split.title()} images: {len(images)}")
        print(f"{split.title()} labels: {len(labels)}")

        if not images:
            raise UserFacingError(f"No images were found in {image_dir}")
        if len(images) != len(labels):
            print(
                f"WARNING: {split} image/label counts differ "
                f"({len(images)} images, {len(labels)} labels)."
            )
        if image_stems != label_stems:
            missing_labels = sorted(image_stems - label_stems)
            orphan_labels = sorted(label_stems - image_stems)
            details = []
            if missing_labels:
                details.append("missing labels: " + ", ".join(missing_labels))
            if orphan_labels:
                details.append("orphan labels: " + ", ".join(orphan_labels))
            raise UserFacingError(
                f"{split.title()} images and labels are not one-to-one; "
                + "; ".join(details)
            )

    return dataset_root, counts


def save_config(
    run_dir: Path,
    config: dict[str, Any],
    run_name: str,
    start_time: datetime,
    end_time: datetime | None,
) -> Path:
    """Save the complete CONFIG plus lifecycle metadata."""
    record = dict(config)
    record.update(
        {
            "run_name": run_name,
            "start_time": start_time.isoformat(timespec="seconds"),
            "end_time": (
                end_time.isoformat(timespec="seconds") if end_time is not None else None
            ),
            "duration_seconds": (
                round((end_time - start_time).total_seconds(), 3)
                if end_time is not None
                else None
            ),
        }
    )
    config_path = run_dir / "experiment_config.json"
    config_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return config_path


def save_environment(run_dir: Path, environment: dict[str, str]) -> Path:
    environment_path = run_dir / "environment.txt"
    environment_path.write_text(
        "\n".join(f"{key}: {value}" for key, value in environment.items()) + "\n",
        encoding="utf-8",
    )
    return environment_path


def run_training(
    YOLO: Any, config: dict[str, Any], run_name: str
) -> tuple[Any, Path, Path]:
    print(f"\nStarting {config['model']} 512-image TELEA training...")
    model = YOLO(config["model"])
    train_args = {
        key: value for key, value in config.items() if key not in NON_TRAIN_CONFIG_KEYS
    }
    train_args["name"] = run_name
    if config.get("gamma_p", 0) > 0 or config.get("noise_p", 0) > 0:
        train_args["trainer"] = build_xray_trainer(config)
    results = model.train(**train_args)

    expected_run_dir = Path(config["project"]) / run_name
    reported_save_dir = getattr(results, "save_dir", None)
    if reported_save_dir is None and getattr(model, "trainer", None) is not None:
        reported_save_dir = getattr(model.trainer, "save_dir", None)
    actual_run_dir = Path(reported_save_dir or expected_run_dir).resolve()
    if actual_run_dir != expected_run_dir.resolve():
        raise UserFacingError(
            f"Unexpected training output directory: {actual_run_dir} "
            f"(expected {expected_run_dir})"
        )

    best_path = expected_run_dir / "weights" / "best.pt"
    if not best_path.is_file():
        raise UserFacingError(
            f"Training finished, but best.pt was not created: {best_path}"
        )
    print(f"\nBest model:\n{best_path}")
    return model, expected_run_dir, best_path


def _evaluation_args(run_dir: Path) -> dict[str, Any]:
    return {
        "data": CONFIG["data"],
        "device": CONFIG["device"],
        "imgsz": CONFIG["imgsz"],
        "batch": CONFIG["batch"],
        "workers": CONFIG["workers"],
        "project": str(run_dir),
        "exist_ok": False,
        "plots": True,
        "verbose": True,
    }


def _extract_metrics(results: Any, split: str) -> dict[str, float]:
    box = getattr(results, "box", None)
    if box is None:
        raise UserFacingError(
            f"Ultralytics returned no detection metrics for the {split} split."
        )
    try:
        return {
            "precision": float(box.mp),
            "recall": float(box.mr),
            "map50": float(box.map50),
            "map50_95": float(box.map),
        }
    except (AttributeError, TypeError, ValueError) as exc:
        raise UserFacingError(
            f"Could not extract precision/recall/mAP metrics for the {split} split."
        ) from exc


def run_validation(best_model: Any, run_dir: Path) -> dict[str, float]:
    results = best_model.val(
        split="val", name="validation", **_evaluation_args(run_dir)
    )
    return _extract_metrics(results, "validation")


def run_test(best_model: Any, run_dir: Path) -> dict[str, float]:
    results = best_model.val(
        split="test", name="test_evaluation", **_evaluation_args(run_dir)
    )
    return _extract_metrics(results, "test")


def run_prediction(best_model: Any, dataset_root: Path, run_dir: Path) -> Path:
    prediction_name = "test_predictions"
    prediction_dir = run_dir / prediction_name
    best_model.predict(
        source=str(dataset_root / "images" / "test"),
        conf=CONFIG["conf"],
        imgsz=CONFIG["imgsz"],
        device=CONFIG["device"],
        save=True,
        project=str(run_dir),
        name=prediction_name,
        exist_ok=False,
        verbose=True,
    )
    if not prediction_dir.is_dir():
        raise UserFacingError(
            f"Test prediction output was not created: {prediction_dir}"
        )
    return prediction_dir


def _find_best_epoch(run_dir: Path) -> int | None:
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        return None
    with results_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    normalized_rows = [
        {str(key).strip(): value for key, value in row.items()} for row in rows
    ]
    try:
        best_row = max(
            normalized_rows,
            key=lambda row: float(row["metrics/mAP50-95(B)"]),
        )
        return int(float(best_row["epoch"]))
    except (KeyError, TypeError, ValueError):
        return None


def save_summary(
    run_dir: Path,
    run_name: str,
    best_path: Path,
    counts: dict[str, int],
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
) -> tuple[Path, dict[str, Any]]:
    summary: dict[str, Any] = {
        "run_name": run_name,
        "best_model": str(best_path),
        "best_epoch": _find_best_epoch(run_dir),
        "train_images": counts["train_images"],
        "val_images": counts["val_images"],
        "test_images": counts["test_images"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"],
        "val_map50": val_metrics["map50"],
        "val_map50_95": val_metrics["map50_95"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_map50": test_metrics["map50"],
        "test_map50_95": test_metrics["map50_95"],
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary_path, summary


def append_experiment_index(
    index_path: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    start_time: datetime,
) -> None:
    """Append one completed experiment without changing existing rows."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not index_path.exists() or index_path.stat().st_size == 0
    if not write_header:
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            existing_header = next(csv.reader(handle), [])
        if tuple(existing_header) != EXPERIMENT_INDEX_FIELDS:
            raise UserFacingError(
                f"Experiment index header is incompatible and was not modified: {index_path}"
            )
    row = {
        "run_name": summary["run_name"],
        "date": start_time.isoformat(timespec="seconds"),
        "model": config["model"],
        "epochs": config["epochs"],
        "imgsz": config["imgsz"],
        "batch": config["batch"],
        "optimizer": config["optimizer"],
        "lr0": config["lr0"],
        "weight_decay": config["weight_decay"],
        "augmentation": augmentation_label(config),
        "seed": config["seed"],
        "val_precision": summary["val_precision"],
        "val_recall": summary["val_recall"],
        "val_map50": summary["val_map50"],
        "val_map50_95": summary["val_map50_95"],
        "test_precision": summary["test_precision"],
        "test_recall": summary["test_recall"],
        "test_map50": summary["test_map50"],
        "test_map50_95": summary["test_map50_95"],
        "best_epoch": summary["best_epoch"],
        "best_model": summary["best_model"],
    }
    encoding = "utf-8-sig" if write_header else "utf-8"
    with index_path.open("a", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_INDEX_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _print_metrics(title: str, metrics: dict[str, float]) -> None:
    print(f"\n{title}:")
    print(f"Precision = {metrics['precision']:.6f}")
    print(f"Recall    = {metrics['recall']:.6f}")
    print(f"mAP50     = {metrics['map50']:.6f}")
    print(f"mAP50-95  = {metrics['map50_95']:.6f}")


def run() -> None:
    validate_xray_augmentation(CONFIG)
    start_time = datetime.now().astimezone()
    base_run_name = build_run_name(CONFIG)
    run_name = get_unique_run_name(Path(CONFIG["project"]), base_run_name)
    if not run_name.startswith(base_run_name):
        raise UserFacingError("Generated run name does not match the current CONFIG.")
    if (
        all(float(CONFIG[key]) == 0.0 for key in AUGMENTATION_KEYS)
        and CONFIG["gamma_p"] == 0
        and CONFIG["noise_p"] == 0
        and "noaug" not in run_name
    ):
        raise UserFacingError(
            "All augmentations are disabled, but run name lacks 'noaug'."
        )
    print(f"Generated run name:\n{run_name}")

    torch, torchvision, ultralytics, YOLO, default_cfg = load_dependencies()
    validate_ultralytics_arguments(default_cfg)
    try:
        import yaml
    except ImportError as exc:
        raise UserFacingError("PyYAML is required to read dataset.yaml.") from exc

    environment = collect_environment(torch, torchvision, ultralytics)
    dataset_root, counts = preflight_dataset(yaml)
    if CONFIG["optimizer"] == "auto":
        print(
            "\nNOTE: optimizer='auto' lets Ultralytics choose the optimizer and may "
            "override the requested lr0 and momentum values."
        )

    expected_run_dir = Path(CONFIG["project"]) / run_name
    if expected_run_dir.exists():
        raise UserFacingError(
            f"Generated run directory already exists: {expected_run_dir}"
        )

    _, run_dir, best_path = run_training(YOLO, CONFIG, run_name)
    config_path = save_config(run_dir, CONFIG, run_name, start_time, None)
    environment_path = save_environment(run_dir, environment)

    best_model = YOLO(str(best_path))
    val_metrics = run_validation(best_model, run_dir)
    test_metrics = run_test(best_model, run_dir)

    print(f"\nDataset: {dataset_root}")
    print(
        f"Split images: train={counts['train_images']}, "
        f"val={counts['val_images']}, test={counts['test_images']}"
    )
    prediction_dir = run_prediction(best_model, dataset_root, run_dir)

    end_time = datetime.now().astimezone()
    config_path = save_config(run_dir, CONFIG, run_name, start_time, end_time)
    summary_path, summary = save_summary(
        run_dir, run_name, best_path, counts, val_metrics, test_metrics
    )
    append_experiment_index(EXPERIMENT_INDEX, CONFIG, summary, start_time)

    print("\n========================================")
    print("YOLO EXPERIMENT COMPLETE")
    print("========================================")
    print(f"\nRun:\n{run_name}")
    print(f"\nOutput:\n{run_dir}")
    print(f"\nConfig:\n{config_path}")
    print(f"\nEnvironment:\n{environment_path}")
    print(f"\nSummary:\n{summary_path}")
    print(f"\nBest model:\n{best_path}")
    _print_metrics("Validation", val_metrics)
    _print_metrics("Test", test_metrics)
    print(f"\nTest predictions:\n{prediction_dir}")
    print(f"\nExperiment index:\n{EXPERIMENT_INDEX}")
    print("========================================")


def main() -> int:
    try:
        run()
        return 0
    except UserFacingError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nERROR: Training was interrupted by the user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"\nERROR: YOLO experiment failed: {exc}", file=sys.stderr)
        print(
            "Review the message above and the Ultralytics console output for details.",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
