from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import itertools
from typing import Dict, List

import cv2
import mlflow
import pandas as pd

from Metrics.brisque_metric import BRISQUEMetric
from Metrics.niqe_metric import NIQEMetric
from Metrics.piqe_metric import compute_piqe
from Preprocessing.BilateralFilter import apply_bilateral_filter


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluacion de filtro bilateral con MLflow")
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--mlruns_dir", type=str, required=True)
    parser.add_argument("--experiment_name", type=str, default="bilateral_filter_experiment")
    parser.add_argument("--output_dir", type=str, default="outputs_bilateral_filter")
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_condition_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def list_images(condition_dir: Path) -> List[Path]:
    return sorted([p for p in condition_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])


def bgr_to_rgb_uint8(image_bgr):
    if image_bgr is None:
        raise ValueError("image_bgr es None")
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(f"Se esperaba imagen BGR HxWx3, recibido shape={getattr(image_bgr, 'shape', None)}")
    if image_bgr.dtype != "uint8":
        image_bgr = image_bgr.astype("uint8")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def compute_metrics_from_bgr(image_bgr, niqe_metric: NIQEMetric, brisque_metric: BRISQUEMetric) -> Dict[str, float]:
    image_rgb = bgr_to_rgb_uint8(image_bgr)
    return {
        "niqe": float(niqe_metric.score(image_rgb)),
        "brisque": float(brisque_metric.score(image_rgb)),
        "piqe": float(compute_piqe(image_rgb)),
    }


def build_parameter_grid() -> List[Dict]:
    configs = []
    for d, sigma_color, sigma_space, color_space, preserve_luminance in itertools.product(
        [5, 7, 9],
        [25, 50, 75],
        [25, 50, 75],
        ["bgr", "ycrcb", "lab"],
        [True, False],
    ):
        configs.append({
            "d": d,
            "sigma_color": sigma_color,
            "sigma_space": sigma_space,
            "color_space": color_space,
            "preserve_luminance": preserve_luminance,
        })
    return configs


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    mlruns_dir = Path(args.mlruns_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(mlruns_dir)
    mlflow.set_tracking_uri(f"file:{mlruns_dir.resolve()}")
    mlflow.set_experiment(args.experiment_name)

    condition_dirs = list_condition_dirs(dataset_root)
    parameter_grid = build_parameter_grid()
    all_run_rows = []
    niqe_metric = NIQEMetric()
    brisque_metric = BRISQUEMetric()

    for condition_dir in condition_dirs:
        image_paths = list_images(condition_dir)
        if not image_paths:
            continue
        for cfg in parameter_grid:
            run_name = (
                f"{condition_dir.name}_d_{cfg['d']}_sc_{cfg['sigma_color']}_ss_{cfg['sigma_space']}_"
                f"cs_{cfg['color_space']}_pl_{str(cfg['preserve_luminance']).lower()}"
            )
            rows = []
            with mlflow.start_run(run_name=run_name):
                for k, v in cfg.items():
                    mlflow.log_param(k, v)
                mlflow.log_param("condition", condition_dir.name)
                mlflow.log_param("technique", "bilateral_filter")
                for img_path in image_paths:
                    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        continue
                    processed_bgr = apply_bilateral_filter(
                        image_bgr=img_bgr,
                        d=cfg["d"],
                        sigma_color=cfg["sigma_color"],
                        sigma_space=cfg["sigma_space"],
                        color_space=cfg["color_space"],
                        preserve_luminance=cfg["preserve_luminance"],
                    )
                    metrics = compute_metrics_from_bgr(processed_bgr, niqe_metric, brisque_metric)
                    rows.append({
                        "condition": condition_dir.name,
                        "image_name": img_path.name,
                        **cfg,
                        **metrics,
                    })
                if not rows:
                    continue
                df = pd.DataFrame(rows)
                summary = {
                    "condition": condition_dir.name,
                    **cfg,
                    "num_images": int(len(df)),
                    "mean_niqe": float(df["niqe"].mean()),
                    "mean_brisque": float(df["brisque"].mean()),
                    "mean_piqe": float(df["piqe"].mean()),
                }
                summary["composite_score"] = summary["mean_niqe"] + summary["mean_brisque"] + summary["mean_piqe"]
                for metric_name in ["mean_niqe", "mean_brisque", "mean_piqe", "composite_score"]:
                    mlflow.log_metric(metric_name, summary[metric_name])
                run_output_dir = output_dir / "runs" / run_name
                ensure_dir(run_output_dir)
                df.to_csv(run_output_dir / "per_image_metrics.csv", index=False)
                pd.DataFrame([summary]).to_csv(run_output_dir / "summary.csv", index=False)
                all_run_rows.append(summary)

    final_df = pd.DataFrame(all_run_rows)
    if final_df.empty:
        raise RuntimeError("No se generaron resultados.")
    final_df = final_df.sort_values(by=["condition", "composite_score"])
    final_df.to_csv(output_dir / "bilateral_filter_all_runs_summary.csv", index=False)
    best_by_condition = final_df.groupby("condition", as_index=False).first()
    best_by_condition.to_csv(output_dir / "best_bilateral_filter_by_condition.csv", index=False)
    print(best_by_condition.to_string(index=False))


if __name__ == "__main__":
    main()
