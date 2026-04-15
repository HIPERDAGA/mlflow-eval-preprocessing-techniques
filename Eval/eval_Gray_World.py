"""Full evaluation script for the Gray World experiment.

Run example in Colab:
python Eval/eval_Gray_World.py \
  --dataset_root /content/drive/MyDrive/Datasets/test \
  --mlruns_dir /content/drive/MyDrive/mlruns \
  --experiment_name gray_world_all_conditions
"""
from __future__ import annotations

import argparse
import itertools
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List

import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


import mlflow
import pandas as pd
from PIL import Image
from tqdm import tqdm

from Metrics.brisque_metric import BRISQUEMetric
from Metrics.niqe_metric import NIQEMetric
from Metrics.piqe_metric import compute_piqe
from Preprocessing.Gray_World import apply_gray_world, read_image_rgb, save_image_rgb

CONDITIONS: List[str] = [
    "fog_day", "fog_night", "fog_twilight",
    "rain_day", "rain_night", "rain_twilight",
    "sand_day", "sand_night", "sand_twilight",
    "snow_day", "snow_night", "snow_twilight",
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Gray World white balance with MLflow.")
    parser.add_argument("--dataset_root", required=True, help="Root path that contains the 12 condition folders")
    parser.add_argument("--mlruns_dir", required=True, help="Directory where MLflow will store runs")
    parser.add_argument("--experiment_name", default="gray_world_all_conditions", help="MLflow experiment name")
    parser.add_argument("--save_examples", type=int, default=3, help="Number of processed example images to log per run")
    return parser.parse_args()


def list_images(folder: str) -> List[str]:
    files: List[str] = []
    for root, _, names in os.walk(folder):
        for name in names:
            suffix = Path(name).suffix.lower()
            if suffix in IMAGE_EXTENSIONS:
                files.append(os.path.join(root, name))
    return sorted(files)


def aggregate_scores(values: Iterable[float]) -> float:
    series = pd.Series(list(values), dtype="float64").dropna()
    return float(series.mean()) if not series.empty else float("nan")


def run_single_configuration(
    dataset_root: str,
    condition: str,
    preserve_luminance: bool,
    channel_gain_limit: float,
    niqe_metric: NIQEMetric,
    brisque_metric: BRISQUEMetric,
    save_examples: int,
) -> Dict[str, float]:
    condition_dir = os.path.join(dataset_root, condition)
    image_paths = list_images(condition_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in: {condition_dir}")

    run_name = (
        f"{condition}__gray_world__pl_{str(preserve_luminance).lower()}"
        f"__cgl_{channel_gain_limit}"
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("condition", condition)
        mlflow.log_param("technique", "white_balance")
        mlflow.log_param("method", "gray_world")
        mlflow.log_param("preserve_luminance", preserve_luminance)
        mlflow.log_param("channel_gain_limit", channel_gain_limit)
        mlflow.log_param("dataset_path", condition_dir)
        mlflow.log_param("num_images", len(image_paths))

        rows: List[Dict[str, float]] = []
        artifact_dir = tempfile.mkdtemp(prefix="gray_world_eval_")
        examples_dir = os.path.join(artifact_dir, "examples")
        os.makedirs(examples_dir, exist_ok=True)

        for index, image_path in enumerate(tqdm(image_paths, desc=run_name)):
            image_rgb = read_image_rgb(image_path)
            processed_rgb = apply_gray_world(
                image_rgb=image_rgb,
                preserve_luminance=preserve_luminance,
                channel_gain_limit=channel_gain_limit,
            )

            niqe = niqe_metric.score(processed_rgb)
            brisque = brisque_metric.score(processed_rgb)
            piqe = compute_piqe(processed_rgb)

            rows.append({
                "file": os.path.basename(image_path),
                "path": image_path,
                "niqe": niqe,
                "brisque": brisque,
                "piqe": piqe,
            })

            if index < save_examples:
                output_example = os.path.join(examples_dir, f"{index:02d}_{os.path.basename(image_path)}")
                save_image_rgb(output_example, processed_rgb)

        results_df = pd.DataFrame(rows)
        mean_niqe = aggregate_scores(results_df["niqe"])
        mean_brisque = aggregate_scores(results_df["brisque"])
        mean_piqe = aggregate_scores(results_df["piqe"])

        mlflow.log_metric("mean_niqe", mean_niqe)
        mlflow.log_metric("mean_brisque", mean_brisque)
        mlflow.log_metric("mean_piqe", mean_piqe)

        per_image_csv = os.path.join(artifact_dir, "per_image_metrics.csv")
        summary_csv = os.path.join(artifact_dir, "summary.csv")

        results_df.to_csv(per_image_csv, index=False)
        pd.DataFrame([{
            "condition": condition,
            "method": "gray_world",
            "preserve_luminance": preserve_luminance,
            "channel_gain_limit": channel_gain_limit,
            "num_images": len(image_paths),
            "mean_niqe": mean_niqe,
            "mean_brisque": mean_brisque,
            "mean_piqe": mean_piqe,
        }]).to_csv(summary_csv, index=False)

        mlflow.log_artifact(per_image_csv, artifact_path="tables")
        mlflow.log_artifact(summary_csv, artifact_path="tables")
        mlflow.log_artifacts(examples_dir, artifact_path="examples")

        shutil.rmtree(artifact_dir, ignore_errors=True)

        return {
            "condition": condition,
            "preserve_luminance": preserve_luminance,
            "channel_gain_limit": channel_gain_limit,
            "mean_niqe": mean_niqe,
            "mean_brisque": mean_brisque,
            "mean_piqe": mean_piqe,
            "num_images": len(image_paths),
        }


def main() -> None:
    args = parse_args()

    os.makedirs(args.mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri(f"file://{args.mlruns_dir}")
    mlflow.set_experiment(args.experiment_name)

    param_grid = list(itertools.product([True, False], [1.2, 1.5, 2.0]))

    niqe_metric = NIQEMetric()
    brisque_metric = BRISQUEMetric()

    all_results: List[Dict[str, float]] = []
    for condition in CONDITIONS:
        for preserve_luminance, channel_gain_limit in param_grid:
            result = run_single_configuration(
                dataset_root=args.dataset_root,
                condition=condition,
                preserve_luminance=preserve_luminance,
                channel_gain_limit=channel_gain_limit,
                niqe_metric=niqe_metric,
                brisque_metric=brisque_metric,
                save_examples=args.save_examples,
            )
            all_results.append(result)

    results_df = pd.DataFrame(all_results)
    final_csv = os.path.join(args.mlruns_dir, "gray_world_global_results.csv")
    results_df.to_csv(final_csv, index=False)
    print(f"Saved global results to: {final_csv}")

    best_rows = []
    for condition, group in results_df.groupby("condition"):
        ranked = group.copy()
        ranked["rank_niqe"] = ranked["mean_niqe"].rank(method="min", ascending=True)
        ranked["rank_brisque"] = ranked["mean_brisque"].rank(method="min", ascending=True)
        ranked["rank_piqe"] = ranked["mean_piqe"].rank(method="min", ascending=True)
        ranked["rank_sum"] = ranked["rank_niqe"] + ranked["rank_brisque"] + ranked["rank_piqe"]
        best_rows.append(ranked.sort_values(["rank_sum", "mean_niqe", "mean_brisque", "mean_piqe"]).iloc[0])

    best_df = pd.DataFrame(best_rows)
    best_csv = os.path.join(args.mlruns_dir, "gray_world_best_by_condition.csv")
    best_df.to_csv(best_csv, index=False)
    print(f"Saved best-by-condition results to: {best_csv}")


if __name__ == "__main__":
    main()
