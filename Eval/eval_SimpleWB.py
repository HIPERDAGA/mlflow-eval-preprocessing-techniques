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

from Metrics.brisque_metric import compute_brisque
from Metrics.niqe_metric import compute_niqe
from Metrics.piqe_metric import compute_piqe
from Preprocessing.SimpleWB import apply_simple_wb


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación de Simple WB con MLflow")
    parser.add_argument(
        "--dataset_root",
        type=str,
        required=True,
        help="Ruta raíz del dataset. Debe contener carpetas por condición.",
    )
    parser.add_argument(
        "--mlruns_dir",
        type=str,
        required=True,
        help="Ruta donde MLflow almacenará los runs.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="simple_wb_experiment",
        help="Nombre del experimento en MLflow.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directorio donde se guardan CSVs y previews.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_condition_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def list_images(condition_dir: Path) -> List[Path]:
    return sorted(
        [
            p for p in condition_dir.iterdir()
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]
    )


def compute_metrics(image_bgr) -> Dict[str, float]:
    return {
        "niqe": float(compute_niqe(image_bgr)),
        "brisque": float(compute_brisque(image_bgr)),
        "piqe": float(compute_piqe(image_bgr)),
    }


def save_preview_examples(
    original_bgr,
    processed_bgr,
    save_dir: Path,
    stem: str,
) -> None:
    ensure_dir(save_dir)
    cv2.imwrite(str(save_dir / f"{stem}_original.png"), original_bgr)
    cv2.imwrite(str(save_dir / f"{stem}_processed.png"), processed_bgr)


def main() -> None:
    args = parse_args()

    dataset_root = Path(args.dataset_root)
    mlruns_dir = Path(args.mlruns_dir)
    output_dir = Path(args.output_dir)

    ensure_dir(output_dir)
    ensure_dir(mlruns_dir)

    mlflow.set_tracking_uri(f"file:{mlruns_dir.resolve()}")
    mlflow.set_experiment(args.experiment_name)

    if not dataset_root.exists():
        raise FileNotFoundError(f"No existe dataset_root: {dataset_root}")

    condition_dirs = list_condition_dirs(dataset_root)
    if not condition_dirs:
        raise FileNotFoundError(f"No se encontraron carpetas de condición en: {dataset_root}")

    clip_values = [0, 1, 2, 5]
    preserve_values = [True, False]
    gain_values = [1.2, 1.5, 2.0]

    all_run_rows = []

    for condition_dir in condition_dirs:
        condition_name = condition_dir.name
        image_paths = list_images(condition_dir)

        if not image_paths:
            print(f"[WARN] Sin imágenes en condición: {condition_name}")
            continue

        grid = list(itertools.product(clip_values, preserve_values, gain_values))

        for clip_percent, preserve_luminance, channel_gain_limit in grid:
            run_name = (
                f"{condition_name}__simple_wb__"
                f"clip_{clip_percent}__"
                f"pl_{str(preserve_luminance).lower()}__"
                f"cgl_{channel_gain_limit}"
            )

            per_image_rows = []
            preview_saved = False

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("condition", condition_name)
                mlflow.log_param("technique", "white_balance")
                mlflow.log_param("method", "simple_wb")
                mlflow.log_param("clip_percent", clip_percent)
                mlflow.log_param("preserve_luminance", preserve_luminance)
                mlflow.log_param("channel_gain_limit", channel_gain_limit)
                mlflow.log_param("dataset_path", str(condition_dir))
                mlflow.log_param("num_images", len(image_paths))

                for img_path in image_paths:
                    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        print(f"[WARN] No se pudo leer: {img_path}")
                        continue

                    processed_bgr = apply_simple_wb(
                        image_bgr=img_bgr,
                        clip_percent=clip_percent,
                        preserve_luminance=preserve_luminance,
                        channel_gain_limit=channel_gain_limit,
                    )

                    metrics = compute_metrics(processed_bgr)

                    row = {
                        "condition": condition_name,
                        "image_name": img_path.name,
                        "method": "simple_wb",
                        "clip_percent": clip_percent,
                        "preserve_luminance": preserve_luminance,
                        "channel_gain_limit": channel_gain_limit,
                        "niqe": metrics["niqe"],
                        "brisque": metrics["brisque"],
                        "piqe": metrics["piqe"],
                    }
                    per_image_rows.append(row)

                    if not preview_saved:
                        preview_dir = output_dir / "previews" / run_name
                        save_preview_examples(
                            original_bgr=img_bgr,
                            processed_bgr=processed_bgr,
                            save_dir=preview_dir,
                            stem=img_path.stem,
                        )
                        preview_saved = True

                if not per_image_rows:
                    print(f"[WARN] Run vacío: {run_name}")
                    continue

                per_image_df = pd.DataFrame(per_image_rows)

                summary = {
                    "condition": condition_name,
                    "method": "simple_wb",
                    "clip_percent": clip_percent,
                    "preserve_luminance": preserve_luminance,
                    "channel_gain_limit": channel_gain_limit,
                    "num_images": int(len(per_image_df)),
                    "mean_niqe": float(per_image_df["niqe"].mean()),
                    "mean_brisque": float(per_image_df["brisque"].mean()),
                    "mean_piqe": float(per_image_df["piqe"].mean()),
                }

                summary["composite_score"] = (
                    summary["mean_niqe"]
                    + summary["mean_brisque"]
                    + summary["mean_piqe"]
                )

                mlflow.log_metric("mean_niqe", summary["mean_niqe"])
                mlflow.log_metric("mean_brisque", summary["mean_brisque"])
                mlflow.log_metric("mean_piqe", summary["mean_piqe"])
                mlflow.log_metric("composite_score", summary["composite_score"])

                run_output_dir = output_dir / "runs" / run_name
                ensure_dir(run_output_dir)

                per_image_csv = run_output_dir / "per_image_metrics.csv"
                summary_csv = run_output_dir / "summary.csv"

                per_image_df.to_csv(per_image_csv, index=False)
                pd.DataFrame([summary]).to_csv(summary_csv, index=False)

                mlflow.log_artifact(str(per_image_csv))
                mlflow.log_artifact(str(summary_csv))

                preview_dir = output_dir / "previews" / run_name
                if preview_dir.exists():
                    for f in preview_dir.iterdir():
                        mlflow.log_artifact(str(f), artifact_path="previews")

                all_run_rows.append(summary)

                print(
                    f"[OK] {run_name} | "
                    f"NIQE={summary['mean_niqe']:.4f} | "
                    f"BRISQUE={summary['mean_brisque']:.4f} | "
                    f"PIQE={summary['mean_piqe']:.4f}"
                )

    if not all_run_rows:
        raise RuntimeError("No se generaron resultados.")

    all_runs_df = pd.DataFrame(all_run_rows)
    all_runs_csv = output_dir / "all_runs_summary.csv"
    all_runs_df.to_csv(all_runs_csv, index=False)

    best_by_condition = (
        all_runs_df.sort_values(
            by=["condition", "composite_score", "mean_niqe", "mean_brisque", "mean_piqe"],
            ascending=[True, True, True, True, True],
        )
        .groupby("condition", as_index=False)
        .first()
    )

    best_csv = output_dir / "best_config_by_condition.csv"
    best_by_condition.to_csv(best_csv, index=False)

    print("\n===== RESULTADOS FINALES =====")
    print(f"Resumen global: {all_runs_csv}")
    print(f"Mejor configuración por condición: {best_csv}")


if __name__ == "__main__":
    main()
