from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import itertools
import tempfile
from typing import Dict, List, Tuple

import cv2
import mlflow
import pandas as pd

from Metrics.brisque_metric import compute_brisque
from Metrics.niqe_metric import compute_niqe
from Metrics.piqe_metric import compute_piqe
from Preprocessing.HistogramEqualization import apply_histogram_equalization


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluación de Ecualización de Histograma con MLflow")
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
        default="histogram_equalization_experiment",
        help="Nombre del experimento en MLflow.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_histogram_equalization",
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


def save_preview_examples(
    original_bgr,
    processed_bgr,
    save_dir: Path,
    stem: str,
) -> None:
    ensure_dir(save_dir)
    cv2.imwrite(str(save_dir / f"{stem}_original.png"), original_bgr)
    cv2.imwrite(str(save_dir / f"{stem}_processed.png"), processed_bgr)


def compute_metrics_from_bgr(image_bgr) -> Dict[str, float]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "temp_metric_image.png"
        ok = cv2.imwrite(str(tmp_path), image_bgr)
        if not ok:
            raise IOError("No se pudo guardar la imagen temporal para métricas.")

        return {
            "niqe": float(compute_niqe(tmp_path)),
            "brisque": float(compute_brisque(tmp_path)),
            "piqe": float(compute_piqe(tmp_path)),
        }


def build_parameter_grid() -> List[Dict]:
    configs: List[Dict] = []

    # Ecualización global: 2 espacios x 2 preserve = 4 configuraciones
    for color_space, preserve_luminance in itertools.product(
        ["ycrcb", "lab"],
        [True, False],
    ):
        configs.append(
            {
                "method": "global_he",
                "color_space": color_space,
                "preserve_luminance": preserve_luminance,
                "clip_limit": None,
                "tile_grid_size": None,
            }
        )

    # CLAHE: 4 clip_limits x 3 tiles x 2 espacios x 2 preserve = 48 configuraciones
    for clip_limit, tile_grid_size, color_space, preserve_luminance in itertools.product(
        [1.0, 2.0, 3.0, 4.0],
        [(4, 4), (8, 8), (16, 16)],
        ["ycrcb", "lab"],
        [True, False],
    ):
        configs.append(
            {
                "method": "clahe",
                "color_space": color_space,
                "preserve_luminance": preserve_luminance,
                "clip_limit": clip_limit,
                "tile_grid_size": tile_grid_size,
            }
        )

    return configs


def format_run_name(
    condition_name: str,
    method: str,
    color_space: str,
    preserve_luminance: bool,
    clip_limit,
    tile_grid_size,
) -> str:
    preserve_tag = str(preserve_luminance).lower()

    if method == "global_he":
        return (
            f"{condition_name}__global_he__"
            f"cs_{color_space}__"
            f"pl_{preserve_tag}"
        )

    tile_tag = f"{tile_grid_size[0]}x{tile_grid_size[1]}"
    return (
        f"{condition_name}__clahe__"
        f"clip_{clip_limit}__"
        f"tile_{tile_tag}__"
        f"cs_{color_space}__"
        f"pl_{preserve_tag}"
    )


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

    parameter_grid = build_parameter_grid()
    all_run_rows = []

    for condition_dir in condition_dirs:
        condition_name = condition_dir.name
        image_paths = list_images(condition_dir)

        if not image_paths:
            print(f"[WARN] Sin imágenes en condición: {condition_name}")
            continue

        for cfg in parameter_grid:
            method = cfg["method"]
            color_space = cfg["color_space"]
            preserve_luminance = cfg["preserve_luminance"]
            clip_limit = cfg["clip_limit"]
            tile_grid_size = cfg["tile_grid_size"]

            run_name = format_run_name(
                condition_name=condition_name,
                method=method,
                color_space=color_space,
                preserve_luminance=preserve_luminance,
                clip_limit=clip_limit,
                tile_grid_size=tile_grid_size,
            )

            per_image_rows = []
            preview_saved = False

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("condition", condition_name)
                mlflow.log_param("technique", "histogram_equalization")
                mlflow.log_param("method", method)
                mlflow.log_param("color_space", color_space)
                mlflow.log_param("preserve_luminance", preserve_luminance)
                mlflow.log_param("dataset_path", str(condition_dir))
                mlflow.log_param("num_images", len(image_paths))

                if clip_limit is not None:
                    mlflow.log_param("clip_limit", clip_limit)
                if tile_grid_size is not None:
                    mlflow.log_param("tile_grid_size", f"{tile_grid_size[0]}x{tile_grid_size[1]}")

                for img_path in image_paths:
                    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        print(f"[WARN] No se pudo leer: {img_path}")
                        continue

                    processed_bgr = apply_histogram_equalization(
                        image_bgr=img_bgr,
                        method=method,
                        color_space=color_space,
                        preserve_luminance=preserve_luminance,
                        clip_limit=clip_limit if clip_limit is not None else 2.0,
                        tile_grid_size=tile_grid_size if tile_grid_size is not None else (8, 8),
                    )

                    metrics = compute_metrics_from_bgr(processed_bgr)

                    row = {
                        "condition": condition_name,
                        "image_name": img_path.name,
                        "method": method,
                        "color_space": color_space,
                        "preserve_luminance": preserve_luminance,
                        "clip_limit": clip_limit,
                        "tile_grid_size": None if tile_grid_size is None else f"{tile_grid_size[0]}x{tile_grid_size[1]}",
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
                    "technique": "histogram_equalization",
                    "method": method,
                    "color_space": color_space,
                    "preserve_luminance": preserve_luminance,
                    "clip_limit": clip_limit,
                    "tile_grid_size": None if tile_grid_size is None else f"{tile_grid_size[0]}x{tile_grid_size[1]}",
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
