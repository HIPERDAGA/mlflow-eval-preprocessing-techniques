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
from Preprocessing.GammaCorrection import apply_gamma_correction


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluación de Gamma Correction con MLflow"
    )
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
        default="gamma_correction_experiment",
        help="Nombre del experimento en MLflow.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_gamma_correction",
        help="Directorio donde se guardan CSVs y previews.",
    )
    parser.add_argument(
        "--gamma_values",
        type=float,
        nargs="+",
        default=[0.6, 0.8, 1.0, 1.2, 1.4],
        help="Valores gamma a evaluar.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_condition_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def list_images(condition_dir: Path) -> List[Path]:
    return sorted(
        [
            p
            for p in condition_dir.iterdir()
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


def bgr_to_rgb_uint8(image_bgr):
    if image_bgr is None:
        raise ValueError("image_bgr es None")

    if not hasattr(image_bgr, "ndim") or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            f"Se esperaba imagen BGR HxWx3, recibido shape={getattr(image_bgr, 'shape', None)}"
        )

    if image_bgr.dtype != "uint8":
        image_bgr = image_bgr.astype("uint8")

    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def compute_metrics_from_bgr(
    image_bgr,
    niqe_metric: NIQEMetric,
    brisque_metric: BRISQUEMetric,
) -> Dict[str, float]:
    image_rgb = bgr_to_rgb_uint8(image_bgr)

    return {
        "niqe": float(niqe_metric.score(image_rgb)),
        "brisque": float(brisque_metric.score(image_rgb)),
        "piqe": float(compute_piqe(image_rgb)),
    }


def build_parameter_grid() -> List[Dict]:
    configs: List[Dict] = []

    for gamma, color_space, preserve_luminance in itertools.product(
        [0.6, 0.8, 1.0, 1.2, 1.4],
        ["rgb", "ycrcb", "lab"],
        [True, False],
    ):
        configs.append(
            {
                "gamma": gamma,
                "color_space": color_space,
                "preserve_luminance": preserve_luminance,
            }
        )

    return configs


def format_run_name(
    condition_name: str,
    gamma: float,
    color_space: str,
    preserve_luminance: bool,
) -> str:
    gamma_tag = str(gamma).replace(".", "_")
    preserve_tag = str(preserve_luminance).lower()

    return (
        f"{condition_name}_gamma_"
        f"g_{gamma_tag}_"
        f"cs_{color_space}_"
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
        raise FileNotFoundError(
            f"No se encontraron carpetas de condición en: {dataset_root}"
        )

    parameter_grid = build_parameter_grid()
    all_run_rows = []

    niqe_metric = NIQEMetric()
    brisque_metric = BRISQUEMetric()

    for condition_dir in condition_dirs:
        condition_name = condition_dir.name
        image_paths = list_images(condition_dir)

        if not image_paths:
            print(f"[WARN] Sin imágenes en condición: {condition_name}")
            continue

        for cfg in parameter_grid:
            gamma = cfg["gamma"]
            color_space = cfg["color_space"]
            preserve_luminance = cfg["preserve_luminance"]

            run_name = format_run_name(
                condition_name=condition_name,
                gamma=gamma,
                color_space=color_space,
                preserve_luminance=preserve_luminance,
            )

            per_image_rows = []
            preview_saved = False

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param("condition", condition_name)
                mlflow.log_param("technique", "gamma_correction")
                mlflow.log_param("gamma", gamma)
                mlflow.log_param("color_space", color_space)
                mlflow.log_param("preserve_luminance", preserve_luminance)
                mlflow.log_param("dataset_path", str(condition_dir))
                mlflow.log_param("num_images", len(image_paths))

                for img_path in image_paths:
                    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        print(f"[WARN] No se pudo leer: {img_path}")
                        continue

                    processed_bgr = apply_gamma_correction(
                        image_bgr=img_bgr,
                        gamma=gamma,
                        color_space=color_space,
                        preserve_luminance=preserve_luminance,
                    )

                    metrics = compute_metrics_from_bgr(
                        processed_bgr,
                        niqe_metric=niqe_metric,
                        brisque_metric=brisque_metric,
                    )

                    row = {
                        "condition": condition_name,
                        "image_name": img_path.name,
                        "method": "gamma_correction",
                        "gamma": gamma,
                        "color_space": color_space,
                        "preserve_luminance": preserve_luminance,
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
                    mlflow.end_run(status="FAILED")
                    continue

                per_image_df = pd.DataFrame(per_image_rows)

                summary = {
                    "condition": condition_name,
                    "method": "gamma_correction",
                    "gamma": gamma,
                    "color_space": color_space,
                    "preserve_luminance": preserve_luminance,
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

                mlflow.log_artifact(str(per_image_csv), artifact_path="tables")
                mlflow.log_artifact(str(summary_csv), artifact_path="tables")

                preview_dir = output_dir / "previews" / run_name
                if preview_dir.exists():
                    for preview_file in preview_dir.glob("*"):
                        mlflow.log_artifact(str(preview_file), artifact_path="previews")

                all_run_rows.append(summary)

                print(
                    f"[OK] {run_name} | "
                    f"NIQE={summary['mean_niqe']:.4f} | "
                    f"BRISQUE={summary['mean_brisque']:.4f} | "
                    f"PIQE={summary['mean_piqe']:.4f}"
                )

    if not all_run_rows:
        raise RuntimeError("No se generaron resultados. Verifica dataset y parámetros.")

    final_df = pd.DataFrame(all_run_rows).sort_values(
        by=["condition", "composite_score", "mean_niqe", "mean_brisque", "mean_piqe"],
        ascending=[True, True, True, True, True],
    )

    final_csv = output_dir / "gamma_correction_all_runs_summary.csv"
    final_df.to_csv(final_csv, index=False)

    best_by_condition = (
        final_df.groupby("condition", as_index=False)
        .first()
        .sort_values(by="condition", ascending=True)
    )
    best_csv = output_dir / "best_gamma_by_condition.csv"
    best_by_condition.to_csv(best_csv, index=False)

    print(f"\nResumen global guardado en: {final_csv}")
    print(f"Mejor configuración por condición: {best_csv}")
    print("\nTop configuraciones por condición:")
    print(best_by_condition.to_string(index=False))


if __name__ == "__main__":
    main()
