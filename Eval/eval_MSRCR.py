from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import argparse
import cv2
import mlflow
import pandas as pd
from typing import Dict, List

from Metrics.brisque_metric import BRISQUEMetric
from Metrics.niqe_metric import NIQEMetric
from Metrics.piqe_metric import compute_piqe
from Preprocessing.MSRCR import apply_msrcr

VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluación de MSRCR con MLflow')
    parser.add_argument('--dataset_root', type=str, required=True)
    parser.add_argument('--mlruns_dir', type=str, required=True)
    parser.add_argument('--experiment_name', type=str, default='msrcr_experiment')
    parser.add_argument('--output_dir', type=str, default='outputs_msrcr')
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def list_condition_dirs(dataset_root: Path) -> List[Path]:
    return sorted([p for p in dataset_root.iterdir() if p.is_dir()])


def list_images(condition_dir: Path) -> List[Path]:
    return sorted([p for p in condition_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS])


def bgr_to_rgb_uint8(image_bgr):
    if image_bgr is None:
        raise ValueError('image_bgr es None')
    if image_bgr.dtype != 'uint8':
        image_bgr = image_bgr.astype('uint8')
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def compute_metrics_from_bgr(image_bgr, niqe_metric: NIQEMetric, brisque_metric: BRISQUEMetric) -> Dict[str, float]:
    image_rgb = bgr_to_rgb_uint8(image_bgr)
    return {
        'niqe': float(niqe_metric.score(image_rgb)),
        'brisque': float(brisque_metric.score(image_rgb)),
        'piqe': float(compute_piqe(image_rgb)),
    }


def save_preview_examples(original_bgr, processed_bgr, save_dir: Path, stem: str) -> None:
    ensure_dir(save_dir)
    cv2.imwrite(str(save_dir / f'{stem}_original.png'), original_bgr)
    cv2.imwrite(str(save_dir / f'{stem}_processed.png'), processed_bgr)


def build_parameter_grid() -> List[Dict]:
    return [
        {'sigmas': (15, 80, 250), 'weights': (1/3, 1/3, 1/3), 'alpha': 100.0, 'beta': 20.0, 'gain': 1.0, 'offset': 0.0, 'preserve_luminance': True},
        {'sigmas': (15, 80, 250), 'weights': (1/3, 1/3, 1/3), 'alpha': 125.0, 'beta': 46.0, 'gain': 1.0, 'offset': 0.0, 'preserve_luminance': True},
        {'sigmas': (15, 80, 250), 'weights': (1/3, 1/3, 1/3), 'alpha': 150.0, 'beta': 60.0, 'gain': 1.0, 'offset': 0.0, 'preserve_luminance': True},
        {'sigmas': (10, 50, 150), 'weights': (0.2, 0.5, 0.3), 'alpha': 125.0, 'beta': 46.0, 'gain': 1.0, 'offset': 0.0, 'preserve_luminance': True},
        {'sigmas': (20, 100, 280), 'weights': (0.2, 0.5, 0.3), 'alpha': 125.0, 'beta': 46.0, 'gain': 1.0, 'offset': 0.0, 'preserve_luminance': False},
    ]


def format_run_name(condition_name: str, cfg: Dict) -> str:
    sigmas = '-'.join(str(int(s)) for s in cfg['sigmas'])
    return (
        f"{condition_name}_msrcr_s_{sigmas}_a_{int(cfg['alpha'])}_b_{int(cfg['beta'])}_"
        f"pl_{str(cfg['preserve_luminance']).lower()}"
    )


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    mlruns_dir = Path(args.mlruns_dir)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    ensure_dir(mlruns_dir)
    mlflow.set_tracking_uri(f'file:{mlruns_dir.resolve()}')
    mlflow.set_experiment(args.experiment_name)

    niqe_metric = NIQEMetric()
    brisque_metric = BRISQUEMetric()
    all_run_rows = []

    for condition_dir in list_condition_dirs(dataset_root):
        image_paths = list_images(condition_dir)
        if not image_paths:
            continue
        for cfg in build_parameter_grid():
            run_name = format_run_name(condition_dir.name, cfg)
            per_image_rows = []
            preview_saved = False
            with mlflow.start_run(run_name=run_name):
                for k, v in cfg.items():
                    mlflow.log_param(k, str(v) if isinstance(v, tuple) else v)
                mlflow.log_param('condition', condition_dir.name)
                mlflow.log_param('technique', 'msrcr')
                mlflow.log_param('num_images', len(image_paths))
                for img_path in image_paths:
                    img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
                    if img_bgr is None:
                        continue
                    processed_bgr = apply_msrcr(
                        image_bgr=img_bgr,
                        sigmas=cfg['sigmas'],
                        weights=cfg['weights'],
                        alpha=cfg['alpha'],
                        beta=cfg['beta'],
                        gain=cfg['gain'],
                        offset=cfg['offset'],
                        preserve_luminance=cfg['preserve_luminance'],
                    )
                    metrics = compute_metrics_from_bgr(processed_bgr, niqe_metric, brisque_metric)
                    per_image_rows.append({
                        'condition': condition_dir.name,
                        'image_name': img_path.name,
                        'method': 'msrcr',
                        'sigmas': str(cfg['sigmas']),
                        'weights': str(cfg['weights']),
                        'alpha': cfg['alpha'],
                        'beta': cfg['beta'],
                        'gain': cfg['gain'],
                        'offset': cfg['offset'],
                        'preserve_luminance': cfg['preserve_luminance'],
                        **metrics,
                    })
                    if not preview_saved:
                        preview_dir = output_dir / 'previews' / run_name
                        save_preview_examples(img_bgr, processed_bgr, preview_dir, img_path.stem)
                        preview_saved = True

                if not per_image_rows:
                    continue

                df = pd.DataFrame(per_image_rows)
                summary = {
                    'condition': condition_dir.name,
                    'method': 'msrcr',
                    'sigmas': str(cfg['sigmas']),
                    'weights': str(cfg['weights']),
                    'alpha': cfg['alpha'],
                    'beta': cfg['beta'],
                    'gain': cfg['gain'],
                    'offset': cfg['offset'],
                    'preserve_luminance': cfg['preserve_luminance'],
                    'num_images': int(len(df)),
                    'mean_niqe': float(df['niqe'].mean()),
                    'mean_brisque': float(df['brisque'].mean()),
                    'mean_piqe': float(df['piqe'].mean()),
                }
                summary['composite_score'] = summary['mean_niqe'] + summary['mean_brisque'] + summary['mean_piqe']
                for m in ['mean_niqe', 'mean_brisque', 'mean_piqe', 'composite_score']:
                    mlflow.log_metric(m, summary[m])
                run_output_dir = output_dir / 'runs' / run_name
                ensure_dir(run_output_dir)
                per_image_csv = run_output_dir / 'per_image_metrics.csv'
                summary_csv = run_output_dir / 'summary.csv'
                df.to_csv(per_image_csv, index=False)
                pd.DataFrame([summary]).to_csv(summary_csv, index=False)
                mlflow.log_artifact(str(per_image_csv), artifact_path='tables')
                mlflow.log_artifact(str(summary_csv), artifact_path='tables')
                all_run_rows.append(summary)

    final_df = pd.DataFrame(all_run_rows)
    final_df = final_df.sort_values(by=['condition', 'composite_score'], ascending=[True, True])
    final_df.to_csv(output_dir / 'msrcr_all_runs_summary.csv', index=False)
    best_by_condition = final_df.groupby('condition', as_index=False).first().sort_values(by='condition')
    best_by_condition.to_csv(output_dir / 'best_msrcr_by_condition.csv', index=False)
    print(best_by_condition.to_string(index=False))


if __name__ == '__main__':
    main()
