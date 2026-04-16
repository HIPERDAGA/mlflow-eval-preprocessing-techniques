from __future__ import annotations

import argparse
import itertools
import os
import tempfile
from pathlib import Path
from typing import Iterable

import mlflow
import pandas as pd
from PIL import Image
from tqdm import tqdm

from Metrics.brisque_metric import compute_brisque
from Metrics.niqe_metric import compute_niqe
from Metrics.piqe_metric import compute_piqe
from Preprocessing.SimpleWB import process_image

CONDITIONS = [
    'fog_day', 'fog_night', 'fog_twilight',
    'rain_day', 'rain_night', 'rain_twilight',
    'sand_day', 'sand_night', 'sand_twilight',
    'snow_day', 'snow_night', 'snow_twilight',
]


def list_images(folder: Path) -> list[Path]:
    valid_ext = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in valid_ext])


def save_example_artifacts(original_path: Path, processed_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    Image.open(original_path).save(output_dir / f'original__{original_path.name}')
    Image.open(processed_path).save(output_dir / f'processed__{processed_path.name}')


def run_condition(
    dataset_root: Path,
    condition: str,
    clip_percent: float,
    preserve_luminance: bool,
    channel_gain_limit: float,
    experiment_name: str,
) -> dict:
    condition_dir = dataset_root / condition
    images = list_images(condition_dir)
    if not images:
        raise FileNotFoundError(f'No se encontraron imagenes en {condition_dir}')

    mlflow.set_experiment(experiment_name)
    run_name = (
        f'{condition}__simple_wb__clip_{clip_percent}'
        f'__pl_{str(preserve_luminance).lower()}__cgl_{channel_gain_limit}'
    )

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({
            'condition': condition,
            'technique': 'white_balance',
            'method': 'simple_wb',
            'clip_percent': clip_percent,
            'preserve_luminance': preserve_luminance,
            'channel_gain_limit': channel_gain_limit,
            'dataset_path': str(condition_dir),
            'num_images': len(images),
        })

        rows = []
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            example_dir = tmpdir_path / 'examples'
            for idx, image_path in enumerate(tqdm(images, desc=run_name)):
                processed_path = tmpdir_path / image_path.name
                process_image(
                    input_path=image_path,
                    output_path=processed_path,
                    clip_percent=clip_percent,
                    preserve_luminance=preserve_luminance,
                    channel_gain_limit=channel_gain_limit,
                )
                niqe = compute_niqe(processed_path)
                brisque = compute_brisque(processed_path)
                piqe = compute_piqe(processed_path)
                rows.append({
                    'image_name': image_path.name,
                    'condition': condition,
                    'method': 'simple_wb',
                    'clip_percent': clip_percent,
                    'preserve_luminance': preserve_luminance,
                    'channel_gain_limit': channel_gain_limit,
                    'niqe': niqe,
                    'brisque': brisque,
                    'piqe': piqe,
                })
                if idx < 3:
                    save_example_artifacts(image_path, processed_path, example_dir)

            df = pd.DataFrame(rows)
            summary = {
                'condition': condition,
                'method': 'simple_wb',
                'clip_percent': clip_percent,
                'preserve_luminance': preserve_luminance,
                'channel_gain_limit': channel_gain_limit,
                'mean_niqe': float(df['niqe'].mean()),
                'mean_brisque': float(df['brisque'].mean()),
                'mean_piqe': float(df['piqe'].mean()),
                'median_niqe': float(df['niqe'].median()),
                'median_brisque': float(df['brisque'].median()),
                'median_piqe': float(df['piqe'].median()),
            }

            mlflow.log_metrics({
                'mean_niqe': summary['mean_niqe'],
                'mean_brisque': summary['mean_brisque'],
                'mean_piqe': summary['mean_piqe'],
                'median_niqe': summary['median_niqe'],
                'median_brisque': summary['median_brisque'],
                'median_piqe': summary['median_piqe'],
            })

            per_image_csv = tmpdir_path / 'per_image_metrics.csv'
            summary_csv = tmpdir_path / 'summary.csv'
            df.to_csv(per_image_csv, index=False)
            pd.DataFrame([summary]).to_csv(summary_csv, index=False)
            mlflow.log_artifact(str(per_image_csv))
            mlflow.log_artifact(str(summary_csv))
            if example_dir.exists():
                mlflow.log_artifacts(str(example_dir), artifact_path='examples')

        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Evaluacion de Simple White Balance por condicion.')
    parser.add_argument('--dataset_root', type=str, required=True, help='Ruta raiz del dataset test.')
    parser.add_argument('--mlruns_dir', type=str, default='./mlruns', help='Ruta para almacenamiento MLflow.')
    parser.add_argument('--experiment_name', type=str, default='simple_wb_experiment')
    parser.add_argument('--conditions', nargs='*', default=CONDITIONS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    mlruns_dir = Path(args.mlruns_dir)
    mlruns_dir.mkdir(parents=True, exist_ok=True)

    os.environ['MLFLOW_TRACKING_URI'] = f'file:{mlruns_dir.resolve()}'
    mlflow.set_tracking_uri(os.environ['MLFLOW_TRACKING_URI'])

    grid = list(itertools.product([0, 1, 2, 5], [True, False], [1.2, 1.5, 2.0]))
    all_summaries = []
    for condition in args.conditions:
        for clip_percent, preserve_luminance, channel_gain_limit in grid:
            summary = run_condition(
                dataset_root=dataset_root,
                condition=condition,
                clip_percent=clip_percent,
                preserve_luminance=preserve_luminance,
                channel_gain_limit=channel_gain_limit,
                experiment_name=args.experiment_name,
            )
            all_summaries.append(summary)

    summary_df = pd.DataFrame(all_summaries)
    summary_df['rank_sum'] = (
        summary_df.groupby('condition')['mean_niqe'].rank(method='min') +
        summary_df.groupby('condition')['mean_brisque'].rank(method='min') +
        summary_df.groupby('condition')['mean_piqe'].rank(method='min')
    )
    best_by_condition = summary_df.sort_values(['condition', 'rank_sum', 'mean_niqe', 'mean_brisque', 'mean_piqe']).groupby('condition').head(1)

    outputs_dir = Path('outputs')
    outputs_dir.mkdir(exist_ok=True)
    summary_df.to_csv(outputs_dir / 'all_runs_summary.csv', index=False)
    best_by_condition.to_csv(outputs_dir / 'best_config_by_condition.csv', index=False)

    print('Experimento completado.')
    print(f'Resumen global: {outputs_dir / "all_runs_summary.csv"}')
    print(f'Mejor configuracion por condicion: {outputs_dir / "best_config_by_condition.csv"}')


if __name__ == '__main__':
    main()
