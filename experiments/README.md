# Experiment Tracking

This folder records the reproducibility information that can be stated from the committed project files.

The latest committed evaluation is in `evaluation/latest/`. The final Raspberry Pi thresholds are in `raspberry_pi_twostage_deploy/config.json`. Raw images, generated dataset folders, and full training runs are intentionally not committed.

## Current Reproducibility Status

- Default seed: `42`.
- Split method: grouped, class-aware split from `training/prepare_twostage_dataset.py`.
- Split ratio behavior: about 70% train, 20% validation, and remaining images for test, applied per related image group.
- Generated split manifest path: `training/road_sign_twostage_dataset/reports/splits.json`.
- Exact historical split manifest: not committed in this repository. The script writes it when the dataset is prepared.

That last point is important. The repository documents the deterministic split logic and seed, but it does not pretend to contain a split manifest that is not actually present.

## Training Flow

```bash
python training/prepare_twostage_dataset.py --raw-dir <pascal-voc-road-sign-dataset> --recreate --seed 42
python training/train_twostage_detector.py --candidate pi --seed 42 --epochs 150 --patience 30
python training/build_classifier_proposal_dataset.py --detector <detector-best.pt> --seed 42
python training/train_twostage_classifier.py --classifier-subdir classifier_crops_14class --epochs 40 --batch-size 64 --export-onnx
python training/evaluate_twostage_pipeline.py --split test --det-conf-threshold 0.15 --classifier-threshold 0.88
```

The commands above follow the committed script defaults and deployed thresholds. If a new training run is made, commit the generated `prep_report.json`, `splits.json` summary or hash, training command, model selection notes, and evaluation outputs under a dated experiment folder.

## Latest Committed Result

See `latest_evaluation_manifest.json` and `metrics_summary.csv` for the latest committed metrics. The recall is intentionally reported as-is; this project should be described as an edge-deployable prototype, not a production traffic-safety system.
