# Road Sign Detection with YOLO and ONNX

Two-stage deep learning pipeline for detecting and recognizing road signs in images, video, and Raspberry Pi camera streams.

This is a computer vision / deep learning project, not a generic ML project. It is prepared as an edge-AI prototype for Raspberry Pi 4B deployment: the pipeline first detects road-sign regions using a YOLO detector, then classifies each crop with an ONNX classifier.

## Pipeline

1. YOLO road-sign detector finds candidate sign bounding boxes.
2. Crop classifier recognizes the sign class or rejects it as `other_sign`.
3. Raspberry Pi inference script runs the final two-stage model on camera, image, or video input.

## Included Final Models

- `raspberry_pi_twostage_deploy/models/detector_ncnn/best_ncnn_model/` - final YOLO NCNN detector export.
- `raspberry_pi_twostage_deploy/models/detector_ncnn_416/best_ncnn_model/` - 416px YOLO NCNN detector export used by the balanced Pi profile.
- `raspberry_pi_twostage_deploy/models/detector_pt/best.pt` - PyTorch detector fallback.
- `raspberry_pi_twostage_deploy/models/classifier/best_classifier.onnx` - ONNX crop classifier.
- `raspberry_pi_twostage_deploy/models/classifier/best_classifier.onnx.data` - ONNX classifier weights sidecar.

## Repository Structure

- `training/` - dataset preparation, detector training, classifier training, hard-negative mining, evaluation, and two-stage inference scripts.
- `notebooks/` - compact two-stage project notebook.
- `evaluation/latest/` - latest evaluation metrics and failure-case summary.
- `raspberry_pi_twostage_deploy/` - final deployable Raspberry Pi package with scripts, config, requirements, and model artifacts.
- `MODEL_CARD.md` - model summary, classes, thresholds, and evaluation notes.

## Latest Evaluation Snapshot

From `evaluation/latest/metrics.json`:

- Test images: 9,062
- End-to-end precision: 0.783
- End-to-end recall: 0.481
- End-to-end F1: 0.596
- Macro F1: 0.595
- `other_sign` reject rate: 0.999

The deployed thresholds are stored in `raspberry_pi_twostage_deploy/config.json`.

This evaluation is reported honestly. The prototype has strong rejection behavior for unknown/other signs, but recall is still the main improvement area. It should be presented as an edge-deployable research/engineering prototype, not as a production-ready traffic-safety system.

## Raspberry Pi Usage

```bash
cd raspberry_pi_twostage_deploy
bash install_pi.sh
bash run_camera_416_highres.sh
```

For more deployment commands, see `raspberry_pi_twostage_deploy/README_PI.md`.

## Training Notes

The `training/` scripts are included to show the full ML engineering workflow. Raw datasets and training run folders are excluded, so users need to provide their own Pascal VOC-style dataset with `--raw-dir`.

Detector training uses Ultralytics model references such as `yolo26n.pt` and `yolo26s.pt`; these are not committed as source files. The final deployment folder includes the trained detector exports and ONNX classifier used on the Raspberry Pi 4B.

## Not Included

Raw image datasets, old experiments, virtual environments, training cache folders, and generated run outputs are intentionally excluded. The final deployable model artifacts are included because they are small enough for the repository and make the project reproducible.
