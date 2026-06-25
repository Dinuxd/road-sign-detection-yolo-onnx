# Model Card

## Project Type

Computer vision / deep learning road-sign detection and recognition.

## Architecture

- Stage 1: YOLO detector for road-sign bounding boxes.
- Stage 2: ONNX crop classifier for sign recognition and `other_sign` rejection.
- Deployment target: Raspberry Pi 4B edge device with NCNN detector export and ONNX Runtime classifier.

## Classes

`tls-g`, `sls-40`, `tls-e`, `tls-y`, `sls-50`, `sls-100`, `sls-80`, `no honking`, `tls-r`, `sls-60`, `sls-70`, `sls-15`, `tls-c`, `other_sign`

## Final Deploy Artifacts

- Detector NCNN: `raspberry_pi_twostage_deploy/models/detector_ncnn/best_ncnn_model`
- Detector NCNN 416 profile: `raspberry_pi_twostage_deploy/models/detector_ncnn_416/best_ncnn_model`
- Detector PyTorch fallback: `raspberry_pi_twostage_deploy/models/detector_pt/best.pt`
- Classifier ONNX: `raspberry_pi_twostage_deploy/models/classifier/best_classifier.onnx`
- Classifier ONNX sidecar: `raspberry_pi_twostage_deploy/models/classifier/best_classifier.onnx.data`

## Deployment Thresholds

- Detector confidence: `0.15`
- Detector IoU: `0.70`
- Classifier threshold: `0.88`
- Crop margin: `0.30`

## Latest Evaluation

Evaluation folder: `evaluation/latest/`

- Test images: 9,062
- Raw predictions: 8,251
- Accepted predictions: 1,527
- Known ground-truth signs: 2,484
- Other-sign ground-truth objects: 8,880
- End-to-end precision: 0.783
- End-to-end recall: 0.481
- End-to-end F1: 0.596
- Macro F1: 0.595
- Other-sign reject rate: 0.999

## Limitations

- Recall is still the main improvement area, especially for smaller or less frequent classes.
- This should be described as an edge-deployable prototype, not a production-ready traffic-safety system.
- Raw training images and generated run folders are not included in this repository.
- For production use, model binaries should be tracked with Git LFS or attached as release assets.
