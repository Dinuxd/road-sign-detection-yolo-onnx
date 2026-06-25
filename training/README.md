# Training Scripts

These scripts document the training and evaluation workflow used before exporting the final Raspberry Pi 4B deployment package.

## Order

1. `prepare_twostage_dataset.py` - convert Pascal VOC annotations into YOLO detector data and classifier crop folders.
2. `train_twostage_detector.py` - train YOLO detector candidates.
3. `build_classifier_proposal_dataset.py` - collect detector proposals for classifier training.
4. `train_twostage_classifier.py` - train and calibrate the crop classifier.
5. `mine_twostage_hard_negatives.py` - add difficult negative crops.
6. `twostage_infer.py` - run two-stage inference with training checkpoints.
7. `evaluate_twostage_pipeline.py` - calculate precision, recall, F1, and failure cases.

## Notes

Raw datasets, training runs, and intermediate checkpoints are not included in this repository.

The final edge deployment is in `../raspberry_pi_twostage_deploy/` and uses:

- YOLO NCNN detector export.
- ONNX crop classifier.
- Raspberry Pi 4B run scripts.
