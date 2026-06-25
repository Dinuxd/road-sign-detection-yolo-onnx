# Raspberry Pi 4B Deployment

This folder contains the final two-stage road-sign pipeline:

- Detector: YOLO NCNN export in `models/detector_ncnn/best_ncnn_model`
- Detector fallback: PyTorch checkpoint in `models/detector_pt/best.pt`
- Classifier: ONNX model in `models/classifier/best_classifier.onnx`
- Classifier weights sidecar: `models/classifier/best_classifier.onnx.data`

## Copy To Raspberry Pi

Copy the full `raspberry_pi_twostage_deploy` folder to the Raspberry Pi. Keep the folder structure unchanged.

## Install

On Raspberry Pi OS 64-bit:

```bash
cd raspberry_pi_twostage_deploy
bash install_pi.sh
```

If you use the official Raspberry Pi Camera, first check the camera:

```bash
rpicam-hello
```

## Run With USB Webcam

```bash
cd raspberry_pi_twostage_deploy
bash run_camera.sh
```

Press `q` to stop the display window.

## Run With Raspberry Pi Camera

```bash
cd raspberry_pi_twostage_deploy
bash run_picamera.sh
```

## Recommended Pi 4B Balanced Profile

This is the combined setup for real-world testing on Raspberry Pi 4B:

- Camera frame: `1280x720`
- Detector input: `416x416` NCNN
- Classifier crop: taken from the original `1280x720` frame
- Inference: every 2nd frame

This keeps the YOLO detector lighter while still giving the classifier a cleaner crop.

USB webcam:

```bash
cd raspberry_pi_twostage_deploy
bash run_camera_416_highres.sh
```

Raspberry Pi Camera:

```bash
cd raspberry_pi_twostage_deploy
bash run_picamera_416_highres.sh
```

If it is still slow, run every 3rd frame:

```bash
bash run_picamera_416_highres.sh --frame-skip 3
```

## Run On Image Or Folder

```bash
cd raspberry_pi_twostage_deploy
bash run_image.sh /home/pi/test_images
```

Annotated outputs and `predictions.csv` are written to `outputs/`.

## Useful Real-World Options

Lower resolution for more FPS:

```bash
bash run_picamera.sh --width 640 --height 480 --frame-skip 2
```

Save an annotated video:

```bash
bash run_picamera.sh --save-video road_test.mp4
```

Show rejected detections for debugging:

```bash
bash run_picamera.sh --draw-rejected
```

The default thresholds come from the latest two-stage evaluation:

- detector confidence: `0.15`
- detector IoU: `0.70`
- classifier threshold: `0.88`
- crop margin: `0.30`

Edit `config.json` if you want to tune these later.
