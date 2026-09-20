
# Car Instance Segmentation with YOLOv8-seg

Fine-tuned YOLOv8s-seg model for car instance segmentation, fine-tuned on a small CCTV-domain dataset and tested live against a public traffic CCTV feed in Yogyakarta, Indonesia.

![Test-set inference](images/test-set-inference.jpg)

## Overview

- **Task**: instance segmentation (not just bounding boxes) of cars
- **Base model**: `yolov8s-seg.pt` (Ultralytics, COCO-pretrained)
- **Training data**: [`car-detection`](https://universe.roboflow.com/) dataset on Roboflow (201 train / 19 validation images, single class: `car`)
- **Live testing**: public ATCS traffic CCTV feed (Sugeng Jeroni, Yogyakarta) via HLS stream
- **Extras**: live car counting with two post-processing filters (minimum box area, mask fill-ratio) to reduce false positives from non-car objects

## Results (verified on validation set)

Trained for 30 epochs at `imgsz=640`, batch size 32.

| Metric | Box | Mask |
|---|---|---|
| Precision | 0.860 | 0.895 |
| Recall | 0.867 | 0.797 |
| mAP50 | 0.885 | 0.878 |
| mAP50-95 | 0.668 | 0.604 |

Sample inference on a held-out test image above: 6 of 7 visible cars detected correctly at high confidence (0.86-0.94); one small, partly-occluded car near the motorcycles on the right is caught only weakly (0.53), consistent with the recall gaps discussed under **Limitations**.

![Live CCTV detection](images/live-cctv-detection.jpg)

## Live CCTV counting

The notebook connects directly to a public CCTV HLS stream, runs frame-by-frame inference, and counts cars in real time. A sample 600-frame run at `conf=0.62`:

- Average cars per frame: ~2.1
- Max cars in a single frame: 4
- Processing speed: ~20-70 FPS depending on scene complexity (Colab T4/A100 GPU)

Output includes a rendered `.mp4` with mask overlays and a `.json` file with per-run statistics.

https://github.com/user-attachments/assets/c756e8a6-aba9-4951-80a5-4c7af73b843c

## Post-processing filters

Because the model is single-class (`car` only), it has no explicit negative examples for other vehicle types. Two lightweight filters are applied on top of raw model output to reduce false positives (e.g. a motorcycle rider with bulky cargo or a thick jacket occasionally resembling a car-shaped blob):

1. **Minimum box area** — discards detections with implausibly small bounding boxes for a car at this camera's distance.
2. **Mask fill-ratio** — a real car's segmentation mask fills most of its bounding box; a motorcycle + rider's mask is much sparser. Detections below a fill-ratio threshold are discarded.

These filters reduce, but do not eliminate, the false-positive rate. See **Limitations** below.

## Limitations

This is documented honestly rather than glossed over:

- **Single-class model**: the model was never trained on a "not a car" negative class. Two separate attempts to retrain with an explicit `motorcycle` class were tried and did not hold up:
  - A large, diverse public dataset (Vehicle Classification V2, ~3.7k images) gave strong validation metrics (car mAP50 0.90, motorcycle mAP50 0.85) but **zero real-world detections** on the CCTV feed — a clean example of the domain-shift problem in computer vision.
  - A merged dataset (this project's CCTV data + a small CCTV-sourced dataset with a motorcycle class) had only 22 motorcycle instances total — too few to learn a usable class.
- **Occasional false positives remain**: on confidence-threshold tuning, a strict precision/recall tradeoff appeared — raising the confidence threshold to suppress motorcycle false positives also caused some genuine cars (especially dark-colored ones with strong glare) to be missed entirely. The final `conf=0.62` is a deliberate middle ground, not a perfect solution.
- **Small training set** (201 images): limited diversity in vehicle body types, angles, and lighting conditions likely explains some of the recall gaps observed on live footage.
- **Video timing**: recorded `.mp4` output is encoded at a fixed frame rate regardless of the CCTV feed's actual real-time capture speed, so playback duration doesn't map 1:1 to wall-clock capture time.

## Repository structure

```
├── car_instance_segmentation.ipynb   # full pipeline: dataset, training, inference, live CCTV
├── images/                           # sample segmentation results
├── README.md
├── README.id.md                      # Indonesian version
└── LICENSE
```

## Setup

1. Open the notebook in Google Colab (GPU runtime recommended).
2. Add a `ROBOFLOW_API_KEY` secret in Colab (Settings → Secrets), or enter it when prompted.
3. Run cells in order. If a previously trained model already exists in your Google Drive at `car_instance_segmentation/best.pt`, the notebook will load it directly instead of retraining.
4. The live CCTV cell targets a public Yogyakarta traffic camera by default — swap in any HLS (`.m3u8`) stream URL to test elsewhere.

Model weight files (`*.pt`) are not included in this repository — see `.gitignore`.

## License

MIT — see [LICENSE](LICENSE).
