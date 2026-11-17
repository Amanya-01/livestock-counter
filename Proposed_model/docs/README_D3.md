# Livestock Detection and Counting: Deliverable 3 (same dataset as D2)

Deployed configuration: **F. Baseline + gated size-aware tiling** (counting threshold 0.35).
Best improved variant on validation: **F. Baseline + gated size-aware tiling**.
Settings are in `config_d3.json`; inference code is in `pipeline.py`. The ONNX file holds the detector only; tiling runs through `pipeline.py`.

## Conclusions
- Best improved variant on validation: F. Baseline + gated size-aware tiling. Configuration deployed: F. Baseline + gated size-aware tiling.
- Small-animal recall at the chosen thresholds: 0.48 for the baseline and 0.66 for F. Baseline + gated size-aware tiling; precision changes from 0.62 to 0.59.
- Overall counting MAE: 1.28 to 1.47 (change +0.19, 95% interval -0.07 to +0.43): not clearly different from the baseline.
- MAE on photos with more than 5 animals: 3.42 to 3.56 (change +0.14, interval -1.28 to +1.31): not clearly different from the baseline.
- Mean AP@50 change +0.015 (interval -0.023 to +0.050): not clearly different from the baseline.
- Counting MAE by herd size is lower than the baseline for: no group; higher for: 1 to 2, 3 to 5, 6 to 10, more than 10.
- Classes with a lower AP@50 than the baseline: Sheep.
- Higher resolution alone (variant C) gives mean AP@50 0.477 against 0.560 for the baseline.
- Cost: 8.1 times the CPU time and 4.1 times the GPU time per image of the baseline.

## Results on the held-out test photos
| Variant | AP50 mean | Recall small | MAE@0.25 | MAE@chosen | MAE (>5 animals)@chosen | Bias@chosen |
|---|---|---|---|---|---|---|
| A. Baseline (640 px) | 0.560 | 0.502 | 1.347 | 1.276 | 3.417 | -0.236 |
| B. Baseline + tiling | 0.554 | 0.715 | 2.622 | 1.440 | 3.472 | -0.098 |
| C. Higher resolution (960 px) | 0.477 | 0.492 | 1.693 | 1.693 | 3.694 | 0.333 |
| D. Higher resolution + tiling | 0.483 | 0.647 | 2.707 | 1.449 | 3.472 | -0.409 |
| E. Baseline + size-aware tiling | 0.567 | 0.719 | 2.182 | 1.476 | 3.278 | 0.116 |
| F. Baseline + gated size-aware tiling | 0.575 | 0.695 | 1.902 | 1.467 | 3.556 | 0.187 |

## Precision and recall at the validation-chosen thresholds
| Variant | Threshold | Precision | Recall | Recall small | Recall medium | Recall large |
|---|---|---|---|---|---|---|
| A. Baseline (640 px) | 0.300 | 0.616 | 0.577 | 0.478 | 0.578 | 0.694 |
| B. Baseline + tiling | 0.500 | 0.616 | 0.600 | 0.624 | 0.568 | 0.609 |
| C. Higher resolution (960 px) | 0.250 | 0.539 | 0.588 | 0.492 | 0.613 | 0.673 |
| D. Higher resolution + tiling | 0.500 | 0.625 | 0.555 | 0.569 | 0.575 | 0.516 |
| E. Baseline + size-aware tiling | 0.400 | 0.592 | 0.611 | 0.661 | 0.523 | 0.653 |
| F. Baseline + gated size-aware tiling | 0.350 | 0.588 | 0.618 | 0.658 | 0.537 | 0.665 |

## Change in counting MAE against the baseline (95% bootstrap interval)
| Variant | MAE change (all) | low | high | MAE change (>5 animals) | low (>5) | high (>5) |
|---|---|---|---|---|---|---|
| B. Baseline + tiling | 0.164 | -0.058 | 0.382 | 0.056 | -1.000 | 0.944 |
| C. Higher resolution (960 px) | 0.418 | 0.213 | 0.649 | 0.278 | -0.389 | 0.944 |
| D. Higher resolution + tiling | 0.173 | -0.022 | 0.356 | 0.056 | -0.612 | 0.694 |
| E. Baseline + size-aware tiling | 0.200 | -0.085 | 0.489 | -0.139 | -1.529 | 0.945 |
| F. Baseline + gated size-aware tiling | 0.191 | -0.067 | 0.431 | 0.139 | -1.278 | 1.306 |

## Hypothesis check
| Criterion | Baseline | F. Baseline + gated size-aware tiling | Met? |
|---|---|---|---|
| 1. Recall for small animals goes up | 0.502 | 0.695 | True |
| 2. Counting MAE on images with >5 animals goes down | 3.42 | 3.56 | False |
| 3. mAP@50 not below the baseline | 0.560 | 0.575 | True |
| (extra) Overall counting MAE goes down | 1.28 | 1.47 | False |

## Speed and memory
| Variant | GPU ms/image | GPU images/s | Peak GPU memory (MB) | CPU ms/image | CPU images/s | Weights (MB) |
|---|---|---|---|---|---|---|
| A. Baseline (640 px) | 12.46 | 80.28 | 47.29 | 69.61 | 14.37 | 6.48 |
| B. Baseline + tiling | 91.22 | 10.96 | 457.78 | 1025.13 | 0.98 | 6.48 |
| C. Higher resolution (960 px) | 17.48 | 57.22 | 73.29 | 123.84 | 8.07 | 6.52 |
| D. Higher resolution + tiling | 121.02 | 8.26 | 716.86 | 1256.85 | 0.80 | 6.52 |
| E. Baseline + size-aware tiling | 92.41 | 10.82 | 457.77 | 958.80 | 1.04 | 6.48 |
| F. Baseline + gated size-aware tiling | 51.28 | 19.50 | 386.90 | 565.74 | 1.77 | 6.48 |

## Use the deployed model in Python
```python
import json, cv2
from ultralytics import YOLO
from pipeline import run_pipeline
cfg = json.load(open("config_d3.json"))
model = YOLO(cfg["weights"])
pred, counts = run_pipeline(model, cv2.imread("farm.jpg"), cfg)
print(counts)
```
