# Livestock Detection and Counting: Deliverable 2 Baseline

**YOLOv8n baseline for detecting and counting Cattle, Goats and Sheep in farm images (Open Images V7 subset).**
Soroti University, Digital Image and Video Processing, Capstone Deliverable 2.
Team: Amanya Edwin, Kato Francis, Ahuura Ivan.

## 1. Package contents

| File / folder | Description |
|---|---|
| `livestock_deliverable2.ipynb` | Source code. Runs every step from data download to this package (Run All) |
| `README.md` | This file |
| `config.json`, `env.json`, `requirements.txt` | All settings, exact software versions and the hardware used |
| `data.yaml`, `splits/` | Dataset definition and the exact image lists of train / val / test |
| `tables/` | Every result as CSV: quality audit, dataset summary, detection, counting, speed, threshold sweep, confusion |
| `figures/` | Figures used in the report |
| `sample_predictions/` | 12 annotated test images produced by the baseline |
| `weights/best.pt` | Trained baseline weights (6.5 MB) |
| `training_results/` | Ultralytics training plots and `results.csv` |

## 2. How to run

**Kaggle (used for the reported results)**
1. Create a new Kaggle notebook and upload `livestock_deliverable2.ipynb`.
2. Right sidebar: Accelerator = **GPU T4**, Internet = **On**.
3. Click **Run All** (about 15 to 20 minutes). If the data are not attached, section A1 downloads 1,500 images from Open Images V7 with FiftyOne.
4. Click **Save Version, then Save & Run All (Commit)**. Open the version's **Output** tab and download `deliverable2_package.zip`.

**Local machine**
1. `pip install -r requirements.txt` (a CUDA GPU is strongly recommended; on CPU training is very slow).
2. Open the notebook and run all cells. Folders are created next to the notebook.

**Use the trained weights directly**
```python
from ultralytics import YOLO
model = YOLO("weights/best.pt")
r = model.predict("your_image.jpg", conf=0.25, iou=0.7, imgsz=640)[0]
print(len(r.boxes), "animals detected")      # total count
```

## 3. Dataset

* **Source:** Open Images V7 (public), train split, images containing Cattle, Goat or Sheep. 1,500 images audited, 1,500 kept, 5,437 boxes.
* **Classes:** 0 Cattle, 1 Goat, 2 Sheep. Annotations are bounding boxes converted to YOLO format.
* **Quality checks:** unreadable images, missing or empty labels, malformed or out-of-range boxes, duplicate files (name and hash), tiny boxes, possible group boxes. Result: nothing removed; 46 images contain tiny boxes (under 0.05% of the image) and 447 images contain a box covering at least 40% of the image (possible group box).
* **Preprocessing:** labels exported to YOLO format and restricted to the three classes; Ultralytics resizes images to 640 px (aspect ratio preserved); default training augmentation (mosaic, flip, colour jitter) is applied to the training split only.
* **Split:** by image, stratified by dominant class, 70/15/15, seed 42. The notebook asserts that no file name or file hash appears in more than one split. The test split is evaluated once.
* **Licence:** annotations CC BY 4.0. Image licences are set per image and must be verified before any redistribution.

| Split | Images | Boxes | Cattle | Goat | Sheep |
|---|---|---|---|---|---|
| Train | 1,050 | 3,788 | 2,561 | 497 | 730 |
| Validation | 225 | 819 | 518 | 86 | 215 |
| Test | 225 | 830 | 574 | 100 | 156 |
| Total | 1,500 | 5,437 | 3,653 | 683 | 1,101 |

## 4. Experimental setup

* **Baseline:** `yolov8n.pt` (COCO-pretrained), fine-tuned with the default Ultralytics recipe.
* **Settings:** 50 epochs (early stopping patience 10, all 50 ran), image size 640, batch 16, seed 42. Counting threshold 0.25, NMS IoU 0.7.
* **Hardware and software:** Kaggle Tesla T4, Python 3.13, PyTorch 2.11.0, Ultralytics 8.4.205, NumPy 2.1.3, OpenCV 4.14.0 (see `env.json`).
* **Metrics:** precision, recall, mAP@50, mAP@50-95; counting MAE, RMSE, bias, % exact, % within +/-1 animal; latency, parameters, weight size, GPU memory.
* **Protocol:** train on the train split; validation split used for early stopping and best-epoch selection (epoch 48); final numbers computed once on the test split. The counting threshold sweep uses validation only.

## 5. Results on the held-out test set (225 images)

**Detection**

| Class | Test boxes | Precision | Recall | mAP@50 | mAP@50-95 |
|---|---|---|---|---|---|
| Cattle | 574 | 0.664 | 0.593 | 0.637 | 0.347 |
| Goat | 100 | 0.560 | 0.585 | 0.566 | 0.365 |
| Sheep | 156 | 0.607 | 0.519 | 0.555 | 0.350 |
| **All** | 830 | 0.611 | 0.566 | **0.586** | **0.354** |

**Counting (predicted vs true animals per image)**

| Subset | Images | True | Predicted | MAE | RMSE | Bias | Exact | Within +/-1 |
|---|---|---|---|---|---|---|---|---|
| All test images | 225 | 830 | 839 | 1.35 | 2.75 | +0.04 | 45.8% | 69.3% |
| Without a possible group box | 145 | 678 | 655 | 1.52 | 2.90 | -0.16 | 40.7% | 64.8% |
| With a possible group box | 80 | 152 | 184 | 1.02 | 2.44 | +0.40 | 55.0% | 77.5% |

**Counting error by number of animals in the image**

| True animals | Images | MAE | Bias |
|---|---|---|---|
| 1 to 2 | 129 | 0.63 | +0.52 |
| 3 to 5 | 60 | 1.60 | +0.07 |
| 6 to 10 | 21 | 3.00 | -2.14 |
| More than 10 | 15 | 4.20 | -1.13 |

**Speed and memory (T4, batch 1):** 22.6 ms per image end to end (44 images/s); 3.14 M parameters; 6.5 MB weights; 156 MB peak GPU memory.

## 6. Findings and limitations

* The main weakness is **missed animals**: about 30 to 35% of every class is missed (Cattle 31.9%, Goat 30.0%, Sheep 34.6%). Class confusion is smaller (12.0% of goats are labelled cattle and 6.0% sheep; 5.8% of sheep are labelled goat).
* **Small animals** are missed most: approximate recall 0.51 for animals under 1% of the image, 0.60 medium, 0.73 large.
* The baseline **undercounts dense herds** (bias -2.14 for 6 to 10 animals) and slightly overcounts sparse images (+0.52 for 1 to 2 animals). Worst case: 52 true animals, 29 predicted.
* Some counting errors are **label errors**: Open Images sometimes draws one "group" box around a flock, and not every animal is labelled. The automatic flag (box of at least 40% of the image) is only a screening aid; flagged images do not have higher error, because it also catches close-ups of single large animals.
* Raising the counting threshold from 0.25 to 0.30 (chosen on validation) changes test MAE from 1.35 to 1.28 and bias from +0.04 to -0.24. It does not fix dense-herd undercounting.
* Limitations: about 1,500 images; Goat is the rarest class; single run, no confidence intervals; web photographs, not Ugandan farm-camera images; near-duplicate photographs across splits were not checked; counting is per photograph with no tracking.

## 7. Proposed improvement for Deliverable 4

**Higher-resolution and tiled inference for small and dense animals.** The evidence points to missed animals, not class confusion: 30 to 35% of animals are missed, recall is 0.51 for small animals against 0.73 for large ones, and counting bias becomes negative above five animals. The plan is to train and test at a larger input size (for example 960 or 1,280 px) and compare with tiled (sliced) inference. Both will be compared with this baseline on the same split, seed and metrics.

Success criteria:
1. Higher recall for small animals and lower counting MAE on images with more than five animals than the baseline values above.
2. mAP@50 on the test set not below 0.586.
3. Latency and memory reported alongside accuracy.

Supporting actions: re-label a small test subset so groups are counted animal by animal; address goats mistaken for cattle with class-balanced sampling; keep the validation-selected counting threshold (0.30) as a minor adjustment.

## 8. Reproducibility notes

Seeds are fixed for Python, NumPy and PyTorch and for training. GPU training can still vary slightly between runs and library versions, and a fresh download of Open Images may return a different set of images, so the saved split lists (`splits/`) and `env.json` are the reference for the reported results.

## 9. References

1. A. Kuznetsova et al., "The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection at scale," *Int. J. Comput. Vis.*, vol. 128, pp. 1956-1981, 2020.
2. J. Redmon, S. Divvala, R. Girshick and A. Farhadi, "You Only Look Once: Unified, real-time object detection," in *Proc. IEEE CVPR*, 2016, pp. 779-788.
3. G. Jocher, A. Chaurasia and J. Qiu, "Ultralytics YOLOv8," 2023. https://github.com/ultralytics/ultralytics
4. T.-Y. Lin et al., "Microsoft COCO: Common objects in context," in *Proc. ECCV*, 2014, pp. 740-755.
5. B. E. Moore and J. J. Corso, "FiftyOne," 2020. https://github.com/voxel51/fiftyone
