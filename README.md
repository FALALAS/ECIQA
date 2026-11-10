# Beyond Ground-Truth Fidelity: Rethinking Exposure Correction through Input-Conditioned Perceptual Evaluation

**Ao Li, Zhenyu Wang<sup>†</sup>, Aobo Li, Zhenqi Fu, Tao Huang, Yufan Zhu, Fangfang Wu, Jinjian Wu, Weisheng Dong<sup>†</sup>**

Xidian University

<sup>†</sup> Corresponding authors.

Official repository for **ECIQA** and **ECIQA-Pref**.

We study exposure correction through perceptual evaluation conditioned on the original input. This repository provides ECIQA-Pref, a dataset of human preferences for exposure correction, and ECIQA, a model that assesses a corrected image together with its degraded input.

## Installation

Use Python 3.10 or later and install a PyTorch build suitable for your device.

```bash
git clone https://github.com/FALALAS/ECIQA.git
cd ECIQA
python -m pip install -e .
```

## Dataset and pretrained models

ECIQA-Pref contains **5,200 images from 1,040 exposure instances**. Each instance includes an original input and four corrected images. The preference labels describe relative quality within each exposure instance.

Download **ECIQA-assets** (dataset images and pretrained weights) from [Baidu Netdisk](https://pan.baidu.com/s/13V898VOabdk6LJrmFmsxVw?pwd=ng41). Extraction code: `ng41`.

After downloading the images and pretrained weights, place the complete `scenes/` folder in `data/ECIQA-Pref/` and both checkpoint `.pt` files directly in `weights/`. All paths below are relative to the repository root. The manifest and split files are already included in the repository.

If the downloaded bundle contains `data/` and `weights/` folders, merge them into the repository root. Keep the scene subfolders and filenames unchanged, without adding an extra directory level.

```text
data/ECIQA-Pref/
├── scenes/          # Original inputs and corrected images
├── manifest.csv     # Image paths, preference labels, and splits
└── splits/          # Training and test manifests

weights/
├── eciqa_baseline_epoch015.pt  # ECIQA checkpoint for inference
└── eciqa_stage2_init.pt        # Initialization for training
```

The provided split contains 4,160 training images and 1,040 test images. Splitting is by exposure instance; 84 underlying source-content groups occur in both partitions.

The manifest keeps 12 fields used by training, evaluation, annotation, and split checks:

| Fields | Purpose |
|---|---|
| `image_id` | Unique image identifier |
| `scene_id`, `content_id` | Exposure instance and underlying source content |
| `relative_path`, `input_relative_path` | Candidate and original input paths |
| `split`, `elo_scene_0_100` | Partition and scene-relative preference target |
| `dataset`, `method` | Source dataset and candidate identity for annotation |
| `width`, `height` | Image dimensions for the annotation layout |
| `exposure` | Underexposure/overexposure grouping for evaluation |

After placing the downloaded files, verify them with `python scripts/verify_assets.py`.

## Inference

Evaluate a corrected image using its original input:

```bash
python -m eciqa.infer \
  --weights weights/eciqa_baseline_epoch015.pt \
  --source input.png \
  --candidate corrected.png \
  --device cuda:0 --output score.json
```

`source` is the original degraded input and `candidate` is the image to evaluate. A higher `score_0_100` indicates better predicted quality. The score reflects learned scene-relative preferences rather than globally calibrated MOS.

Use `--device cpu` for CPU inference. Add `--view-chunk 1 --cross-query-chunk 512` to reduce GPU memory use.

## Training

Train on ECIQA-Pref from the supplied initialization:

```bash
python -m eciqa.train \
  --init-weights weights/eciqa_stage2_init.pt \
  --config configs/baseline.json \
  --manifest data/ECIQA-Pref/manifest.csv \
  --project-root data/ECIQA-Pref \
  --output-dir runs/eciqa --selection-split test \
  --device cuda:0 --precision auto --workers 4 \
  --stop-after-epoch 15
```

This reproduces the selected epoch within the original 100-epoch schedule. Omit `--stop-after-epoch 15` to complete all 100 epochs. The supplied checkpoint protocol uses the test split for selection; use a separate validation split and `--selection-split val` for new experiments.

## Evaluation

```bash
python -m eciqa.evaluate \
  --weights weights/eciqa_baseline_epoch015.pt \
  --manifest data/ECIQA-Pref/manifest.csv \
  --project-root data/ECIQA-Pref \
  --output-dir evaluation/eciqa --split test \
  --device cuda:0 --precision fp32
```

Evaluation reports SRCC and PLCC for underexposure, overexposure, and the overall test set.

## Annotation tool

The browser-based tool displays the original input and two corrected candidates for pairwise preference annotation. It supports ties, synchronized local magnification, randomized candidate order, progress recovery, CSV/JSON export, and a separate review page.

Start the tool from the repository root; it uses only the Python standard library:

```bash
python annotation_tool/serve.py
```

Open **http://127.0.0.1:8010** in Chrome or Edge. The tool loads the images from `data/ECIQA-Pref/`.

1. Enter an annotator ID, select a Group (A–E), and click **Choose Folder** to choose a local results folder.
2. Select a pair mode and click **Start**. `Compact` uses the original candidate-pair selection; `All pairs` includes every pair of corrected candidates.
3. Compare Input vs A, Input vs B, and A vs B, then click **Save & Next**. Inconsistent rankings must be corrected before saving.
4. Use **Export CSV/JSON** to export the session, or **Open review tool** to review saved annotations from the same results folder.

Annotations are saved locally as `group_A.csv` through `group_E.csv`; review edits are stored separately in `group_*_reviewed.csv`. Groups represent independent coverage layers. Use a stable annotator ID and keep the results folder to resume work; avoid simultaneous writes to the same folder. No annotations are uploaded by the tool.

To use your own image collection, supply a manifest with `dataset`, `scene_id`, `method`, `relative_path`, `input_relative_path`, `width`, and `height` columns. Include one row with `method=input` and at least two corrected candidates per scene. Paths are relative to the image directory; preference labels are not required:

```bash
python annotation_tool/serve.py --manifest path/to/manifest.csv --data-root path/to/images
```

## Acknowledgements

We sincerely thank the following annotators for the time, concentration, and
careful judgments they have contributed to ECIQA. Their work provides the
foundation for the dataset's subjective quality labels. The order below does not indicate a
contribution ranking.

| Name | Gender | Age | Affiliation |
| --- | --- | ---: | --- |
| Jinghao Bian | Male | 26 | Xidian University |
| Yishan Chen | Female | 23 | Xidian University |
| Bihui Cheng | Male | 23 | Xidian University |
| Junyan Guo | Male | 23 | Jiangsu Normal University |
| Zihan Jia | Female | 25 | Bayer (Shanghai) |
| Zhiming Jiang | Male | 24 | Xidian University |
| Yuantao Ju | Male | 25 | Shanghai Customs |
| Hang Li | Female | 28 | Durr (Shanghai) |
| Jicai Li | Male | 24 | Xidian University |
| Yiqi Li | Male | 26 | Xidian University |
| Youxiang Li | Male | 53 | Jiangsu Liji High School |
| Mengzu Liu | Male | 27 | Xidian University |
| Min Qiu | Female | 53 | Suining Go Academy |
| Jiawei Sun | Male | 21 | Xidian University |
| Yujia Sun | Female | 31 | Xidian University |
| Ruiao Tang | Male | 24 | Nanjing University of Finance & Economics |
| Zhengyang Wang | Female | 23 | Xidian University |
| Chentian Wei | Male | 25 | Tsinghua University |
| Ping Wu | Female | 25 | Xi'an Jiaotong University |
| Le Zhang | Female | 23 | Huawei (Edinburgh) |
| Tingjia Zhang | Male | 24 | Shanghai Jiao Tong University |
| Xin Zhang | Male | 25 | Nanjing Customs |
| Yapeng Zhang | Male | 25 | Xidian University |
| Yongchuan Zhang | Male | 24 | Xidian University |
| Zijun Zhang | Male | 24 | Air Force Engineering University |
| Jingzhu Zheng | Male | 25 | Xidian University |
