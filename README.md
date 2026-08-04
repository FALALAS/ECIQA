# ECIQA

**Exposure Correction Image Quality Assessment Dataset**

> **Project status:** ECIQA is under active construction. Image organization,
> subjective annotation, quality control, and score aggregation are still in
> progress. The dataset is not yet available for download, and the current raw
> annotations should not be treated as finalized mean opinion score (MOS)
> labels.

## Overview

ECIQA is a large-scale subjective image quality dataset for evaluating exposure
correction results. It is designed to capture how human observers judge the
perceptual quality of corrected images, including overall exposure, local
visibility, color fidelity, contrast, naturalness, and correction artifacts.

The current protocol collects pairwise preferences instead of asking annotators
to assign an isolated numerical score. Each judgment compares an original input
with two corrected candidates. The resulting preferences will be checked,
aggregated, and calibrated before the final quality scores and benchmark splits
are released.

## Current Working Snapshot

The following numbers describe the internal image bundle generated on
July 31, 2026. They are a construction snapshot, not a commitment to the final
release size.

| Item | Current count or status |
| --- | ---: |
| Source datasets | MSEC and SICEV2 |
| Total scenes | 4,784 |
| MSEC scenes | 4,724 |
| SICEV2 scenes | 60 |
| Total images | 23,920 |
| Input images | 4,784 |
| Candidate images | 19,136 |
| Image storage | Independent physical copies |
| Resizing or re-encoding | None |
| Annotation status | In progress |

As of August 5, 2026, the live annotation files contain 1,084 completed trial
records covering 1,035 unique candidate pairs and 868 scenes. These records are
raw pairwise judgments; reliability analysis, observer screening, score
aggregation, and train/validation/test splitting have not yet been finalized.

## Data Composition

Each scene contains one input image and the available exposure-correction
outputs. The current MSEC subset includes outputs from:

- CoTF (original configuration)
- CoTF with a 0.5x MEF-SSIM objective and Mertens-reference color correction
- FECNet (original configuration)
- FECNet with a 0.5x MEF-SSIM objective

The current SICEV2 subset treats low-exposure and over-exposure inputs as
separate scenes. It includes four candidate families:

- CoTF with middle-exposure supervision
- CoTF with fused-reference supervision
- FECNet with middle-exposure supervision
- FECNet with fused-reference supervision

MSEC inputs whose filenames end in `_0` are excluded from the current annotation
pool because pilot annotators could not reliably perceive an exposure error in
those inputs. This inclusion rule will be documented again when the final
protocol is frozen.

Images retain their original file format and dimensions. Some FECNet outputs
contain model padding and are one to three pixels larger than the corresponding
input; these differences are recorded in the scene manifest rather than removed
by resizing.

## Annotation Protocol

Each trial displays three images:

1. the original input;
2. Candidate A; and
3. Candidate B.

The annotator completes all three pairwise comparisons:

- Input versus Candidate A
- Input versus Candidate B
- Candidate A versus Candidate B

For every comparison, the annotator may select the left image, the right image,
or `Similar`. The tool derives a ranking with possible ties, such as
`Input>A=B`. A trial cannot be saved if the three choices form a cyclic or
logically inconsistent preference.

The interface provides synchronized local magnification across all three images
to support inspection of details at the same relative position. Candidate
left/right placement is reproducibly varied to reduce position bias.

## Coverage and Sampling

Annotation progress is stored in five independent coverage layers, Group A
through Group E. These groups are coverage layers rather than five fixed
annotators. The target is for each canonical candidate pair to receive one
independent judgment in every layer. Annotators retain a stable expert ID and
assigned group, while completed pair IDs are skipped automatically.

The internal tool currently supports four sampling modes:

- `Compact`: uses a curated set of candidate-pair families for each source
  dataset.
- `Unseen`: prioritizes MSEC scenes with no result in any coverage layer and
  always includes the color-corrected CoTF candidate.
- `All pairs`: enumerates every pair of candidate outputs available in a scene.
- `Marked only`: restores a manually marked review queue.

The `Unseen` opponent is selected reproducibly from the remaining candidate
methods using the seed and scene ID. SICEV2 is not included in this mode because
it does not contain the corresponding color-corrected candidate.

## Recorded Annotation Fields

The raw annotation tables currently retain:

- expert ID and coverage group;
- canonical pair ID, trial ID, dataset, and scene ID;
- input and candidate method identifiers;
- relative image paths;
- all three pairwise decisions;
- the automatically derived ranking;
- annotation duration; and
- completion timestamp.

The release version will document any observer filtering, consistency metrics,
preference aggregation method, and transformation from raw judgments to the
published quality scores.

## Internal Working Layout

The active annotation bundle is organized as follows. Raw images and annotations
are maintained outside this Git repository while construction is ongoing.

```text
MSEC_SICEV2_MOS/
|-- scenes/                       # Scene-oriented image bundle
|-- mos_label_tool/               # Static browser annotation tool
|-- working_folder/
|   |-- group_A.csv
|   |-- group_B.csv
|   |-- group_C.csv
|   |-- group_D.csv
|   |-- group_E.csv
|   `-- marked_comparisons.csv
|-- scene_manifest.csv
|-- candidate_manifest.csv
`-- organization_summary.json
```

The two manifests describe scene membership, method identity, item type,
display order, dimensions, and file locations. The group CSV files are the raw
handoff and aggregation records used during annotation.

## Planned Release

The public release is expected to add:

- the frozen dataset composition and inclusion criteria;
- the complete subjective-study protocol;
- anonymized and quality-controlled annotations;
- aggregated quality scores and observer agreement statistics;
- official training, validation, and test splits;
- baseline IQA results and evaluation scripts;
- data access and redistribution terms; and
- a formal citation entry.

No release date, license, or redistribution permission is claimed by this
construction README.

## Acknowledgements

We sincerely thank the following annotators for the time, concentration, and
careful judgments they have contributed to ECIQA. Their work provides the
foundation for the dataset's subjective quality labels. The order below follows
the information supplied by the project organizer and does not indicate a
contribution ranking.

| Name | Gender | Age at annotation | Affiliation |
| --- | --- | ---: | --- |
| Junyan Guo | Male | 23 | Jiangsu Normal University |
| Min Qiu | Female | 53 | Suining Chess Academy |
| Youxiang Li | Male | 53 | Jiangsu Suining Liji Senior High School |
| Hang Li | Female | 28 | Durr Paintshop Systems Engineering (Shanghai) Co., Ltd. |
| Chentian Wei | Male | 25 | Tsinghua University |
| Tingjia Zhang | Male | 24 | Shanghai Jiao Tong University |
| Yiqi Li | Male | 26 | Xidian University |
| Jinghao Bian | Male | 26 | Xidian University |
| Yuantao Ju | Male | 25 | Shanghai Customs |

The personal details in this table were provided by the project organizer.
Consent to publish names, gender, age, and affiliations should be reconfirmed
with every participant before this repository or dataset is publicly released.

## Citation

A formal citation will be provided with the first public dataset release.
