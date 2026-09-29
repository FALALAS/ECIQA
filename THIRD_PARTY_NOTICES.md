# Third-party notices and provenance

Audit date: 2026-09-24. This document records the dependencies of the extracted final model. A repository code license does not automatically establish redistribution permission for separately downloaded datasets or checkpoints.

## GRMP-IQA

- Source: [official GRMP-IQA repository](https://github.com/LXDxmu/GRMP-IQA).
- Paper: *Few-Shot Image Quality Assessment via Adaptation of Vision-Language Models*, ICCV 2025.
- Use: prompt-capable CLIP implementation, text-prompt encoding, and prompt-pretrained initialization underlying the downstream model.
- License evidence: the official [README license section](https://github.com/LXDxmu/GRMP-IQA#-license) states MIT. The local source tree and the inspected upstream root do not contain a standalone LICENSE file.
- Release preparation status: preserve attribution and upstream provenance; obtain the complete intended copyright/license notice before redistributing GRMP-derived files. This document does not invent an upstream copyright holder or replace a missing notice.
- Local vendoring change: `eciqa/vendor/clip_model.py` removes only two commented-out lines containing the original author's private home-directory paths. Its remaining lines and computational definitions are unchanged from the inspected GRMP source.
- Upstream resources: [GRMP-IQA model repository](https://huggingface.co/zzhowe/GRMP-IQA). Checkpoint distribution terms must be checked independently from the code's README declaration.

## OpenAI CLIP

- Source: [OpenAI CLIP](https://github.com/openai/CLIP).
- Use: CLIP model architecture and text tokenization, directly or through the GRMP-IQA derivative.
- License: [MIT](https://github.com/openai/CLIP/blob/main/LICENSE).
- Copyright notice: Copyright (c) 2021 OpenAI.
- The complete upstream MIT notice must accompany any redistributed CLIP-derived implementation. Preserve that notice together with the vendored files. GRMP-specific modifications require their own provenance in addition to the OpenAI notice.

## Restormer

- Source: [official Restormer repository](https://github.com/swz30/Restormer).
- Paper: *Restormer: Efficient Transformer for High-Resolution Image Restoration*, CVPR 2022.
- Use: Restormer encoder architecture underlying the distilled qMap representation. Downstream ECIQA uses encoder-level features rather than the image-restoration decoder.
- License: [MIT](https://github.com/swz30/Restormer/blob/main/LICENSE.md).
- Copyright notice: Copyright (c) 2022 Syed Waqas Zamir and contributors.
- Preserve the full upstream MIT notice with the vendored implementation. The qMap pretraining target, training data, and derived weights have provenance separate from the architecture license.

## Runtime packages

PyTorch, NumPy, SciPy, Pillow, and einops retain their own upstream licenses. They are declared as runtime dependencies rather than relicensed by this project. The exact package list and version requirements are defined by the package metadata.

## ECIQA code, checkpoints, and data

The project authors have not yet selected a project-wide license in this preparation directory. No blanket MIT claim is made for new ECIQA code, derived checkpoints, preference annotations, or third-party images. Public data/model/tool availability is planned after double-blind review; add distribution terms and a final license before publishing.

The final baseline does not depend on the experimental MANIQA, MUSIQ, SCLIP, ClearCLIP, or CrossScore branches. Do not copy those historical vendor trees into the release unless their actual code becomes a runtime dependency and its corresponding attribution is reviewed.
