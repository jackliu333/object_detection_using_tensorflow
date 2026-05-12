# Generalizing Bayesian Human–AI Collaboration

Code and data accompanying the IJCAI–ECAI 2026 paper
**"Generalizing Bayesian Human–AI Collaboration: Theory and Application in Data-Scarce Environments."**

The repository fuses three heterogeneous classifiers — an in-house vision model (TensorFlow Object Detection API with a ResNet backbone), human volunteer predictions collected through a custom R Shiny labelling app, and GPT-4o predictions on video frames — under a Bayesian combination framework with auxiliary ordinal feedback (confidence, image clarity, surface reflection).

## Repository layout

```
.
├── NN_model/                 In-house vision model: ResNet-backbone object
│                             detector fine-tuned on FBS products (TF Object
│                             Detection API, Colab notebooks)
├── bayesian_model/           Bayesian combination model (R + JAGS) and
│                             GPT-4o experiments
│   ├── v1/                   Main working version: R scripts, notebooks,
│   │                         intermediate result CSVs
│   └── v3/                   Latest "mode 5" Bayesian notebook
├── rshiny/                   Volunteer labelling app — randomly samples
│                             product images and records the six attributes
├── video_recognition/        360° video → frame extraction pipeline used
│                             in the ChatGPT + ResNet experiment
├── master_list.csv           Product catalogue: filename, product type,
│                             weight class, halal status, etc.
└── master_list_manual.csv    Manually corrected catalogue (ground-truth labels)
```

## Key files mapped to the paper

| Component                                  | Where it lives                                                                |
| ------------------------------------------ | ----------------------------------------------------------------------------- |
| ResNet training (Sec. 3.1, Sec. 5.1)       | `NN_model/Foodbank_Object_Detection_Model_Training_v8.ipynb`                  |
| ResNet logit outputs                       | `bayesian_model/v1/all_imgs_results_big_model.csv` and `_small_model.csv`     |
| Bayesian combination model (Sec. 3.2)      | `bayesian_model/v1/working v2.R` (sequential), `working v2 parallel.R` (parallel chains) |
| ChatGPT + ResNet variant (Sec. 5)          | `bayesian_model/v1/working v2 chatgpt.R`, `chatgpt api.ipynb`, `chatgpt_prediction.csv` |
| Latest "mode 5" Bayesian model             | `bayesian_model/v3/Bayesian model for mode 5.ipynb`                           |
| Volunteer labelling interface              | `rshiny/` (run `shiny::runApp()` from the project root)                       |
| Video → frame extraction                   | `video_recognition/real_time_object_detection.py`                             |

## Data

- **Human + ResNet experiment.** 306 product images, each labelled by five volunteers with three direct predictions (product type, weight class, halal status) and three auxiliary ordinal ratings (confidence, image clarity, surface reflection), each on a Low / Medium / High scale. Product metadata is in `master_list.csv`; the manually-corrected version (used as ground truth) is `master_list_manual.csv`.
- **ChatGPT + ResNet experiment.** 292 products captured as 360° video; ten equally-spaced frames per product are passed to the GPT-4o API together with a fixed classification prompt. Predictions are stored in `bayesian_model/v1/chatgpt_prediction.csv`.

The label space follows FBS's operational taxonomy: 70+ product-type classes, 2 halal classes, 3 weight classes.

## Vision model

Trained via transfer learning on the TensorFlow Object Detection API with a ResNet backbone, run end-to-end in `NN_model/Foodbank_Object_Detection_Model_Training_v8.ipynb` (Google Colab). Trained weights live under `NN_model/model_weights/`. Logit scores exported for the Bayesian step are in `bayesian_model/v1/all_imgs_results_big_model.csv` and `all_imgs_results_small_model.csv`, where "big" and "small" refer to the two training-set sizes used to probe data scarcity.

## Bayesian model

Implemented in R via JAGS (`rjags`). The active script is `bayesian_model/v1/working v2.R`; `working v2 parallel.R` runs the three MCMC chains in parallel for speed. Default settings: 3 chains, 2,000-iteration burn-in, 5,000 post-burn-in samples. Per-volunteer models are fitted separately, yielding the 200 total models reported in the paper (4 train–test ratios × 5 volunteers × 10 seeds). The latest variant is `bayesian_model/v3/Bayesian model for mode 5.ipynb`.

## How to reproduce

1. **Train the vision model.** Open `NN_model/Foodbank_Object_Detection_Model_Training_v8.ipynb` in Colab, mount your data folder, run end-to-end. Weights write to `NN_model/model_weights/`. Export per-image logits to `bayesian_model/v1/all_imgs_results_*_model.csv`.
2. **Collect human labels.** Run the Shiny app in `rshiny/` (`shiny::runApp()` from R). Each volunteer's session writes the six attributes per image; outputs are merged into `master_list.csv`.
3. **Run GPT-4o predictions.** Open `bayesian_model/v1/chatgpt api.ipynb`, set `OPENAI_API_KEY`, run end-to-end. Output: `chatgpt_prediction.csv`.
4. **Fit the Bayesian combination model.** From R, source `bayesian_model/v1/working v2.R` (Human + ResNet) or `working v2 chatgpt.R` (ChatGPT + ResNet). Use `working v2 parallel.R` for full sweeps across splits, volunteers, and seeds.

## Dependencies

- **Vision model.** TensorFlow Object Detection API (TF2), Python ≥ 3.8. Easiest in Google Colab.
- **Bayesian model.** R ≥ 4.0 with `rjags`, `coda`, `dplyr`, `parallel`. JAGS ≥ 4.3 must be installed separately (`apt install jags` / `brew install jags`).
- **ChatGPT pipeline.** Python with `openai`; set `OPENAI_API_KEY` in your environment.
- **Volunteer labelling app.** R with `shiny`, `DT`, `shinyjs`.

## Citation

```bibtex
@inproceedings{liu2026generalizing,
  title     = {Generalizing {Bayesian} {Human--AI} Collaboration:
               Theory and Application in Data-Scarce Environments},
  author    = {Liu, Jack and others},
  booktitle = {Proceedings of the 35th International Joint Conference on
               Artificial Intelligence (IJCAI--ECAI)},
  year      = {2026}
}
```

## Acknowledgements

We thank Food Bank Singapore and its volunteers for supporting data collection and labelling.
