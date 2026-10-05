# Zero Trust Enhanced Defense Mechanisms for Mitigating Social Engineering Exploitation in Online Social Networks

Code, frozen configuration, cached model outputs and results for the paper
*Zero Trust Enhanced Defense Mechanisms for Mitigating Social Engineering Exploitation in Online Social Networks*
(D. Saravana Kumar, Burugapalli Rohan, Raja Muthalagu, Somasundaram A., Pranav M. Pawar; BITS Pilani, Dubai Campus).

## Overview

The system turns conversational social-engineering (SE) detection into Zero Trust access decisions:

1. **SecureBERT** scores the most recent turns of a conversation (last 512 tokens).
2. **ModernBERT** scores the full conversation (up to 8,192 tokens).
3. A **3 → 16 → 1 fusion MLP** (81 parameters) combines the two temperature-calibrated probabilities and their disagreement `|f1 − f2|` into a trust score `T ∈ [0, 100]`.
4. A **Policy Decision Point** (NIST SP 800-207, Tenets 4 and 6) maps `T` to **Full Access**, **Restricted** or **Blocked**, using `T_block` from Youden's J and `T_permit` = 75th percentile of benign validation scores.

All fitted quantities (early stopping, temperatures, fusion MLP, thresholds) are derived on a validation split; the SEConvo test set is scored once.

## Headline results (SEConvo test set, strict transfer)

| Metric | Value |
|---|---|
| F1 (95% CI) | 0.764 [0.715, 0.810] |
| AUROC (95% CI) | 0.825 [0.778, 0.864] |
| Median latency (T4 GPU) | 51.3 ms per conversation |
| LoveFraud02 scams detected | 79 / 83 (95.2%) |
| …of which before the first payment request | 31 / 78 (39.7%) |

All numbers in the paper are in `results_v3_final.json`.

## Repository contents

| File | Description |
|---|---|
| `Zero_Trust__SE_Detection.ipynb` | Full pipeline: data split, training, calibration, fusion, thresholds, evaluation, ablation, robustness, explainability, LoveFraud02 and length-aware thresholds |
| `results_v3_final.json` | Every reported number |
| `gen_numbers.py` | Converts the results file into the LaTeX macros used by the paper: `python gen_numbers.py results_v3_final.json > numbers.tex` |
| `split.json` | Indices of the fitting and validation splits (seed 42) |
| `frozen_config.json` | Temperatures and Zero Trust thresholds |
| `fusion_mlp_full_v3.pt`, `fusion_mlp_noconflict_v3.pt` | Trained fusion layers (with and without the conflict feature) |
| `securebert_v3_history.json`, `modernbert_v3_untruncated_history.json` | Encoder training logs |
| `seconvo_test_logits.npz`, `sevsim_test_logits.npz` | Cached encoder outputs on the test sets |
| `val_prefix_logits.npz`, `test_prefix_logits.npz`, `lovefraud_prefix_logits.npz` | Cached encoder outputs on conversation prefixes (online evaluation) |
| `robustness_logits.npz`, `robustness_results.json` | Perturbation experiments |
| `mistral_test_cache_recovered.json`, `mistral_val_cache_recovered.json`, `mistral7b_baseline_results_recovered.json` | Raw Mistral-7B-Instruct responses and baseline results |

The fine-tuned encoder checkpoints (SecureBERT ≈ 0.5 GB, ModernBERT ≈ 0.6 GB) exceed GitHub's file-size limit and are available from the authors on request. The cached logits are sufficient to reproduce every number without them.

## Data

The datasets are not redistributed here; obtain them from their sources:

- **SEConvo** (Ai et al., 2024) — downloaded automatically by the notebook from [Zenodo record 12170260](https://zenodo.org/records/12170260).
- **SE-VSim** (Kumarage et al., 2025) — request from the authors; place `train_data.json` and `test_data.json` in `MyDrive/thesis_models/sevsim/`.
- **LoveFraud02** (Faber, 2024) — [Mendeley Data, doi:10.17632/kmhvb4x5d8.1](https://doi.org/10.17632/kmhvb4x5d8.1), converted to SEConvo format before use.

## Reproducing

**From cached outputs (no GPU):** the notebook loads saved checkpoints and cached logits whenever they exist, so the evaluation sections can be re-run without retraining.

**From scratch:** open the notebook in Google Colab with a T4 GPU, mount Google Drive, and run the cells in order. Artefacts are written to `MyDrive/thesis_models/v3_leakfree/` (set `BASE_DIR` in Section 3 to change this). Environment: Python 3.12, PyTorch 2, Transformers 4.51, scikit-learn, statsmodels, SHAP 0.51, LIME 0.2.

GPU kernels on Colab are not bit-for-bit reproducible; retraining the encoders can shift cross-fitted F1 by about 0.06 (see the paper's threats-to-validity section).

## Citation

The paper is under review at *Online Social Networks and Media*. Citation details will be added on publication.

## License

MIT (code). The datasets remain under their original licences.
