# Submission: Soup RMSNorm + tiled cross-entropy candidate

**DRAFT / NOT SUBMITTED / NOT GPU-VALIDATED**

Prepared on 2026-11-04, after the stated 2026-10-31 deadline. No extension is confirmed.

## Links

- Repository, final solution commit: PENDING
- Public Kaggle notebook, saved version with COMMIT set to the solution SHA: PENDING
- Demo video, no more than 3 minutes: PENDING
- Presentation: PENDING

Add the saved Kaggle URL to README too. Fill submission.json for the completeness checker.

## Team (up to 4 people)

| Name | Role | GitHub | Kaggle | Contact (captain) |
|---|---|---|---|---|
| PENDING | PENDING | PENDING | PENDING | PENDING |

## Our kernels

| Kernel | File | Replaces | Training-time share |
|---|---|---|---|
| ce_tile_lse | solution/kernels.py | Per-tile log-sum-exp | Not measured |
| ce_row_loss | solution/kernels.py | Per-row CE loss | Not measured |
| ce_tile_backward | solution/kernels.py | Log-softmax backward intermediate | Not measured |
| rms_forward | solution/kernels.py | Frozen-weight RMSNorm forward | Not measured |
| rms_backward | solution/kernels.py | Frozen-weight RMSNorm input gradient | Not measured |

Written with Codex assistance using the log-sum-exp identity, softmax derivative and
RMSNorm chain rule. Integration semantics checked against
[transformers v5.16.1](https://github.com/huggingface/transformers/tree/v5.16.1).
The unchanged SwiGLU example is from the official starter, **not our own kernel**.
No third-party kernel library is used.

## Results on Kaggle T4

| | Base Soup in our session | Ours | Ratio |
|---|---|---|---|
| Whole soup train, seconds | Not measured | Not measured | Not measured |
| Peak GPU memory, MiB | Not measured | Not measured | Not measured |
| Public eval loss | Not measured | Not measured | Not measured |

- [ ] All 10 official checks passed on the submitted code and saved Kaggle version.
- [x] README describes implementation, setup, limitations and missing measurements.
- [ ] Public notebook URL added to both README and SUBMISSION.
- [ ] Team, final commit, presentation and demo links supplied.

## Presentation

Outline: docs/PRESENTATION_OUTLINE_RU.md. Recording script: docs/DEMO_SCRIPT_RU.md.
These are preparation documents, not a finished presentation or video.

## Limitations

CUDA correctness, kernel share, training/eval quality and performance are unverified.
The 5% own-kernel time requirement may fail. Full logits remain for Soup metrics.
Specialized CE: fp32 CUDA index-label classification, no weighting or smoothing.
Frozen RMSNorm: fp16/bf16/fp32 inputs, fp32 weights, widths up to 8192.
Other supported PyTorch cases use original operations. No higher-order GPU gradients.
