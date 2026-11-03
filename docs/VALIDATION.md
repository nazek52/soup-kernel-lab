# Validation record

Date: 2026-11-04. Host: Windows, Python 3.12, torch 2.15.0+cpu.

## Executed locally

- `python -m pytest solution -q`: **18 passed, 20 skipped**.
- The skipped cases require CUDA, including the official SwiGLU tests.
- CPU cases cover the fallback loss and gradient, ignored labels, irregular shapes,
  frozen/trainable norm weights and invalid labels.
- Ruff check of new solution modules: passed.
- Official source analyzer: five original kernels detected; both starter SwiGLU kernels
  correctly excluded. No close rewrites, banned imports or syntax errors reported.
- `harness/`, `task/`, `tests/`: no differences from the starter commit.
- All six executable notebook cells parsed as Python; outputs remain empty.
- Submission completeness checker reports INCOMPLETE as expected (missing links/results).

These are CPU fallback tests, not execution of the Triton kernels. PyTorch emitted a warning
about optional NumPy being absent; it did not affect these tensor-only tests.

## Not executed

- Triton compilation and GPU forward/backward tests.
- Official 64-step training, held-out evaluation and 79 Soup regression tests.
- Own-kernel time share and all ten official checks.
- Performance measurements on Kaggle T4.

No GPU speed, memory, quality or competition score is claimed. Source comparison against
starter commit c7c2fcd15972ce8b898d8d3ff732ee192c817ceb must leave task/, harness/ and tests/
unchanged. The repository is a candidate awaiting GPU validation and publication details.
