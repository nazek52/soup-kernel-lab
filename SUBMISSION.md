# Soup Kernel Lab — описание проекта

Самостоятельный проект. Не зарегистрирован как команда и не подан на конкурс.
Создан 4 ноября 2026, после указанного в FAQ дедлайна. GPU-проверка не выполнена.

## Материалы

- Репозиторий: https://github.com/nazek52/soup-kernel-lab
- Проверенная CI версия кода: `8d63b2445d68c909758679262d8bb10c8ae7d8ed`.
- Владелец: [nazek52](https://github.com/nazek52), Kaggle: loolaey.
- Kaggle CPU, успешный сохранённый запуск: https://www.kaggle.com/code/loolaey/soup-kernel-lab-cpu-demo?scriptVersionId=365396899
- Kaggle требует телефон для Public; владелец выбрал работу без проверки телефона.
- Публичная копия с outputs: [выполненный notebook](notebooks/kaggle_cpu_demo_executed.ipynb).
- [Демо-видео, 2:39](artifacts/Soup_Kernel_Lab_Demo.mp4), русская синтетическая озвучка.
- [Презентация PowerPoint](artifacts/Soup_Kernel_Lab.pptx), 6 редактируемых слайдов.
- [JSON результата Kaggle](evidence/kaggle-cpu-result.json).

## Подтверждённые проверки

GitHub Actions: 20 CPU-тестов и 10 Triton interpreter тестов прошли.
Kaggle CPU: 3 из 3 сравнений loss и градиентов прошли; Python 3.13.18,
PyTorch 2.11.0+cpu, seed 2026. Это проверка математического прототипа.
CUDA-компиляция, полное обучение и требования официального harness не проверены.

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
- [x] Сохранённый CPU notebook и его ограничение доступа описаны в README и SUBMISSION.
- [x] Репозиторий, проверенный commit, презентация и демо доступны.
- [ ] Официальная регистрация команды и конкурсная сдача (не выполнялись).

## Limitations

CUDA correctness, kernel share, training/eval quality and performance are unverified.
The 5% own-kernel time requirement may fail. Full logits remain for Soup metrics.
Specialized CE: fp32 CUDA index-label classification, no weighting or smoothing.
Frozen RMSNorm: fp16/bf16/fp32 inputs, fp32 weights, widths up to 8192.
Other supported PyTorch cases use original operations. No higher-order GPU gradients.
