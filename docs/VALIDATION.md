# Фактическая проверка

Дата: 4 ноября 2026, Казахстан.

## GitHub Actions

https://github.com/nazek52/soup-kernel-lab/actions/runs/41710054016
Код: 8d63b2445d68c909758679262d8bb10c8ae7d8ed.
Ubuntu 24.04, Python 3.12.16, torch 2.8.0+cpu, triton 3.4.0.
- Ruff: PASS.
- pytest solution dev_tests: 20 passed, 30 skipped (20 CUDA, 10 interpreter).
- TRITON_INTERPRET=1 pytest dev_tests/test_interpreter.py: 10 passed.
- CPU HTML/JSON report: создан; workflow artifact cpu-demonstration.

Интерпретатор выполняет настоящие тела собственных ядер, проверяет арифметику
и индексы. Он не проверяет CUDA-компиляцию, асинхронное выполнение или скорость.

## Локальный запуск

Windows, Python 3.12.14, torch 2.15.0+cpu. 20 passed, 30 skipped.
Демо seed2026: 3/3 численных сравнения прошли. Числа: evidence/local-cpu-result.json.
Презентация и видео показывают именно локальные числа.

## Kaggle

https://www.kaggle.com/code/loolaey/soup-kernel-lab-cpu-demo?scriptVersionId=365396899
Save & Run All: successful, 24.6 секунды. Python3.13.18, torch2.11.0+cpu.
3/3 сравнений loss и градиентов прошли; evidence/kaggle-cpu-result.json.
Настоящий скачанный notebook с outputs: notebooks/kaggle_cpu_demo_executed.ipynb.
Kaggle требует телефон для Public и GPU. По выбору владельца телефон не проверяли.
Ноутбук остался приватным, его копия с outputs опубликована на GitHub.

## Материалы

PowerPoint: 6 слайдов, структура/геометрия/шрифты/повторный импорт прошли,
все слайды визуально проверены. Видео1280×720, H.264/AAC,158.9секунды,
синтетическая русская озвучка, полный файл успешно декодирован ffmpeg.

## Непроверенное

T4 CUDA-тесты; скорость и память полного обучения; качество адаптера;
10 официальных правил и доля собственных ядер≥5%. Эти результаты не заявлены.
`scripts/check_submission.py` намеренно остаётся строгой проверкой конкурсной
комплектности и выдаёт INCOMPLETE без команды/официальных GPU-результатов.
