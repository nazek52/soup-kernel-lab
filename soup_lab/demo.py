"""Small real CPU experiment; separate from the fixed Soup training task."""

import hashlib
import html
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F


def tiled_reference(x, labels, block=4096):
    """Readable PyTorch model of our algorithm, not the Triton implementation."""
    parts = [torch.logsumexp(part.float(), dim=-1) for part in x.split(block, dim=-1)]
    lse = torch.logsumexp(torch.stack(parts, dim=-1), dim=-1)
    selected = x.gather(1, labels.clamp_min(0)[:, None]).squeeze(1)
    return torch.where(labels != -100, lse - selected, 0).sum() / (labels != -100).sum()


def experiment():
    generator = torch.Generator().manual_seed(2026)
    rows = []
    for vocab in (257, 4099, 151936):
        x = torch.randn(8, vocab, generator=generator, requires_grad=True)
        labels = torch.randint(0, vocab, (8,), generator=generator)
        labels[::3] = -100
        reference = F.cross_entropy(x, labels)
        actual = tiled_reference(x, labels)
        ref_grad = torch.autograd.grad(reference, x)[0]
        actual_grad = torch.autograd.grad(actual, x)[0]
        torch.testing.assert_close(actual, reference, atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(actual_grad, ref_grad, atol=2e-5, rtol=2e-5)
        rows.append({"vocabulary": vocab, "rows": 8, "reference_loss": reference.item(),
                     "tiled_loss": actual.item(), "loss_abs_error": abs((actual-reference).item()),
                     "gradient_max_abs_error": (actual_grad-ref_grad).abs().max().item(),
                     "passed": True})
    return {"scope": "CPU PyTorch mathematical reference, not Triton or full Soup training",
            "created_utc": datetime.now(timezone.utc).isoformat(), "seed": 2026,
            "python": platform.python_version(), "torch": torch.__version__, "cases": rows,
            "gpu_benchmark": "not run", "official_checks": "not run"}


def render_report(result):
    rows = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(value))}</td>" for value in (
            case["vocabulary"], f'{case["reference_loss"]:.7f}', f'{case["tiled_loss"]:.7f}',
            f'{case["loss_abs_error"]:.2e}', f'{case["gradient_max_abs_error"]:.2e}', "PASS"
        )) + "</tr>" for case in result["cases"]
    )
    return f'''<!doctype html><html lang="ru"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Soup Kernel Lab — проверка вычислений</title>
<style>
:root {{ color-scheme:light; font-family:Segoe UI,Arial,sans-serif; color:#172837; background:#f5f4ef; }}
body {{ margin:0; }} main {{ max-width:1100px; margin:auto; padding:64px 32px; }}
h1 {{ font-size:clamp(36px,5vw,64px); letter-spacing:-2px; margin:16px 0; }}
h2 {{ font-size:28px; margin-top:48px; }} p {{ line-height:1.65; max-width:850px; font-size:18px; }}
.eyebrow {{ color:#426e61; letter-spacing:3px; font-size:13px; font-weight:700; }}
.lead {{ font-size:22px; }} .table-wrap {{ overflow-x:auto; }}
table {{ width:100%; border-collapse:collapse; margin:28px 0; background:white; }}
td,th {{ text-align:left; padding:18px 16px; border-bottom:1px solid #d6ded8; }}
th {{ font-size:13px; color:#426e61; }} td {{ font-variant-numeric:tabular-nums; }}
code {{ font-family:Consolas,monospace; }} .muted {{ color:#5e6b71; font-size:15px; }}
a {{ color:#246452; }} details {{ padding:20px 0; border-top:1px solid #bdccc4; }}
</style><main>
<div class="eyebrow">SOUP KERNEL LAB / NUMERICAL EXPERIMENT</div>
<h1>Тот же loss.<br>Вычисление по блокам.</h1>
<p class="lead">Реальные результаты небольшой проверки cross-entropy на CPU.
Сравниваем формулу по блокам словаря с PyTorch, включая градиенты.</p>
<p><strong>3 из 3 сравнений прошли.</strong> Это проверка математического прототипа.
Triton-ядра и полное обучение Soup проверяются отдельно.</p>
<h2>Измеренная численная ошибка</h2><div class="table-wrap"><table>
<thead><tr><th>Словарь</th><th>PyTorch loss</th><th>По блокам</th><th>Ошибка loss</th>
<th>Макс. ошибка градиента</th><th>Статус</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Что делает алгоритм</h2>
<p>Каждый блок из 4096 классов вычисляет log-sum-exp. Второе сокращение объединяет
блоки строки. Затем вычитается logit целевого класса. При обратном проходе градиент
равен вероятности softmax минус индикатор правильного класса.</p>
<details open><summary>Граница доказательства</summary><p>Здесь нет замера ускорения GPU,
потребления VRAM или качества адаптера. Полный конкурсный запуск использует фиксированную
Qwen2.5-1.5B-Instruct, Kaggle T4 и неизменённый официальный harness.</p></details>
<h2>Следующая проверка</h2><p><code>python -m soup_lab gpu-test</code> проверит CUDA-ядра.
<code>python -m soup_lab benchmark</code> выполнит официальный эксперимент на T4.</p>
<p><a href="https://github.com/nazek52/soup-kernel-lab">Исходный код и Kaggle notebook</a>
 · <a href="result.json">JSON с результатами</a></p>
<p class="muted">Seed {result['seed']} · Python {html.escape(result['python'])} ·
PyTorch {html.escape(result['torch'])}<br>{html.escape(result['created_utc'])}</p>
</main></html>'''


def run_demo(output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    result = experiment()
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report = output / "index.html"
    report.write_text(render_report(result), encoding="utf-8")
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (output / "result.json", report)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return report
