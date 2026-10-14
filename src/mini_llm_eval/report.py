import json
from collections import defaultdict

from mini_llm_eval.registry import REPORTS_DIR, RESULTS_DIR, SUITES_DIR, load_registry, load_suite


def latest_runs(results_dir=RESULTS_DIR):
    runs = {}
    for meta_path in sorted(results_dir.glob("*/*/*/meta.json")):
        run = meta_path.parent
        runs[(run.parents[1].name, run.parents[0].name)] = run
    return runs


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def metric(results, task, name):
    entry = results["results"].get(task, {})
    return number(entry.get(f"{name},none")), number(entry.get(f"{name}_stderr,none"))


def collect(suite, registry, results_dir=RESULTS_DIR):
    rows = []
    for (model_name, suite_name), run in latest_runs(results_dir).items():
        if suite_name != suite.name or model_name not in registry:
            continue
        results = json.loads((run / "results.json").read_text())
        meta = json.loads((run / "meta.json").read_text())
        model = registry[model_name]
        rows.append(
            {
                "model": model,
                "meta": meta,
                "scores": {t.name: metric(results, t.name, t.primary) for t in suite.tasks},
            }
        )
    order = list(registry)
    rows.sort(key=lambda r: order.index(r["model"].name))
    return rows


def consistency_warnings(rows):
    warnings = []
    for label, getter in [
        ("lm-eval version", lambda m: m["lm_eval"]),
        ("protocol version", lambda m: m["protocol"]),
        ("few-shot count", lambda m: m["suite"]["num_fewshot"]),
        ("example limit", lambda m: m["suite"]["limit"]),
        ("device", lambda m: m["device"]),
    ]:
        values = sorted({str(getter(r["meta"])) for r in rows})
        if len(values) > 1:
            warnings.append(f"Runs use different values of **{label}**: {', '.join(values)}.")
    if any(r["meta"]["mini_llm_eval"].get("dirty") is not False for r in rows):
        warnings.append("Some runs come from a modified or untracked working tree.")
    if any(r["meta"]["suite"]["limit"] for r in rows):
        warnings.append("Some runs use only the first examples of each benchmark (limit set).")
    return warnings


def format_cell(value, stderr):
    if value is None:
        return ""
    return f"{value:.3f} ± {stderr:.3f}" if stderr is not None else f"{value:.3f}"


def markdown_table(suite, rows):
    header = ["Model", "Non-embedding params", "Tokens seen"] + [f"{t.name} ({t.primary})" for t in suite.tasks]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        m = r["model"]
        cells = [m.name, f"{m.non_embedding_params / 1e6:.1f}M", f"{m.tokens_seen / 1e9:.2f}B"]
        cells += [format_cell(*r["scores"][t.name]) for t in suite.tasks]
        lines.append("| " + " | ".join(cells) + " |")
    chance = ["Random guessing", "", ""] + ["" if t.chance is None else f"{t.chance:.3f}" for t in suite.tasks]
    lines.append("| " + " | ".join(chance) + " |")
    return "\n".join(lines)


def plot_scores(suite, rows, x_attr, xlabel, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tasks = [t for t in suite.tasks if any(r["scores"][t.name][0] is not None for r in rows)]
    if not tasks:
        return None
    cols = min(3, len(tasks))
    nrows = -(-len(tasks) // cols)
    fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 3.6 * nrows), squeeze=False)
    families = defaultdict(list)
    for r in rows:
        families[r["model"].family].append(r)
    for ax, task in zip(axes.flat, tasks):
        for family, members in families.items():
            points = sorted(
                (getattr(r["model"], x_attr), r["scores"][task.name]) for r in members if r["scores"][task.name][0] is not None
            )
            if not points:
                continue
            xs = [p[0] for p in points]
            ys = [p[1][0] for p in points]
            errs = [p[1][1] or 0 for p in points]
            ax.errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=family)
        if task.chance is not None:
            ax.axhline(task.chance, linestyle="--", color="gray", linewidth=1)
        ax.set_xscale("log")
        ax.set_title(f"{task.name} ({task.primary})", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(tasks) :]:
        ax.axis("off")
    axes.flat[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def build_report(suite_name, results_dir=RESULTS_DIR, reports_dir=REPORTS_DIR, registry=None, suites_dir=SUITES_DIR):
    registry = registry or load_registry()
    suite = load_suite(suite_name, suites_dir)
    rows = collect(suite, registry, results_dir)
    if not rows:
        raise SystemExit(f"No results for suite {suite_name!r}. Run `mini-eval run {suite_name} MODEL` first.")
    reports_dir.mkdir(parents=True, exist_ok=True)
    tokens_png = plot_scores(suite, rows, "tokens_seen", "tokens seen", reports_dir / f"{suite.name}_vs_tokens.png")
    params_png = plot_scores(
        suite, rows, "non_embedding_params", "non-embedding parameters", reports_dir / f"{suite.name}_vs_params.png"
    )
    meta = rows[0]["meta"]
    parts = [
        f"# Results: {suite.name}",
        "",
        suite.description,
        "",
        f"Protocol v{meta['protocol']}, lm-eval {meta['lm_eval']}, {suite.num_fewshot}-shot. "
        "Values are the primary metric of each task with their standard error.",
        "",
    ]
    warnings = consistency_warnings(rows)
    if warnings:
        parts += ["> **Warnings**", ">"] + [f"> - {w}" for w in warnings] + [""]
    parts += [markdown_table(suite, rows), ""]
    if tokens_png:
        parts += ["## Score against tokens seen", "", f"![]({tokens_png.name})", ""]
    if params_png:
        parts += ["## Score against non-embedding parameters", "", f"![]({params_png.name})", ""]
    out = reports_dir / f"{suite.name}.md"
    out.write_text("\n".join(parts))
    return out