import os
from datetime import datetime
from pathlib import Path

from common.paths import attach_mix_ratio_fields

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_AGGREGATE_MD = _PROJECT_ROOT / "results_main.md"


def _aggregate_result_md_path() -> Path:
    env = os.environ.get("UNIMISS_AGGREGATE_RESULT_MD", "").strip()
    if not env:
        return _DEFAULT_AGGREGATE_MD
    p = Path(env).expanduser()
    return p.resolve() if p.is_absolute() else (Path.cwd() / p).resolve()


def _result_section_title(model_name: str, agg: dict) -> str:
    parts = [model_name, str(agg["dataset"])]
    if agg.get("mask_type") == "mix" and agg.get("mix_ratio_label"):
        parts.append(f"mix · MAR:MNAR {agg['mix_ratio_label']}")
    else:
        parts.append(str(agg["mask_type"]))
    line = " | ".join(parts) + f" | Missing Rate: {agg['missing_rate']}"
    run = agg.get("run_name")
    if run and str(run) != "full":
        line += f" | run={run}"
    return f"## {line}\n"


def _mix_meta_lines(agg: dict) -> list[str]:
    if agg.get("mask_type") != "mix" or not agg.get("mix_ratio_label"):
        return []
    mr = float(agg["mar_ratio"])
    lines = [
        f"> **Mix MAR:MNAR = {agg['mix_ratio_label']}** "
        f"(``mar_ratio={mr:.2f}``, MAR share {int(round(mr * 100))}% of mix mechanism)\n",
    ]
    run = agg.get("run_name")
    if run and str(run) != "full":
        lines.append(f"> Run name: `{run}`\n")
    tag = agg.get("mix_output_tag")
    if tag:
        lines.append(f"> Output dir tag: `{tag}`\n")
    return lines


def _metrics_table_lines(agg: dict, *, for_summary: bool = False) -> list[str]:
    runs = agg["runs"]
    lines: list[str] = []
    nparm = agg.get("n_parameters")
    if for_summary and nparm is not None:
        lines.append(f"**Parameters:** {int(nparm):,}\n")
    lines.append("| Seed | MAE | RMSE | MRE | NRMSE | Eval Points |")
    lines.append("|:----:|:-------:|:-------:|:-------:|:-------:|:-----------:|")
    for run in runs:
        lines.append(
            f"| {run['seed']} "
            f"| {run['mae']:.6f} "
            f"| {run['rmse']:.6f} "
            f"| {run['mre']:.6f} "
            f"| {run['nrmse']:.6f} "
            f"| {int(run['n_points'])} |"
        )
    a_mae = agg["mae"]
    a_rmse = agg["rmse"]
    a_mre = agg["mre"]
    a_nrmse = agg["nrmse"]
    a_np = agg["n_points"]
    lines.append(
        f"| **Mean±Std** "
        f"| **{float(a_mae['mean']):.3f}±{float(a_mae['std']):.3f}** "
        f"| **{float(a_rmse['mean']):.3f}±{float(a_rmse['std']):.3f}** "
        f"| **{float(a_mre['mean']):.3f}±{float(a_mre['std']):.3f}** "
        f"| **{float(a_nrmse['mean']):.3f}±{float(a_nrmse['std']):.3f}** "
        f"| **{float(a_np['mean']):.0f}±{float(a_np['std']):.0f}** |"
    )
    lines.append("\n---\n")
    return lines


def _table_block(model_name: str, agg: dict, *, for_summary: bool = False) -> list[str]:
    lines = [_result_section_title(model_name, agg)]
    lines.extend(_mix_meta_lines(agg))
    lines.extend(_metrics_table_lines(agg, for_summary=for_summary))
    return lines


def write_result_md(
    model_name: str,
    agg: dict,
    result_path: str | Path | None = None,
    *,
    mar_ratio: float | None = None,
):
    md_path = Path(result_path) if result_path is not None else _DEFAULT_AGGREGATE_MD
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    mr = float(mar_ratio if mar_ratio is not None else agg.get("mar_ratio", 0.5))
    agg = attach_mix_ratio_fields(dict(agg), mar_ratio=mr)

    def artifact_section(include_doc_header: bool) -> list[str]:
        lines: list[str] = []
        if include_doc_header:
            lines.append("# Experiment Results\n")
        lines.append(_result_section_title(model_name, agg))
        lines.append(f"> Recorded at: {timestamp}\n")
        lines.extend(_mix_meta_lines(agg))
        te = agg.get("training_epochs")
        if te is not None:
            lines.append(f"> Training epochs: {te}\n")
        nparm = agg.get("n_parameters")
        if nparm is not None:
            lines.append(f"> Parameters: {int(nparm):,}\n")
        lines.extend(_metrics_table_lines(agg))
        return lines

    def summary_section(include_doc_header: bool) -> list[str]:
        lines: list[str] = []
        if include_doc_header:
            lines.append("# Experiment Results\n")
        lines.extend(_table_block(model_name, agg, for_summary=True))
        return lines

    aggregate_md = _aggregate_result_md_path()
    targets: list[Path] = []
    if md_path.resolve() != aggregate_md.resolve():
        targets.append(md_path)
    targets.append(aggregate_md)

    printed = []
    for target in targets:
        need_header = not target.exists() or target.stat().st_size == 0
        if target.resolve() == aggregate_md.resolve():
            text = "\n".join(summary_section(need_header)) + "\n"
        else:
            text = "\n".join(artifact_section(need_header)) + "\n"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as f:
            f.write(text)
        printed.append(target.resolve())

    mix_hint = ""
    if agg.get("mix_ratio_label"):
        mix_hint = f" | mix MAR:MNAR={agg['mix_ratio_label']} (mar_ratio={agg['mar_ratio']:.2f})"
    run_hint = ""
    if agg.get("run_name") and str(agg["run_name"]) != "full":
        run_hint = f" | run={agg['run_name']}"
    print(f"\n[Result] Appended {model_name}{mix_hint}{run_hint} @ mr={agg['missing_rate']} to:")
    for p in printed:
        tip = " (aggregate)" if p == aggregate_md.resolve() else ""
        print(f"  - {p}{tip}")
