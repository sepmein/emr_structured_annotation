"""Compare paired full-response benchmark records and optionally export a figure.

Analysis uses stdlib. --plot additionally needs matplotlib in the caller's
environment. No inference, network calls, or training. Capacity is an explicit
assumption-based scenario, never inferred from serial latency or API timings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile


def number(value, name, minimum=0, maximum=None, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if value < minimum or (positive and value == 0) or (maximum is not None and value > maximum):
        raise ValueError(f"{name} is outside its permitted range")
    return value


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def identifier(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
        raise ValueError("Stable task identifiers are required")
    return str(value)


def quantile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def task_fingerprint(payload):
    try:
        value = payload["inputs"]["tasks"]["sha256"]
    except (KeyError, TypeError):
        raise ValueError("Both benchmarks require inputs.tasks.sha256") from None
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Task fingerprint must be a lowercase SHA256")
    return value


def summarize(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list) or not payload["samples"]:
        raise ValueError("Expected a benchmark object with nonempty formal samples")
    fingerprint = task_fingerprint(payload)
    repeats = integer(payload.get("repeats"), "repeats", 1)
    concurrency = integer(payload.get("concurrency"), "concurrency", 1)
    integer(payload.get("tasks_per_request"), "tasks_per_request", 1)
    if payload["tasks_per_request"] != 1:
        raise ValueError("This comparison requires one complete record per request")
    if not isinstance(payload.get("measurement_scope"), str) or not payload["measurement_scope"].strip():
        raise ValueError("An explicit measurement_scope is required")
    grouped, seen, versions = {}, set(), set()
    success_latencies, all_latencies = [], []
    for sample in payload["samples"]:
        if not isinstance(sample, dict) or sample.get("phase") != "measured":
            raise ValueError("samples must contain formal measured requests only")
        task_id = identifier(sample.get("task_id"))
        repeat = integer(sample.get("repeat"), "repeat")
        if repeat >= repeats or (task_id, repeat) in seen:
            raise ValueError("Duplicate or out-of-range task/repeat")
        seen.add((task_id, repeat))
        chars = integer(sample.get("text_chars"), "text_chars", 1)
        elapsed = number(sample.get("latency_ms"), "latency_ms")
        status = sample.get("status")
        if status not in ("ok", "failed"):
            raise ValueError("Every request must explicitly be ok or failed")
        row = grouped.setdefault(task_id, {"task_id": task_id, "text_chars": chars, "successful_latencies_ms": [], "failed_runs": 0})
        if chars != row["text_chars"]:
            raise ValueError("Text length changed between repetitions")
        all_latencies.append(elapsed)
        if status == "ok":
            success_latencies.append(elapsed)
            row["successful_latencies_ms"].append(elapsed)
            version = sample.get("response_model_version")
            if version is not None and (not isinstance(version, str) or not version.strip()):
                raise ValueError("Reported model versions must be nonempty strings or null")
            if version:
                versions.add(version)
        else:
            row["failed_runs"] += 1
    if len(versions) > 1 or payload.get("multiple_model_versions_seen") is True:
        raise ValueError("Mixed reported model versions require separate runs")
    if len(seen) != len(grouped) * repeats:
        raise ValueError("Every task needs one record per formal repetition, including failures")
    if integer(payload.get("unique_tasks"), "unique_tasks", 1) != len(grouped) or integer(payload.get("measured_task_runs"), "measured_task_runs", 1) != len(seen):
        raise ValueError("Declared task/request totals disagree with the sample records")
    wall = number(payload.get("measured_wall_seconds"), "measured_wall_seconds", positive=True)
    if wall * 1000 + 1e-6 < max(all_latencies):
        raise ValueError("Measured wall time is shorter than a complete request")
    if concurrency == 1 and wall * 1000 + 1e-6 < sum(all_latencies):
        raise ValueError("Serial wall time must include all successful and failed requests")
    points = []
    for row in sorted(grouped.values(), key=lambda r: (r["text_chars"], r["task_id"])):
        points.append({"task_id": row["task_id"], "text_chars": row["text_chars"],
                       "median_success_latency_ms": statistics.median(row["successful_latencies_ms"]) if row["successful_latencies_ms"] else None,
                       "successful_runs": len(row["successful_latencies_ms"]), "failed_runs": row["failed_runs"]})
    failed = len(seen) - len(success_latencies)
    kind = payload.get("dataset_kind", "unverified")
    if kind not in ("unverified", "synthetic_demo", "human_test_set"):
        raise ValueError("Unknown dataset_kind declaration")
    return {"task_source_sha256": fingerprint, "unique_tasks": len(grouped), "formal_requests": len(seen),
            "repeats": repeats, "concurrency": concurrency, "measurement_scope": payload["measurement_scope"],
            "successful_requests": len(success_latencies), "failed_requests": failed, "failure_rate": failed / len(seen),
            "all_failed_tasks": sum(p["successful_runs"] == 0 for p in points),
            "p50_success_latency_ms": quantile(success_latencies, .5), "p95_success_latency_ms": quantile(success_latencies, .95),
            "measured_wall_seconds": wall, "observed_successful_requests_per_second": len(success_latencies) / wall,
            "dataset_kind": kind, "reported_versions": sorted(versions),
            "token_coverage": payload.get("token_coverage", {"status": "unknown"}), "points": points}


def read_context(context):
    if not isinstance(context, dict):
        raise ValueError("Context must contain ours and baseline descriptions")
    result = {}
    for role in ("ours", "baseline"):
        row = context.get(role)
        if not isinstance(row, dict):
            raise ValueError(f"Missing {role} context")
        label = row.get("label")
        if not isinstance(label, str) or not label.strip() or len(label) > 60 or any(c in label for c in "\n\r|<>"):
            raise ValueError("Use short plain-text model labels")
        if row.get("deployment") not in ("local", "external_api"):
            raise ValueError("deployment must be local or external_api")
        scope = row.get("task_scope_id")
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("task_scope_id must declare the extraction task, e.g. entity_only_v1")
        budget = row.get("resource_budget_id")
        if budget is not None and (not isinstance(budget, str) or not budget.strip()):
            raise ValueError("resource_budget_id must be nonempty when supplied")
        result[role] = {"label": label, "deployment": row["deployment"], "task_scope_id": scope, "resource_budget_id": budget}
    if result["ours"]["task_scope_id"] != result["baseline"]["task_scope_id"]:
        raise ValueError("The two runs must perform the same declared extraction task")
    return result


def capacity_scenario(scenario, context, synthetic):
    if not isinstance(scenario, dict):
        raise ValueError("Capacity scenario must be an object")
    if any(context[role]["deployment"] != "local" for role in context):
        raise ValueError("Same-local-resource hospital estimates cannot use external API comparisons")
    budget = context["ours"]["resource_budget_id"]
    if not budget or budget != context["baseline"]["resource_budget_id"]:
        raise ValueError("Declare the same resource_budget_id for local capacity scenarios")
    if scenario.get("basis") != "complete_records_per_second" or scenario.get("reviewed_quality_and_full_text") is not True:
        raise ValueError("Capacity requires complete-record rates and explicit quality/full-text review declaration")
    rates = scenario.get("sustained_rates")
    if not isinstance(rates, dict):
        raise ValueError("Explicit sustained_rates are required; serial observed throughput is never used automatically")
    hours = number(scenario.get("hours_per_day"), "hours_per_day", maximum=24, positive=True)
    fraction = number(scenario.get("planning_fraction"), "planning_fraction", maximum=1, positive=True)
    volume = number(scenario.get("records_per_hospital_day"), "records_per_hospital_day", positive=True)
    peak = number(scenario.get("peak_records_per_hospital_second"), "peak_records_per_hospital_second", positive=True)
    rationale = scenario.get("assumptions")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("Record the scenario assumptions, rate source, and planning margin rationale")
    outputs = {}
    for role in ("ours", "baseline"):
        q = number(rates.get(role), f"sustained_rates.{role}")
        daily = q * hours * 3600 * fraction
        by_day, by_peak = math.floor(daily / volume), math.floor(q * fraction / peak)
        outputs[role] = {"declared_sustained_full_records_per_second": q, "daily_planning_records": daily,
                         "hospitals_by_daily_volume": by_day, "hospitals_by_peak": by_peak, "planning_hospitals": min(by_day, by_peak)}
    ratio = rates["ours"] / rates["baseline"] if rates["baseline"] else None
    return {"status": "assumption_based_planning_only", "synthetic_demo": synthetic, "resource_budget_id": budget,
            "hours_per_day": hours, "planning_fraction": fraction, "records_per_hospital_day": volume,
            "peak_records_per_hospital_second": peak, "assumptions": rationale, "models": outputs,
            "declared_sustained_rate_ratio": ratio,
            "notes": ["Capacity is a declared scenario, not independently verified deployment evidence.",
                      "Rates are supplied separately and are never derived from latency or serial benchmark throughput.",
                      "Full-record rates must already include segmentation and the declared retry policy.",
                      "The smaller of daily-volume and peak limits is used; other workflow bottlenecks are not modeled."]}


def compare(ours, baseline, context, scenario=None):
    descriptions = read_context(context)
    runs = {"ours": summarize(ours), "baseline": summarize(baseline)}
    if runs["ours"]["task_source_sha256"] != runs["baseline"]["task_source_sha256"]:
        raise ValueError("Input task file hashes differ; freeze one common task file")
    maps = {role: {p["task_id"]: p["text_chars"] for p in run["points"]} for role, run in runs.items()}
    if maps["ours"] != maps["baseline"]:
        raise ValueError("Task IDs or character lengths differ between runs")
    kinds = {run["dataset_kind"] for run in runs.values()}
    if len(kinds) != 1:
        raise ValueError("Dataset-kind declarations disagree")
    synthetic = kinds == {"synthetic_demo"}
    aligned = all(runs["ours"][key] == runs["baseline"][key] for key in ("measurement_scope", "concurrency", "repeats"))
    success_keys = [{(identifier(s["task_id"]), s["repeat"]) for s in payload["samples"] if s["status"] == "ok"}
                    for payload in (ours, baseline)]
    same_success_population = success_keys[0] == success_keys[1]
    ratios = {}
    for metric in ("p50_success_latency_ms", "p95_success_latency_ms"):
        denominator, numerator = runs["ours"][metric], runs["baseline"][metric]
        ratios[metric] = numerator / denominator if aligned and same_success_population and denominator and numerator is not None else None
    result = {"protocol_version": 1, "dataset_kind": next(iter(kinds)), "synthetic_demo": synthetic,
              "context": descriptions, "runs": runs, "aligned_latency_protocol_declared": aligned,
              "same_successful_request_population": same_success_population,
              "descriptive_latency_ratio_baseline_over_ours": ratios,
              "capacity": capacity_scenario(scenario, descriptions, synthetic) if scenario is not None else None,
              "notes": ["The figure shows per-record successful-repeat medians; overall P50/P95 use all successful formal requests.",
                        "Failures remain in the request denominator and measured wall time; all-failed tasks have no latency point.",
                        "Character length is the common x axis; model-specific token counts are not interchangeable.",
                        "Task/measurement/resource declarations are not independent proof of quality, coverage, weights or hardware.",
                        "Descriptive latency ratios and serial observed throughput are not hospital-capacity multipliers."]}
    return result


def markdown(report):
    title = "虚构记录演示；非模型性能" if report["synthetic_demo"] else "模型测速分布对照；来源与条件需复核"
    lines = [f"# {title}", "", "每点为一条病历的成功重复请求耗时中位数；总体P50/P95按正式成功请求计算。", "",
             "| 项目 | 我们的模型 | 对照模型 |", "|---|---:|---:|"]
    for label, key in (("唯一病历数", "unique_tasks"), ("正式请求数", "formal_requests"), ("失败请求数", "failed_requests"),
                       ("全部失败病历数", "all_failed_tasks"), ("成功P50（ms）", "p50_success_latency_ms"),
                       ("成功P95（ms）", "p95_success_latency_ms"), ("观察到的请求吞吐（条/秒，非容量）", "observed_successful_requests_per_second")):
        values = [report["runs"][role][key] for role in ("ours", "baseline")]
        formatted = ["—" if v is None else f"{v:.3f}" if isinstance(v, float) else str(v) for v in values]
        lines.append(f"| {label} | {' | '.join(formatted)} |")
    lines += ["", "失败率、逐病历点位、输入校验值及完整口径见comparison.json。不同计时范围/并发/重复设计，或成功请求不是同一任务/轮次集合时，不计算延迟比值。",
              "", "## 有失败请求的任务", "", "| 方案 | 任务ID | 字符数 | 失败请求/正式请求 |", "|---|---|---:|---:|"]
    for role in ("ours", "baseline"):
        for point in report["runs"][role]["points"]:
            if point["failed_runs"]:
                safe_id = point["task_id"].replace("|", "\\|").replace("\n", " ").replace("\r", " ")
                lines.append(f"| {report['context'][role]['label']} | {safe_id} | {point['text_chars']} | {point['failed_runs']}/{point['successful_runs'] + point['failed_runs']} |")
    lines += [
              "", "## 医院容量", ""]
    if report["capacity"] is None:
        lines.append("未提供同资源本地持续负载及业务假设，不计算医院承载量。")
    else:
        cap = report["capacity"]
        lines += ["以下为声明输入的情景估算，不是已验证部署能力。", "", "| 方案 | 日总量限制 | 高峰限制 | 规划医院数 |", "|---|---:|---:|---:|"]
        for role in ("ours", "baseline"):
            values = cap["models"][role]
            lines.append(f"| {report['context'][role]['label']} | {values['hospitals_by_daily_volume']} | {values['hospitals_by_peak']} | {values['planning_hospitals']} |")
        lines += ["", cap["assumptions"]]
    return "\n".join(lines) + "\n"


def render_figure(report, output_dir, log_y=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    preferred = next((n for n in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS") if n in installed), "DejaVu Sans")
    plt.rcParams.update({"font.family": preferred, "axes.unicode_minus": False, "svg.fonttype": "path", "font.size": 11})
    fig = plt.figure(figsize=(15, 8.5), facecolor="white", layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=(1, .83, 1), height_ratios=(3, 1))
    left = fig.add_subplot(grid[0, 0])
    middle = fig.add_subplot(grid[0, 1])
    right = fig.add_subplot(grid[0, 2], sharex=left, sharey=left)
    bottom = fig.add_subplot(grid[1, :2] if report["capacity"] else grid[1, :])
    colors = {"ours": "#285E8E", "baseline": "#B76B25"}
    all_points = [p for r in report["runs"].values() for p in r["points"]]
    maximum_chars = max(p["text_chars"] for p in all_points)
    maximum_ms = max((p["median_success_latency_ms"] for p in all_points if p["median_success_latency_ms"] is not None), default=1)
    positive_ms = [p["median_success_latency_ms"] for p in all_points if p["median_success_latency_ms"] is not None]
    if log_y and (not positive_ms or min(positive_ms) <= 0):
        raise ValueError("Log latency axes require positive successful record medians")
    for role, axis in (("ours", left), ("baseline", right)):
        points = report["runs"][role]["points"]
        valid = [p for p in points if p["median_success_latency_ms"] is not None]
        axis.scatter([p["text_chars"] for p in valid], [p["median_success_latency_ms"] / 1000 for p in valid],
                     color=colors[role], s=42, alpha=.75, edgecolors="white", linewidths=.4)
        axis.set(xlabel="原文字符数", ylabel="完整响应耗时中位数（秒）", xlim=(0, maximum_chars * 1.05))
        if log_y:
            axis.set_yscale("log")
            axis.set_ylim(min(positive_ms) * .7 / 1000, maximum_ms * 1.5 / 1000)
            axis.set_ylabel("完整响应耗时中位数（秒，对数刻度）")
        else:
            axis.set_ylim(0, max(maximum_ms * 1.12 / 1000, .001))
        axis.set_title(report["context"][role]["label"], fontweight="bold", color=colors[role])
        axis.grid(alpha=.15)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        failed_points = [p for p in points if p["successful_runs"] == 0]
        axis.scatter([p["text_chars"] for p in failed_points], [.94] * len(failed_points), transform=axis.get_xaxis_transform(),
                     marker="x", color="#414A53", s=55)
        axis.text(.02, .98, f"× 全部失败：{len(failed_points)}条（顶端标记不表示耗时）", transform=axis.transAxes, va="top", fontsize=8.5)
    middle.axis("off")
    blocks = ["正式请求统计", "我们 / 对照"]
    for title, key, unit, divisor in (("典型耗时 P50", "p50_success_latency_ms", "秒", 1000), ("较慢请求 P95", "p95_success_latency_ms", "秒", 1000),
                                      ("观察吞吐（非容量）", "observed_successful_requests_per_second", "条/秒", 1)):
        vals = [report["runs"][r][key] for r in ("ours", "baseline")]
        blocks += ["", title, " / ".join("—" if v is None else f"{v / divisor:,.3f}" for v in vals) + f" {unit}"]
    blocks += ["", "失败率", " / ".join(f"{report['runs'][r]['failure_rate']:.1%}" for r in ("ours", "baseline")),
               "", "描述性P50 / P95比值（对照÷我们）",
               " / ".join("—" if v is None else f"{v:.2f}倍" for v in report["descriptive_latency_ratio_baseline_over_ours"].values()),
               "", f"唯一病历：{report['runs']['ours']['unique_tasks']}条", "质量 / 全文覆盖：另附评价", "容量：须另做持续负载测试"]
    if not report["aligned_latency_protocol_declared"]:
        blocks += ["计时/并发/重复条件未对齐"]
    if not report["same_successful_request_population"]:
        blocks += ["成功样本不同，未计算比值"]
    middle.text(.5, .98, "\n".join(blocks), ha="center", va="top", fontsize=10.5, linespacing=1.15)
    lengths = [p["text_chars"] for p in report["runs"]["ours"]["points"]]
    bottom.hist(lengths, bins=min(12, max(1, len(set(lengths)))), color="#818B95", edgecolor="white")
    bottom.set(xlabel="两侧共同的原文字符数", ylabel="唯一病历数", title="输入长度分布（每条病历计一次）")
    bottom.spines[["top", "right"]].set_visible(False)
    if report["capacity"]:
        capacity_axis = fig.add_subplot(grid[1, 2])
        values = [report["capacity"]["models"][r]["planning_hospitals"] for r in ("ours", "baseline")]
        capacity_axis.barh(["我们", "对照"], values, color=[colors[r] for r in ("ours", "baseline")])
        rates = report["capacity"]["models"]
        supplied = " / ".join(f"{rates[r]['declared_sustained_full_records_per_second']:g}" for r in ("ours", "baseline"))
        capacity_axis.set(xlim=(0, max(1, max(values) * 1.25)), xlabel="规划医院数（家）")
        capacity_axis.set_title(f"假设情景 · 非已验证部署容量\n另填持续吞吐：{supplied}条/秒", fontsize=10)
        for index, value in enumerate(values):
            capacity_axis.text(value, index, f" {value}", va="center")
        capacity_axis.spines[["top", "right"]].set_visible(False)
    if report["synthetic_demo"]:
        title = "虚构测速记录 · 仅演示图形与计算，不代表模型性能"
    elif any(c["deployment"] == "external_api" for c in report["context"].values()):
        title = "本地模型与外部API的服务耗时对照 · 不代表同算力容量"
    else:
        title = "本地模型服务耗时分布 · 硬件与任务条件另附记录"
    fig.suptitle(title, fontsize=17, fontweight="bold")
    for suffix in ("png", "svg"):
        fig.savefig(output_dir / f"latency_distribution.{suffix}", dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ours", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--capacity-scenario", type=Path)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--log-y", action="store_true", help="Use the same labeled log latency scale on both plots")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.log_y and not args.plot:
            raise ValueError("--log-y requires --plot")
        if args.output_dir.exists():
            raise ValueError("Use a new output directory; existing comparisons are never overwritten")
        paths = {key: getattr(args, key) for key in ("ours", "baseline", "context", "capacity_scenario") if getattr(args, key) is not None}
        data = {key: path.read_bytes() for key, path in paths.items()}
        parsed = {key: json.loads(value.decode("utf-8-sig")) for key, value in data.items()}
        report = compare(parsed["ours"], parsed["baseline"], parsed["context"], parsed.get("capacity_scenario"))
        report.update({"generated_at_utc": datetime.now(timezone.utc).isoformat(),
                       "sources": {key: {"file": paths[key].name, "sha256": hashlib.sha256(value).hexdigest()} for key, value in data.items()},
                       "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".benchmark-comparison-", dir=args.output_dir.parent) as temp:
            stage = Path(temp) / "result"
            stage.mkdir()
            (stage / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            (stage / "comparison.md").write_text(markdown(report), encoding="utf-8")
            if args.plot:
                render_figure(report, stage, args.log_y)
            stage.rename(args.output_dir)
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    print(f"Paired unique records={report['runs']['ours']['unique_tasks']}; dataset={report['dataset_kind']}; hospital capacity={'scenario only' if report['capacity'] else 'not estimated'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
