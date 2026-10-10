"""Measure a ModernBERT Label Studio /predict service without training it.

Uses one task per request to avoid assigning positional responses to wrong tasks.
The first measured response for each task is preserved for entity evaluation.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
import xml.etree.ElementTree as ET

from emr_annotation.evaluation.entity_predictions import normalize_records, read_label_groups
from emr_annotation.evaluation.regex_entity_baseline import quantile


class RequestFailure(Exception):
    """Safe error category: never carries response bodies, URLs or credentials."""

    def __init__(self, category: str, http_status: int | None = None):
        super().__init__(category)
        self.category = category
        self.http_status = http_status


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not silently send task text or Authorization to a different endpoint.
        return None


def make_transport(endpoint: str, timeout: float, username: str | None, password: str | None) -> Callable:
    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Endpoint must be an HTTP(S) URL without credentials, query or fragment")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Request timeout must be finite and positive")
    if bool(username) != bool(password):
        raise ValueError("Provide both basic-auth environment variables, or neither")
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if username:
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        headers["Authorization"] = "Basic " + token
    opener = build_opener(NoRedirect())

    def request_one(payload):
        request = Request(endpoint, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"), headers=headers, method="POST")
        try:
            with opener.open(request, timeout=timeout) as response:
                status = response.status
                raw = response.read(16 * 1024 * 1024 + 1)
            if len(raw) > 16 * 1024 * 1024:
                raise RequestFailure("response_too_large", status)
            try:
                return json.loads(raw.decode("utf-8-sig")), status
            except (ValueError, UnicodeError):
                raise RequestFailure("invalid_json", status) from None
        except HTTPError as exc:
            status = exc.code
            exc.close()
            raise RequestFailure("http_error", status) from None
        except (TimeoutError, socket.timeout):
            raise RequestFailure("timeout") from None
        except URLError as exc:
            category = "timeout" if isinstance(exc.reason, (TimeoutError, socket.timeout)) else "connection_error"
            raise RequestFailure(category) from None
        except OSError:
            raise RequestFailure("connection_error") from None

    return request_one


def parse_prediction(payload: Any, task, groups: dict[str, list[str]], targets: dict[str, str]) -> tuple[list[dict], str | None, int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list) or len(payload["results"]) != 1:
        raise ValueError("A single request must return exactly one prediction")
    prediction = payload["results"][0]
    if isinstance(prediction, list):
        if len(prediction) != 1:
            raise ValueError("Multiple candidate predictions are ambiguous")
        prediction = prediction[0]
    if not isinstance(prediction, dict) or not isinstance(prediction.get("result"), list):
        raise ValueError("Prediction requires an explicit result array")
    version = prediction.get("model_version")
    if version is not None and (not isinstance(version, str) or len(version) > 256):
        raise ValueError("Invalid model version")
    entities, ignored = [], 0
    for item in prediction["result"]:
        if not isinstance(item, dict):
            raise ValueError("Prediction result must be an object")
        kind = item.get("type")
        if kind in {"relation", "choices"}:
            # This tool validates entity output only, not relation/Choices quality.
            ignored += 1
            continue
        if kind != "labels":
            raise ValueError("Unsupported result type")
        control = item.get("from_name")
        value = item.get("value")
        if not isinstance(control, str) or control not in groups or item.get("to_name") != targets[control] or not isinstance(value, dict):
            raise ValueError("Entity controls do not match the frozen XML")
        labels = value.get("labels")
        if not isinstance(labels, list) or len(labels) != 1 or labels[0] not in groups[control]:
            raise ValueError("Entity label does not match its XML control")
        entity = {"start": value.get("start"), "end": value.get("end"), "label": labels[0]}
        if "text" in value:
            entity["text"] = value["text"]
        entities.append(entity)
    # Keep duplicate predictions; the evaluator counts their false positives.
    normalize_records([{"task_id": task.task_id, "text": task.text, "entities": entities}],
                      {label for group in groups.values() for label in group}, reference=False, source="response")
    return entities, version, ignored


def read_targets(label_xml: str, groups: dict[str, list[str]]) -> dict[str, str]:
    root = ET.fromstring(label_xml)
    texts = {node.attrib.get("name"): node.attrib.get("value") for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Text"}
    targets = {node.attrib["name"]: node.attrib.get("toName") for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "Labels"}
    if set(targets) != set(groups) or any(texts.get(target) != "$text" for target in targets.values()):
        raise ValueError("Benchmark expects every Labels control to target Text value=$text")
    return targets


def benchmark_service(payload: Any, groups: dict[str, list[str]], label_xml: str, project: str,
                      request_one: Callable, repeats: int, warmup: int) -> tuple[list[dict], dict]:
    if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
        raise ValueError("Tasks must be a normalized task array")
    # Never submit reference entities, annotations or patient metadata to the service.
    tasks = normalize_records([{"task_id": row.get("task_id"), "text": row.get("text"), "entities": []} for row in payload],
                              {label for group in groups.values() for label in group}, reference=True, source="tasks")
    targets = read_targets(label_xml, groups)
    if type(repeats) is not int or repeats < 1 or type(warmup) is not int or warmup < 0:
        raise ValueError("repeats must be positive and warmup nonnegative")
    if not isinstance(project, str) or not project.strip():
        raise ValueError("An explicit project identifier is required")
    samples, warmup_samples, predictions = [], [], []
    first_signatures = {}
    changed, wall_seconds = 0, 0.0

    def measure(task, repeat, phase):
        start = time.perf_counter_ns()
        entities, version, ignored, error, http_status = [], None, 0, None, None
        try:
            result, http_status = request_one({"tasks": [{"id": task.task_id, "data": {"text": task.text}}], "project": project, "label_config": label_xml, "params": {}})
            try:
                entities, version, ignored = parse_prediction(result, task, groups, targets)
            except (ValueError, TypeError):
                raise RequestFailure("invalid_entity_response", http_status) from None
        except RequestFailure as exc:
            error, http_status = exc.category, exc.http_status
        elapsed = (time.perf_counter_ns() - start) / 1_000_000
        return entities, {
            "task_id": task.task_id, "repeat": repeat, "phase": phase, "latency_ms": elapsed,
            "text_chars": len(task.text), "status": "ok" if error is None else "failed",
            "error_category": error, "http_status": http_status, "entity_count": len(entities),
            "response_model_version": version, "ignored_nonentity_results": ignored,
        }

    for repeat in range(warmup):
        for task in tasks.values():
            _, sample = measure(task, repeat, "warmup")
            warmup_samples.append(sample)
    for repeat in range(repeats):
        wall_start = time.perf_counter_ns()
        for task in tasks.values():
            entities, sample = measure(task, repeat, "measured")
            samples.append(sample)
            signature = (sample["status"], tuple(sorted((e["start"], e["end"], e["label"]) for e in entities)))
            if repeat == 0:
                first_signatures[task.task_id] = signature
                predictions.append({"task_id": task.task_id, "text": task.text, "entities": entities, "status": sample["status"]})
            elif signature != first_signatures[task.task_id]:
                changed += 1
        wall_seconds += (time.perf_counter_ns() - wall_start) / 1_000_000_000
    successful = [s for s in samples if s["status"] == "ok"]
    latencies = [s["latency_ms"] for s in successful]
    lengths = [len(t.text) for t in tasks.values()]
    versions = Counter(s["response_model_version"] or "unreported" for s in successful)
    return predictions, {
        "protocol_version": 1, "measurement_scope": "serial_http_request_through_entity_validation",
        "notes": [
            "Latency includes request serialization, transport, service execution, response parsing and entity validation.",
            "File I/O, initial input validation and warmup are excluded; loop bookkeeping is included in throughput wall time.",
            "This measures the deployed NER+RE request, not an isolated entity-only forward pass.",
            "First measured predictions (including failures) are used for quality evaluation, with no success retry selection.",
            "Later repeats are timing samples, not additional independent medical records.",
            "A client timeout does not prove server work stopped; there are no automatic retries.",
            "Token counts, actual maximum length and truncation coverage are not reported by this endpoint and remain unknown.",
            "Only entity output is validated; ignored relation/Choices results are not scored.",
            "Reported model version and declared artifact/environment do not prove which weights the server loaded.",
        ],
        "client_environment": {"python": platform.python_version(), "platform": platform.platform(), "machine": platform.machine()},
        "concurrency": 1, "tasks_per_request": 1, "unique_tasks": len(tasks), "repeats": repeats,
        "warmup_passes": warmup, "warmup_task_runs": len(warmup_samples),
        "warmup_failed_runs": sum(s["status"] != "ok" for s in warmup_samples),
        "measured_task_runs": len(samples), "successful_task_runs": len(successful),
        "failed_task_runs": len(samples) - len(successful), "failure_rate": 1 - len(successful) / len(samples),
        "error_categories": dict(Counter(s["error_category"] for s in samples if s["status"] != "ok")),
        "p50_success_latency_ms": quantile(latencies, 0.5) if latencies else None,
        "p95_success_latency_ms": quantile(latencies, 0.95) if latencies else None,
        "mean_success_latency_ms": statistics.mean(latencies) if latencies else None,
        "measured_wall_seconds": wall_seconds,
        "successful_tasks_per_second": len(successful) / wall_seconds if wall_seconds else None,
        "text_chars": {"min": min(lengths), "max": max(lengths), "p50": quantile(lengths, .5), "p95": quantile(lengths, .95)},
        "token_coverage": {"status": "unknown_not_returned_by_service", "truncation_rate": None},
        "successful_response_model_versions": dict(versions), "multiple_model_versions_seen": len(versions) > 1,
        "later_runs_differing_from_first_prediction": changed,
        "warmup_samples": warmup_samples, "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--label-config", required=True, type=Path)
    parser.add_argument("--endpoint", required=True, help="Explicit backend /predict URL")
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-name", choices=("before", "after"), required=True)
    parser.add_argument("--model-artifact-id", required=True, help="Operator-declared model artifact revision/checksum")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--dataset-kind", choices=("unverified", "synthetic_demo", "human_test_set"), default="unverified")
    args = parser.parse_args(argv)
    try:
        if not args.model_artifact_id.strip():
            raise ValueError("A nonempty declared model artifact identifier is required")
        inputs = {"tasks": args.tasks, "label_config": args.label_config}
        outputs = [args.output, args.benchmark]
        if len({p.resolve() for p in outputs}) != len(outputs) or {p.resolve() for p in outputs} & {p.resolve() for p in inputs.values()}:
            raise ValueError("Outputs must be distinct and must not overwrite inputs")
        # Refuse overwriting earlier measurements; choose a fresh run directory.
        if any(p.exists() for p in outputs):
            raise ValueError("Output already exists; choose a new run directory")
        groups = read_label_groups(args.label_config)
        transport = make_transport(args.endpoint, args.timeout, os.getenv("EMR_BENCHMARK_BASIC_USER"), os.getenv("EMR_BENCHMARK_BASIC_PASSWORD"))
        predictions, report = benchmark_service(json.loads(args.tasks.read_text(encoding="utf-8-sig")), groups,
                                                args.label_config.read_text(encoding="utf-8-sig"), args.project,
                                                transport, args.repeats, args.warmup)
        report.update({"run_name": args.run_name, "declared_model_artifact_id": args.model_artifact_id,
                       "dataset_kind": args.dataset_kind, "measured_at_utc": datetime.now(timezone.utc).isoformat(),
                       "request_timeout_seconds": args.timeout,
                       "endpoint_sha256": hashlib.sha256(args.endpoint.encode("utf-8")).hexdigest(),
                       "project_sha256": hashlib.sha256(args.project.encode("utf-8")).hexdigest(),
                       "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        report["inputs"] = {name: {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in inputs.items()}
        for path, payload in ((args.output, predictions), (args.benchmark, report)):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except (ValueError, OSError, ET.ParseError) as exc:
        parser.error(str(exc))
    print(f"{args.run_name}: tasks={len(predictions)}, measured requests={report['measured_task_runs']}, failures={report['failed_task_runs']}; dataset={args.dataset_kind}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
