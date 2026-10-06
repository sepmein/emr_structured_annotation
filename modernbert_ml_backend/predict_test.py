import csv
import statistics
import time
from datetime import datetime
from pathlib import Path

import requests
import urllib3


LS_URL = "http://192.168.204.97:8080/labelstudio"
ML_BACKEND_ID = 6
PROJECT_ID = 104

COOKIE_HEADER = "csrftoken=HA67Rkdkt51bY5SATaKboImNhTnoJFt4; sessionid=.eJxVT0tugzAQvYvXYA2Diccsu-8Z0PgHbpAdYZDaRL17Q5VNlu-v9xBH8mIUxiL6qHwbMJpWdbpvDUfXgqfBeAMEuheNKNvMOd15TyVPt6sYu0asXPdpLXPKT6jJAGgalIQLGULViImPfZmOGrbpf6oD8UZadteQT8V_cZ6LdCXvW7LytMiXWuVn8WH9eHnfChauy5m2FMlh1EzIvUVyOgZmRSqigg4N-4tjYyP0QzS99RSUt9Hi8zpHpc7SGmo9r4XvW9p-xIiDQQAJv39B1F0i:1x4YwU:YzDWtytVvfwmpCpn5eyRjk9B_pPiD-RlcgPJ2f7hOgc"

VERIFY_TLS = False

# 预测测试轮数。每一轮执行 PREDICT_RUNS 次预测。
TEST_BATCH = 1
PREDICT_RUNS = 20
REQUEST_TIMEOUT = (30, 600)

OUTPUT_DIR = Path("benchmark_results")


def parse_cookie_header(cookie_header):
    cookie_header = cookie_header.strip().replace("\\_", "_")

    if cookie_header.lower().startswith("cookie:"):
        cookie_header = cookie_header.split(":", 1)[1].strip()

    cookies = {}
    for item in cookie_header.split(";"):
        name, separator, value = item.strip().partition("=")
        if separator and name:
            cookies[name.strip()] = value.strip()

    missing = {"csrftoken", "sessionid"} - cookies.keys()
    if missing:
        raise ValueError(f"Cookie 缺少字段：{', '.join(sorted(missing))}")

    return cookie_header, cookies


def build_headers(cookie_header, cookies, method):
    headers = {
        "Accept": "application/json",
        "Cookie": cookie_header,
    }

    if method.upper() == "POST":
        headers.update(
            {
                "Content-Type": "application/json",
                "X-CSRFToken": cookies["csrftoken"],
                "Origin": LS_URL.removesuffix("/labelstudio"),
                "Referer": f"{LS_URL}/projects/{PROJECT_ID}/settings/ml",
            }
        )

    return headers


def print_response(label, response, elapsed):
    print(f"\n{'=' * 80}")
    print(f"[{label}]")
    print(f"完整的响应：{response.text}")
    print(f"请求方法: {response.request.method}")
    print(f"请求地址: {response.url}")
    print(f"状态码: {response.status_code}")
    print(f"耗时: {elapsed:.3f} 秒")
    print(f"{'=' * 80}\n")


def get_error_message(response, response_data):
    if response.ok:
        return ""

    if isinstance(response_data, dict):
        return str(
            response_data.get("detail")
            or response_data.get("error_message")
            or response_data.get("error")
            or f"HTTP {response.status_code}"
        )

    return f"HTTP {response.status_code}: {response.text[:1000]}"


def percentile(values, ratio):
    values = sorted(values)
    index = min(round((len(values) - 1) * ratio), len(values) - 1)
    return values[index]


def main():
    if "<" in COOKIE_HEADER or ">" in COOKIE_HEADER:
        raise ValueError("请填写实际的 COOKIE_HEADER。")

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    cookie_header, cookies = parse_cookie_header(COOKIE_HEADER)

    session = requests.Session()
    session.verify = VERIFY_TLS

    whoami = session.get(
        f"{LS_URL}/api/current-user/whoami",
        headers=build_headers(cookie_header, cookies, "GET"),
        timeout=(30, 30),
    )
    print_response("whoami 登录验证", whoami, whoami.elapsed.total_seconds())
    whoami.raise_for_status()

    results = []

    for batch in range(1, TEST_BATCH + 1):
        print(f"开始第 {batch}/{TEST_BATCH} 轮预测测试")

        for run in range(1, PREDICT_RUNS + 1):
            started_at = time.perf_counter()

            try:
                response = session.post(
                    f"{LS_URL}/api/ml/{ML_BACKEND_ID}/predict/test?random=true",
                    headers=build_headers(cookie_header, cookies, "POST"),
                    json={},
                    timeout=REQUEST_TIMEOUT,
                )
                elapsed = time.perf_counter() - started_at
                print_response(f"第 {batch} 轮预测，第 {run} 次", response, elapsed)

                try:
                    response_data = response.json()
                except ValueError:
                    response_data = None

                error = get_error_message(response, response_data)
                predict_text = ""

                if response.ok:
                    try:
                        predict_text = response_data["task"]["data"]["text"]
                        if not isinstance(predict_text, str):
                            predict_text = str(predict_text)
                    except (KeyError, TypeError) as exc:
                        error = f"预测响应中未找到 task.data.text：{exc}"

                results.append(
                    {
                        "batch": batch,
                        "run": run,
                        "operation": "predict",
                        "url": response.url,
                        "status_code": response.status_code,
                        "elapsed_seconds": round(elapsed, 3),
                        "error": error,
                        "predict_text": predict_text,
                        "predict_text_length": len(predict_text),
                    }
                )

            except requests.RequestException as exc:
                elapsed = time.perf_counter() - started_at
                results.append(
                    {
                        "batch": batch,
                        "run": run,
                        "operation": "predict",
                        "url": f"{LS_URL}/api/ml/{ML_BACKEND_ID}/predict/test?random=true",
                        "status_code": "",
                        "elapsed_seconds": round(elapsed, 3),
                        "error": str(exc),
                        "predict_text": "",
                        "predict_text_length": 0,
                    }
                )
                print(f"第 {batch} 轮预测，第 {run} 次请求异常：{exc}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / f"predict_benchmark_{datetime.now():%Y%m%d_%H%M%S}.csv"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    durations = [
        float(item["elapsed_seconds"])
        for item in results
        if not item["error"]
    ]

    if durations:
        print(
            f"\n预测汇总：次数={len(durations)}，"
            f"平均={statistics.mean(durations):.3f}s，"
            f"P50={percentile(durations, 0.50):.3f}s，"
            f"P95={percentile(durations, 0.95):.3f}s，"
            f"P99={percentile(durations, 0.99):.3f}s"
        )

    print(f"结果文件：{csv_path.resolve()}")


if __name__ == "__main__":
    main()