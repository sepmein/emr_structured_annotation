"""Fetch WHO Disease Outbreak News records and render them to PDF.

The script uses WHO's public Disease Outbreak News API, writes a JSON snapshot
and an HTML report, then uses local Chrome/Edge in headless mode to print the
HTML report to PDF.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import html
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import warnings


WHO_DON_API = "https://www.who.int/api/emergencies/diseaseoutbreaknews"
WHO_DON_PAGE = "https://www.who.int/emergencies/disease-outbreak-news"
API_QUERY = (
    "sf_provider=dynamicProvider372"
    "&sf_culture=en"
    "&$orderby=PublicationDateAndTime%20desc"
    "&$expand=EmergencyEvent"
    "&$select=Title,TitleSuffix,OverrideTitle,UseOverrideTitle,ItemDefaultUrl,"
    "FormattedDate,PublicationDateAndTime,Summary,DonId"
)


def clean_text(value: str | None) -> str:
    if not value:
        return ""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        text = BeautifulSoup(value, "html.parser").get_text(" ", strip=True)
    return re.sub(r"\s+", " ", text).strip()


def item_title(item: dict) -> str:
    if item.get("UseOverrideTitle") and item.get("OverrideTitle"):
        return item["OverrideTitle"]
    title = item.get("Title") or "Disease outbreak news"
    suffix = item.get("TitleSuffix") or ""
    return f"{title} - {suffix}" if suffix else title


def item_url(item: dict) -> str:
    path = item.get("ItemDefaultUrl") or ""
    if path.startswith("http"):
        return path
    return f"{WHO_DON_PAGE}/item{path}"


def fetch_total_count() -> int:
    url = f"{WHO_DON_API}?sf_provider=dynamicProvider372&sf_culture=en&$count=true&$top=0"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    count = response.json().get("@odata.count")
    if not isinstance(count, int):
        raise RuntimeError("WHO API did not return @odata.count")
    return count


def fetch_batch(skip: int, top: int) -> tuple[int, list[dict]]:
    url = f"{WHO_DON_API}?{API_QUERY}&$top={top}&$skip={skip}"
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=45)
            response.raise_for_status()
            return skip, response.json().get("value", [])
        except Exception as exc:  # retry transient network/API failures
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Failed to fetch WHO DON batch skip={skip}") from last_error


def fetch_records(batch_size: int = 100, workers: int = 8) -> list[dict]:
    total = fetch_total_count()
    skips = list(range(0, total, batch_size))
    items: list[dict] = []

    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = [executor.submit(fetch_batch, skip, batch_size) for skip in skips]
        for index, job in enumerate(futures.as_completed(jobs), 1):
            skip, batch = job.result()
            print(f"Fetched batch {index}/{len(jobs)} skip={skip} rows={len(batch)}")
            items.extend(batch)

    items.sort(key=lambda row: row.get("PublicationDateAndTime") or "", reverse=True)
    records = []
    for index, item in enumerate(items, 1):
        records.append(
            {
                "index": index,
                "don_id": item.get("DonId") or "",
                "date": item.get("FormattedDate") or (item.get("PublicationDateAndTime") or "")[:10],
                "title": item_title(item),
                "url": item_url(item),
                "summary": clean_text(item.get("Summary")),
            }
        )
    return records


def build_html(records: list[dict], generated_at: str) -> str:
    css = """
@page { size: A4; margin: 14mm 12mm; }
* { box-sizing: border-box; }
body { font-family: Arial, "Noto Sans SC", "Microsoft YaHei", sans-serif; color: #202124; line-height: 1.42; }
h1 { font-size: 22px; margin: 0 0 6px; color: #005eb8; }
.meta { font-size: 10px; color: #5f6368; margin-bottom: 12px; }
.notice { border-left: 4px solid #005eb8; padding: 8px 10px; background: #eef6ff; font-size: 10.5px; margin-bottom: 14px; }
.item { break-inside: avoid; border-top: 1px solid #d9e1e8; padding: 8px 0 7px; }
.h { font-size: 12px; font-weight: 700; margin-bottom: 2px; }
.date { color: #005eb8; font-size: 10px; font-weight: 700; }
.url { font-size: 9px; color: #3867a6; overflow-wrap: anywhere; }
.summary { font-size: 10px; margin-top: 4px; }
.no-summary { color: #7a7a7a; font-style: italic; }
"""
    parts = [
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        "<title>WHO Disease Outbreak News Index</title>"
        f"<style>{css}</style></head><body>",
        "<h1>WHO Disease Outbreak News</h1>",
        (
            '<div class="meta">'
            f"Source: {html.escape(WHO_DON_PAGE)}<br>"
            f"Generated: {html.escape(generated_at)}<br>"
            f"Total records captured from WHO API: {len(records)}"
            "</div>"
        ),
        (
            '<div class="notice">'
            "This PDF is an index of WHO Disease Outbreak News records: publication date, "
            "title, original WHO link, and the public summary field when available. "
            "It is not a verbatim republication of full WHO articles."
            "</div>"
        ),
    ]

    for record in records:
        summary = record["summary"] or "No summary field available from the listing API."
        summary_class = "summary" if record["summary"] else "summary no-summary"
        don_id = f" ({html.escape(record['don_id'])})" if record["don_id"] else ""
        parts.append(
            '<section class="item">'
            f'<div class="date">{html.escape(record["date"])}{don_id}</div>'
            f'<div class="h">{record["index"]}. {html.escape(record["title"])}</div>'
            f'<div class="url">{html.escape(record["url"])}</div>'
            f'<div class="{summary_class}">{html.escape(summary)}</div>'
            "</section>"
        )

    parts.append("</body></html>")
    return "\n".join(parts)


def find_browser(explicit_path: str | None = None) -> str:
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)

    candidates.extend(
        [
            shutil.which("chrome"),
            shutil.which("chrome.exe"),
            shutil.which("msedge"),
            shutil.which("msedge.exe"),
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ]
    )

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    raise FileNotFoundError("Could not find Chrome or Edge. Pass --browser-path explicitly.")


def print_pdf(browser_path: str, html_path: Path, pdf_path: Path) -> None:
    file_url = html_path.resolve().as_uri()
    command = [
        browser_path,
        "--headless",
        "--disable-gpu",
        "--no-sandbox",
        f"--print-to-pdf={pdf_path.resolve()}",
        file_url,
    ]
    subprocess.run(command, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="output", help="Output directory. Default: output")
    parser.add_argument("--stamp", default=datetime.now().strftime("%Y-%m-%d"), help="Filename date stamp.")
    parser.add_argument("--browser-path", help="Path to chrome.exe or msedge.exe.")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API fetch workers.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"who_disease_outbreak_news_index_{args.stamp}"

    records = fetch_records(workers=args.workers)
    json_path = out_dir / f"{stem}.json"
    html_path = out_dir / f"{stem}.html"
    pdf_path = out_dir / f"{stem}.pdf"

    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(build_html(records, datetime.now().isoformat(timespec="seconds")), encoding="utf-8")

    browser_path = find_browser(args.browser_path)
    print_pdf(browser_path, html_path, pdf_path)

    print(f"Records: {len(records)}")
    print(f"JSON: {json_path.resolve()}")
    print(f"HTML: {html_path.resolve()}")
    print(f"PDF: {pdf_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
