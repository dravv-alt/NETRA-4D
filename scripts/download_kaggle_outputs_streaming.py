"""Download Kaggle kernel outputs without buffering whole videos in memory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kernel", help="owner/kernel-slug")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--only", action="append", default=[], help="Download only paths containing this text")
    args = parser.parse_args()

    from kaggle import api

    owner, slug = args.kernel.split("/", 1)
    response = api.process_response(api.kernel_output_with_http_info(owner, slug))
    args.destination.mkdir(parents=True, exist_ok=True)
    for item in response["files"]:
        if args.only and not any(value in item["fileName"] for value in args.only):
            continue
        destination = args.destination / item["fileName"]
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"present: {item['fileName']} ({destination.stat().st_size} bytes)", flush=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".part")
        offset = temporary.stat().st_size if temporary.is_file() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        with requests.get(item["url"], headers=headers, stream=True, timeout=(30, 300)) as download:
            download.raise_for_status()
            if offset and download.status_code != 206:
                offset = 0
            content_range = download.headers.get("content-range", "")
            expected_total = (
                int(content_range.rsplit("/", 1)[1])
                if "/" in content_range and content_range.rsplit("/", 1)[1].isdigit()
                else offset + int(download.headers.get("content-length", 0))
            )
            written = offset
            with temporary.open("ab" if offset else "wb") as output:
                for chunk in download.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        output.write(chunk)
                        written += len(chunk)
        if expected_total and written != expected_total:
            raise RuntimeError(
                f"Incomplete download for {item['fileName']}: {written} of {expected_total} bytes"
            )
        os.replace(temporary, destination)
        print(f"downloaded: {item['fileName']} ({written} bytes)", flush=True)

    log = response.get("log")
    if log:
        (args.destination / f"{slug}.log").write_text(log, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
