"""YouTube comment extraction for the NLP+RAG data pipeline.

This module scrapes top-level YouTube comments from a list of video IDs and
outputs a normalized tabular dataset with the exact schema consumed by
`1_preprocess.py`:

- `video_id`
- `comment_id`
- `author`
- `comment_text`
- `likes`
- `published_at`

Inputs:
- YouTube Data API key (from env or `.env`)
- List of video IDs
- Target row count

Output:
- A CSV file (`youtube_comments_10k_v2.csv` by default) and a pandas DataFrame.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


LOGGER = logging.getLogger(__name__)

DEFAULT_VIDEO_IDS: list[str] = [
    # Tech/ML creator content with active engagement and broad NLP-friendly discourse.
    "cYwioeHu_OU",
    # AI/productivity topic with high practical-comment density.
    "Lfzu74XDyco",
    # Engineering/technology discussion variety for topical breadth.
    "TiS6vnju_mI",
    # General educational/analysis-oriented audience comments for domain diversity.
    "QOcP5OvSwlI",
]

SCHEMA_COLUMNS: list[str] = [
    "video_id",
    "comment_id",
    "author",
    "comment_text",
    "likes",
    "published_at",
]


def _load_api_key() -> str:
    env_var_name = "YOUTUBE_API_KEY"
    api_key = os.environ.get(env_var_name, "").strip()
    if api_key:
        return api_key

    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == env_var_name:
                parsed = value.strip().strip('"').strip("'")
                if parsed:
                    return parsed

    raise ValueError(
        f"Missing required API key: {env_var_name}. Checked os.environ and {env_path}."
    )


def _http_error_reason(error: HttpError) -> str:
    try:
        payload = json.loads(error.content.decode("utf-8"))
        return str(payload.get("error", {}).get("message", "unknown reason"))
    except Exception:
        return str(error)


def _execute_with_retry(request: Any, video_id: str) -> dict[str, Any]:
    backoff_seconds = [1, 2, 4]
    for attempt in range(1, 4):
        try:
            return request.execute()
        except HttpError as error:
            status = getattr(error.resp, "status", None)
            reason = _http_error_reason(error)

            if status in {400, 403}:
                raise

            if status is not None and 500 <= status < 600 and attempt < 3:
                wait_for = backoff_seconds[attempt - 1]
                LOGGER.warning(
                    "Transient API error for video_id=%s (status=%s, reason=%s). "
                    "Retrying in %ss (attempt %s/3).",
                    video_id,
                    status,
                    reason,
                    wait_for,
                    attempt,
                )
                time.sleep(wait_for)
                continue

            raise RuntimeError(
                "YouTube API request failed for "
                f"video_id={video_id}, status={status}, reason={reason}"
            ) from error

    raise RuntimeError(f"Unexpected retry loop exit for video_id={video_id}")


def scrape_comments(
    api_key: str,
    video_ids: list[str],
    target_count: int = 10000,
) -> pd.DataFrame:
    youtube = build("youtube", "v3", developerKey=api_key)
    rows: list[dict[str, Any]] = []

    for video_id in video_ids:
        if len(rows) >= target_count:
            break

        LOGGER.info("Scraping video_id=%s", video_id)
        page_token: str | None = None

        while len(rows) < target_count:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=page_token,
                textFormat="plainText",
            )

            try:
                response = _execute_with_retry(request, video_id)
            except HttpError as error:
                status = getattr(error.resp, "status", None)
                reason = _http_error_reason(error)

                if status == 403 and "disable" in reason.lower():
                    LOGGER.info(
                        "Skipping video_id=%s because comments are disabled or unavailable "
                        "(status=%s, reason=%s)",
                        video_id,
                        status,
                        reason,
                    )
                    break

                raise RuntimeError(
                    "YouTube API request failed for "
                    f"video_id={video_id}, status={status}, reason={reason}"
                ) from error

            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                top_level = snippet.get("topLevelComment", {}).get("snippet", {})

                text_display = (top_level.get("textDisplay") or "").strip()
                if not text_display:
                    continue

                rows.append(
                    {
                        "video_id": snippet.get("videoId"),
                        "comment_id": item.get("id"),
                        "author": top_level.get("authorDisplayName"),
                        "comment_text": text_display,
                        "likes": top_level.get("likeCount"),
                        "published_at": top_level.get("publishedAt"),
                    }
                )

                if len(rows) >= target_count:
                    break

            page_token = response.get("nextPageToken")
            if not page_token:
                break

    return pd.DataFrame(rows, columns=SCHEMA_COLUMNS)


def run_extraction(
    api_key: str,
    video_ids: list[str],
    target_count: int = 10000,
    output_csv_path: str = "youtube_comments_10k_v2.csv",
) -> pd.DataFrame:
    comments_df = scrape_comments(api_key=api_key, video_ids=video_ids, target_count=target_count)
    comments_df.to_csv(output_csv_path, index=False, encoding="utf-8-sig")
    return comments_df


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    start = time.perf_counter()
    api_key = _load_api_key()
    comments_df = run_extraction(
        api_key=api_key,
        video_ids=DEFAULT_VIDEO_IDS,
        target_count=10000,
        output_csv_path="youtube_comments_10k_v2.csv",
    )
    elapsed_seconds = time.perf_counter() - start
    per_video_counts = comments_df["video_id"].value_counts().to_dict() if not comments_df.empty else {}

    print(
        "\n".join(
            [
                "--- Scraping Summary ---",
                f"Total rows: {len(comments_df)}",
                f"Per-video counts: {per_video_counts}",
                f"Elapsed seconds: {elapsed_seconds:.2f}",
                "Output CSV: youtube_comments_10k_v2.csv",
            ]
        )
    )


if __name__ == "__main__":
    main()

