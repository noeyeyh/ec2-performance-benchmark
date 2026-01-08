# -*- coding: utf-8 -*-
"""
Naver News Benchmark Crawler (full text included)

- Collect URLs from Naver news search result pages
- Fetch articles concurrently (aiohttp) with polite random delay
- Parse full text:
    * New UI: article#dic_area  (n.news.naver.com)
    * Old UI fallback: #articleBodyContents (news.naver.com)
- Save outputs:
    * <out>/articles.jsonl  (includes full text)
    * <out>/summary.json

Usage example:
  python crawl.py --query "삼성전자" --pages 3 --per_page 10 --concurrency 2 --min_delay_ms 400 --max_delay_ms 900 --out out_samsung

Then inspect:
  python - <<'PY'
import json
p="out_samsung/articles.jsonl"
with open(p,encoding="utf-8") as f:
    for line in f:
        r=json.loads(line)
        if r["ok"]:
            print(r["url"])
            print(r["text_len"])
            print(r["text"][:500])
            break
PY
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import random
import re
import socket
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

import aiohttp
from bs4 import BeautifulSoup


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://search.naver.com/",
}


def now_ts() -> int:
    return int(time.time())


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    xs = sorted(values)
    if len(xs) == 1:
        return float(xs[0])
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return float(xs[f])
    d0 = xs[f] * (c - k)
    d1 = xs[c] * (k - f)
    return float(d0 + d1)


def clean_text(s: str) -> str:
    s = s.replace("\u200b", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def normalize_naver_news_url(url: str) -> Optional[str]:
    if not url:
        return None
    u = url.split("#")[0]
    try:
        parsed = urlparse(u)
    except Exception:
        return None

    host = (parsed.netloc or "").lower()

    # Prefer canonical paths, drop tracking query when possible
    if "n.news.naver.com" in host:
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    if "m.news.naver.com" in host:
        # mobile still often has dic_area; keep path
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    if "news.naver.com" in host:
        # old style sometimes needs query; keep query
        if parsed.query:
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{parsed.query}"
        return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

    return None


def make_search_urls(query: str, pages: int) -> List[str]:
    q = quote(query)
    urls = []
    for i in range(pages):
        start = 1 + i * 10
        urls.append(
            f"https://search.naver.com/search.naver?where=news&sm=tab_pge&query={q}&start={start}"
        )
    return urls


def parse_search_links(html: str, per_page_cap: int) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    found: List[str] = []

    # Collect only Naver news domains (this is important to avoid “본문 없음” pages)
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        if ("n.news.naver.com" in href) or ("news.naver.com" in href) or ("m.news.naver.com" in href):
            norm = normalize_naver_news_url(href)
            if norm:
                found.append(norm)

    # Deduplicate, keep order
    seen = set()
    uniq: List[str] = []
    for u in found:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    return uniq[:per_page_cap]


def extract_title(soup: BeautifulSoup) -> str:
    el = soup.select_one("h2#title_area")
    if el:
        return clean_text(el.get_text(" ", strip=True))

    el = soup.select_one("h3#articleTitle")
    if el:
        return clean_text(el.get_text(" ", strip=True))

    if soup.title:
        return clean_text(soup.title.get_text(" ", strip=True))
    return ""


def extract_published(soup: BeautifulSoup) -> str:
    el = soup.select_one("span.media_end_head_info_datestamp_time")
    if el:
        return clean_text(el.get_text(" ", strip=True))

    el = soup.select_one(".t11")
    if el:
        return clean_text(el.get_text(" ", strip=True))
    return ""


def extract_press(soup: BeautifulSoup) -> str:
    el = soup.select_one("a.media_end_head_top_logo img")
    if el and el.get("alt"):
        return clean_text(el.get("alt"))

    el = soup.select_one("em.media_end_linked_more_point")
    if el:
        return clean_text(el.get_text(" ", strip=True))

    el = soup.select_one("#footer address a")
    if el:
        return clean_text(el.get_text(" ", strip=True))

    return ""


def extract_body_text(soup: BeautifulSoup) -> str:
    # New UI (your screenshot): <article id="dic_area">
    body = soup.select_one("article#dic_area")
    if body:
        for tag in body.select("script, style, iframe"):
            tag.decompose()
        return clean_text(body.get_text("\n", strip=True))

    # Old UI: #articleBodyContents
    body = soup.select_one("#articleBodyContents")
    if body:
        for tag in body.select("script, style, iframe"):
            tag.decompose()
        txt = body.get_text("\n", strip=True)
        txt = txt.replace("flash 오류를 우회하기 위한 함수 추가 function _flash_removeCallback() {}", "")
        return clean_text(txt)

    return ""


@dataclass
class ArticleResult:
    url: str
    status: int
    ok: bool
    title: str
    published: str
    press: str
    text: str
    text_len: int
    fetch_s: float
    parse_s: float
    err: str = ""


async def fetch_text(session: aiohttp.ClientSession, url: str, timeout_s: float) -> Tuple[int, str]:
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.get(url, timeout=timeout, allow_redirects=True) as resp:
        status = resp.status
        text = await resp.text(errors="ignore")
        return status, text


async def worker_one(
    session: aiohttp.ClientSession,
    url: str,
    timeout_s: float,
    delay_ms_range: Tuple[int, int],
    sem: asyncio.Semaphore,
    debug_why: bool = False,
) -> ArticleResult:
    async with sem:
        lo, hi = delay_ms_range
        if hi > 0:
            await asyncio.sleep(random.uniform(lo, hi) / 1000.0)

        t0 = time.perf_counter()
        try:
            status, html = await fetch_text(session, url, timeout_s)
        except Exception as e:
            t_fetch = time.perf_counter() - t0
            return ArticleResult(
                url=url,
                status=0,
                ok=False,
                title="",
                published="",
                press="",
                text="",
                text_len=0,
                fetch_s=t_fetch,
                parse_s=0.0,
                err=f"fetch_error: {type(e).__name__}: {e}",
            )
        t_fetch = time.perf_counter() - t0

        t1 = time.perf_counter()
        try:
            soup = BeautifulSoup(html, "lxml")
            title = extract_title(soup)
            published = extract_published(soup)
            press = extract_press(soup)
            text = extract_body_text(soup)
            text_len = len(text)
            ok = (status == 200) and (text_len > 0)
            t_parse = time.perf_counter() - t1

            if debug_why and status == 200 and text_len == 0:
                dic = soup.select_one("article#dic_area")
                old = soup.select_one("#articleBodyContents")
                page_title = soup.title.get_text(" ", strip=True) if soup.title else "NO_TITLE"
                print(
                    f"[WHY] url={url} dic_area={bool(dic)} old_body={bool(old)} "
                    f"title={page_title[:50]}"
                )

            return ArticleResult(
                url=url,
                status=status,
                ok=ok,
                title=title,
                published=published,
                press=press,
                text=text,
                text_len=text_len,
                fetch_s=t_fetch,
                parse_s=t_parse,
                err="",
            )
        except Exception as e:
            t_parse = time.perf_counter() - t1
            return ArticleResult(
                url=url,
                status=status,
                ok=False,
                title="",
                published="",
                press="",
                text="",
                text_len=0,
                fetch_s=t_fetch,
                parse_s=t_parse,
                err=f"parse_error: {type(e).__name__}: {e}",
            )


async def collect_urls(query: str, pages: int, per_page: int, timeout_s: float) -> List[str]:
    urls: List[str] = []
    search_urls = make_search_urls(query, pages)

    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS) as session:
        for su in search_urls:
            try:
                status, html = await fetch_text(session, su, timeout_s)
                if status != 200:
                    continue
                got = parse_search_links(html, per_page_cap=per_page)
                urls.extend(got)
            except Exception:
                continue

    seen = set()
    uniq: List[str] = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            uniq.append(u)
    return uniq


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_summary(
    query: str,
    urls_total: int,
    results: List[ArticleResult],
    total_seconds: float,
    concurrency: int,
    delay_ms_range: Tuple[int, int],
) -> Dict[str, Any]:
    ok_cnt = sum(1 for r in results if r.ok)
    fail_cnt = urls_total - ok_cnt
    success_rate = (ok_cnt / urls_total * 100.0) if urls_total else 0.0

    fetch_times = [r.fetch_s for r in results if r.fetch_s > 0]
    parse_times = [r.parse_s for r in results if r.ok and r.parse_s > 0]

    return {
        "hostname": socket.gethostname(),
        "arch": platform.machine(),
        "platform": platform.platform(),
        "timestamp": now_ts(),
        "query": query,
        "urls_total": urls_total,
        "ok": ok_cnt,
        "fail": fail_cnt,
        "success_rate": round(success_rate, 2),
        "total_seconds": round(total_seconds, 3),
        "throughput_urls_per_s": round((urls_total / total_seconds) if total_seconds > 0 else 0.0, 3),
        "fetch_avg_s": round(sum(fetch_times) / len(fetch_times), 4) if fetch_times else 0.0,
        "fetch_p95_s": round(percentile(fetch_times, 0.95), 4) if fetch_times else 0.0,
        "fetch_p99_s": round(percentile(fetch_times, 0.99), 4) if fetch_times else 0.0,
        "parse_avg_s": round(sum(parse_times) / len(parse_times), 4) if parse_times else 0.0,
        "parse_p95_s": round(percentile(parse_times, 0.95), 4) if parse_times else 0.0,
        "parse_p99_s": round(percentile(parse_times, 0.99), 4) if parse_times else 0.0,
        "concurrency": concurrency,
        "delay_ms_range": [delay_ms_range[0], delay_ms_range[1]],
    }


async def run_bench(args: argparse.Namespace) -> None:
    ensure_dir(args.out)

    # 1) URL collection
    urls = await collect_urls(args.query, args.pages, args.per_page, args.timeout)
    print(f"[INFO] collected_urls={len(urls)}")

    # 2) Fetch + parse
    sem = asyncio.Semaphore(args.concurrency)
    t0 = time.perf_counter()

    async with aiohttp.ClientSession(headers=DEFAULT_HEADERS) as session:
        results = await asyncio.gather(
            *[
                worker_one(
                    session=session,
                    url=u,
                    timeout_s=args.timeout,
                    delay_ms_range=(args.min_delay_ms, args.max_delay_ms),
                    sem=sem,
                    debug_why=args.debug_why,
                )
                for u in urls
            ]
        )

    total_seconds = time.perf_counter() - t0

    # 3) Save outputs (includes full text)
    rows: List[Dict[str, Any]] = []
    for r in results:
        rows.append(
            {
                "url": r.url,
                "status": r.status,
                "ok": r.ok,
                "title": r.title,
                "published": r.published,
                "press": r.press,
                "text_len": r.text_len,
                "text": r.text,  # ✅ full article text saved here
                "fetch_s": round(r.fetch_s, 6),
                "parse_s": round(r.parse_s, 6),
                "err": r.err,
            }
        )

    summary = build_summary(
        query=args.query,
        urls_total=len(urls),
        results=results,
        total_seconds=total_seconds,
        concurrency=args.concurrency,
        delay_ms_range=(args.min_delay_ms, args.max_delay_ms),
    )

    write_jsonl(os.path.join(args.out, "articles.jsonl"), rows)
    write_json(os.path.join(args.out, "summary.json"), summary)

    print("[DONE] summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, help="Search keyword (e.g., 삼성전자)")
    p.add_argument("--pages", type=int, default=3, help="Number of search pages to fetch")
    p.add_argument("--per_page", type=int, default=10, help="Max links per search page (cap)")
    p.add_argument("--concurrency", type=int, default=3, help="Concurrent article requests")
    p.add_argument("--timeout", type=float, default=10.0, help="HTTP timeout seconds")
    p.add_argument("--min_delay_ms", type=int, default=200, help="Min random delay before each request")
    p.add_argument("--max_delay_ms", type=int, default=600, help="Max random delay before each request")
    p.add_argument("--out", type=str, default="out", help="Output directory")
    p.add_argument(
        "--debug_why",
        action="store_true",
        help="Print WHY lines when status=200 but body text is empty",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_delay_ms < 0 or args.max_delay_ms < 0 or args.max_delay_ms < args.min_delay_ms:
        raise ValueError("Invalid delay range")
    asyncio.run(run_bench(args))


if __name__ == "__main__":
    main()