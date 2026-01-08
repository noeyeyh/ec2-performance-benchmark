# brief_local.py
# -*- coding: utf-8 -*-

import argparse
import json
import time
import subprocess
from pathlib import Path


SYSTEM_PROMPT = """\
너는 한국어로만 응답하는 금융 애널리스트다.
절대로 중국어, 영어를 사용하지 말고 반드시 한국어로만 작성하라.

다음은 '신한투자증권 및 신한금융그룹 관련 최신 뉴스 기사 전문'이다.

이 기사들을 종합해
"투자자용 데일리 기업 브리핑"을 작성하라.
너는 질문은 하지마.

[출력 형식]
1. 한 줄 결론 (한국어)
2. 핵심 요약 5줄 (한국어, 중복 제거)


[중요 규칙]
- 출력 언어: 한국어 ONLY
- 기사에 없는 사실을 추정하지 말 것
- 간결하지만 분석적으로
"""
# ====== CONFIG ======
INPUT_JSONL = "out_shinhan/articles.jsonl"
OUTPUT_JSON = "out_shinhan/summary.json"
MODEL_NAME = "mistral:7b-instruct-q4_0"
MAX_ARTICLES = 10
# ====================

def load_articles(jsonl_path: Path, max_articles=10):
    articles = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            articles.append(json.loads(line))
            if len(articles) >= max_articles:
                break
    return articles


def build_prompt(articles):
    lines = []
    for i, a in enumerate(articles, 1):
        lines.append(f"[기사 {i}]")
        lines.append(f"제목: {a['title']}")
        lines.append(f"언론사: {a.get('press','')}")
        lines.append(f"발행일: {a.get('published','')}")
        lines.append(f"본문:\n{a['text']}\n")
    return SYSTEM_PROMPT + "\n\n" + "\n".join(lines)


def run_ollama(model: str, prompt: str) -> str:
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt,
        text=True,
        capture_output=True
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return proc.stdout.strip()

def main():
    articles = load_articles(Path(INPUT_JSONL), MAX_ARTICLES)
    prompt = build_prompt(articles)

    t0 = time.perf_counter()
    result = run_ollama(MODEL_NAME, prompt)
    elapsed = time.perf_counter() - t0

    output = {
        "model": MODEL_NAME,
        "articles_used": len(articles),
        "elapsed_seconds": round(elapsed, 3),
        "summary": result
    }

    Path(OUTPUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_JSON).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"[DONE] local LLM summary generated in {output['elapsed_seconds']}s")

if __name__ == "__main__":
    main()